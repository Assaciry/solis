
from __future__ import annotations

from typing import Optional, Union, Dict, Callable, List, Any, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F

import os
import numpy as np
import pandas as pd

class SimulatedTrajectoryDataset(Dataset):
    """
    A simple trajectory dataset for an ODEEquation.

    Generates n_trajectories rollouts over a given time interval, using either:
      - SciPy IVP solve (high-accuracy ground truth, CPU/NumPy)
      - Torch RK4/Euler (fast, differentiable in PyTorch)

    Returns PyTorch tensors with consistent shapes:
      t:  (T, 1)
      y:  (T, D)           -- simulated states in first-order coordinates
      y0: (D,)             -- initial state
      exo: Optional[dict]  -- each exogenous signal as (T,1) tensor
      gt:  Optional (T, D) -- user-provided ground-truth trajectory (if supplied)

    Notes
    -----
    - D = ode_equation.total_state_dim
    - If you want trajectories to keep autograd graphs (e.g., for gradient
      through integration), set keep_graph=True and use backend='torch_rk4'.
      Beware of memory usage when keeping graphs for many long trajectories.
    """

    def __init__(
        self,
        ode_equation,
        n_trajectories: int,
        t0: float,
        tf: float,
        *,
        T: Optional[int] = None,
        t_eval: Optional[np.ndarray] = None,
        backend: str = "torch",                  # 'torch' or 'numpy'
        method: str = "rk4",                     # 'rk4' or 'euler' (torch backend)
        dt: Optional[float] = None,              # only used if T/t_eval not given (torch backend)
        y0_sampler=None,                         # callable -> np.ndarray OR torch.Tensor of shape (D,)
        y0_set: Optional[Union[np.ndarray, torch.Tensor]] = None,  # (N,D) or (D,)
        exogenous_np: Optional[Dict[str, Callable[[float], float]]] = None,
        exogenous_torch: Optional[Dict[str, Callable]] = None,     # refine signature if you like
        params_override: Optional[Dict] = None,
        ground_truth: Optional[torch.Tensor] = None,               # shape (N,T,D)
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
        keep_graph: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        traj_id_base: int = 0, # Used in case of multi-simulated trajectory to assign proper trajectory IDs
    ):
        super().__init__()
        self.eq = ode_equation
        self.N = int(n_trajectories)
        self.params_override = params_override
        self.backend = backend.lower()
        self.method = method.lower()
        self.keep_graph = keep_graph
        self.memory_size = None

        if self.backend not in ("torch", "numpy"):
            raise ValueError("backend must be 'torch' or 'numpy'")
        if self.method not in ("rk4", "euler"):
            raise ValueError("method must be 'rk4' or 'euler'")

        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)
        self.dtype = dtype

        # Compile once to know state dimension
        D = self.eq.total_state_dim
        self.state_dim = D

        # Trajectory ID base
        self.traj_id_base = traj_id_base

        # Time grid
        if t_eval is not None:
            t_np = np.asarray(t_eval, dtype=float)
            assert t_np.ndim == 1, "t_eval must be 1D"
            assert t_np[0] == t0 and t_np[-1] == tf, "t_eval must span [t0, tf]"
        else:
            if T is None:
                if dt is None:
                    raise ValueError("Provide T or t_eval (or dt for torch backend).")
                T = int(np.floor((tf - t0) / float(dt))) + 1
            t_np = np.linspace(t0, tf, int(T))
        self.T = int(t_np.shape[0])
        self.t = torch.as_tensor(t_np, dtype=self.dtype, device=self.device).reshape(self.T, 1)

        # Exogenous providers
        self.exo_names = list(self.eq.exogenous_functions) if self.eq.exogenous_functions else []
        self.exo_np = exogenous_np or {}
        self.exo_torch = exogenous_torch or {}

        # Y0 setup
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._rng = rng

        def _default_sampler():
            return np.zeros((D,), dtype=float)

        if y0_sampler is None and y0_set is None:
            y0_sampler = _default_sampler

        self._y0_sampler = y0_sampler
        self._y0_set = None
        if y0_set is not None:
            y0_arr = torch.as_tensor(y0_set, dtype=self.dtype, device=self.device) \
                        if isinstance(y0_set, torch.Tensor) else torch.as_tensor(y0_set, dtype=self.dtype)
            if y0_arr.ndim == 1:
                y0_arr = y0_arr.unsqueeze(0)  # (1,D)
            if y0_arr.shape[1] != D:
                raise ValueError(f"y0_set second dim must be {D}")
            self._y0_set = y0_arr

        # Optional ground truth supplied by user
        if ground_truth is not None:
            if not isinstance(ground_truth, torch.Tensor):
                ground_truth = torch.as_tensor(ground_truth, dtype=self.dtype)
            if ground_truth.ndim != 3:
                raise ValueError("ground_truth must have shape (N, T, D)")
            if ground_truth.shape[1:] != (self.T, D):
                raise ValueError(f"ground_truth must have shape (N, {self.T}, {D})")
            self.gt = ground_truth.to(device=self.device, dtype=self.dtype)
            if self.gt.shape[0] != self.N:
                raise ValueError("ground_truth first dim must match n_trajectories")
        else:
            self.gt = None

        # Pre-generate all trajectories
        self._t_list: List[torch.Tensor] = []
        self._y_list: List[torch.Tensor] = []
        self._y0_list: List[torch.Tensor] = []
        self._exo_list: List[Dict[str, torch.Tensor]] = []
        self._source: List[str] = []

        self._generate_all()

        self.total_data = sum([len(d) for d in self._t_list])
        
        if verbose:
            print(f"[SimulatedTrajectoryDataset] Trajectories={self.N} | Total Data={self.total_data} | Average Data per Traj={self.total_data/self.N:.1f}.")

    # -----------------------------
    # Public Dataset API
    # -----------------------------
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        Supports:
        - int: returns a single trajectory dict
        - slice: returns a list of trajectory dicts (e.g., ds[1:3] -> [1, 2])
        - list/np.ndarray/torch.Tensor of ints: returns list of trajectory dicts
        Notes:
        - Negative indices are handled (Python semantics).
        - For slices, step other than 1 is allowed but will skip accordingly.
        """
        # ---- helpers ----
        def _single_item(i: int):
            # normalize negatives
            if i < 0:
                i += self.N
            if not (0 <= i < self.N):
                raise IndexError(f"Index {i} out of range for N={self.N}")

            t = self._t_list[i]
            y = self._y_list[i]
            y0 = self._y0_list[i]
            exo = self._exo_list[i] if self._exo_list else None

            sample = {"t": t, "y": y, "y0": y0, "source": self._source[i], "traj_id": int(self.traj_id_base + i)}

            if exo:
                sample["exo"] = exo
            if self.gt is not None:
                sample["gt"] = self.gt[i]
            
            return sample

        # ---- dispatch by idx type ----
        # 1) integer
        if isinstance(idx, int):
            return _single_item(idx)

        # 2) slice
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.N)
            return [_single_item(i) for i in range(start, stop, step)]

        # 3) fancy indexing (sequence of ints)
        if isinstance(idx, (list, tuple)):
            return [_single_item(int(i)) for i in idx]

        if isinstance(idx, np.ndarray):
            if idx.dtype.kind not in ("i", "u"):  # integer array only
                raise TypeError("Only integer arrays supported for indexing.")
            return [_single_item(int(i)) for i in idx.tolist()]

        if torch.is_tensor(idx):
            if not idx.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("Only integer tensors supported for indexing.")
            return [_single_item(int(i)) for i in idx.view(-1).tolist()]

        raise TypeError(f"Unsupported index type: {type(idx)}")

    # -----------------------------
    # Generation helpers
    # -----------------------------
    def _sample_y0(self, i: int) -> torch.Tensor:
        if self._y0_set is not None:
            j = i if i < self._y0_set.shape[0] else (i % self._y0_set.shape[0])
            y0 = self._y0_set[j]
            return y0.to(device=self.device, dtype=self.dtype)
        else:
            y0 = self._y0_sampler()
            if isinstance(y0, torch.Tensor):
                return y0.to(device=self.device, dtype=self.dtype).reshape(-1)
            else:
                return torch.as_tensor(y0, dtype=self.dtype, device=self.device).reshape(-1)

    def _eval_exogenous_np(self, tt: np.ndarray) -> Dict:
        out = {}
        for name in self.exo_names:
            fn = self.exo_np.get(name, None)
            if fn is None:
                out[name] = lambda t: 0.0
            else:
                out[name] = fn
        return out

    def _eval_exogenous_torch_over_grid(self, t_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate exogenous torch callables over t_grid, broadcasting as needed.
        Returns dict[name] -> (T,1) tensors.
        """
        out: dict[str, torch.Tensor] = {}
        if not self.exo_names:
            return out

        T = t_grid.shape[0]
        for name in self.exo_names:
            fn = self.exo_torch.get(name, None)
            if fn is None:
                out[name] = torch.zeros_like(t_grid)
                continue

            val = fn(t_grid)  # ideally (T,1)
            if not torch.is_tensor(val):
                val = torch.as_tensor(val, dtype=self.dtype, device=self.device)

            # Normalize shapes to (T,1)
            if val.ndim == 0:
                # scalar -> broadcast
                val = val.reshape(1, 1).expand(T, 1)
            elif val.ndim == 1:
                if val.shape[0] == 1:
                    val = val.reshape(1, 1).expand(T, 1)
                elif val.shape[0] == T:
                    val = val.reshape(T, 1)
                else:
                    raise ValueError(f"Exogenous '{name}' has length {val.shape[0]} != T={T}.")
            elif val.ndim == 2 and val.shape[1] == 1:
                if val.shape[0] == 1:
                    val = val.expand(T, 1)
                elif val.shape[0] != T:
                    raise ValueError(f"Exogenous '{name}' has length {val.shape[0]} != T={T}.")
            else:
                raise ValueError(f"Exogenous '{name}' must be scalar, (T,), or (T,1); got {tuple(val.shape)}.")

            out[name] = val.to(device=self.device, dtype=self.dtype)
        return out

    def _rollout_torch(
        self,
        y0: torch.Tensor,
        t_grid: torch.Tensor,
        method: str,
        exo_series: Optional[Dict[str, "torch.Tensor"]],
    ) -> torch.Tensor:
        """
        Roll out a trajectory using ODEEquation.simulate_one_step to avoid duplicating integrators.
        Keeps the autograd graph if self.keep_graph=True.
        """
        if self.eq.compiled is None:
            self.eq.compiled = self.eq._compile_first_order()

        # Build exogenous callables for simulate_one_step
        if self.exo_names:
            if self.exo_torch:  # prefer user-provided continuous functions
                exo_call = {name: self.exo_torch.get(name, lambda tau: torch.zeros_like(tau))
                            for name in self.exo_names}
            elif exo_series:    # otherwise, interpolate from the provided grid values
                exo_call = {name: (lambda tau, name=name: _interp_like_grid(t_grid, exo_series[name], tau))
                            for name in self.exo_names}
            else:
                exo_call = {name: (lambda tau: torch.zeros_like(tau)) for name in self.exo_names}
        else:
            exo_call = None

        T, D = t_grid.shape[0], self.state_dim
        y_list = [y0.reshape(1, D)]
        for n in range(T - 1):
            ti = t_grid[n:n+1, :]              # (1,1)
            hi = t_grid[n+1:n+2, :] - ti       # (1,1)
            xi = y_list[-1]                    # (1,D)
            xn = self.eq.simulate_one_step(
                t=ti,
                x=xi,
                dt=hi,
                exogenous=exo_call,
                params_override=self.params_override,
                method=method,
            )
            y_list.append(xn if self.keep_graph else xn.detach())

        return torch.vstack(y_list)  # (T, D)

    def _generate_all(self):
        self._t_list.clear()
        self._y_list.clear()
        self._y0_list.clear()
        self._exo_list.clear()
        self._source.clear()

        t_grid = self.t  # (T,1)
        T = self.T

        for i in range(self.N):
            y0 = self._sample_y0(i)  # (D,)
            self._y0_list.append(y0)

            # Exogenous series (for return)
            exo_series = self._eval_exogenous_torch_over_grid(t_grid)

            if self.backend == "numpy":
                # Solve with SciPy via the ODEEquation API (CPU), then convert to torch
                exo_np = self._eval_exogenous_np(t_grid.cpu().numpy().ravel())
                sol = self.eq.solve_ivp(
                    t_span=(float(t_grid[0].item()), float(t_grid[-1].item())),
                    y0=y0.detach().cpu().numpy(),
                    exogenous=exo_np if exo_np else None,
                    params_override=self.params_override,
                    t_eval=t_grid.cpu().numpy().ravel(),
                )
                
                Y = torch.as_tensor(sol.y.T, dtype=self.dtype, device=self.device)  # (T, D)

                self._y_list.append(Y)
                self._t_list.append(t_grid)
                self._exo_list.append(exo_series)
                self._source.append("ivp_numpy")

            else:  # torch
                X = self._rollout_torch(y0.to(self.device, self.dtype), t_grid, self.method, exo_series=exo_series)

                self._y_list.append(X)
                self._t_list.append(t_grid)
                self._exo_list.append(exo_series)
                self._source.append(self.method)
    
    # -----------------------------
    # Memory accounting
    # -----------------------------
    def _iter_owned_tensors(self):
        """Walk all containers we own and yield tensors we directly store."""
        if hasattr(self, "t") and torch.is_tensor(self.t):
            yield self.t
        # listeler
        for lst in (getattr(self, "_t_list", []),
                    getattr(self, "_y_list", []),
                    getattr(self, "_y0_list", [])):
            for t in lst:
                if torch.is_tensor(t):
                    yield t

        for d in getattr(self, "_exo_list", []):
            if isinstance(d, dict):
                for v in d.values():
                    if torch.is_tensor(v):
                        yield v

        gt = getattr(self, "gt", None)
        if torch.is_tensor(gt):
            yield gt

    @staticmethod
    def _humanize_bytes(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        f = float(n)
        for u in units:
            if f < 1024.0:
                return f"{f:.2f} {u}"
            f /= 1024.0
        return f"{f:.2f} EB"

    def memory_bytes(self):
        seen = set()  
        cpu_total = 0
        cuda_totals = {}

        breakdown = {"t_list": 0, "y_list": 0, "y0_list": 0, "exo_list": 0, "gt": 0, "other": 0}

        def _acc_tensor(t: torch.Tensor, tag: str):
            nonlocal cpu_total
            try:
                stor = t.untyped_storage()
                key = (t.device, stor.data_ptr())
                nbytes = int(stor.nbytes())
            except Exception:
                key = (t.device, int(t.data_ptr()))
                nbytes = int(t.element_size() * t.nelement())

            if key in seen:
                return
            seen.add(key)

            if t.is_cuda:
                idx = t.device.index if t.device.index is not None else 0
                cuda_totals[idx] = cuda_totals.get(idx, 0) + nbytes
            else:
                cpu_total += nbytes
            breakdown[tag] = breakdown.get(tag, 0) + nbytes

        if hasattr(self, "t") and torch.is_tensor(self.t):
            _acc_tensor(self.t, "other")  

        # Lists
        for tag, lst in (("t_list", getattr(self, "_t_list", [])),
                         ("y_list", getattr(self, "_y_list", [])),
                         ("y0_list", getattr(self, "_y0_list", []))):
            for t in lst:
                if torch.is_tensor(t):
                    _acc_tensor(t, tag)

        for d in getattr(self, "_exo_list", []):
            if isinstance(d, dict):
                for v in d.values():
                    if torch.is_tensor(v):
                        _acc_tensor(v, "exo_list")

        if torch.is_tensor(getattr(self, "gt", None)):
            _acc_tensor(self.gt, "gt")

        total = cpu_total + sum(cuda_totals.values())
        return {
            "cpu": cpu_total,
            "cuda": cuda_totals,
            "total": total,
            "detail": breakdown
        }

    def memory_report(self) -> str:
        """Returns a human-readable report."""
        info = self.memory_bytes()
        lines = []
        lines.append(f"Total Memory Usage: {self._humanize_bytes(info['total'])}")
        lines.append(f"CPU: {self._humanize_bytes(info['cpu'])}")
        if info["cuda"]:
            for dev, b in sorted(info["cuda"].items()):
                lines.append(f"CUDA device {dev}: {self._humanize_bytes(b)}")
        lines.append("Component Breakdown:")
        for k, v in info["detail"].items():
            if v:
                lines.append(f"  - {k}: {self._humanize_bytes(v)}")
        return "\n".join(lines)

# --------- small utility for torch exogenous interpolation ----------
def _interp_like_grid(t_grid: torch.Tensor, v_grid: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    """
    Piecewise-constant interpolation from a known grid (t_grid, v_grid) to t_query.
    Assumes t_grid is increasing, shapes: t_grid (T,1), v_grid (T,1), t_query (N,1).
    """
    # Broadcast compare: for each query, find index i s.t. t_grid[i] <= t_query < t_grid[i+1]
    # Use torch.bucketize for efficiency
    edges = t_grid.reshape(-1)  # (T,)
    q = t_query.reshape(-1)
    idx = torch.bucketize(q, edges, right=True) - 1
    idx = idx.clamp(min=0, max=edges.numel() - 1)
    return v_grid.reshape(-1, 1)[idx].reshape(t_query.shape)


@dataclass(frozen=True)
class SubtrajIndex:
    parent_traj_id: int
    start: int          # inclusive
    end: int            # exclusive


class SubtrajectoryDataset(Dataset):
    """
    Wrap a multi-trajectory dataset and expose many subtrajectories.

    Each returned sample is trajectory-like:
      - t_abs: (L,1) absolute time slice from parent
      - t:     (L,1) either absolute or relative time (configurable)
      - y:     (L,D) state slice
      - y0:    (D,)  initial condition for THIS subtrajectory (= y[start])
      - exo:   dict[name]->(L,1) sliced (if present)
      - gt:    (L,D) sliced (if present)
      - bookkeeping fields: parent_traj_id, start_idx, end_idx, t0_abs

    Why both t_abs and t?
      - If your ODE / exogenous forcing depends on absolute time, you must preserve it.
      - If you want time normalization in the networks, relative time is often nicer.
    """

    def __init__(
        self,
        base_ds: Dataset,
        *,
        # Choose ONE of the following window specs:
        subseq_len: Optional[int] = None,      # number of samples per subtrajectory
        subseq_T: Optional[float] = None,      # time duration per subtrajectory (approx on grid)

        # Window movement:
        stride: Optional[int] = None,          # step in samples between windows (default: subseq_len i.e. non-overlapping)
        overlap: Optional[float] = None,       # alternative to stride, e.g. 0.5 -> 50% overlap (used if stride is None)

        # Truncation / filtering:
        drop_last: bool = True,                # drop tail windows that don't fit exactly
        min_len: int = 2,                      # reject windows shorter than this (if drop_last=False)
        Tf: Optional[float] = None,            # optional max absolute time bound per parent before windowing

        # Time output control:
        return_relative_time: bool = True,     # returned key "t" will be relative if True else absolute
        also_return_t_abs: bool = True,        # always useful for debugging / absolute-time forcing

        # Exogenous handling:
        include_exogenous: bool = True,

        # Safety checks:
        require_monotone_time: bool = True,
    ):
        super().__init__()
        self.base = base_ds
        self.include_exo = bool(include_exogenous)
        self.drop_last = bool(drop_last)
        self.min_len = int(min_len)
        self.Tf = Tf
        self.return_relative_time = bool(return_relative_time)
        self.also_return_t_abs = bool(also_return_t_abs)
        self.require_monotone_time = bool(require_monotone_time)

        if (subseq_len is None) == (subseq_T is None):
            raise ValueError("Provide exactly one of subseq_len or subseq_T.")

        if subseq_len is not None:
            self.subseq_len = int(subseq_len)
            if self.subseq_len <= 1:
                raise ValueError("subseq_len must be >= 2.")
        else:
            self.subseq_len = None

        if subseq_T is not None:
            self.subseq_T = float(subseq_T)
            if self.subseq_T <= 0:
                raise ValueError("subseq_T must be > 0.")
        else:
            self.subseq_T = None

        # Determine stride
        if stride is not None and overlap is not None:
            raise ValueError("Use either stride or overlap, not both.")
        if stride is not None:
            self.stride = int(stride)
            if self.stride <= 0:
                raise ValueError("stride must be positive.")
        else:
            if overlap is None:
                # default: non-overlapping windows
                self.stride = int(self.subseq_len) if self.subseq_len is not None else None
            else:
                ov = float(overlap)
                if not (0.0 <= ov < 1.0):
                    raise ValueError("overlap must be in [0,1).")
                if self.subseq_len is None:
                    raise ValueError("overlap requires subseq_len mode (samples), not subseq_T.")
                self.stride = max(1, int(round(self.subseq_len * (1.0 - ov))))

        # Build window index map
        self._index: List[SubtrajIndex] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        info = self._index[i]
        parent = self.base[info.parent_traj_id]

        # Required fields from base
        t_full: torch.Tensor = parent["t"]          # (T,1)
        y_full: torch.Tensor = parent["y"]          # (T,D)

        t_abs = t_full[info.start:info.end]
        y = y_full[info.start:info.end]

        # IC: state at the start of the slice
        y0 = y_full[info.start].reshape(-1)

        # Time output choice
        if self.return_relative_time:
            t = t_abs - t_abs[:1]
        else:
            t = t_abs

        out: Dict[str, Any] = {
            "t": t,
            "y": y,
            "y0": y0,
            "parent_traj_id": int(info.parent_traj_id) + parent.get("traj_id", 0),
            "start_idx": int(info.start),
            "end_idx": int(info.end),
            "t0_abs": float(t_abs[0].item()),
            "source": parent.get("source", "subtraj"),
        }

        if self.also_return_t_abs:
            out["t_abs"] = t_abs

        # Exogenous
        if self.include_exo and ("exo" in parent) and isinstance(parent["exo"], dict):
            exo_out: Dict[str, torch.Tensor] = {}
            for name, v in parent["exo"].items():
                vv = v if torch.is_tensor(v) else torch.as_tensor(v, device=t_full.device, dtype=t_full.dtype)
                if vv.ndim == 1:
                    vv = vv.reshape(-1, 1)
                exo_out[name] = vv[info.start:info.end]
            out["exo"] = exo_out

        # Ground-truth optional
        if "gt" in parent and parent["gt"] is not None:
            gt_full = parent["gt"]
            out["gt"] = gt_full[info.start:info.end]

        return out

    # -------------------------
    # internal: build index
    # -------------------------
    def _build_index(self) -> None:
        self._index.clear()

        # We need a representative device/dtype for intermediate computations
        # but we’ll just use each trajectory’s tensors directly.
        for traj_id in range(len(self.base)):
            item = self.base[traj_id]
            t: torch.Tensor = item["t"]
            if t.ndim != 2 or t.shape[1] != 1:
                raise ValueError(f"base[{traj_id}]['t'] must be (T,1). Got {tuple(t.shape)}")

            # optional truncation by Tf (absolute time)
            T_i = t.shape[0]
            if self.Tf is not None:
                # keep indices with t <= Tf (inclusive-ish)
                # find last index with t <= Tf; if none, skip
                mask = (t.squeeze(-1) <= float(self.Tf))
                if not bool(mask.any()):
                    continue
                T_i = int(mask.nonzero(as_tuple=False)[-1].item()) + 1

            t_use = t[:T_i]

            if self.require_monotone_time:
                dt = t_use[1:] - t_use[:-1]
                if not bool((dt > 0).all()):
                    raise ValueError(f"Time is not strictly increasing for traj {traj_id} (after Tf truncation).")

            # Determine window length in samples
            if self.subseq_len is not None:
                L = self.subseq_len
            else:
                # subseq_T mode: choose L as the smallest sample count whose duration >= subseq_T
                # Note: this is grid-dependent and approximate.
                target = float(self.subseq_T)
                # Precompute for each possible start the max end via searching.
                # We'll do a simple two-pointer scan O(T).
                L = None  # variable per start in this mode

            if self.subseq_len is not None:
                stride = int(self.stride) if self.stride is not None else L
                self._add_fixed_len_windows(traj_id, T_i, L, stride)
            else:
                self._add_time_len_windows(traj_id, t_use, target=float(self.subseq_T))

        if len(self._index) == 0:
            raise RuntimeError("SubtrajectoryDataset produced zero windows. Check subseq settings, Tf, and base dataset length.")

    def _add_fixed_len_windows(self, traj_id: int, T_i: int, L: int, stride: int) -> None:
        if T_i < self.min_len:
            return

        start = 0
        while start < T_i:
            end = start + L
            if end <= T_i:
                self._index.append(SubtrajIndex(traj_id, start, end))
            else:
                if not self.drop_last:
                    # keep the remainder if it's long enough
                    if (T_i - start) >= self.min_len:
                        self._index.append(SubtrajIndex(traj_id, start, T_i))
                break
            start += stride

    def _add_time_len_windows(self, traj_id: int, t: torch.Tensor, target: float) -> None:
        T_i = t.shape[0]
        if T_i < self.min_len:
            return

        # Two-pointer: for each start, advance end until duration >= target
        start = 0
        end = 1
        while start < T_i - 1:
            t0 = t[start].item()

            # ensure end at least start+1
            if end < start + 1:
                end = start + 1

            # advance end until duration condition or end hits T_i
            while end < T_i and (t[end - 1].item() - t0) < target:
                end += 1

            if end <= T_i and (t[end - 1].item() - t0) >= target:
                self._index.append(SubtrajIndex(traj_id, start, end))
            else:
                # couldn't satisfy target duration
                if not self.drop_last:
                    if (T_i - start) >= self.min_len:
                        self._index.append(SubtrajIndex(traj_id, start, T_i))
                break

            # move start; in subseq_T mode stride is ambiguous, so use "one-step" shift
            # (you can change this to a bigger hop if you want fewer windows)
            start += 1

class SubtrajectoryView(Dataset):
    """
    Adapts SubtrajectoryDataset (or any dataset returning trajectory dicts)
    into a MultiTrajectoryDataset-like object expected by PINNTrainDataset.

    It materializes all subtrajectories into:
      - N
      - state_dim
      - _t_list, _y_list, _y0_list, _exo_list
      - optional gt
    """

    def __init__(
        self,
        sub_ds: Dataset,
        *,
        use_abs_time_key: str = "t_abs",   # prefer absolute time if available
        use_time_key_fallback: str = "t",
        y_key: str = "y",
        y0_key: str = "y0",
        exo_key: str = "exo",
        gt_key: str = "gt",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.sub_ds = sub_ds

        self._t_list: List[torch.Tensor] = []
        self._y_list: List[torch.Tensor] = []
        self._y0_list: List[torch.Tensor] = []
        self._exo_list: List[Dict[str, torch.Tensor]] = []
        self._source: List[str] = []
        self.gt: Optional[torch.Tensor] = None
        self._parent_traj_id: List[int] = []
        self._t0_abs: List[float] = []   # optional but useful

        # Materialize
        gt_list = []
        for i in range(len(sub_ds)):
            it: Dict[str, Any] = sub_ds[i]

            # time: prefer absolute time to keep forcing consistent
            if use_abs_time_key in it:
                t = it[use_abs_time_key]
            else:
                t = it[use_time_key_fallback]

            y = it[y_key]
            y0 = it[y0_key]

            if not torch.is_tensor(t):  t = torch.as_tensor(t)
            if not torch.is_tensor(y):  y = torch.as_tensor(y)
            if not torch.is_tensor(y0): y0 = torch.as_tensor(y0)

            if device is not None:
                t = t.to(device=device)
                y = y.to(device=device)
                y0 = y0.to(device=device)
            if dtype is not None:
                t = t.to(dtype=dtype)
                y = y.to(dtype=dtype)
                y0 = y0.to(dtype=dtype)

            # normalize shapes
            if t.ndim == 1: t = t.reshape(-1, 1)
            if y0.ndim != 1: y0 = y0.reshape(-1)

            self._t_list.append(t)
            self._y_list.append(y)
            self._y0_list.append(y0)

            exo_out = {}
            if exo_key in it and isinstance(it[exo_key], dict):
                for name, v in it[exo_key].items():
                    vv = v if torch.is_tensor(v) else torch.as_tensor(v)
                    if device is not None: vv = vv.to(device=device)
                    if dtype is not None:  vv = vv.to(dtype=dtype)
                    if vv.ndim == 1: vv = vv.reshape(-1, 1)
                    exo_out[name] = vv
            self._exo_list.append(exo_out)

            self._source.append(it.get("source", "subtraj"))

            if gt_key in it and it[gt_key] is not None:
                g = it[gt_key]
                g = g if torch.is_tensor(g) else torch.as_tensor(g)
                if device is not None: g = g.to(device=device)
                if dtype is not None:  g = g.to(dtype=dtype)
                gt_list.append(g)

            pid = int(it.get("parent_traj_id", -1))
            self._parent_traj_id.append(pid)

            self._t0_abs.append(float(it.get("t0_abs", 0.0)))

        # Required attributes for PINNTrainDataset
        self.N = len(self._t_list)
        if self.N == 0:
            raise RuntimeError("No subtrajectories to adapt (N=0).")

        self.state_dim = int(self._y_list[0].shape[1])

        # optional: stack gt if it exists for all items
        if len(gt_list) == self.N:
            self.gt = torch.stack(gt_list, dim=0)  # (N, L, D) but L may vary -> only works if equal length
        else:
            self.gt = None

        if verbose:
            lens = torch.tensor([t.shape[0] for t in self._t_list])
            print(f"[SubtrajectoryView] N={self.N} | "
                  f"state_dim={self.state_dim} | "
                  f"len(min/mean/max)={int(lens.min())}/{float(lens.float().mean()):.1f}/{int(lens.max())}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        # Return a MultiTrajectoryDataset-like dict
        sample = {
            "t": self._t_list[idx],
            "y": self._y_list[idx],
            "y0": self._y0_list[idx],
            "source": self._source[idx],
            "parent_traj_id": self._parent_traj_id[idx],
            "t0_abs": self._t0_abs[idx],  # optional
        }
        if self._exo_list and self._exo_list[idx]:
            sample["exo"] = self._exo_list[idx]
        if self.gt is not None:
            sample["gt"] = self.gt[idx]
        return sample

# =============================================================================
# LoadedTrajectoryDataset (CSV -> trajectories)
# =============================================================================
class LoadedTrajectoryDataset(Dataset):
    """
    Load trajectories from a single CSV that contains:
      - traj_id column (trajectory id)
      - t column (timestamp)
      - y columns (state signals)
      - optional exogenous columns (exo)
      - optional gt columns

    Each __getitem__ returns:
      {
        "t":  (T,1) tensor,
        "y":  (T,D) tensor,
        "y0": (D,) tensor,
        "exo": {name: (T,1)} optional,
        "gt": (T,D_gt) optional,
        "source": str
      }
    """

    def __init__(
        self,
        csv_path: str,
        *,
        traj_id_col: str = "traj_id",
        t_col: str = "t",
        y_cols: Sequence[str] = ("y",),
        exo_cols: Optional[Dict[str, str]] = None,   # exo_name -> csv column
        gt_cols: Optional[Sequence[str]] = None,     # optional columns for gt (same length T)
        sort_by_time: bool = True,
        ensure_strictly_increasing_time: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.traj_id_col = traj_id_col
        self.t_col = t_col
        self.y_cols = list(y_cols)
        self.exo_cols = dict(exo_cols) if exo_cols is not None else {}
        self.gt_cols = list(gt_cols) if gt_cols is not None else None
        self.sort_by_time = bool(sort_by_time)
        self.ensure_strict = bool(ensure_strictly_increasing_time)

        if device is None:
            device = torch.device("cpu")
        self.device = torch.device(device)
        self.dtype = dtype

        df = pd.read_csv(csv_path)

        # Validate columns
        needed = [traj_id_col, t_col] + list(self.y_cols)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        for _, col in self.exo_cols.items():
            if col not in df.columns:
                raise ValueError(f"Missing exogenous column '{col}' in CSV.")

        if self.gt_cols is not None:
            missing_gt = [c for c in self.gt_cols if c not in df.columns]
            if missing_gt:
                raise ValueError(f"Missing gt columns in CSV: {missing_gt}")

        # Materialize into lists for fast access
        self._t_list: List[torch.Tensor] = []
        self._y_list: List[torch.Tensor] = []
        self._y0_list: List[torch.Tensor] = []
        self._exo_list: List[Dict[str, torch.Tensor]] = []
        self._gt_list: List[Optional[torch.Tensor]] = []
        self._source: List[str] = []

        for traj_id, g in df.groupby(traj_id_col):
            if self.sort_by_time:
                g = g.sort_values(t_col, kind="mergesort")

            t_np = g[t_col].to_numpy(dtype=float).reshape(-1, 1)
            y_np = g[self.y_cols].to_numpy(dtype=float)

            t = torch.as_tensor(t_np, dtype=self.dtype, device=self.device)
            y = torch.as_tensor(y_np, dtype=self.dtype, device=self.device)

            if t.shape[0] < 2:
                # Skip degenerate trajectory
                continue

            if self.ensure_strict:
                dt = t[1:] - t[:-1]
                if not bool((dt > 0).all()):
                    raise ValueError(f"Trajectory {traj_id}: time is not strictly increasing.")

            y0 = y[0].reshape(-1)

            exo = {}
            for name, col in self.exo_cols.items():
                v_np = g[col].to_numpy(dtype=float).reshape(-1, 1)
                exo[name] = torch.as_tensor(v_np, dtype=self.dtype, device=self.device)

            gt = None
            if self.gt_cols is not None:
                gt_np = g[self.gt_cols].to_numpy(dtype=float)
                gt = torch.as_tensor(gt_np, dtype=self.dtype, device=self.device)

            self._t_list.append(t)
            self._y_list.append(y)
            self._y0_list.append(y0)
            self._exo_list.append(exo)
            self._gt_list.append(gt)

            self._source.append(f"csv:{os.path.basename(csv_path)}:{traj_id}")

        if len(self._t_list) == 0:
            raise RuntimeError("LoadedTrajectoryDataset: no valid trajectories found.")

        self.N = len(self._t_list)
        self.state_dim = int(self._y_list[0].shape[1])

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = {
            "t": self._t_list[idx],
            "y": self._y_list[idx],
            "y0": self._y0_list[idx],
            "source": self._source[idx],
        }
        if self._exo_list[idx]:
            out["exo"] = self._exo_list[idx]
        if self._gt_list[idx] is not None:
            out["gt"] = self._gt_list[idx]
        return out

# =============================================================================
# Helper: Zero-Order Hold Interpolator
# =============================================================================
class ZOHInterpolator:
    """
    Efficiently queries u(t) for a sorted time grid using step-function logic.
    Assumes t_grid is sorted ascending.
    """
    def __init__(self, t_grid: torch.Tensor, u_grid: torch.Tensor):
        self.t_grid = t_grid  # (T, 1)
        self.u_grid = u_grid  # (T, D_u)
        self.T_max = t_grid[-1].item()
        self.T_min = t_grid[0].item()

    def __call__(self, t_query: torch.Tensor) -> torch.Tensor:
        """
        t_query: (N, 1) arbitrary times
        Returns: (N, D_u) corresponding inputs
        """
        # 1. Clamp query times to valid range to avoid index errors
        t_q = t_query.clamp(self.T_min, self.T_max)

        # 2. Search sorted: find index where t_grid[i] <= t_q < t_grid[i+1]
        # side='right' returns index i such that t_grid[i-1] <= t < t_grid[i]
        # so we subtract 1 to get the "holding" index.
        idx = torch.searchsorted(self.t_grid[:, 0], t_q[:, 0], right=True) - 1
        
        # 3. Safety clamp indices (handles t_q < t_min or numerical edge cases)
        idx = idx.clamp(0, len(self.u_grid) - 1)

        return self.u_grid[idx]

# =============================================================================
# PINNTrainDataset (Full Implementation)
# =============================================================================
class PINNTrainDataset(Dataset):
    def __init__(
        self,
        multi_trajectory_dataset: Dataset,
        *,
        # Collocation controls
        global_collocation: bool = False,
        collocation_frac: Optional[float] = None,
        num_collocation: Optional[int] = None,

        # Data (supervised) sampling controls
        datapoint_frac: Optional[float] = None,
        num_data_per_traj: Optional[int] = None,

        # New Controls for Continuous Physics
        continuous_collocation: bool = True,   # If True, jitters time off-grid
        collocation_jitter_scale: float = 1.0, # 1.0 = jitter up to full dt step

        # Sampling Strategy
        use_importance_sampling: bool = True,  
        importance_u_weight: float = 5.0,      # High weight for input switches
        importance_y_weight: float = 2.0,      # Moderate weight for output transients
        importance_blur_sigma: float = 2.0,    # Smear importance to neighbors

        # Misc
        disjoint_sets: bool = True,
        seed: Optional[int] = None,
        include_exogenous: bool = True,
        Tf: Optional[float] = None,
        use_uniform_grid: bool = False,
        add_noise_std: float = 0.0,
    ):
        super().__init__()
        self.src = multi_trajectory_dataset
        self.include_exo = include_exogenous
        self.disjoint_sets = disjoint_sets
        
        # Continuous Physics Settings
        self.continuous_collocation = continuous_collocation
        self.collocation_jitter_scale = collocation_jitter_scale

        # Importance sampling config
        self.use_importance_sampling = use_importance_sampling
        self.importance_u_weight = importance_u_weight
        self.importance_y_weight = importance_y_weight
        self.importance_blur_sigma = importance_blur_sigma

        # ---------- validate collocation knobs ----------
        if global_collocation:
            if collocation_frac is not None:
                raise ValueError("Global mode: collocation_frac not allowed; use num_collocation only.")
            if num_collocation is None or int(num_collocation) <= 0:
                raise ValueError("Global mode: num_collocation is required and must be positive.")
        else:
            if (collocation_frac is None) and (num_collocation is None):
                raise ValueError("Per-trajectory mode: specify exactly one of collocation_frac or num_collocation.")
            if (collocation_frac is not None) and (num_collocation is not None):
                raise ValueError("Per-trajectory mode: provide only one of collocation_frac or num_collocation.")
            if collocation_frac is not None and not (0.0 < float(collocation_frac) <= 1.0):
                raise ValueError("Per-trajectory mode: collocation_frac must be in (0, 1].")
            if num_collocation is not None and int(num_collocation) <= 0:
                raise ValueError("Per-trajectory mode: num_collocation must be positive.")

        # ---------- store knobs ----------
        self.global_collocation = bool(global_collocation)
        self.collocation_frac = float(collocation_frac) if collocation_frac is not None else None
        self.num_collocation = int(num_collocation) if num_collocation is not None else None

        self.datapoint_frac = float(datapoint_frac) if datapoint_frac is not None else None
        self.num_data_per_traj = int(num_data_per_traj) if num_data_per_traj is not None else None
        self.num_data = self.num_data_per_traj

        self.N_traj = len(self.src)

        # Inspect one item to infer D/device/dtype
        it0 = self.src[0]
        y0_temp = it0["y"]
        if not torch.is_tensor(y0_temp):
            y0_temp = torch.as_tensor(y0_temp)
        self.D = int(y0_temp.shape[1])
        
        t0_temp = it0["t"]
        if not torch.is_tensor(t0_temp):
            t0_temp = torch.as_tensor(t0_temp)
        self.device = t0_temp.device
        self.dtype = t0_temp.dtype

        self.Tf = Tf
        self.use_uniform_grid = bool(use_uniform_grid)
        self.add_noise_std = float(add_noise_std)

        self.t_col_master: Optional[torch.Tensor] = None 

        # RNG
        self._g = torch.Generator(device="cpu")
        if seed is not None:
            self._g.manual_seed(int(seed))

        # ---------------------------------------------------------------------
        # Pre-build Interpolators and Lengths
        # ---------------------------------------------------------------------
        self.T_list: List[int] = []
        self.exo_interpolators: List[Dict[str, ZOHInterpolator]] = []
        self.dt_list: List[float] = [] # Average dt per trajectory

        for i in range(self.N_traj):
            # 1. Extract Full Tensors (using helper)
            t_full = self._to_tensor(self.src[i]["t"])
            
            # Handle Tf truncation
            if Tf is None:
                T_i = len(t_full)
            else:
                # Ensure we don't go out of bounds if Tf is larger than data
                idx_end = int(torch.argmin(torch.abs(t_full - float(Tf))).item())
                T_i = max(1, min(len(t_full), idx_end + 1))
            self.T_list.append(T_i)
            
            # Calculate dt for jittering
            if T_i > 1:
                dt_avg = (t_full[T_i-1] - t_full[0]) / (T_i - 1)
                self.dt_list.append(dt_avg.item())
            else:
                self.dt_list.append(0.1) # Fallback

            # 2. Build Interpolators for this trajectory
            interps = {}
            if self.include_exo and "exo" in self.src[i]:
                # Robustly handle exo dictionary
                exo_dict = self.src[i]["exo"]
                if isinstance(exo_dict, dict):
                    for k, v in exo_dict.items():
                        u_full = self._to_tensor(v)[:T_i]
                        t_use = t_full[:T_i]
                        # Create interpolator
                        interps[k] = ZOHInterpolator(t_use, u_full)
            self.exo_interpolators.append(interps)

        # ---------- global collocation master time grid ----------
        if self.global_collocation:
            t_starts, t_ends = [], []
            for i, T_i in enumerate(self.T_list):
                t_i = self._to_tensor(self.src[i]["t"])
                t_starts.append(float(t_i[0].item()))
                t_ends.append(float(t_i[T_i - 1].item()))
            t_min = float(min(t_starts))
            t_max = float(max(t_ends))
            if not (t_min < t_max):
                raise ValueError("Union time span is degenerate.")

            Nc_total = self.num_collocation
            if self.use_uniform_grid:
                t_vals = torch.linspace(t_min, t_max, steps=Nc_total, dtype=self.dtype, device=self.device)
            else:
                u = torch.rand(Nc_total, generator=self._g, dtype=self.dtype)  # CPU
                t_vals = (t_min + (t_max - t_min) * u).to(self.device)
                t_vals, _ = torch.sort(t_vals)
            self.t_col_master = t_vals.reshape(-1, 1)

        # ---------- pre-sample indices ----------
        self.idx_col_per_traj: List[torch.Tensor] = []
        self.idx_data_per_traj: List[torch.Tensor] = []
        self._sample_all_indices()

        # Logging
        self.total_colocation_points = sum(int(len(d)) for d in self.idx_col_per_traj)
        self.total_datapoints = sum(int(len(d)) for d in self.idx_data_per_traj)

        mode = "GLOBAL" if self.global_collocation else "PER-TRAJ"
        avg_per_traj = (self.total_colocation_points / self.N_traj) if self.N_traj > 0 else 0.0
        print(
            f"[PINNTrainDataset] Collocation Mode={mode} | "
            f"Trajectories={self.N_traj} | "
            f"Total Collocation={self.total_colocation_points} | "
            f"Data={self.total_datapoints} | "
            f"Average Collocation per Traj={avg_per_traj:.0f}"
        )

    def _to_tensor(self, x) -> torch.Tensor:
        """Helper to ensure data is tensor, on correct device/dtype, and 2D."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def _get_importance_weights(self, i: int, T_i: int) -> torch.Tensor:
        """
        Calculates a probability mass function (PMF) over indices [0, T_i-1].
        High gradient in u or y => High probability.
        """
        if not self.use_importance_sampling:
            return torch.ones(T_i, dtype=torch.float32)

        item = self.src[i]
        
        # Extract y (target) - move to CPU for weight calc to avoid VRAM fragmentation
        y = item["y"]
        if not torch.is_tensor(y): y = torch.as_tensor(y)
        y = y[:T_i].to(device="cpu", dtype=torch.float32)
        
        # Extract u (input) if exists
        u = None
        if self.include_exo and "exo" in item and isinstance(item["exo"], dict) and "u" in item["exo"]:
            u = item["exo"]["u"]
            if not torch.is_tensor(u): u = torch.as_tensor(u)
            u = u[:T_i].to(device="cpu", dtype=torch.float32)

        # Base weight (uniform baseline so steady states aren't ignored)
        weights = torch.ones(T_i, dtype=torch.float32)

        # 1. Gradient of u (Input switches are critical)
        if u is not None and self.importance_u_weight > 0:
            # Central difference gradient, pad to keep shape same
            # Transpose to (D, T) for padding, then back
            if u.ndim == 1: u = u.unsqueeze(1)
            u_padded = F.pad(u.transpose(0,1), (1,1), mode='replicate').transpose(0,1)
            du = (u_padded[2:] - u_padded[:-2]).abs().sum(dim=1) # Magnitude sum over dims
            
            # Normalize to 0..1 to keep scaling consistent
            if du.max() > 1e-6:
                du = du / du.max()
            
            weights += self.importance_u_weight * du

        # 2. Gradient of y (Fast dynamics are critical)
        if self.importance_y_weight > 0:
            if y.ndim == 1: y = y.unsqueeze(1)
            y_padded = F.pad(y.transpose(0,1), (1,1), mode='replicate').transpose(0,1)
            dy = (y_padded[2:] - y_padded[:-2]).abs().sum(dim=1)
            
            if dy.max() > 1e-6:
                dy = dy / dy.max()
                
            weights += self.importance_y_weight * dy

        # 3. Blur the weights (Dilation)
        if self.importance_blur_sigma > 0:
            # 1D gaussian kernel approx (simple box/avg pool here)
            kernel_size = int(self.importance_blur_sigma * 4) + 1
            if kernel_size % 2 == 0: kernel_size += 1
            
            w_reshaped = weights.view(1, 1, -1)
            blurred = F.avg_pool1d(w_reshaped, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            weights = blurred.view(-1)
            # Recover magnitude
            weights = weights / (weights.max() + 1e-8) * (self.importance_u_weight + self.importance_y_weight + 1)

        return weights

    def _sample_all_indices(self):
        self.idx_col_per_traj.clear()
        self.idx_data_per_traj.clear()

        # Handle GLOBAL MODE separately 
        if self.global_collocation:
            assert self.t_col_master is not None
            t_master = self.t_col_master.squeeze(-1)

            for i, T_i in enumerate(self.T_list):
                t_i = self._to_tensor(self.src[i]["t"])[:T_i].squeeze(-1)

                tmin_i = t_i[0]
                tmax_i = t_i[-1]
                mask = (t_master >= tmin_i) & (t_master <= tmax_i)
                t_use = t_master[mask]

                if t_use.numel() == 0:
                    idx_col = torch.tensor([0], dtype=torch.long)
                else:
                    # Map global times to nearest indices in trajectory
                    idx_right = torch.bucketize(t_use, t_i, right=False)
                    idx_left = torch.clamp(idx_right - 1, min=0)
                    idx_right = torch.clamp(idx_right, max=T_i - 1)

                    dist_left = torch.abs(t_use - t_i[idx_left])
                    dist_right = torch.abs(t_use - t_i[idx_right])
                    idx_nearest = torch.where(dist_left <= dist_right, idx_left, idx_right)
                    idx_col = torch.unique(idx_nearest, sorted=True)

                if self.datapoint_frac is not None:
                    Nd_i = max(0, int(round(self.datapoint_frac * T_i)))
                else:
                    Nd_i = max(0, min(int(self.num_data or 0), T_i))

                if self.disjoint_sets:
                    all_idx = torch.arange(T_i, dtype=torch.long)
                    mask_dat = torch.ones(T_i, dtype=torch.bool)
                    mask_dat[idx_col] = False
                    pool = all_idx[mask_dat]
                    Nd_i = min(Nd_i, int(pool.numel()))
                    if Nd_i > 0:
                        perm = pool[torch.randperm(pool.numel(), generator=self._g)[:Nd_i]]
                        idx_dat = torch.sort(perm).values
                    else:
                        idx_dat = torch.empty(0, dtype=torch.long)
                else:
                    idx_dat = (torch.randperm(T_i, generator=self._g)[:Nd_i].sort().values
                            if Nd_i > 0 else torch.empty(0, dtype=torch.long))

                self.idx_col_per_traj.append(idx_col)
                self.idx_data_per_traj.append(idx_dat)
            
            # RETURN to ensure we don't run the PER-TRAJ loop
            return 

        # PER-TRAJECTORY MODE
        for i, T_i in enumerate(self.T_list):
            
            # 1. Determine Counts
            if self.collocation_frac is not None:
                Nc_i = max(1, int(round(self.collocation_frac * T_i)))
            else:
                Nc_i = int(self.num_collocation)

            if self.datapoint_frac is not None:
                Nd_i = max(0, int(round(self.datapoint_frac * T_i)))
            else:
                Nd_i = max(0, min(int(self.num_data or 0), T_i))

            # 2. Compute Importance Weights
            weights = self._get_importance_weights(i, T_i)

            # 3. Sample
            total_samples = Nc_i + (Nd_i if self.disjoint_sets else 0)
            total_samples = min(total_samples, T_i)
            
            if self.use_importance_sampling:
                # Weighted sampling WITHOUT replacement
                sampled_indices = torch.multinomial(
                    weights, 
                    num_samples=total_samples, 
                    replacement=False, 
                    generator=self._g
                )
            else:
                sampled_indices = torch.randperm(T_i, generator=self._g)[:total_samples]

            # 4. Split into Collocation / Data
            if self.disjoint_sets:
                shuffled_sampled = sampled_indices[torch.randperm(total_samples, generator=self._g)]
                idx_col = shuffled_sampled[:Nc_i]
                idx_dat = shuffled_sampled[Nc_i:Nc_i+Nd_i]
            else:
                # Overlapping sets
                idx_col = sampled_indices[:Nc_i]
                
                # Resample data from importance weights if possible, or just take random
                if Nd_i > 0:
                    if self.use_importance_sampling:
                        idx_dat = torch.multinomial(weights, Nd_i, replacement=False, generator=self._g)
                    else:
                        idx_dat = torch.randperm(T_i, generator=self._g)[:Nd_i]
                else:
                    idx_dat = torch.empty(0, dtype=torch.long)

            idx_col, _ = torch.sort(idx_col)
            idx_dat, _ = torch.sort(idx_dat)
            
            self.idx_col_per_traj.append(idx_col)
            self.idx_data_per_traj.append(idx_dat)
            
    def __len__(self) -> int:
        return self.N_traj

    def __getitem__(self, traj_id: int) -> dict:
        item = self.src[traj_id]
        T_i = self.T_list[traj_id]

        # 1. Get Indices
        idx_col = self.idx_col_per_traj[traj_id]
        idx_dat = self.idx_data_per_traj[traj_id]

        # 2. Get Raw Tensors (Truncated to Tf)
        t_raw = self._to_tensor(item["t"])[:T_i]
        y_raw = self._to_tensor(item["y"])[:T_i]
        y0 = self._to_tensor(item["y0"]).reshape(-1)

        # 3. Prepare Data Points (Exact measurements required)
        t_data = t_raw[idx_dat]
        y_data = y_raw[idx_dat]
        if self.add_noise_std > 0:
            y_data = y_data + torch.randn_like(y_data) * self.add_noise_std

        # 4. Prepare Collocation Points (Continuous Physics!)
        t_col_basis = t_raw[idx_col]
        
        if self.continuous_collocation:
            # Jitter: t_new = t_old + Uniform(-0.5*dt, 0.5*dt) * scale
            dt = self.dt_list[traj_id]
            jitter_range = dt * self.collocation_jitter_scale
            # (rand - 0.5) gives [-0.5, 0.5]
            noise = (torch.rand_like(t_col_basis) - 0.5) * jitter_range
            t_col = t_col_basis + noise
            # Clamp to domain bounds to avoid going outside [0, Tf]
            t_col = t_col.clamp(t_raw[0], t_raw[-1])
        else:
            t_col = t_col_basis

        # 5. Handle Exogenous Inputs via Interpolation
        exo_col = {}
        exo_data = {}
        
        interps = self.exo_interpolators[traj_id]
        
        for name, interp_obj in interps.items():
            # Interpolate for collocation (handles off-grid t_col)
            exo_col[name] = interp_obj(t_col)
            
            # For Data: Use the stored grid for exact indices
            u_full = interp_obj.u_grid 
            exo_data[name] = u_full[idx_dat]

        # prefer parent trajectory id if present
        parent_id = item.get("parent_traj_id", None)
        if parent_id is None or int(parent_id) < 0:
            parent_id = traj_id

        return {
            "traj_id": int(parent_id),        # <-- embedding id (physical trajectory)
            "window_id": int(traj_id),        # <-- optional
            "t_col": t_col,
            "t_data": t_data,
            "y_data": y_data,
            "y0": y0,
            "exo_col": exo_col,
            "exo_data": exo_data,
            # include indices for debugging if needed
            "idx_col": idx_col,
            "idx_data": idx_dat
        }

    def resample(self, seed: Optional[int]):
        if seed is not None:
            self._g.manual_seed(int(seed))
        self._sample_all_indices()

# ==========================================================================
#
#
#  DATA LOADER
#
#
# ==========================================================================
def make_pinn_dataloader(
    pinn_ds: Dataset,
    n_traj_samples: int,
    n_data_samples: int,
    *,
    shuffle_trajs: bool = True,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    sampler = RandomSampler(pinn_ds) if shuffle_trajs else None
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    def _len0(x: torch.Tensor) -> int:
        return int(x.shape[0]) if x.ndim >= 1 else 1

    def _stack_truncate(batch_list, L, *, is_y=False, device=None, dtype=None, to_long=False):
        B = len(batch_list)
        if L <= 0:
            return torch.empty(
                (B, 0, 1),
                dtype=(torch.long if to_long else (dtype or torch.float32)),
                device=device,
            )

        outs = []
        for xi in batch_list:
            if device is not None and xi.device != device:
                xi = xi.to(device=device)
            if dtype is not None and not to_long:
                xi = xi.to(dtype=dtype)

            if xi.ndim == 1:
                xi = xi[:L].reshape(-1, 1)
            else:
                if is_y:
                    xi = xi[:L]
                else:
                    xi = xi[:L, :1]
            outs.append(xi)

        return torch.stack(outs, dim=0)

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        ref = batch[0]
        dev = ref["t_data"].device if isinstance(ref["t_data"], torch.Tensor) else None
        dtp = ref["t_data"].dtype if isinstance(ref["t_data"], torch.Tensor) else torch.float32

        L_col = min(n_data_samples, min(_len0(it["t_col"]) for it in batch)) if "t_col" in ref else 0
        L_dat = min(n_data_samples, min(_len0(it["t_data"]) for it in batch)) if "t_data" in ref else 0

        traj_id = torch.tensor([it["traj_id"] for it in batch], dtype=torch.long, device=dev)

        t_col = _stack_truncate([it["t_col"] for it in batch], L_col, device=dev, dtype=dtp)
        t_data = _stack_truncate([it["t_data"] for it in batch], L_dat, device=dev, dtype=dtp)

        y_data = _stack_truncate([it["y_data"] for it in batch], L_dat, device=dev, dtype=dtp, is_y=True)

        idx_col = _stack_truncate([it["idx_col"] for it in batch], L_col, device=dev, to_long=True)
        idx_data = _stack_truncate([it["idx_data"] for it in batch], L_dat, device=dev, to_long=True)

        D = int(batch[0]["y0"].numel())
        y0 = torch.stack([it["y0"] for it in batch], dim=0).to(device=dev).reshape(B, 1, D)

        exo_col_names = set().union(*[set(it["exo_col"].keys()) for it in batch])
        exo_data_names = set().union(*[set(it["exo_data"].keys()) for it in batch])

        exo_col = {}
        for name in sorted(exo_col_names):
            seq = [it["exo_col"].get(name, torch.zeros((_len0(it["t_col"]), 1), device=dev, dtype=dtp)) for it in batch]
            exo_col[name] = _stack_truncate(seq, L_col, device=dev, dtype=dtp)

        exo_data = {}
        for name in sorted(exo_data_names):
            seq = [it["exo_data"].get(name, torch.zeros((_len0(it["t_data"]), 1), device=dev, dtype=dtp)) for it in batch]
            exo_data[name] = _stack_truncate(seq, L_dat, device=dev, dtype=dtp)

        return {
            "traj_id": traj_id,
            "t_col": t_col, "t_data": t_data, "y_data": y_data, "y0": y0,
            "exo_col": exo_col, "exo_data": exo_data,
            "idx_col": idx_col, "idx_data": idx_data,
        }

    return DataLoader(
        pinn_ds,
        batch_size=n_traj_samples,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        generator=generator,
    )