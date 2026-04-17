

from typing import Callable, Dict, List, Optional, Union
from .ode import ODEEquation
import torch
import numpy as np


class TorchODEResidual:
    """
    Residual evaluator for ODE systems represented in symbolic form (via ODEEquation)
    and compiled into a PyTorch-compatible right-hand side (RHS).

    This class uses autograd to compute time-derivatives of network outputs and
    compares them against the symbolic ODE equations to form residuals suitable
    for physics-informed learning.
    """

    def __init__(self, ode: "ODEEquation"):
        """
        Parameters
        ----------
        ode : ODEEquation
            Symbolic ODE system
            Must support `.compile_first_order()` to produce a torch-based RHS.
        """
        self._ode = ode

        # Compile RHS into torch function if not already available
        if self._ode.compiled is None:
            self._ode.compiled = self._ode._compile_first_order()
        self.f_torch = ode.compiled.f_torch
        if self.f_torch is None:
            raise RuntimeError("Torch RHS is unavailable for this equation.")

        # Metadata from ODE system
        self.n_eqs = len(self._ode.eqs)
        self.num_deps = len(self._ode.dependent_variables)
        self.orders = [int(i) for i in self._ode.orders]

        # Indices used for extracting derivatives
        self.rhs_indices: List[int] = self._cumsum_minus1(self.orders)

        # Track indices of highest derivatives (for each dependent variable)
        ctr = self.orders[0]
        self.highest_derivative_indices: List[int] = [ctr]
        for var_index in range(1, self.num_deps):
            max_order = self.orders[var_index]
            ctr += (max_order + 1)
            self.highest_derivative_indices.append(ctr)

        # Indices corresponding to lower-order derivatives (state variables)
        self._ode_state_indices: List[int] = [
            idx for idx in range(ctr) if idx not in self.highest_derivative_indices
        ]

    @staticmethod
    def _cumsum_minus1(L: List[int]) -> List[int]:
        cumsum = []
        s = 0
        for v in L:
            s += v
            cumsum.append(s - 1)
        return cumsum

    def __call__(
        self,
        t: "torch.Tensor",  # (N, 1)
        y: "torch.Tensor",  # (N, M): M = number of dependent variables
        exogenous: Optional[Dict[str, Union[Callable[["torch.Tensor"], "torch.Tensor"], torch.Tensor]]] = None,
        params_override: Optional[Dict[str, float]] = None,
        T_scale : float = 1.
    ) -> "torch.Tensor":
        """
        Compute ODE residuals using PyTorch autograd.

        Parameters
        ----------
        t : torch.Tensor, shape (N, 1)
            Time samples. Must require gradients to enable autograd.
        y : torch.Tensor, shape (N, M)
            Predicted trajectories for each dependent variable, aligned with t.
        exogenous : dict[name -> callable | tensor], optional
            Values of exogenous inputs. Two supported modes:
            - Callable: lambda tau -> (N,1) tensor
            - Tensor: (N,), (N,1), or scalar; will be broadcast into a callable
        params_override : dict, optional
            Optional overrides for parameters defined in `ode.params`.
        T_scale: float = 1
            Scaling factor of time axis. Set to 1 if there is no prescaling is done.
        Returns
        -------
        torch.Tensor, shape (N, n_eqs)
            Residuals stacked column-wise. Each column corresponds to
            (LHS - RHS) of the matching equation in `ode.eqs`.
        """

        # --- Shape checks ---
        if t.ndim != 2:
            raise ValueError("t must be 2D: (N,1).")
        if not t.requires_grad:
            raise ValueError("t must require gradients for autograd.")
        if y.ndim != 2 or y.shape[1] != self.num_deps or y.shape[0] != t.shape[0]:
            raise ValueError(
                f"y has shape {tuple(y.shape)} but expected ({t.shape[0]}, {self.num_deps})"
            )

        # --- Wrap exogenous inputs into callables ---
        exo_call = None
        if self._ode.exogenous_functions:
            assert type(exogenous) == dict, "Exogenous inputs should be given as dictionary."
            exo_call = {}
            device, dtype = t.device, t.dtype
            for name in self._ode.exogenous_functions:
                val = None if exogenous is None else exogenous.get(name, None)
                if callable(val):
                    exo_call[name] = val
                elif isinstance(val, torch.Tensor):
                    v = val.to(device=device, dtype=dtype)
                    exo_call[name] = (lambda vv=v: (lambda tau: vv.expand_as(tau)))()
                else:
                    exo_call[name] = lambda tau: torch.zeros_like(tau)

        # --- Build list of derivatives for each dependent variable ---
        grads = []
        for var_index in range(self.num_deps):
            max_order = self.orders[var_index]

            # Order 0: No scaling needed (y is y)
            current_var = y[:, var_index:var_index + 1]
            grads.append(current_var)

            # We need to maintain the 'raw' gradient graph w.r.t normalized 't'
            # to compute higher order derivatives correctly via autograd.
            raw_grad = current_var

            for i in range(1, max_order + 1):
                # 1. Compute Raw Derivative w.r.t. Normalized Time (d/d_tau)
                raw_grad = torch.autograd.grad(
                    raw_grad, t, grad_outputs=torch.ones_like(raw_grad),
                    create_graph=True, retain_graph=True
                )[0]

                # 2. Apply Chain Rule Scaling to get Physical Derivative
                # d^i y / dt^i = (1 / T_scale^i) * d^i y / d_tau^i
                
                # We apply scaling only when storing it for the ODE check.
                # We keep 'raw_grad' unscaled for the next iteration of autograd loop.
                scale_factor = 1.0 / (T_scale ** i)
                phys_grad = raw_grad * scale_factor
                
                grads.append(phys_grad)

        # Reconstruct state vector (lower-order derivatives only)
        Y = torch.hstack([grads[idx] for idx in self._ode_state_indices])

        # Highest-order derivatives (LHS of the ODEs)
        Ydot_auto = torch.hstack([grads[idx] for idx in self.highest_derivative_indices])

        # Evaluate symbolic RHS using compiled torch function
        rhs_all = self.f_torch(t, Y, exo_call, params_override)  # (N, state_dim)

        # Extract RHS corresponding to highest-derivative components
        Ydot_ode = rhs_all[:, self.rhs_indices]

        return Ydot_ode - Ydot_auto # Reference - Autograd