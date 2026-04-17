
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from typing import Optional, Tuple 


def save_model_package(model, config_dict, filepath):
    """
    Saves both the weights and the configuration args.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        "config": config_dict,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Saved model to: {filepath}")

def load_checkpoint(model_class, filepath, device="cpu"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
        
    ckpt = torch.load(filepath, map_location=device)
    
    # 1. Initialize empty model using saved config
    model = model_class(**ckpt["config"])
    
    # 2. Load weights AND buffers (t_min, y_mean are restored here)
    model.load_state_dict(ckpt["state_dict"])
    
    model.to(device)
    model.eval()
    return model

# ==============
# HELPERS
# ==============
@torch.no_grad()
def get_global_time_scale(dataset):
    """
    Iterates through a dataset to find the global min and max time values
    across both collocation and measurement points.
    """
    t_min = float('inf')
    t_max = float('-inf')

    print("Calculating global time scale...")
    # Iterate through the dataset (indices)
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Check colocation time
        tc = sample["t_col"]
        if tc.numel() > 0:
            t_min = min(t_min, tc.min().item())
            t_max = max(t_max, tc.max().item())
            
        # Check measurement time
        td = sample["t_data"]
        if td.numel() > 0:
            t_min = min(t_min, td.min().item())
            t_max = max(t_max, td.max().item())
            
    t_span = t_max - t_min
    if t_span < 1e-6: t_span = 1.0
    
    print(f"Global Time Stats: Min={t_min:.4f}, Max={t_max:.4f}, Span={t_span:.4f}")
    return t_min, t_span

@torch.no_grad()
def get_global_signal_stats(dataset, eps=1e-8):
    """
    Computes global mean and std for y, v, and u across a PINNTrainDataset.
    
    Uses both collocation and data points.
    Assumes:
      - y_data[:,0] = y
      - y_data[:,1] = v
      - exo_col["u"], exo_data["u"] exist if forcing is used
    """

    # Running sums
    sum_y = 0.0
    sum_v = 0.0
    sum_u = 0.0

    sumsq_y = 0.0
    sumsq_v = 0.0
    sumsq_u = 0.0

    count_yv = 0
    count_u = 0

    print("Calculating global signal statistics...")

    for i in range(len(dataset)):
        sample = dataset[i]

        # -------- states from data points --------
        y_data = sample["y_data"]
        if y_data.numel() > 0:
            y = y_data[:, 0]
            v = y_data[:, 1]

            sum_y += y.sum().item()
            sum_v += v.sum().item()
            sumsq_y += (y * y).sum().item()
            sumsq_v += (v * v).sum().item()
            count_yv += y.numel()

        # -------- forcing from data + collocation --------
        if "exo_col" in sample and "u" in sample["exo_col"]:
            u = sample["exo_col"]["u"]
            if u.numel() > 0:
                sum_u += u.sum().item()
                sumsq_u += (u * u).sum().item()
                count_u += u.numel()

        if "exo_data" in sample and "u" in sample["exo_data"]:
            u = sample["exo_data"]["u"]
            if u.numel() > 0:
                sum_u += u.sum().item()
                sumsq_u += (u * u).sum().item()
                count_u += u.numel()

    if count_yv == 0:
        raise RuntimeError("No y/v samples found when computing global statistics.")

    # Means
    mean_y = sum_y / count_yv
    mean_v = sum_v / count_yv
    mean_u = (sum_u / count_u) if count_u > 0 else 0.0

    # Variances
    var_y = (sumsq_y / count_yv) - mean_y**2
    var_v = (sumsq_v / count_yv) - mean_v**2
    var_u = ((sumsq_u / count_u) - mean_u**2) if count_u > 0 else 1.0

    # Stds
    std_y = (var_y + eps) ** 0.5
    std_v = (var_v + eps) ** 0.5
    std_u = (var_u + eps) ** 0.5

    print(
        f"Global Signal Stats:\n"
        f"  y: mean={mean_y:.4e}, std={std_y:.4e}\n"
        f"  v: mean={mean_v:.4e}, std={std_v:.4e}\n"
        f"  u: mean={mean_u:.4e}, std={std_u:.4e}"
    )

    return {
        "y_mean": mean_y,
        "y_std":  std_y,
        "v_mean": mean_v,
        "v_std":  std_v,
        "u_mean": mean_u,
        "u_std":  std_u,
    }

# ============================================================
# 1) Fourier Features 
# ============================================================
class FourierFeatures(nn.Module):
    """
    Random Fourier Features for scalar or vector inputs.

    Forward:
      x: (N, D)
      returns: (N, out_dim) where out_dim = include_input*D + 2*n_frequencies
    """
    def __init__(
        self,
        in_dim: int,
        n_frequencies: int = 8,
        sigma: float = 1.0,
        trainable: bool = False,
        include_input: bool = True,
    ):
        super().__init__()
        assert in_dim >= 1
        assert n_frequencies >= 1
        self.in_dim = in_dim
        self.n_frequencies = n_frequencies
        self.include_input = include_input

        B = torch.randn(n_frequencies, in_dim) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return (self.in_dim if self.include_input else 0) + 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"FourierFeatures expected x shape (N,{self.in_dim}), got {tuple(x.shape)}")

        proj = (2.0 * torch.pi) * (x @ self.B.t())  # (N, n_freq)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

        return torch.cat([x, ff], dim=1) if self.include_input else ff


# ============================================================
# 2) ParamNetFeatures
# ============================================================
class ParamNetFeatures(nn.Module):
    """
    Hand-crafted features for parameter manifold learning.
    Input: state [y, v] and optional input u.
    Output: feature vector (LayerNorm optional).
    """
    def __init__(
        self,
        include_u: bool = True,
        poly_order: int = 3,
        include_abs: bool = False,
        include_energy: bool = False,
        include_cross: bool = False,
        layernorm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert poly_order in (1, 2, 3)
        self.include_u = include_u
        self.poly_order = poly_order
        self.include_abs = include_abs
        self.include_energy = include_energy
        self.include_cross = include_cross
        self.eps = eps

        feat_dim = self._infer_feat_dim()
        self.ln = nn.LayerNorm(feat_dim) if layernorm else None

    def _infer_feat_dim(self) -> int:
        dim = 0
        dim += 2  # y, v
        if self.include_abs:
            dim += 2  # |y|, |v|
        if self.include_energy:
            dim += 1  # energy
        if self.poly_order >= 2:
            dim += 2  # y^2, v^2
            if self.include_cross:
                dim += 1  # y*v
        if self.poly_order >= 3:
            dim += 2  # y^3, v^3
            if self.include_cross:
                dim += 2  # y^2*v, y*v^2

        if self.include_u:
            dim += 1
            if self.poly_order >= 2:
                dim += 1
                if self.include_cross:
                    dim += 2  # u*y, u*v

        return dim

    def forward(self, yv: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = yv[:, 0:1]
        v = yv[:, 1:2]

        feats = [y, v]

        if self.include_abs:
            feats += [y.abs(), v.abs()]

        if self.include_energy:
            E = 0.5 * v * v + 0.5 * y * y
            feats += [E]

        if self.poly_order >= 2:
            y2 = y * y
            v2 = v * v
            feats += [y2, v2]
            if self.include_cross:
                feats += [y * v]

        if self.poly_order >= 3:
            y3 = y2 * y
            v3 = v2 * v
            feats += [y3, v3]
            if self.include_cross:
                feats += [y2 * v, y * v2]

        if self.include_u:
            if u is None:
                raise ValueError("include_u=True but u is None")
            feats += [u]
            if self.poly_order >= 2:
                feats += [u * u]
                if self.include_cross:
                    feats += [u * y, u * v]

        out = torch.cat(feats, dim=1)
        return self.ln(out) if self.ln is not None else out


# ============================================================
# 3) Context Encoder 
# ============================================================

class ContextEncoderGRU(nn.Module):
    """
    Encodes a window/sequence context from u(t) (and optionally Δu(t)).

    Input:
      u_seq:  (B, L, 1)
      du_seq: (B, L, 1) optional
    Output:
      c:      (B, C)
    """
    def __init__(
        self,
        context_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        include_du: bool = True,
        layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.include_du = include_du
        in_dim = 1 + (1 if include_du else 0)

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, context_dim)
        self.ln = nn.LayerNorm(context_dim) if layernorm else None

    def forward(self, u_seq: torch.Tensor, du_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if u_seq.ndim != 3 or u_seq.shape[-1] != 1:
            raise ValueError(f"u_seq must be (B,L,1), got {tuple(u_seq.shape)}")

        if self.include_du:
            if du_seq is None:
                # default: finite difference inside (cheap, consistent)
                du = torch.zeros_like(u_seq)
                du[:, 1:, :] = u_seq[:, 1:, :] - u_seq[:, :-1, :]
                du_seq = du
            if du_seq.ndim != 3 or du_seq.shape[-1] != 1:
                raise ValueError(f"du_seq must be (B,L,1), got {tuple(du_seq.shape)}")
            x = torch.cat([u_seq, du_seq], dim=-1)  # (B,L,2)
        else:
            x = u_seq  # (B,L,1)

        out, hN = self.gru(x)               # hN: (num_layers, B, hidden_dim)
        h_last = hN[-1]                     # (B, hidden_dim)
        c = self.proj(h_last)               # (B, context_dim)
        return self.ln(c) if self.ln is not None else c


# ============================================================
# 4) FiLM blocks for solution net 
# ============================================================

class FiLMLayer(nn.Module):
    """
    Produces FiLM parameters (gamma, beta) from conditioning vector c.

    Identity init: gamma_raw=0, beta=0 => h' = (1+0)*h + 0 = h
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: (N,H), c: (N,C)
        gb = self.fc(c)                     # (N, 2H)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return (1.0 + gamma) * h + beta


class FiLMMLP(nn.Module):
    """
    Solution net with FiLM conditioning at each hidden layer.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 2,
        n_layers: int = 3,
        cond_dim: int = 32,
        activation: str = "silu",
    ):
        super().__init__()
        assert n_layers >= 1
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        act = nn.SiLU() if activation == "silu" else nn.Tanh()

        layers = []
        films = []

        # input -> hidden
        layers.append(nn.Linear(in_dim, hidden_dim))
        films.append(FiLMLayer(cond_dim, hidden_dim))
        layers.append(act)

        # hidden blocks
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            films.append(FiLMLayer(cond_dim, hidden_dim))
            layers.append(act)

        self.layers = nn.ModuleList(layers)
        self.films = nn.ModuleList(films)

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        c: (N, cond_dim)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be (N,in_dim), got {tuple(x.shape)}")
        if c.ndim != 2 or c.shape[0] != x.shape[0]:
            raise ValueError(f"c must be (N,cond_dim) matching x, got {tuple(c.shape)}")

        film_idx = 0
        h = x
        i = 0
        while i < len(self.layers):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                h = layer(h)
                # apply FiLM after each Linear in the hidden stack (not output)
                h = self.films[film_idx](h, c)
                film_idx += 1
                i += 1
            else:
                h = layer(h)
                i += 1

        return self.out(h)
    


class SOLIS(nn.Module):
    """
    FiLM-conditioned solution net + parameter net.

    Output:
      out: (N, 2+3) = [y, v, k, d, g] if use_intercept=False
      out: (N, 2+4) = [y, v, k, d, g, c] if use_intercept=True
    """
    def __init__(
        self,
        sol_net_hidden_dim: int = 256,
        sol_net_layers: int = 3,

        # context encoder
        context_dim: int = 32,
        context_hidden_dim: int = 64,
        context_include_du: bool = True,
        
        # param net
        param_net_hidden_dim: int = 128,
        num_experts: int = 8,
        use_moe: bool = True,
        include_u_in_params: bool = True,
        poly_order: int = 3,
        ensure_positive_wn: bool = False,
        use_intercept: bool = False,       
        include_abs: bool = False,
        include_energy: bool = False,
        include_cross: bool = False,
        layernorm: bool = True,

        # time features
        x_dim: int = 4,
        use_u_in_sol_net: bool = True,
        use_relative_time: bool = True,
        include_abs_time: bool = False,
        use_fourier_time: bool = True,
        time_fourier_frequencies: int = 8,
        time_fourier_sigma: float = 1.0,
        time_fourier_trainable: bool = False,
        time_include_raw: bool = True,

        # trajectory embedding (optional)
        num_trajectories: int = 1,
        traj_emb_dim: int = 0,
        traj_emb_init_scale: float = 0.01,

        # normalization
        use_input_normalization: bool = False,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.ensure_positive_wn = ensure_positive_wn
        self.x_dim = x_dim
        self.use_intercept = use_intercept
        self.num_params = 4 if self.use_intercept else 3  # <--- Determine param output size

        # ---------- normalization buffers ----------
        self.register_buffer("_y_mean", torch.zeros(1))
        self.register_buffer("_y_std",  torch.ones(1))
        self.register_buffer("_v_mean", torch.zeros(1))
        self.register_buffer("_v_std",  torch.ones(1))
        self.register_buffer("_u_mean", torch.zeros(1))
        self.register_buffer("_u_std",  torch.ones(1))
        self.use_input_normalization = bool(use_input_normalization)

        # ---------- trajectory embedding ----------
        self.num_trajectories = int(num_trajectories)
        self.traj_emb_dim = int(traj_emb_dim)
        if self.traj_emb_dim > 0:
            if self.num_trajectories <= 0:
                raise ValueError("num_trajectories must be >= 1 if traj_emb_dim > 0")
            self.traj_emb = nn.Embedding(self.num_trajectories, self.traj_emb_dim)
            nn.init.uniform_(self.traj_emb.weight, -traj_emb_init_scale, traj_emb_init_scale)
        else:
            self.traj_emb = None

        # ---------- context encoder ----------
        self.context_dim = int(context_dim)
        self.context_encoder = ContextEncoderGRU(
            context_dim=context_dim,
            hidden_dim=context_hidden_dim,
            num_layers=1,
            include_du=context_include_du,
            layernorm=True,
            dropout=0.0,
        )

        # Use IC inside context
        self.cond_include_ic_y = True
        self.cond_include_ic_v = (self.x_dim==4) # 4 means t,y0,v0,t0

        self.additional_context_dim = 0
        if self.cond_include_ic_y: self.additional_context_dim += 1
        if self.cond_include_ic_v: self.additional_context_dim += 1

        # conditioning vector for FiLM:[context, traj_emb, (y0,v0)]
        self.cond_dim = self.context_dim + self.traj_emb_dim + self.additional_context_dim

        # ---------- time channels ----------
        self.use_relative_time = bool(use_relative_time)
        self.include_abs_time = bool(include_abs_time)
        n_time = int(self.use_relative_time) + int(self.include_abs_time)
        if n_time == 0:
            raise ValueError("At least one of use_relative_time or include_abs_time must be True")
        self.n_time = n_time

        self.use_fourier_time = bool(use_fourier_time)
        if self.use_fourier_time:
            self.t_ff = FourierFeatures(
                in_dim=n_time,
                n_frequencies=time_fourier_frequencies,
                sigma=time_fourier_sigma,
                trainable=time_fourier_trainable,
                include_input=time_include_raw,
            )
            t_feat_dim = self.t_ff.out_dim
        else:
            self.t_ff = None
            t_feat_dim = n_time

        # time bounds buffer (for absolute-time normalization)
        self.register_buffer("_t_min", torch.tensor(0.0))
        self.register_buffer("_t_max", torch.tensor(1.0))

        # ---------- FiLM solution net ----------
        # pointwise plant input: [time_feat, y0, v0, t0, u]
        # (embedding is NOT concatenated now; it's used in FiLM cond)
        self.use_u_in_sol_net = use_u_in_sol_net
        plant_in_dim = (t_feat_dim + (x_dim - 1))
        if self.use_u_in_sol_net:
            plant_in_dim += 1  # + u

        self.y_net = FiLMMLP(
            in_dim=plant_in_dim,
            hidden_dim=sol_net_hidden_dim,
            out_dim=2,
            n_layers=sol_net_layers,
            cond_dim=self.cond_dim,
            activation="silu",
        )

        # ---------- Param net ----------
        self.param_feat = ParamNetFeatures(
            include_u=include_u_in_params,
            poly_order=poly_order,
            include_abs=include_abs,
            include_energy=include_energy,
            include_cross=include_cross,
            layernorm=layernorm,
        )
        feat_dim = self.param_feat._infer_feat_dim()

        if self.use_moe:
            self.gating_net = nn.Sequential(
                nn.Linear(feat_dim, param_net_hidden_dim), nn.SiLU(),
                nn.Linear(param_net_hidden_dim, num_experts),
                nn.Softmax(dim=-1),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, param_net_hidden_dim // 2), nn.SiLU(),
                    nn.Linear(param_net_hidden_dim // 2, self.num_params), # <--- USE num_params
                ) for _ in range(num_experts)
            ])
            self.param_head = None
        else:
            self.gating_net = None
            self.experts = None
            self.param_head = nn.Sequential(
                nn.Linear(feat_dim, param_net_hidden_dim), nn.SiLU(),
                nn.Linear(param_net_hidden_dim, param_net_hidden_dim), nn.SiLU(),
                nn.Linear(param_net_hidden_dim, self.num_params), # <--- USE num_params
            )

    # -----------------
    # setters / helpers
    # -----------------

    @torch.no_grad()
    def set_time_bounds(self, t_min: float, t_max: float) -> None:
        self._t_min.fill_(float(t_min))
        self._t_max.fill_(float(t_max))

    @torch.no_grad()
    def set_norm_stats(
        self,
        y_mean: float, y_std: float,
        v_mean: float, v_std: float,
        u_mean: float, u_std: float,
    ) -> None:
        self._y_mean.fill_(float(y_mean))
        self._y_std.fill_(float(y_std) if y_std > 0 else 1.0)
        self._v_mean.fill_(float(v_mean))
        self._v_std.fill_(float(v_std) if v_std > 0 else 1.0)
        self._u_mean.fill_(float(u_mean))
        self._u_std.fill_(float(u_std) if u_std > 0 else 1.0)

    def _normalize_time_abs(self, t: torch.Tensor) -> torch.Tensor:
        # map [t_min,t_max] ->[-1,1]
        denom = (self._t_max - self._t_min).clamp_min(1e-8)
        t01 = (t - self._t_min) / denom
        return 2.0 * t01 - 1.0

    def _norm_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._y_mean) / self._y_std.clamp_min(1e-8)

    def _norm_v(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self._v_mean) / self._v_std.clamp_min(1e-8)

    def _norm_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self._u_mean) / self._u_std.clamp_min(1e-8)

    # -----------------
    # context interface
    # -----------------

    def encode_context(self, u_seq: torch.Tensor, du_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u_seq: (B, L, 1)
        returns: c (B, context_dim)
        """
        if self.use_input_normalization:
            u_seq_in = self._norm_u(u_seq)
            if du_seq is not None:
                du_seq_in = du_seq / self._u_std.clamp_min(1e-8) # Only scale by the standard deviation.
            else:
                du_seq_in = None
        else:
            u_seq_in = u_seq
            du_seq_in = du_seq
            
        return self.context_encoder(u_seq_in, du_seq_in)

    def _build_cond(
        self,
        N: int,
        traj_id: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        ic_cond: Optional[torch.Tensor] = None,   
        B: Optional[int] = None,
        L: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build FiLM conditioning vector c_cond: (N, cond_dim)
        context can be:
          - (N, C) already pointwise
          - (B, C) window-level (will be repeated to (B*L,C))
        """
        parts =[]

        # context
        if context is None:
            # still allow conditioning even if only traj_emb exists
            ctx = torch.zeros(N, self.context_dim, device=self._t_min.device, dtype=self._t_min.dtype)
        else:
            if context.ndim != 2:
                raise ValueError(f"context must be 2D, got {tuple(context.shape)}")

            if context.shape[0] == N:
                ctx = context
            else:
                # assume (B,C) and we need to repeat for each point in window
                if B is None or L is None:
                    raise ValueError("If context is (B,C), must provide B and L for broadcast")
                if context.shape[0] != B:
                    raise ValueError(f"context batch mismatch: context has B={context.shape[0]} but provided B={B}")
                ctx = context[:, None, :].expand(B, L, self.context_dim).reshape(N, self.context_dim)

        parts.append(ctx)

        # traj embedding
        if self.traj_emb is not None:
            if traj_id is None:
                raise ValueError("traj_id must be provided when traj_emb_dim > 0")
            if traj_id.ndim == 2 and traj_id.shape[1] == 1:
                traj_id = traj_id.squeeze(1)
            traj_id = traj_id.to(dtype=torch.long)
            e = self.traj_emb(traj_id)  # (N, emb_dim)
            parts.append(e)
        
        # initial-condition conditioning (y0,v0)
        if (self.cond_include_ic_y and self.cond_include_ic_v):
            if ic_cond is None:
                raise ValueError("cond_include_ic=True but ic_cond is None")
            if ic_cond.ndim != 2 or ic_cond.shape[0] != N or ic_cond.shape[1] != 2:
                raise ValueError(f"ic_cond must be (N,2), got {tuple(ic_cond.shape)}")
            parts.append(ic_cond)
        
        elif self.cond_include_ic_y:
            if ic_cond is None:
                raise ValueError("cond_include_ic=True but ic_cond is None")
            if ic_cond.ndim != 2 or ic_cond.shape[0] != N or ic_cond.shape[1] != 1:
                raise ValueError(f"ic_cond must be (N,1), got {tuple(ic_cond.shape)}")
            parts.append(ic_cond)

        return torch.cat(parts, dim=1)  # (N, cond_dim)

    # -----------------
    # parameter head
    # -----------------

    def predict_params(
        self,
        yv: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        detach_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if detach_state:
            yv_in = yv.detach()
            u_in = u.detach() if u is not None else None
        else:
            yv_in = yv
            u_in = u

        feats = self.param_feat(yv_in, u_in)  # (N,F)

        if self.use_moe:
            gate = self.gating_net(feats)  # (N,E)
            expert_preds = torch.stack([exp(feats) for exp in self.experts], dim=1)  # (N,E,num_params)
            raw = torch.sum(gate.unsqueeze(-1) * expert_preds, dim=1)  # (N,num_params)
        else:
            gate = None
            raw = self.param_head(feats)

        k = F.softplus(raw[:, 0:1]) + 1e-4 if self.ensure_positive_wn else raw[:, 0:1]
        d = F.softplus(raw[:, 1:2]) + 1e-6
        g = raw[:, 2:3]

        if self.use_intercept:
            c = raw[:, 3:4]
            params = torch.cat([k, d, g, c], dim=1)
        else:
            params = torch.cat([k, d, g], dim=1)
            
        return params, gate

    # -----------------
    # forward variants
    # -----------------

    def forward_pointwise(
            self,
            x: torch.Tensor,                  # (N,4) =[t, y0, v0, t0]
            u: torch.Tensor,                  # (N,1)
            traj_id: Optional[torch.Tensor] = None,  
            context: Optional[torch.Tensor] = None,  
            B: Optional[int] = None,
            L: Optional[int] = None,
            detach: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        N = x.shape[0]

        # 1. Split inputs
        if self.x_dim == 4:
            t, y0, v0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        elif self.x_dim == 3:
            t, y0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            v0 = None

        # 2. Normalize inputs (FIX 1: Include t0!)
        if self.use_input_normalization:
            y0_in = self._norm_y(y0)
            u_in  = self._norm_u(u)
            t0_in = self._normalize_time_abs(t0) # Normalize t0!
            if self.x_dim == 4:
                v0_in = self._norm_v(v0)
        else:
            y0_in = y0
            u_in = u
            t0_in = t0
            if self.x_dim == 4:
                v0_in = v0

        # Construct rest_in safely
        if self.x_dim == 4:
            rest_in = torch.cat([y0_in, v0_in, t0_in], dim=1)
            ic_cond = torch.cat([y0_in, v0_in], dim=1) if self.cond_include_ic_v else y0_in
        else:
            rest_in = torch.cat([y0_in, t0_in], dim=1)
            ic_cond = y0_in

        # 3. Time features
        time_cols =[]
        norm_cols =[]
        denom = (self._t_max - self._t_min).clamp_min(1e-8)
        
        if self.use_relative_time:
            tau = t - t0
            time_cols.append(tau)
            norm_cols.append(2.0 * (tau / denom)) # roughly [0, 2]
            
        if self.include_abs_time:
            time_cols.append(t)
            norm_cols.append(self._normalize_time_abs(t))

        time_in = torch.cat(time_cols, dim=1)
        time_norm = torch.cat(norm_cols, dim=1)

        # FIX 3: Always use normalized time if normalization is enabled, 
        # even if Fourier features are turned off.
        if self.use_fourier_time:
            t_feat = self.t_ff(time_norm if self.use_input_normalization else time_in)
        else:
            t_feat = time_norm if self.use_input_normalization else time_in

        # 4. Conditioning
        c_cond = self._build_cond(
            N, traj_id=traj_id, context=context, ic_cond=ic_cond, B=B, L=L,
        )

        # 5. Network Forward
        plant_in = torch.cat([t_feat, rest_in, u_in], dim=1) if self.use_u_in_sol_net else torch.cat([t_feat, rest_in], dim=1)
        
        # FIX 4: Output Un-normalization Paradigm
        yv_raw = self.y_net(plant_in, c_cond) 

        if self.use_input_normalization:
            # Assume network learns to output normalized values, we scale them back 
            # to physical units for the PDE residual and parameter network mapping
            y_phys = yv_raw[:, 0:1] * self._y_std.clamp_min(1e-8) + self._y_mean
            v_phys = yv_raw[:, 1:2] * self._v_std.clamp_min(1e-8) + self._v_mean
            yv = torch.cat([y_phys, v_phys], dim=1)
        else:
            yv = yv_raw

        # 6. Parameter Network
        if self.use_input_normalization:
            # The parameter network operates purely in the normalized domain for stability
            y_norm = self._norm_y(yv[:, 0:1])
            v_norm = self._norm_v(yv[:, 1:2])
            yv_for_param = torch.cat([y_norm, v_norm], dim=1)
            u_for_param = self._norm_u(u)
        else:
            yv_for_param = yv
            u_for_param = u

        params, gate = self.predict_params(yv_for_param, u=u_for_param, detach_state=detach)

        # Return yv in physical units (for data loss and physics residual)
        out = torch.cat([yv, params], dim=1)
        return out, gate

    def forward_batched(
        self,
        x_b: torch.Tensor,                    # (B,L,4)
        u_b: torch.Tensor,                    # (B,L,1)
        traj_id_b: Optional[torch.Tensor] = None,   # (B,L) or (B,)
        context_b: Optional[torch.Tensor] = None,   # (B,C) (recommended)
        detach: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convenience wrapper: batched forward without manual flattening.
        Returns:
          out_b:  (B,L,5) or (B,L,6)
          gate_b: (B,L,E) or None
        """
        if x_b.ndim != 3 or x_b.shape[-1] != self.x_dim:
            raise ValueError(f"x_b must be (B,L,{self.x_dim}), got {tuple(x_b.shape)}")
        if u_b.ndim != 3 or u_b.shape[-1] != 1:
            raise ValueError(f"u_b must be (B,L,1), got {tuple(u_b.shape)}")

        B, L, _ = x_b.shape
        x = x_b.reshape(B * L, self.x_dim)
        u = u_b.reshape(B * L, 1)

        if traj_id_b is not None:
            if traj_id_b.ndim == 1:
                # (B,) -> expand to (B,L)
                traj_id_flat = traj_id_b[:, None].expand(B, L).reshape(B * L)
            elif traj_id_b.ndim == 2:
                traj_id_flat = traj_id_b.reshape(B * L)
            else:
                raise ValueError("traj_id_b must be (B,) or (B,L)")
        else:
            traj_id_flat = None

        out, gate = self.forward_pointwise(
            x=x,
            u=u,
            traj_id=traj_id_flat,
            context=context_b,  # (B,C) -> broadcast
            B=B, L=L,
            detach=detach,
        )

        out_b = out.reshape(B, L, -1)
        if gate is not None:
            # gate is (B*L, E) in current implementation
            gate_b = gate.reshape(B, L, -1)
        else:
            gate_b = None
        return out_b, gate_b
    

class MultitrajectoryIPINN(nn.Module):
    """
    Baseline IPINN model for Multi-Trajectory learning.
    
    Architecture:
      1. Solution Net: Identical to PI2NDi (GRU + FiLM + Fourier Features).
         It reconstructs y(t), v(t) for multiple trajectories.
      2. Parameter Block: Instead of a neural network, it learns 
         3 GLOBAL CONSTANTS (k, d, g) shared across all time and states.
    
    Output:
      out: (N, 2+3) = [y, v, k_const, d_const, g_const]
    """
    def __init__(
        self,
        # --- Solution Net Args (Same as PI2NDi) ---
        sol_net_hidden_dim: int = 256,
        sol_net_layers: int = 3,
        context_dim: int = 32,
        context_hidden_dim: int = 64,
        context_include_du: bool = True,
        x_dim: int = 4,
        use_u_in_sol_net: bool = True,
        use_relative_time: bool = True,
        include_abs_time: bool = False,
        use_fourier_time: bool = True,
        time_fourier_frequencies: int = 8,
        time_fourier_sigma: float = 1.0,
        time_fourier_trainable: bool = False,
        time_include_raw: bool = True,
        traj_emb_dim: int = 0,
        num_trajectories: int = 1,
        
        # --- IPINN Specific Args ---
        ensure_positive_k: bool = True, # Enforce k > 0 (Stiffness)
        ensure_positive_d: bool = True, # Enforce d > 0 (Damping)
        
        # Ignored args (kept for compatibility with training script calls)
        **kwargs 
    ):
        super().__init__()
        self.x_dim = x_dim
        self.ensure_positive_k = ensure_positive_k
        self.ensure_positive_d = ensure_positive_d

        # ------------------------------------------------------------------
        # 1. Global Parameters (The "Physics")
        # ------------------------------------------------------------------
        # We register them as parameters so optimizer picks them up.
        # Initialized to 1.0 or 0.1 to avoid zero-gradient issues at start.
        self._raw_k = nn.Parameter(torch.tensor(1.0))
        self._raw_d = nn.Parameter(torch.tensor(0.5))
        self._raw_g = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------------
        # 2. Shared Components (Buffers & Context) - Copied from PI2NDi
        # ------------------------------------------------------------------
        # Normalization buffers
        self.register_buffer("_y_mean", torch.zeros(1))
        self.register_buffer("_y_std",  torch.ones(1))
        self.register_buffer("_v_mean", torch.zeros(1))
        self.register_buffer("_v_std",  torch.ones(1))
        self.register_buffer("_u_mean", torch.zeros(1))
        self.register_buffer("_u_std",  torch.ones(1))
        # We assume True if this class is used, or pass as arg
        self.use_input_normalization = kwargs.get('use_input_normalization', False)

        # Time bounds
        self.register_buffer("_t_min", torch.tensor(0.0))
        self.register_buffer("_t_max", torch.tensor(1.0))

        # Trajectory Embedding
        self.traj_emb_dim = int(traj_emb_dim)
        if self.traj_emb_dim > 0:
            self.traj_emb = nn.Embedding(num_trajectories, self.traj_emb_dim)
            nn.init.uniform_(self.traj_emb.weight, -0.01, 0.01)
        else:
            self.traj_emb = None

        # Context Encoder (GRU)
        self.context_dim = int(context_dim)
        self.context_encoder = ContextEncoderGRU(
            context_dim=context_dim,
            hidden_dim=context_hidden_dim,
            num_layers=1,
            include_du=context_include_du,
            layernorm=True
        )

        # Condition logic
        self.cond_include_ic_y = True
        self.cond_include_ic_v = (self.x_dim == 4)
        
        additional_ctx = 0
        if self.cond_include_ic_y: additional_ctx += 1
        if self.cond_include_ic_v: additional_ctx += 1
        
        self.cond_dim = self.context_dim + self.traj_emb_dim + additional_ctx

        # ------------------------------------------------------------------
        # 3. Time Features - Copied from PI2NDi
        # ------------------------------------------------------------------
        self.use_relative_time = bool(use_relative_time)
        self.include_abs_time = bool(include_abs_time)
        n_time = int(self.use_relative_time) + int(self.include_abs_time)
        
        self.use_fourier_time = bool(use_fourier_time)
        if self.use_fourier_time:
            self.t_ff = FourierFeatures(
                in_dim=n_time,
                n_frequencies=time_fourier_frequencies,
                sigma=time_fourier_sigma,
                trainable=time_fourier_trainable,
                include_input=time_include_raw,
            )
            t_feat_dim = self.t_ff.out_dim
        else:
            self.t_ff = None
            t_feat_dim = n_time

        # ------------------------------------------------------------------
        # 4. Solution Net (FiLM MLP) - Copied from PI2NDi
        # ------------------------------------------------------------------
        self.use_u_in_sol_net = use_u_in_sol_net
        plant_in_dim = (t_feat_dim + (x_dim - 1))
        if self.use_u_in_sol_net:
            plant_in_dim += 1

        self.y_net = FiLMMLP(
            in_dim=plant_in_dim,
            hidden_dim=sol_net_hidden_dim,
            out_dim=2, # y, v
            n_layers=sol_net_layers,
            cond_dim=self.cond_dim,
            activation="silu",
        )

    # ------------------------------------------------------------------
    # Properties for Clean Access
    # ------------------------------------------------------------------
    @property
    def k(self):
        # Softplus + epsilon to ensure strictly positive stiffness if requested
        if self.ensure_positive_k:
            return F.softplus(self._raw_k) + 1e-4
        return self._raw_k

    @property
    def d(self):
        # Softplus + epsilon to ensure strictly positive damping if requested
        if self.ensure_positive_d:
            return F.softplus(self._raw_d) + 1e-6
        return self._raw_d

    @property
    def g(self):
        # Gain is usually signed
        return self._raw_g

    # ------------------------------------------------------------------
    # Standard Setters (Copied from PI2NDi to avoid crashes)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_time_bounds(self, t_min: float, t_max: float) -> None:
        self._t_min.fill_(float(t_min))
        self._t_max.fill_(float(t_max))

    @torch.no_grad()
    def set_norm_stats(self, y_mean, y_std, v_mean, v_std, u_mean, u_std):
        self._y_mean.fill_(float(y_mean))
        self._y_std.fill_(float(y_std) if y_std > 0 else 1.0)
        self._v_mean.fill_(float(v_mean))
        self._v_std.fill_(float(v_std) if v_std > 0 else 1.0)
        self._u_mean.fill_(float(u_mean))
        self._u_std.fill_(float(u_std) if u_std > 0 else 1.0)

    def _normalize_time_abs(self, t):
        denom = (self._t_max - self._t_min).clamp_min(1e-8)
        return 2.0 * ((t - self._t_min) / denom) - 1.0

    def _norm_y(self, y): return (y - self._y_mean) / self._y_std.clamp_min(1e-8)
    def _norm_v(self, v): return (v - self._v_mean) / self._v_std.clamp_min(1e-8)
    def _norm_u(self, u): return (u - self._u_mean) / self._u_std.clamp_min(1e-8)

    # ------------------------------------------------------------------
    # Context Encoding
    # ------------------------------------------------------------------
    def encode_context(self, u_seq: torch.Tensor, du_seq: Optional[torch.Tensor] = None):
        if self.use_input_normalization:
            u_seq_in = self._norm_u(u_seq)
            du_seq_in = None if du_seq is None else self._norm_u(du_seq)
        else:
            u_seq_in = u_seq
            du_seq_in = du_seq
        return self.context_encoder(u_seq_in, du_seq_in)

    def _build_cond(self, N, traj_id, context, ic_cond=None, B=None, L=None):
        parts = []
        # Context
        if context is None:
            ctx = torch.zeros(N, self.context_dim, device=self._t_min.device)
        else:
            if context.shape[0] == N:
                ctx = context
            else:
                ctx = context[:, None, :].expand(B, L, self.context_dim).reshape(N, self.context_dim)
        parts.append(ctx)

        # Traj Emb
        if self.traj_emb is not None:
            if traj_id is None: raise ValueError("traj_id required")
            e = self.traj_emb(traj_id.squeeze(1).long() if traj_id.ndim==2 else traj_id.long())
            parts.append(e)

        # IC
        if (self.cond_include_ic_y or self.cond_include_ic_v):
            if ic_cond is None: raise ValueError("ic_cond required")
            parts.append(ic_cond)
        
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward_pointwise(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        traj_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        B: Optional[int] = None,
        L: Optional[int] = None,
        detach: bool = False, # Ignored, kept for API compatibility
    ):
        N = x.shape[0]

        # 1. Prepare Inputs (Same as PI2NDi)
        if self.x_dim == 4:
            t, y0, v0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
            if self.use_input_normalization:
                y0_in, v0_in, u_in = self._norm_y(y0), self._norm_v(v0), self._norm_u(u)
            else:
                y0_in, v0_in, u_in = y0, v0, u
            rest_in = torch.cat([y0_in, v0_in, t0], dim=1)
        elif self.x_dim == 3:
            t, y0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            if self.use_input_normalization:
                y0_in, u_in = self._norm_y(y0), self._norm_u(u)
            else:
                y0_in, u_in = y0, u
            rest_in = torch.cat([y0_in, t0], dim=1)
        
        # 2. Build Conditioning
        ic_cond = torch.cat([y0_in, v0_in], dim=1) if self.cond_include_ic_v else y0_in
        c_cond = self._build_cond(N, traj_id, context, ic_cond, B, L)

        # 3. Time Features
        time_cols = []
        if self.use_relative_time: time_cols.append(t - t0)
        if self.include_abs_time: time_cols.append(t)
        time_in = torch.cat(time_cols, dim=1)

        if self.use_fourier_time:
            denom = (self._t_max - self._t_min).clamp_min(1e-8)
            norm_cols = []
            idx = 0
            if self.use_relative_time:
                norm_cols.append(2.0 * (time_in[:, idx:idx+1] / denom))
                idx += 1
            if self.include_abs_time:
                norm_cols.append(self._normalize_time_abs(time_in[:, idx:idx+1]))
            t_feat = self.t_ff(torch.cat(norm_cols, dim=1))
        else:
            t_feat = time_in

        # 4. Run Solution Network
        plant_in = torch.cat([t_feat, rest_in, u_in], dim=1) if self.use_u_in_sol_net else torch.cat([t_feat, rest_in], dim=1)
        
        yv = self.y_net(plant_in, c_cond)

        # 5. Append Global Parameters
        # Instead of a param net, we just expand the global constants
        k_batch = self.k.expand(N, 1)
        d_batch = self.d.expand(N, 1)
        g_batch = self.g.expand(N, 1)

        # Output format: [y, v, k, d, g]
        out = torch.cat([yv, k_batch, d_batch, g_batch], dim=1)
        
        # Return None for gate (no MoE)
        return out, None

    # API Compatibility Wrapper
    def predict_params(self, *args, **kwargs):
        # If the training loop tries to call this, return defaults
        # shape [1, 3] usually
        k = self.k.reshape(1,1)
        d = self.d.reshape(1,1)
        g = self.g.reshape(1,1)
        return torch.cat([k,d,g], dim=1), None
    

class SimpleIPINN(nn.Module):
    """
    Vanilla IPINN baseline for Multi-Trajectory learning.
    
    Architecture:
      1. Solution Net: A standard MLP (Feed-Forward Network).
         Input: [t, y0, v0, t0, u] (Concatenated)
         Output: [y, v]
         
         * No GRU context
         * No FiLM conditioning
         * No Fourier Features (Standard coordinate inputs)
         
      2. Parameter Block: Global Learnable Constants (k, d, g).
    """
    def __init__(
        self,
        # --- Network Size ---
        sol_net_hidden_dim: int = 128,
        sol_net_layers: int = 4,
        x_dim: int = 4,           # [t, y0, v0, t0] or [t, y0, t0]
        
        # --- IPINN Params ---
        ensure_positive_k: bool = True,
        ensure_positive_d: bool = True,
        
        # --- Compatibility Args (Ignored but accepted) ---
        context_dim=None, context_hidden_dim=None,
        use_fourier_time=False, traj_emb_dim=0,
        **kwargs
    ):
        super().__init__()
        self.x_dim = x_dim
        self.ensure_positive_k = ensure_positive_k
        self.ensure_positive_d = ensure_positive_d

        # ------------------------------------------------------------------
        # 1. Global Parameters
        # ------------------------------------------------------------------
        self._raw_k = nn.Parameter(torch.tensor(1.0))
        self._raw_d = nn.Parameter(torch.tensor(0.5))
        self._raw_g = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------------
        # 2. Normalization Buffers (Essential for fair comparison)
        # ------------------------------------------------------------------
        self.register_buffer("_y_mean", torch.zeros(1))
        self.register_buffer("_y_std",  torch.ones(1))
        self.register_buffer("_v_mean", torch.zeros(1))
        self.register_buffer("_v_std",  torch.ones(1))
        self.register_buffer("_u_mean", torch.zeros(1))
        self.register_buffer("_u_std",  torch.ones(1))
        self.register_buffer("_t_min", torch.tensor(0.0))
        self.register_buffer("_t_max", torch.tensor(1.0))
        
        self.use_input_normalization = kwargs.get('use_input_normalization', False)

        # ------------------------------------------------------------------
        # 3. The Vanilla MLP
        # ------------------------------------------------------------------
        # Input: x (size x_dim) + u (size 1)
        in_dim = x_dim + 1 
        
        layers = []
        layers.append(nn.Linear(in_dim, sol_net_hidden_dim))
        act_fn = nn.SiLU()
        layers.append(act_fn)
        
        for _ in range(sol_net_layers - 1):
            layers.append(nn.Linear(sol_net_hidden_dim, sol_net_hidden_dim))
            layers.append(act_fn)
            
        layers.append(nn.Linear(sol_net_hidden_dim, 2)) # Output y, v
        
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def k(self):
        if self.ensure_positive_k: return F.softplus(self._raw_k) + 1e-4
        return self._raw_k

    @property
    def d(self):
        if self.ensure_positive_d: return F.softplus(self._raw_d) + 1e-6
        return self._raw_d

    @property
    def g(self): return self._raw_g

    # ------------------------------------------------------------------
    # Normalization Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_time_bounds(self, t_min, t_max):
        self._t_min.fill_(float(t_min))
        self._t_max.fill_(float(t_max))

    @torch.no_grad()
    def set_norm_stats(self, y_mean, y_std, v_mean, v_std, u_mean, u_std):
        self._y_mean.fill_(float(y_mean))
        self._y_std.fill_(float(y_std) if y_std > 0 else 1.0)
        self._v_mean.fill_(float(v_mean))
        self._v_std.fill_(float(v_std) if v_std > 0 else 1.0)
        self._u_mean.fill_(float(u_mean))
        self._u_std.fill_(float(u_std) if u_std > 0 else 1.0)

    def _norm_y(self, y): return (y - self._y_mean) / self._y_std.clamp_min(1e-8)
    def _norm_v(self, v): return (v - self._v_mean) / self._v_std.clamp_min(1e-8)
    def _norm_u(self, u): return (u - self._u_mean) / self._u_std.clamp_min(1e-8)

    # ------------------------------------------------------------------
    # Dummy Context Method (To prevent training loop crash)
    # ------------------------------------------------------------------
    def encode_context(self, u_seq, du_seq=None):
        # Vanilla IPINN does not use context, returns None
        return None 

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward_pointwise(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        traj_id: Optional[torch.Tensor] = None, # Ignored
        context: Optional[torch.Tensor] = None, # Ignored
        B: Optional[int] = None,
        L: Optional[int] = None,
        detach: bool = False,
    ):
        N = x.shape[0]
        
        # 1. Normalize Inputs
        # x is [t, y0, v0, t0] or [t, y0, t0]
        if self.x_dim == 4:
            t, y0, v0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
            if self.use_input_normalization:
                y0 = self._norm_y(y0)
                v0 = self._norm_v(v0)
        else:
            t, y0, t0 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            v0 = None # Not used
            if self.use_input_normalization:
                y0 = self._norm_y(y0)

        if self.use_input_normalization:
            u_in = self._norm_u(u)
        else:
            u_in = u
            
        # 2. Time Normalization (Standard PINN practice: map t to [-1, 1] or similar)
        # Using the same absolute time norm as PI2NDi for fairness
        denom = (self._t_max - self._t_min).clamp_min(1e-8)
        t_norm = 2.0 * ((t - self._t_min) / denom) - 1.0
        
        # 3. Construct Input Vector [t_norm, y0, (v0), t0_norm, u]
        # Note: We normalize t0 same as t
        t0_norm = 2.0 * ((t0 - self._t_min) / denom) - 1.0
        
        if self.x_dim == 4:
            net_in = torch.cat([t_norm, y0, v0, t0_norm, u_in], dim=1)
        else:
            net_in = torch.cat([t_norm, y0, t0_norm, u_in], dim=1)
            
        # 4. Forward MLP
        yv = self.net(net_in)

        # 5. Append Global Params
        k_batch = self.k.expand(N, 1)
        d_batch = self.d.expand(N, 1)
        g_batch = self.g.expand(N, 1)

        out = torch.cat([yv, k_batch, d_batch, g_batch], dim=1)
        return out, None

    # Compatibility Wrapper
    def predict_params(self, *args, **kwargs):
        k = self.k.reshape(1,1)
        d = self.d.reshape(1,1)
        g = self.g.reshape(1,1)
        return torch.cat([k,d,g], dim=1), None