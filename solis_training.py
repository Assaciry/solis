
from pinoco import TorchODEResidual
from solis_nn import MultitrajectoryIPINN, SimpleIPINN

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional 

def rolling_ridge_hint(
    y, v, dv, u,
    window_size: int,
    ridge: float = 1e-4,
    *,
    standardize_X: bool = True,
    adaptive_ridge: bool = True,
    ridge_alpha: float = 1.0,          # scales adaptive ridge
    weight_mode: str = "mineig",       # "mineig" or "cond"
    weight_temp: float = 10.0,         # sharpness for weight squashing
    min_eig_floor: float = 1e-8,       # stability
    eps: float = 1e-8,
    use_intercept: bool = False,       
):
    """
    y,v,dv,u: (B,L,1)
    Returns:
      k_hat, d_hat, g_hat, c_hat, w_hat
      (c_hat is None if use_intercept is False)
    """

    if y.ndim != 3 or v.ndim != 3 or dv.ndim != 3 or u.ndim != 3:
        raise ValueError("y,v,dv,u must be (B,L,1)")
    B, L, _ = y.shape

    # Design matrix: X =[-y, -v, u, (ones)]
    if use_intercept:
        ones = torch.ones_like(y)
        X = torch.cat([-y, -v, u, ones], dim=-1)  # (B,L,4)
    else:
        X = torch.cat([-y, -v, u], dim=-1)        # (B,L,3)
        
    Y = dv                                        # (B,L,1)
    P = X.shape[-1]                               # Number of parameters

    w = int(window_size)
    if w < 2:
        raise ValueError("window_size must be >= 2")
    pad = w // 2

    # Pad along time
    Xp = F.pad(X, (0, 0, pad, pad), mode="replicate")  
    Yp = F.pad(Y, (0, 0, pad, pad), mode="replicate")  

    # Unfold windows
    Xw = Xp.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (B,L,w,P)
    Yw = Yp.unfold(dimension=1, size=w, step=1).permute(0, 1, 3, 2)  # (B,L,w,1)

    # Standardize X per window (Critical: do NOT standardize the intercept 'ones' column)
    if standardize_X:
        if use_intercept:
            Xw_sub = Xw[..., :3]
            mu_sub = Xw_sub.mean(dim=2, keepdim=True)
            var_sub = (Xw_sub - mu_sub).pow(2).mean(dim=2, keepdim=True)
            std_sub = torch.sqrt(var_sub + eps)
            Xw_s_sub = (Xw_sub - mu_sub) / std_sub
            # Re-attach the unstandardized 'ones' column
            Xw_s = torch.cat([Xw_s_sub, Xw[..., 3:4]], dim=-1)
        else:
            mu = Xw.mean(dim=2, keepdim=True)                            
            var = (Xw - mu).pow(2).mean(dim=2, keepdim=True)             
            std = torch.sqrt(var + eps)                                  
            Xw_s = (Xw - mu) / std                                       
    else:
        mu = None
        std = None
        Xw_s = Xw

    # Compute normal equations in standardized space
    Xt = Xw_s.transpose(-1, -2)              # (B,L,P,w)
    XtX = Xt @ Xw_s                          # (B,L,P,P)
    XtY = Xt @ Yw                            # (B,L,P,1)

    # Choose ridge per window
    if adaptive_ridge:
        tr = XtX.diagonal(dim1=-2, dim2=-1).sum(dim=-1)              
        lam = ridge_alpha * ridge * (tr / float(P)).clamp_min(eps)        
    else:
        lam = torch.full((B, L), float(ridge), device=X.device, dtype=X.dtype)

    I = torch.eye(P, device=X.device, dtype=X.dtype).view(1, 1, P, P)
    XtX_reg = XtX + lam[..., None, None] * I

    # Solve
    theta_s = torch.linalg.solve(XtX_reg, XtY).squeeze(-1)  # (B,L,P)

    # Convert coefficients back to physical units
    if standardize_X:
        if use_intercept:
            std_cols = std_sub.squeeze(2)   # (B,L,3)
            mu_cols = mu_sub.squeeze(2)     # (B,L,3)
            
            theta_dyn = theta_s[..., :3] / std_cols.clamp_min(eps)
            
            # y = (th/std)*x +[th_c - sum(th_k * mu_k / std_k)]
            c_hat = theta_s[..., 3:4] - (theta_dyn * mu_cols).sum(dim=-1, keepdim=True)
            
            k_hat = theta_dyn[..., 0:1]
            d_hat = theta_dyn[..., 1:2]
            g_hat = theta_dyn[..., 2:3]
        else:
            std_cols = std.squeeze(2)                                  
            theta = theta_s / std_cols.clamp_min(eps)
            k_hat = theta[..., 0:1]
            d_hat = theta[..., 1:2]
            g_hat = theta[..., 2:3]
            c_hat = None
    else:
        k_hat = theta_s[..., 0:1]
        d_hat = theta_s[..., 1:2]
        g_hat = theta_s[..., 2:3]
        c_hat = theta_s[..., 3:4] if use_intercept else None

    # Reliability weight
    evals = torch.linalg.eigvalsh(XtX)                               
    evals = torch.clamp(evals, min=0.0)

    min_eig = evals[..., 0]                                          
    max_eig = evals[..., -1]                                         
    cond = max_eig / (min_eig + min_eig_floor)                       

    if weight_mode == "mineig":
        tr = evals.sum(dim=-1).clamp_min(eps)                        
        mineig_norm = min_eig / (tr / float(P) + eps)
        w_hat = torch.sigmoid(weight_temp * (mineig_norm - 0.05))    
    elif weight_mode == "cond":
        inv = 1.0 / (cond + 1.0)
        w_hat = torch.sigmoid(weight_temp * (inv - 0.1))
    else:
        raise ValueError("weight_mode must be 'mineig' or 'cond'")

    w_hat = w_hat.unsqueeze(-1)  

    return k_hat, d_hat, g_hat, c_hat, w_hat


def weighted_total_variation(
    t_col: torch.Tensor,      
    params: torch.Tensor,     
    yv_col: torch.Tensor,     
    *,
    mask: Optional[torch.Tensor] = None,   
    eps: float = 1e-6,
    weight_mode: str = "inverse_energy",  
    dt_normalize: bool = True,
    p_norm: int = 1,                      
    reduce: str = "mean",                 
    ) -> torch.Tensor:
    """
    Weighted total variation penalty along collocation time for each batch element.
    Dynamically adjusts to whatever size 'params' is (P=3 or P=4).
    """
    if t_col.ndim != 3 or t_col.shape[-1] != 1:
        raise ValueError(f"t_col must be (B,Lc,1), got {tuple(t_col.shape)}")
    if yv_col.ndim != 3 or yv_col.shape[-1] != 2:
        raise ValueError(f"yv_col must be (B,Lc,2), got {tuple(yv_col.shape)}")
    if params.ndim != 3 or params.shape[:2] != t_col.shape[:2]:
        raise ValueError(f"params must be (B,Lc,P) with same (B,Lc) as t_col, got {tuple(params.shape)}")

    B, Lc, _ = t_col.shape

    t = t_col.squeeze(-1)
    dtheta = params[:, 1:, :] - params[:, :-1, :]
    dt = (t[:, 1:] - t[:, :-1]).clamp_min(eps)

    y_mid = 0.5 * (yv_col[:, 1:, 0] + yv_col[:, :-1, 0])
    v_mid = 0.5 * (yv_col[:, 1:, 1] + yv_col[:, :-1, 1])
    energy = y_mid * y_mid + v_mid * v_mid

    if weight_mode == "inverse_energy":
        w = 1.0 / (eps + energy)
    elif weight_mode == "energy":
        w = energy
    else:
        raise ValueError("weight_mode must be 'inverse_energy' or 'energy'")

    if p_norm == 1:
        tv = dtheta.abs().sum(dim=-1)
    elif p_norm == 2:
        tv = torch.sqrt((dtheta * dtheta).sum(dim=-1) + eps)
    else:
        raise ValueError("p_norm must be 1 or 2")

    if dt_normalize:
        tv = tv / dt

    weighted = w * tv  

    if mask is not None:
        if mask.shape != (B, Lc):
            raise ValueError(f"mask must be (B,Lc), got {tuple(mask.shape)}")
        step_mask = mask[:, 1:] & mask[:, :-1]  
        step_mask_f = step_mask.to(dtype=weighted.dtype)

        weighted = weighted * step_mask_f
        w = w * step_mask_f

    if reduce == "sum":
        return weighted.sum()

    if reduce == "mean":
        denom = w.sum(dim=1).clamp_min(eps)          
        per_b = weighted.sum(dim=1) / denom          
        return per_b.mean()

    raise ValueError("reduce must be 'mean' or 'sum'")


def set_requires_grad(module_or_params, requires_grad):
    if isinstance(module_or_params, nn.Module):
        for p in module_or_params.parameters():
            p.requires_grad = requires_grad
    elif isinstance(module_or_params, (list, tuple, dict)):
        iterable = module_or_params.values() if isinstance(module_or_params, dict) else module_or_params
        for p in iterable:
            p.requires_grad = requires_grad
    else:
        module_or_params.requires_grad = requires_grad


def weighted_mse(pred, target, w, eps=1e-8):
    num = (w * (pred - target).pow(2)).sum()
    den = w.sum().clamp_min(eps)
    return num / den


def train_epoch(
    model, # Assumes duck-typing (LPVParamNetFiLM, IPINN, etc.)
    eq,
    residual,
    loader: torch.utils.data.DataLoader,
    optimizer_plant: torch.optim.Optimizer,
    optimizer_params: torch.optim.Optimizer,
    criterion: nn.MSELoss,
    ic_dim: int = 2,
    data_dim: int = 2,
    phase: int = 1,
    change_phase: bool = False,
    scheduler_plant=None,
    scheduler_params=None,
    lambda_data: float = 10.0,
    lambda_ic: float = 100.0,
    lambda_phys: float = 1.0,
    lambda_phy_hint: float = 1.0,
    lambda_gate_reg: float = 0.1,
    lambda_tv: float = 0.1,
    H_horizon: int = 10,
    lambda_step: float = 10,
    random_window: str = "large"
):
    model.train()
    
    # Safely pull the flag from the model
    use_intercept = getattr(model, "use_intercept", False)

    # =========================
    # 1) Curriculum / freezing
    # =========================
    if phase == 1:
        if change_phase:
            set_requires_grad(model.y_net, True)
            set_requires_grad(model.context_encoder, True)
            if getattr(model, "traj_emb", None) is not None:
                set_requires_grad(model.traj_emb, True)

            if getattr(model, "use_moe", False):
                set_requires_grad(model.gating_net, False)
                set_requires_grad(model.experts, False)
            elif hasattr(model, "param_head"):
                set_requires_grad(model.param_head, False)

        w_phys, w_data, w_ic = lambda_phys, lambda_data, lambda_ic
        w_hint, w_tv, w_step = 0.0, 0.0, 0.0

    elif phase == 2:
        if change_phase:
            set_requires_grad(model.y_net, False)
            set_requires_grad(model.context_encoder, False)
            if getattr(model, "traj_emb", None) is not None:
                set_requires_grad(model.traj_emb, False)

            if getattr(model, "use_moe", False):
                set_requires_grad(model.gating_net, True)
                set_requires_grad(model.experts, True)
            elif hasattr(model, "param_head"):
                set_requires_grad(model.param_head, True)

        w_phys, w_data, w_ic = 10*lambda_phys, 0.0, 0.0
        w_hint, w_tv, w_step = lambda_phy_hint, lambda_tv, lambda_step
    else:
        raise ValueError("phase must be 1 or 2")

    epoch_metrics = {
        "loss": 0.0, "physics_loss": 0.0, "data_loss": 0.0,
        "ic_loss": 0.0, "hint_loss": 0.0, "diversity_loss": 0.0,
        "step_loss": 0.0, "tv_loss": 0.0,
    }

    device = next(model.parameters()).device
    n_batches = 0

    for batch in loader:
        n_batches += 1

        # -------------------------
        # Unpack + make contiguous
        # -------------------------
        t_col = batch["t_col"].to(device).contiguous()         
        t_data = batch["t_data"].to(device).contiguous()       
        y_data = batch["y_data"].to(device).contiguous()       
        y0 = batch["y0"].to(device).contiguous()               

        exo_c = {k: v.to(device).contiguous() for k, v in batch["exo_col"].items()}
        exo_d = {k: v.to(device).contiguous() for k, v in batch["exo_data"].items()}

        B, Lc, _ = t_col.shape
        _, Ld, _ = t_data.shape

        # Normalize y0 inline 
        y0_base = y0.reshape(B, -1)
        if y0_base.shape[1] < ic_dim:
            raise ValueError(f"y0 must contain at least {ic_dim} states [y0,v0].")
        y0_base = y0_base[:, :ic_dim]  
        
        y0_y = y0_base[:, 0:1]    
        if ic_dim == 2:
            y0_v = y0_base[:, 1:2]    

        t0 = t_col[:, 0, :]       

        step_losses = {}

        def closure():
            if phase == 1:
                optimizer_plant.zero_grad(set_to_none=True)
            else:
                optimizer_params.zero_grad(set_to_none=True)

            u_col_seq = exo_c["u"]                  
            context = model.encode_context(u_col_seq)  
            
            # ==========================
            # A) Physics residuals
            # ==========================
            t_col_flat = t_col.reshape(-1, 1).detach().requires_grad_(True)   

            y0_y_flat = y0_y.repeat_interleave(Lc, dim=0)    
            t0_flat   = t0.repeat_interleave(Lc, dim=0)      

            if ic_dim == 2:
                y0_v_flat = y0_v.repeat_interleave(Lc, dim=0)    
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, y0_v_flat, t0_flat], dim=1)  
            else:
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, t0_flat], dim=1)              

            u_col_flat = u_col_seq.reshape(-1, u_col_seq.shape[-1])  
            context_flat = context.repeat_interleave(Lc, dim=0)            

            out_col_flat, gate_col_flat = model.forward_pointwise(
                x_col_flat, u_col_flat,
                traj_id=None,
                context=context_flat,
                detach=True
            )  

            yv_col_flat = out_col_flat[:, 0:2]
            k_col_flat  = out_col_flat[:, 2:3]
            d_col_flat  = out_col_flat[:, 3:4]
            g_col_flat  = out_col_flat[:, 4:5]
            
            exo_phys = {"u": u_col_flat, "k": k_col_flat, "d": d_col_flat, "g": g_col_flat}
            
            # Conditionally extract intercept
            if use_intercept:
                c_col_flat = out_col_flat[:, 5:6]
                exo_phys["c"] = c_col_flat

            physics_res = residual(
                t_col_flat,
                yv_col_flat,
                exogenous=exo_phys,
            )

            loss_phys = criterion(physics_res[:, 0:1], torch.zeros_like(physics_res[:, 0:1])) + \
                        criterion(physics_res[:, 1:2], torch.zeros_like(physics_res[:, 1:2]))

            loss_hint = t_col_flat.new_tensor(0.0)
            loss_tv = t_col_flat.new_tensor(0.0)
            loss_step = t_col_flat.new_tensor(0.0)
            loss_div = t_col_flat.new_tensor(0.0)

            # ==========================
            # Phase 2 extras
            # ==========================
            if phase == 2:
                dv_dt = torch.autograd.grad(
                    outputs=yv_col_flat[:, 1:2],
                    inputs=t_col_flat,
                    grad_outputs=torch.ones_like(yv_col_flat[:, 1:2]),
                    create_graph=True,
                )[0].reshape(B, Lc, 1)

                if random_window == "large":
                    window_size = int(torch.randint(int(Lc/2), Lc, (1,), device=device).item())
                elif random_window == "medium":
                    window_size = int(torch.randint(5, Lc, (1,), device=device).item())
                elif random_window == "small":
                    window_size = int(torch.randint(5, int(Lc/2), (1,), device=device).item())
                
                if window_size % 2 == 0:
                    window_size += 1
                window_size = min(window_size, Lc)

                y_seq = yv_col_flat[:, 0:1].reshape(B, Lc, 1).detach()
                v_seq = yv_col_flat[:, 1:2].reshape(B, Lc, 1).detach()
                u_seq = u_col_flat.reshape(B, Lc, -1).detach()

                k_hat, d_hat, g_hat, c_hat, weights = rolling_ridge_hint(
                    y_seq, v_seq, dv_dt.detach(), u_seq[..., :1], 
                    window_size=window_size,
                    standardize_X=True,
                    ridge_alpha=2.,
                    ridge=2e-4,
                    use_intercept=use_intercept, # Pass the flag down
                )

                weights = weights.reshape(-1, 1).detach() 
                loss_hint = (
                    weighted_mse(k_col_flat, k_hat.reshape(-1,1), weights) +
                    weighted_mse(d_col_flat, d_hat.reshape(-1,1), weights) +
                    weighted_mse(g_col_flat, g_hat.reshape(-1,1), weights)
                )
                if use_intercept:
                    loss_hint += weighted_mse(c_col_flat, c_hat.reshape(-1,1), weights)
                
                # Dynamic slicing based on flag
                param_dim = 6 if use_intercept else 5
                out_col_b = out_col_flat.view(B, Lc, param_dim)          
                t_col_leaf = t_col_flat.view(B, Lc, 1)           

                gate_col_b = None
                if getattr(model, "use_moe", False) and (gate_col_flat is not None):
                    gate_col_b = gate_col_flat.view(B, Lc, -1)

                yv_for_w = out_col_b[..., 0:2].detach()   
                params   = out_col_b[..., 2:param_dim]            
                t_seq    = t_col_leaf.detach()            

                loss_tv = weighted_total_variation(
                    t_seq, params, yv_for_w,
                    mask=None, eps=1e-6,
                    weight_mode="inverse_energy",
                    dt_normalize=True, p_norm=2, reduce="mean",
                )

                # Multi-step rollout: teacher is detached y,v
                x_teacher = out_col_b[..., 0:2].detach()  

                k_seq = k_col_flat.reshape(B, Lc, 1)
                d_seq = d_col_flat.reshape(B, Lc, 1)
                g_seq = g_col_flat.reshape(B, Lc, 1)
                if use_intercept:
                    c_seq = c_col_flat.reshape(B, Lc, 1)
                t_det = t_col.detach()

                max_start = Lc - H_horizon - 1
                if max_start > 0:
                    s = torch.randint(0, max_start, (B,), device=device)
                    x = x_teacher[torch.arange(B, device=device), s, :]  

                    def _const_like(val, t):
                        return val.expand_as(t)

                    roll =[]
                    for h in range(H_horizon):
                        idx = s + h
                        bidx = torch.arange(B, device=device)

                        t_cur = t_det[bidx, idx, :]                 
                        dt = t_det[bidx, idx + 1, :] - t_cur        

                        u_cur = u_seq[bidx, idx, :1]                
                        k_cur = k_seq[bidx, idx, :]
                        d_cur = d_seq[bidx, idx, :]
                        g_cur = g_seq[bidx, idx, :]

                        exo_step = {
                            "u": (lambda t, uu=u_cur: _const_like(uu, t)),
                            "k": (lambda t, kk=k_cur: _const_like(kk, t)),
                            "d": (lambda t, dd=d_cur: _const_like(dd, t)),
                            "g": (lambda t, gg=g_cur: _const_like(gg, t)),
                        }
                        if use_intercept:
                            c_cur = c_seq[bidx, idx, :]
                            exo_step["c"] = (lambda t, cc=c_cur: _const_like(cc, t))

                        x = eq.simulate_one_step(
                            t_cur, x, dt, exogenous=exo_step,
                        )

                        x_tgt = x_teacher[bidx, idx + 1, :]
                        # Using Huber Loss to prevent gradients from exploding on unstable rollouts!
                        roll.append(F.huber_loss(x, x_tgt, delta=1.0))

                    loss_step = torch.stack(roll).mean()

                # MoE diversity
                if getattr(model, "use_moe", False) and gate_col_b is not None:
                    avg_usage = gate_col_b.mean(dim=(0, 1))  
                    uniform = torch.full_like(avg_usage, 1.0 / avg_usage.numel())
                    loss_div = torch.sum(avg_usage * (torch.log(avg_usage + 1e-8) - torch.log(uniform + 1e-8)))

            # ==========================
            # B) Data loss
            # ==========================
            u_data_seq = exo_d["u"]  
            
            if ic_dim == 2:
                x_data_b = torch.cat([t_data,y0_y[:, None, :].expand(B, Ld, 1),y0_v[:, None, :].expand(B, Ld, 1),t0[:,  None, :].expand(B, Ld, 1)],dim=-1,)
            else:
                x_data_b = torch.cat([t_data,y0_y[:, None, :].expand(B, Ld, 1),t0[:,  None, :].expand(B, Ld, 1)],dim=-1,)
            
            out_data_b, _ = model.forward_batched(
                x_b=x_data_b,
                u_b=u_data_seq,
                traj_id_b=None,
                context_b=context,   
                detach=True,
            )

            y_data_2 = y_data.reshape(B, Ld, -1)[..., :data_dim] 
            loss_data = criterion(out_data_b[..., 0:data_dim].reshape(-1, data_dim), y_data_2.reshape(-1, data_dim))

            # ==========================
            # C) IC loss
            # ==========================
            u_ic = u_col_seq[:, 0, :].reshape(B, -1)   
            if ic_dim == 2:
                x_ic = torch.cat([t0, y0_y, y0_v, t0], dim=1)  
            else:
                x_ic = torch.cat([t0, y0_y, t0], dim=1)        

            out_ic_b, _ = model.forward_batched(
                x_b=x_ic[:, None, :],        
                u_b=u_ic[:, None, :],        
                traj_id_b=None,
                context_b=context,
                detach=True,
            )

            pred_ic = out_ic_b[:, 0, 0:ic_dim]  
            loss_ic = criterion(pred_ic, y0_base)

            # ==========================
            # Total loss
            # ==========================
            total_loss = (
                w_phys * loss_phys +
                w_data * loss_data +
                w_ic   * loss_ic +
                w_hint * loss_hint +
                w_tv   * loss_tv +
                w_step * loss_step +
                lambda_gate_reg * loss_div
            )

            total_loss.backward()

            step_losses.update(
                {
                    "loss": float(total_loss.detach().item()),
                    "physics_loss": float(loss_phys.detach().item()),
                    "data_loss": float(loss_data.detach().item()),
                    "ic_loss": float(loss_ic.detach().item()),
                    "hint_loss": float(loss_hint.detach().item()),
                    "tv_loss": float(loss_tv.detach().item()),
                    "step_loss": float(loss_step.detach().item()),
                    "diversity_loss": float(loss_div.detach().item()),
                }
            )
            return total_loss

        if phase == 1:
            try:
                optimizer_plant.step(closure)
            except TypeError:
                loss = closure()
                optimizer_plant.step()
        else:
            try:
                optimizer_params.step(closure)
            except TypeError:
                loss = closure()
                optimizer_params.step()

        for k in epoch_metrics:
            epoch_metrics[k] += step_losses.get(k, 0.0)

    n = max(n_batches, 1)
    final_metrics = {k: v / n for k, v in epoch_metrics.items()}

    if phase == 1 and scheduler_plant is not None:
        try:
            scheduler_plant.step(final_metrics["loss"])
        except Exception:
            scheduler_plant.step()

    if phase == 2 and scheduler_params is not None:
        try:
            scheduler_params.step(final_metrics["loss"])
        except Exception:
            scheduler_params.step()

    return final_metrics


def train_epoch_ipinn(
    model: MultitrajectoryIPINN,
    residual: TorchODEResidual,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.MSELoss,
    scheduler=None,
    ic_dim: int = 2,
    data_dim: int = 2,
    lambda_data: float = 100.0,
    lambda_ic: float = 100.0,
    lambda_phys: float = 1.0,
):
    model.train()

    epoch_metrics = {
        "loss": 0.0,
        "physics_loss": 0.0,
        "data_loss": 0.0,
        "ic_loss": 0.0,
    }

    device = next(model.parameters()).device
    n_batches = 0

    for batch in loader:
        n_batches += 1
        
        # -------------------------
        # Unpack Data
        # -------------------------
        # Collocation
        t_col = batch["t_col"].to(device).contiguous().requires_grad_(True) # (B,Lc,1)
        exo_c = {k: v.to(device).contiguous() for k, v in batch["exo_col"].items()}
        
        # Measurement
        t_data = batch["t_data"].to(device).contiguous()
        y_data = batch["y_data"].to(device).contiguous()
        y0 = batch["y0"].to(device).contiguous()
        exo_d = {k: v.to(device).contiguous() for k, v in batch["exo_data"].items()}

        B, Lc, _ = t_col.shape
        _, Ld, _ = t_data.shape
        
        # Reshape y0
        y0_base = y0.reshape(B, -1)[:, :ic_dim] # (B,2) or (B,1)

        def closure():
            optimizer.zero_grad()
            
            # -------------------------
            # 1. Forward Collocation (Physics)
            # -------------------------
            # Context from u_col
            u_col_seq = exo_c["u"] 
            context = model.encode_context(u_col_seq)

            # Flatten for residual
            t_col_flat = t_col.reshape(-1, 1) # (N,1)
            u_col_flat = u_col_seq.reshape(-1, 1)

            # Expand IC/T0 for flattening
            # Note: MultitrajectoryIPINN.forward_pointwise needs x=(t, y0, v0, t0)
            # We reconstruct x_flat manually similar to PI2NDi training
            y0_y = y0_base[:, 0:1]
            t0 = t_col[:, 0, :]

            # Repeat for flattened batch
            t0_flat = t0.repeat_interleave(Lc, dim=0)
            y0_y_flat = y0_y.repeat_interleave(Lc, dim=0)
            
            if ic_dim == 2:
                y0_v = y0_base[:, 1:2]
                y0_v_flat = y0_v.repeat_interleave(Lc, dim=0)
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, y0_v_flat, t0_flat], dim=1)
            else:
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, t0_flat], dim=1)
            
            context_flat = context.repeat_interleave(Lc, dim=0)

            # Forward
            out_col, _ = model.forward_pointwise(
                x_col_flat, u_col_flat,
                context=context_flat,
                detach=False
            )
            
            # Unpack [y, v, k, d, g]
            # k, d, g are constant per batch but expanded in output
            yv_pred = out_col[:, 0:2]
            k_pred = out_col[:, 2:3]
            d_pred = out_col[:, 3:4]
            g_pred = out_col[:, 4:5]

            # Residual
            # Pass k,d,g as exogenous fields since they are outputs of the model
            # but they act as parameters in the equation
            res_val = residual(
                t_col_flat,
                yv_pred,
                exogenous={"u": u_col_flat, "k": k_pred, "d": d_pred, "g": g_pred},
            )
            
            loss_phys = (res_val**2).mean()

            # -------------------------
            # 2. Forward Data (Fidelity)
            # -------------------------
            u_data_seq = exo_d["u"]
            
            # Construct batched x for data
            # Just like PI2NDi train loop
            if ic_dim == 2:
                y0_v = y0_base[:, 1:2]
                x_data_b = torch.cat([
                    t_data,
                    y0_y[:, None, :].expand(B, Ld, 1),
                    y0_v[:, None, :].expand(B, Ld, 1),
                    t0[:, None, :].expand(B, Ld, 1)
                ], dim=-1)
            else:
                 x_data_b = torch.cat([
                    t_data,
                    y0_y[:, None, :].expand(B, Ld, 1),
                    t0[:, None, :].expand(B, Ld, 1)
                ], dim=-1)

            out_data, _ = model.forward_pointwise(
                x_data_b.reshape(-1, model.x_dim),
                u_data_seq.reshape(-1, 1),
                context=context.repeat_interleave(Ld, dim=0),
                detach=False
            )
            
            # Compare only measured states (usually just y, maybe v)
            y_pred_data = out_data[:, 0:data_dim]
            y_target = y_data.reshape(-1, y_data.shape[-1])[:, :data_dim]
            
            loss_data = criterion(y_pred_data, y_target)

            # -------------------------
            # 3. Forward IC (Initial Condition)
            # -------------------------
            # At t=0 (from t_col start)
            u_ic = u_col_seq[:, 0, :].reshape(-1, 1)
            
            if ic_dim == 2:
                x_ic = torch.cat([t0, y0_y, y0_v, t0], dim=1)
            else:
                x_ic = torch.cat([t0, y0_y, t0], dim=1)
                
            out_ic, _ = model.forward_pointwise(
                x_ic, u_ic,
                context=context, # (B,C)
                detach=False
            )
            
            loss_ic = criterion(out_ic[:, :ic_dim], y0_base)

            # -------------------------
            # Total Loss
            # -------------------------
            total_loss = lambda_data * loss_data + lambda_ic * loss_ic + lambda_phys * loss_phys
            
            total_loss.backward()
            
            return {
                "loss": total_loss.item(),
                "physics_loss": loss_phys.item(),
                "data_loss": loss_data.item(),
                "ic_loss": loss_ic.item()
            }

        # Optimizer Step
        step_losses = closure()
        optimizer.step()

        for k in epoch_metrics:
            epoch_metrics[k] += step_losses.get(k, 0.0)

    # Average
    n = max(n_batches, 1)
    final_metrics = {k: v / n for k, v in epoch_metrics.items()}

    if scheduler:
        try:
            scheduler.step(final_metrics["loss"])
        except Exception:
            scheduler.step()

    return final_metrics

def train_epoch_simple_ipinn(
    model: SimpleIPINN,
    residual: TorchODEResidual,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.MSELoss,
    scheduler=None,
    ic_dim: int = 2,
    data_dim: int = 2,
    lambda_data: float = 100.0,
    lambda_ic: float = 100.0,
    lambda_phys: float = 1.0,
):
    model.train()

    epoch_metrics = {
        "loss": 0.0,
        "physics_loss": 0.0,
        "data_loss": 0.0,
        "ic_loss": 0.0,
    }

    device = next(model.parameters()).device
    n_batches = 0

    for batch in loader:
        n_batches += 1
        
        # -------------------------
        # Unpack Data
        # -------------------------
        # Collocation
        t_col = batch["t_col"].to(device).contiguous().requires_grad_(True) # (B,Lc,1)
        exo_c = {k: v.to(device).contiguous() for k, v in batch["exo_col"].items()}
        
        # Measurement
        t_data = batch["t_data"].to(device).contiguous()
        y_data = batch["y_data"].to(device).contiguous()
        y0 = batch["y0"].to(device).contiguous()
        exo_d = {k: v.to(device).contiguous() for k, v in batch["exo_data"].items()}

        B, Lc, _ = t_col.shape
        _, Ld, _ = t_data.shape
        
        # Reshape y0
        y0_base = y0.reshape(B, -1)[:, :ic_dim] # (B,2) or (B,1)

        def closure():
            optimizer.zero_grad()
            
            # -------------------------
            # 1. Forward Collocation (Physics)
            # -------------------------

            # Flatten for residual
            u_col_seq = exo_c["u"] 
            t_col_flat = t_col.reshape(-1, 1) # (N,1)
            u_col_flat = u_col_seq.reshape(-1, 1)

            # Expand IC/T0 for flattening
            # Note: MultitrajectoryIPINN.forward_pointwise needs x=(t, y0, v0, t0)
            # We reconstruct x_flat manually similar to PI2NDi training
            y0_y = y0_base[:, 0:1]
            t0 = t_col[:, 0, :]

            # Repeat for flattened batch
            t0_flat = t0.repeat_interleave(Lc, dim=0)
            y0_y_flat = y0_y.repeat_interleave(Lc, dim=0)
            
            if ic_dim == 2:
                y0_v = y0_base[:, 1:2]
                y0_v_flat = y0_v.repeat_interleave(Lc, dim=0)
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, y0_v_flat, t0_flat], dim=1)
            else:
                x_col_flat = torch.cat([t_col_flat, y0_y_flat, t0_flat], dim=1)
            
            # Forward
            out_col, _ = model.forward_pointwise(x_col_flat, u_col_flat)
            
            # Unpack [y, v, k, d, g]
            # k, d, g are constant per batch but expanded in output
            yv_pred = out_col[:, 0:2]
            k_pred = out_col[:, 2:3]
            d_pred = out_col[:, 3:4]
            g_pred = out_col[:, 4:5]

            # Residual
            # Pass k,d,g as exogenous fields since they are outputs of the model
            # but they act as parameters in the equation
            res_val = residual(
                t_col_flat,
                yv_pred,
                exogenous={"u": u_col_flat, "k": k_pred, "d": d_pred, "g": g_pred},
            )
            
            loss_phys = (res_val**2).mean()

            # -------------------------
            # 2. Forward Data (Fidelity)
            # -------------------------
            u_data_seq = exo_d["u"]
            
            # Construct batched x for data
            # Just like PI2NDi train loop
            if ic_dim == 2:
                y0_v = y0_base[:, 1:2]
                x_data_b = torch.cat([
                    t_data,
                    y0_y[:, None, :].expand(B, Ld, 1),
                    y0_v[:, None, :].expand(B, Ld, 1),
                    t0[:, None, :].expand(B, Ld, 1)
                ], dim=-1)
            else:
                 x_data_b = torch.cat([
                    t_data,
                    y0_y[:, None, :].expand(B, Ld, 1),
                    t0[:, None, :].expand(B, Ld, 1)
                ], dim=-1)

            out_data, _ = model.forward_pointwise(x_data_b.reshape(-1, model.x_dim),u_data_seq.reshape(-1, 1))
            
            # Compare only measured states (usually just y, maybe v)
            y_pred_data = out_data[:, 0:data_dim]
            y_target = y_data.reshape(-1, y_data.shape[-1])[:, :data_dim]
            
            loss_data = criterion(y_pred_data, y_target)

            # -------------------------
            # 3. Forward IC (Initial Condition)
            # -------------------------
            # At t=0 (from t_col start)
            u_ic = u_col_seq[:, 0, :].reshape(-1, 1)
            
            if ic_dim == 2:
                x_ic = torch.cat([t0, y0_y, y0_v, t0], dim=1)
            else:
                x_ic = torch.cat([t0, y0_y, t0], dim=1)
                
            out_ic, _ = model.forward_pointwise(x_ic, u_ic,)
            loss_ic = criterion(out_ic[:, :ic_dim], y0_base)

            # -------------------------
            # Total Loss
            # -------------------------
            total_loss = lambda_data * loss_data + lambda_ic * loss_ic + lambda_phys * loss_phys
            
            total_loss.backward()
            
            return {
                "loss": total_loss.item(),
                "physics_loss": loss_phys.item(),
                "data_loss": loss_data.item(),
                "ic_loss": loss_ic.item()
            }

        # Optimizer Step
        step_losses = closure()
        optimizer.step()

        for k in epoch_metrics:
            epoch_metrics[k] += step_losses.get(k, 0.0)

    # Average
    n = max(n_batches, 1)
    final_metrics = {k: v / n for k, v in epoch_metrics.items()}

    if scheduler:
        try:
            scheduler.step(final_metrics["loss"])
        except Exception:
            scheduler.step()

    return final_metrics