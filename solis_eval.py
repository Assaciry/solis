import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _paper_rcparams():
    """IEEE Standard Styling"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,           # IEEE figures usually use 8pt font
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def kdg_to_K_zetawn(k, d, g, eps=1e-6):
    wn = torch.sqrt(torch.clamp(k, min=eps))
    zeta = d / (2.0 * wn + eps)
    K = g / (k + eps)
    return K, zeta, wn

def compute_metrics(y_true, y_pred):
    """
    Computes RMSE, NRMSE, R2, and Accuracy %
    y_true, y_pred: numpy arrays of shape (N,) or (N, D)
    """
    # Flatten to treat all dimensions equally (or keep per-dim if preferred)
    # Here we compute aggregate metrics over the whole trajectory
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    
    # Range for NRMSE
    y_range = np.max(y_true) - np.min(y_true)
    if y_range < 1e-6: y_range = 1.0
    
    nrmse = rmse / y_range
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    # "Accuracy" as 1 - NRMSE (clamped 0 to 100%)
    acc = max(0.0, (1 - nrmse)) * 100.0
    
    return {
        "RMSE": rmse,
        "NRMSE": nrmse,
        "Accuracy_pct": acc,
        "R2": r2
    }

#############################
## SOLUTION PLOT
############################
@torch.no_grad()
def eval_model(
    models, 
    dataset,
    pinn_dataset=None,
    save_path=None,   
    max_traj_to_plot=5, 
    t_limit=None,       
    convert_params=True,
    plot_params: bool = True,  
    break_loop: bool = False,
):
    _paper_rcparams()
    
    # --- Input Sanitization ---
    if isinstance(models, torch.nn.Module):
        models = {"Model": models}
    elif isinstance(models, list):
        models = {f"M{i}": m for i, m in enumerate(models)}
    
    # Ensure Eval Mode
    for m in models.values():
        m.eval()
    
    device = next(iter(models.values())).parameters().__next__().device
    
    # --- Grouping Logic ---
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        pid = dataset[idx].get("parent_traj_id", dataset[idx].get("source", idx))
        groups[pid].append(idx)
    
    parent_ids = sorted(list(groups.keys()))
    if len(parent_ids) > max_traj_to_plot:
        indices = np.linspace(0, len(parent_ids)-1, max_traj_to_plot, dtype=int)
        parent_ids = [parent_ids[i] for i in indices]

    # --- Setup Figure (3 Rows) ---
    nrows = 3 if plot_params else 2
    fig, axes = plt.subplots(
        nrows, 1, 
        figsize=(4.0, 5.0), # Wider/Taller for complex legend
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.3] if plot_params else [1,1], 
                     "hspace": 0.15}
    )

    # Colors (Okabe-Ito Palette: Colorblind safe & B/W printable)
    # Replaces light yellow with Vermilion, and standard blue with high-contrast Blue
    c_y = "#0072B2"    # Strong Blue
    c_v = "#D55E00"    # Vermilion (Red-Orange) - much easier to see than Yellow
    c_meas = "#000000" # Pure Black for measurements to stand out
    
    # Parameter Colors: Bluish Green, Reddish Purple, Sky Blue
    c_p = ["#009E73", "#CC79A7", "#56B4E9"] 

    # Line Styles for Models (Cyclic)
    # Added explicit tuple for the 4th style to make 'dotted' visible in print
    line_styles = ['-', '--', '-.', (0, (3, 1, 1, 1))] 
    
    ic_kw = dict(marker="o", s=15, facecolors="white", edgecolors="black", linewidths=1.0, zorder=10)
    meas_kw = dict(marker=".", linestyle="None", markersize=2., alpha=.6, color=c_meas, zorder=1)

    # Metrics Storage
    model_metrics = {name: {'y_true': [], 'y_pred': [], 'v_true': [], 'v_pred': []} for name in models}

    # Initialize Twin Axis ONLY IF needed
    if plot_params:
        ax_params = axes[2]
        ax_params_twin = ax_params.twinx() if (plot_params and convert_params) else None 

    # --- Plotting Loop ---
    for pid in parent_ids:
        indices = groups[pid]
        indices = sorted(indices, key=lambda i: dataset[i]["t"][0].item())

        for idx in indices:
            traj = dataset[idx]
            
            t = traj["t"].to(device)
            y_true = traj["y"].to(device)
            u = traj.get("exo", {}).get("u", torch.zeros_like(t)).to(device)
            y0 = traj["y0"].to(device)
            
            t_cpu = t.detach().cpu().numpy().flatten()
            L = t.shape[0]

            # 1. Plot Ground Truth (Thick Transparent) - ONCE
            axes[0].plot(t_cpu, y_true[:,0].cpu().numpy(), color=c_y, linewidth=3.0, alpha=0.25)
            axes[1].plot(t_cpu, y_true[:,1].cpu().numpy(), color=c_v, linewidth=3.0, alpha=0.25)
            
            axes[0].scatter(t_cpu[0].item(), y0[0].item(), **ic_kw)
            if pinn_dataset is not None:
                y_data = pinn_dataset[idx].get("y_data", None)
                t_data = pinn_dataset[idx].get("t_data", None)
                if t_data is not None and y_data is not None:
                    axes[0].plot(t_data, y_data[:, 0].detach().cpu(), **meas_kw)

            # 2. Iterate Models
            for i, (name, model) in enumerate(models.items()):
                ls = line_styles[i % len(line_styles)]
                
                # Forward Pass
                x_in = torch.cat([t, y0[0].view(1,1).expand(L,1), t[0].expand(L,1)], dim=1) 
                if model.x_dim == 4: 
                     x_in = torch.cat([t, y0[0].view(1,1).expand(L,1), y0[1].view(1,1).expand(L,1), t[0].expand(L,1)], dim=1)
                
                context = model.encode_context(u.reshape(-1,1).unsqueeze(0))
                if context is not None: context = context.expand(L, -1)

                out, _ = model.forward_pointwise(x_in, u.reshape(-1,1), context=context)
                
                y_pred = out[:, 0].detach().cpu().numpy()
                v_pred = out[:, 1].detach().cpu().numpy()
                k, d, g = out[:, 2], out[:, 3], out[:, 4]

                # Metrics
                model_metrics[name]['y_true'].append(y_true[:,0].cpu().numpy())
                model_metrics[name]['y_pred'].append(y_pred)
                model_metrics[name]['v_true'].append(y_true[:,1].cpu().numpy())
                model_metrics[name]['v_pred'].append(v_pred)

                # Plot Row 1 (y) & Row 2 (v)
                axes[0].plot(t_cpu, y_pred, color=c_y, linestyle=ls, linewidth=1.2, alpha=0.9)
                axes[1].plot(t_cpu, v_pred, color=c_v, linestyle=ls, linewidth=1.2, alpha=0.9)

                # Plot Row 3 (Params) 
                if plot_params:  
                    if convert_params:
                        # K, Zeta, Wn
                        p1, p2, p3 = kdg_to_K_zetawn(k, d, g) 
                        
                        # Left Axis: Zeta (Red), Wn (Rose)
                        ax_params.plot(t_cpu, p2.detach().cpu().numpy(), color=c_p[1], linestyle=ls, linewidth=1.2)
                        ax_params.plot(t_cpu, p3.detach().cpu().numpy(), color=c_p[2], linestyle=ls, linewidth=1.2)
                        # Right Axis: K (Green)
                        ax_params_twin.plot(t_cpu, p1.detach().cpu().numpy(), color=c_p[0], linestyle=ls, linewidth=1.2)
                    else:
                        # g, d, k
                        p1, p2, p3 = g, d, k 
                        # All Left Axis
                        ax_params.plot(t_cpu, p1.detach().cpu().numpy(), color=c_p[0], linestyle=ls, linewidth=1.2)
                        ax_params.plot(t_cpu, p2.detach().cpu().numpy(), color=c_p[1], linestyle=ls, linewidth=1.2)
                        ax_params.plot(t_cpu, p3.detach().cpu().numpy(), color=c_p[2], linestyle=ls, linewidth=1.2)

        # Break after first parent to avoid clutter
        if break_loop: break

    # --- Legends ---
    # 1. Top Legend: GT vs Model Styles
    h_row1 = [Line2D([0],[0], color=c_y, lw=4, alpha=0.3, label='GT')]
    for i, name in enumerate(models.keys()):
        ls = line_styles[i % len(line_styles)]
        h_row1.append(Line2D([0],[0], color=c_y, linestyle=ls, lw=1.2, label=name))
    
    axes[0].legend(handles=h_row1, loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), 
                #    mode="expand", 
                   ncol=4, frameon=False, fontsize=7)
    axes[0].set_ylabel("$y$")

    # 2. Middle Legend
    h_row2 = [Line2D([0],[0], color=c_v, lw=4, alpha=0.3, label='GT')]
    for i, name in enumerate(models.keys()):
        ls = line_styles[i % len(line_styles)]
        h_row2.append(Line2D([0],[0], color=c_v, linestyle=ls, lw=1.2, label=name))
    
    axes[1].legend(handles=h_row2, loc='lower right', bbox_to_anchor=(0, 0.92, 1, 0.2), 
            #    mode="expand", 
                ncol=4, frameon=False, fontsize=7)
    
    axes[1].set_ylabel("$v$")

    # 3. Bottom Legend + labels ONLY IF params are plotted
    if plot_params:  # <-- added
        h_row3 = []
        
        if convert_params:
            # Green=K, Red=Zeta, Rose=Wn
            param_defs = [('$K$', c_p[0]), ('$\zeta$', c_p[1]), ('$\omega_n$', c_p[2])]
            ylab_l, ylab_r = "$\zeta, \omega_n$", "$K$"
        else:
            # Green=g, Red=d, Rose=k
            param_defs = [('$g$', c_p[0]), ('$d$', c_p[1]), ('$k$', c_p[2])]
            ylab_l, ylab_r = "$d, k, g$", ""

        for param_name, color in param_defs:
            for i, name in enumerate(models.keys()):
                ls = line_styles[i % len(line_styles)]
                label = f"{param_name} {name}"
                h_row3.append(Line2D([0],[0], color=color, linestyle=ls, lw=1.5, label=label))

        ax_params.legend(handles=h_row3, loc='center', 
                        #  bbox_to_anchor=(0.5, -0.35), 
                         ncol=3, frameon=False, fontsize=5)
        
        ax_params.set_ylabel(ylab_l)
        if ax_params_twin:
            ax_params_twin.set_ylabel(ylab_r)
            ax_params_twin.spines['right'].set_visible(True)

    if plot_params:
        ax_params.set_xlabel("Time (s)")
    else:
        axes[1].set_xlabel("Time (s)")

    if t_limit: axes[0].set_xlim(0, t_limit)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0) 
        # plt.savefig(save_path, dpi=300)
    plt.show()

    # --- Metrics Calculation ---
    final_metrics = {}
    for name in models.keys():
        m_dict = {}
        if len(model_metrics[name]['y_true']) > 0:
            yt = np.concatenate(model_metrics[name]['y_true'])
            yp = np.concatenate(model_metrics[name]['y_pred'])
            vt = np.concatenate(model_metrics[name]['v_true'])
            vp = np.concatenate(model_metrics[name]['v_pred'])
            m_dict['y'] = compute_metrics(yt, yp)
            m_dict['v'] = compute_metrics(vt, vp)
        final_metrics[name] = m_dict

    return final_metrics


########################
## Rollout Plot
########################
def plot_surrogate_rollout(
    models, 
    eq_pred, 
    test_ds, 
    traj_idx=0, 
    save_path=None,
    convert_params=True,
    plot_params: bool = True,   # <-- added
    use_running_average_y=False,
    use_running_average_v=False,
    blend_alpha=0.05,
):
    _paper_rcparams()
    
    if isinstance(models, torch.nn.Module): models = {"Model": models}
    elif isinstance(models, list): models = {f"M{i}": m for i, m in enumerate(models)}
    for m in models.values(): m.eval()

    # Data
    traj_data = test_ds[traj_idx]
    t_span = traj_data["t"].squeeze().cpu().numpy()
    t0_abs = traj_data.get("t0_abs", 0.0)
    if torch.is_tensor(t0_abs): t0_abs = t0_abs.item()
    t_plot = t_span + t0_abs
    
    dt = t_span[1] - t_span[0]
    u_data = traj_data["exo"]["u"]
    y_gt = traj_data["y"].cpu().numpy() 

    # Simulate
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            curr_state = traj_data["y"][0:1, :].clone()
            hist_y, hist_p = [], []

            for index, t_curr in enumerate(t_span):
                u_in = u_data[index].unsqueeze(0) 
                
                # Model Pred Params
                model_state_in = curr_state.clone()
                if hasattr(model, 'use_input_normalization') and model.use_input_normalization:
                    u_in_norm = model._norm_u(u_in)
                    model_state_in[0,0] = model._norm_y(model_state_in[0,0])
                    if model.x_dim > 1: model_state_in[0,1] = model._norm_v(model_state_in[0,1])
                else:
                    u_in_norm = u_in

                p_now, _ = model.predict_params(model_state_in, u=u_in_norm, detach_state=True)
                k, d, g = p_now[0, 0], p_now[0, 1], p_now[0, 2]
                
                if convert_params:
                    v1, v2, v3 = kdg_to_K_zetawn(k, d, g)
                else:
                    v1, v2, v3 = g, d, k

                # Physics Step
                curr_state_pred = eq_pred.simulate_one_step(
                    t_curr, curr_state, dt, 
                    exogenous={"u": lambda t: u_data[index], "k": lambda t: k, "d": lambda t: d, "g": lambda t: g}
                )

                curr_state = curr_state_pred
                if use_running_average_y:
                    curr_state[0,0] = (1-blend_alpha)*curr_state[0,0] + blend_alpha*y_gt[index,0]
                if use_running_average_v:
                    curr_state[0,1] = (1-blend_alpha)*curr_state[0,1] + blend_alpha*y_gt[index,1]

                hist_y.append(curr_state.cpu().numpy())
                hist_p.append([v1.item(), v2.item(), v3.item()])
            
            results[name] = {'y': np.concatenate(hist_y, axis=0), 'p': np.array(hist_p)}

    # --- PLOTTING ---
    nrows = 3 if plot_params else 2
    fig, axes = plt.subplots(
        nrows, 1, 
        figsize=(4.0, 5.0), # Wider/Taller for complex legend
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.3] if plot_params else [1,1], 
                     "hspace": 0.15}
    )

    # Colors (Okabe-Ito Palette: Colorblind safe & B/W printable)
    # Replaces light yellow with Vermilion, and standard blue with high-contrast Blue
    c_y = "#0072B2"    # Strong Blue
    c_v = "#D55E00"    # Vermilion (Red-Orange) - much easier to see than Yellow
    
    # Parameter Colors: Bluish Green, Reddish Purple, Sky Blue
    c_p = ["#009E73", "#CC79A7", "#56B4E9"] 

    # Line Styles for Models (Cyclic)
    # Added explicit tuple for the 4th style to make 'dotted' visible in print
    line_styles = ['-', '--', '-.', (0, (3, 1, 1, 1))] 

    # Row 1: y
    axes[0].plot(t_plot, y_gt[:, 0], color=c_y, lw=3, alpha=0.25)
    for i, (name, res) in enumerate(results.items()):
        ls = line_styles[i % len(line_styles)]
        axes[0].plot(t_plot, res['y'][:, 0], color=c_y, ls=ls, lw=1.2, alpha=0.9)
    axes[0].set_ylabel("$y$")

    # Row 2: v
    axes[1].plot(t_plot, y_gt[:, 1], color=c_v, lw=3, alpha=0.25)
    for i, (name, res) in enumerate(results.items()):
        ls = line_styles[i % len(line_styles)]
        axes[1].plot(t_plot, res['y'][:, 1], color=c_v, ls=ls, lw=1.2, alpha=0.9)
    axes[1].set_ylabel("$v$")

    # Row 3: Params (only if enabled)
    if plot_params:  # <-- added
        ax_params = axes[2]
        ax_params_twin = ax_params.twinx() if (plot_params and convert_params) else None  # <-- changed
        for i, (name, res) in enumerate(results.items()):
            ls = line_styles[i % len(line_styles)]
            p = res['p']
            
            if convert_params:
                # p1=K (Green, Twin), p2=Zeta (Red, Left), p3=Wn (Rose, Left)
                ax_params.plot(t_plot, p[:, 1], color=c_p[1], ls=ls, lw=1.2) # Red
                ax_params.plot(t_plot, p[:, 2], color=c_p[2], ls=ls, lw=1.2) # Rose
                ax_params_twin.plot(t_plot, p[:, 0], color=c_p[0], ls=ls, lw=1.2) # Green
            else:
                # p1=g, p2=d, p3=k (All Left)
                ax_params.plot(t_plot, p[:, 0], color=c_p[0], ls=ls, lw=1.2)
                ax_params.plot(t_plot, p[:, 1], color=c_p[1], ls=ls, lw=1.2)
                ax_params.plot(t_plot, p[:, 2], color=c_p[2], ls=ls, lw=1.2)

        if convert_params:
            ax_params.set_ylabel("$\zeta, \omega_n$")
            ax_params_twin.set_ylabel("$K$")
            ax_params_twin.spines['right'].set_visible(True)
        else:
            ax_params.set_ylabel("$d, k, g$")

    if plot_params:
        ax_params.set_xlabel("Time (s)")
    else:
        axes[1].set_xlabel("Time (s)")

    # Legends
    # 1. Top Legend: GT vs Model Styles
    h_row1 = [Line2D([0],[0], color=c_y, lw=4, alpha=0.3, label='GT')]
    for i, name in enumerate(models.keys()):
        ls = line_styles[i % len(line_styles)]
        h_row1.append(Line2D([0],[0], color=c_y, linestyle=ls, lw=1.2, label=name))
    
    axes[0].legend(handles=h_row1, loc='lower right', bbox_to_anchor=(0, 1.02, 1, 0.2), 
                #    mode="expand", 
                   ncol=4, frameon=False, fontsize=7)
    axes[0].set_ylabel("$y$")

    # 2. Middle Legend
    h_row2 = [Line2D([0],[0], color=c_v, lw=4, alpha=0.3, label='GT')]
    for i, name in enumerate(models.keys()):
        ls = line_styles[i % len(line_styles)]
        h_row2.append(Line2D([0],[0], color=c_v, linestyle=ls, lw=1.2, label=name))
    
    axes[1].legend(handles=h_row2, loc='lower right', bbox_to_anchor=(0, 0.92, 1, 0.2), 
            #    mode="expand", 
                ncol=4, frameon=False, fontsize=7)
    
    axes[1].set_ylabel("$v$")

    # Param Legend (only if enabled)
    if plot_params: 
        h_row3 = []
        if convert_params:
            param_defs = [('$K$', c_p[0]), ('$\zeta$', c_p[1]), ('$\omega_n$', c_p[2])]
        else:
            param_defs = [('$g$', c_p[0]), ('$d$', c_p[1]), ('$k$', c_p[2])]

        for param_name, color in param_defs:
            for i, name in enumerate(models.keys()):
                ls = line_styles[i % len(line_styles)]
                label = f"{param_name} {name}"
                h_row3.append(Line2D([0],[0], color=color, linestyle=ls, lw=1.5, label=label))

        ax_params.legend(handles=h_row3, loc='center', 
                         #bbox_to_anchor=(0.5, -0.35), 
                         ncol=3, frameon=False, fontsize=5)

    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0) 
        # plt.savefig(save_path, dpi=300)
    plt.show()

    # Metrics
    metrics = {}
    for name, res in results.items():
        m_dict = {}
        m_dict['y'] = compute_metrics(y_gt[:, 0], res['y'][:, 0])
        m_dict['v'] = compute_metrics(y_gt[:, 1], res['y'][:, 1])
        metrics[name] = m_dict

    return metrics


################################
## PHASE PLOT 
################################
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    dot = (a * b).sum(dim=1)
    na = a.norm(dim=1)
    nb = b.norm(dim=1)
    return dot / (na * nb + eps)

def _to_numpy_traj(traj):
    if traj is None: return None
    if isinstance(traj, torch.Tensor): return traj.detach().cpu().numpy()
    return np.asarray(traj)

@torch.no_grad()
def plot_phase_comparison(
    model, eq_gt, eq_pred,
    trajs=None,                 
    D=100, 
    y_range=(-2.5, 2.5),
    v_range=(-2.5, 2.5),
    u0=0.0,
    save_path=None,
    device=None,
    dtype=torch.float32,
    detach_state=True
):
    _paper_rcparams()
    device = device or next(model.parameters()).device

    # --- 1. Compile Equations ---
    if eq_gt.compiled is None: eq_gt.compiled = eq_gt._compile_first_order()
    if eq_pred.compiled is None: eq_pred.compiled = eq_pred._compile_first_order()
    f_gt = eq_gt.compiled.f_torch
    f_pr = eq_pred.compiled.f_torch

    # --- 2. Create Grid ---
    y_lin = torch.linspace(y_range[0], y_range[1], D, device=device, dtype=dtype)
    v_lin = torch.linspace(v_range[0], v_range[1], D, device=device, dtype=dtype)
    Y, V = torch.meshgrid(y_lin, v_lin, indexing="ij") 
    X = torch.stack([Y.reshape(-1), V.reshape(-1)], dim=1) 
    N = X.shape[0]
    t0 = torch.zeros((N, 1), device=device, dtype=dtype)

    # --- 3. Compute Fields ---
    # Dummy inputs for autonomous phase portrait (u=u0)
    u_tensor = torch.full((N, 1), float(u0), device=device, dtype=dtype)
    
    # GT Field
    # Note: If your eq_gt definition uses "u", we must provide it.
    exo_gt_dummy = { "u": lambda t: u_tensor }
    F_gt = f_gt(t0, X, exogenous=exo_gt_dummy)

    # Pred Field
    params, _ = model.predict_params(X, u=u_tensor, detach_state=detach_state)
    k, d, g = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    
    # Helper to wrap tensor as callable for ODE equation
    def const_exo_tensor(val): return lambda t: val

    exo_pred = {
        "u": lambda t: u_tensor,
        "k": const_exo_tensor(k),
        "d": const_exo_tensor(d),
        "g": const_exo_tensor(g),
    }
    F_pred = f_pr(t0, X, exogenous=exo_pred)

    # --- 4. Process Data for Plotting ---
    Yc = Y.cpu().numpy()
    Vc = V.cpu().numpy()
    
    # Calculate Magnitudes (Speed) for coloring streamlines
    speed_gt = F_gt.norm(dim=1).reshape(D,D).cpu().numpy()
    speed_pred = F_pred.norm(dim=1).reshape(D,D).cpu().numpy()
    
    # Shared Normalization for fair comparison of colors
    max_speed = max(speed_gt.max(), speed_pred.max())
    norm_speed = Normalize(0, max_speed)

    # Cosine Similarity
    cos_sim = cosine_similarity(F_gt, F_pred).reshape(D,D).cpu().numpy()
    
    # --- 5. PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True, gridspec_kw={'wspace': 0.1})
    
    # Helper for overlays
    def overlay_trajs(ax, label_first=False):
        if trajs is not None:
            for i, tr in enumerate(trajs):
                tr_np = _to_numpy_traj(tr)
                mask = (tr_np[:,0] > y_range[0]) & (tr_np[:,0] < y_range[1]) & \
                       (tr_np[:,1] > v_range[0]) & (tr_np[:,1] < v_range[1])
                if mask.any():
                    lbl = None# if (label_first and i==0) else None
                    ax.plot(tr_np[mask,0], tr_np[mask,1], 'w-', alpha=0.35, linewidth=0.8, label=lbl)

    # -- 1. Learned Surrogate --
    axes[0].streamplot(
        Yc.T, Vc.T, 
        F_pred[:,0].reshape(D,D).cpu().numpy().T, 
        F_pred[:,1].reshape(D,D).cpu().numpy().T,
        color=speed_pred.T, cmap='autumn', norm=norm_speed, 
        density=1.2, linewidth=0.7, arrowsize=0.7
    )
    axes[0].set_title("(a) Learned Surrogate $\dot{\hat{\mathbf{x}}}$")
    axes[0].set_ylabel("$v$")
    axes[0].set_xlabel("$y$")
    overlay_trajs(axes[0], label_first=True)
    axes[0].legend(loc='upper right', frameon=False, fontsize=6)

    # -- 2. Ground Truth --
    st_gt = axes[1].streamplot(
        Yc.T, Vc.T, 
        F_gt[:,0].reshape(D,D).cpu().numpy().T, 
        F_gt[:,1].reshape(D,D).cpu().numpy().T,
        color=speed_gt.T, cmap='autumn', norm=norm_speed,
        density=1.2, linewidth=0.7, arrowsize=0.7
    )
    axes[1].set_title("(b) Ground Truth $\dot{\mathbf{x}}$")
    axes[1].set_xlabel("$y$")
    overlay_trajs(axes[1])
    
    # Colorbar for Speed (Shared 1 & 2)
    # div1 = make_axes_locatable(axes[1])
    # cax1 = div1.append_axes("right", size="5%", pad=0.05)
    # cbar1 = plt.colorbar(st_gt.lines, cax=cax1)
    # cbar1.set_label("Speed $|\dot{\mathbf{x}}|$", fontsize=7)

    # -- 3. Similarity --
    im = axes[2].imshow(
        cos_sim.T, 
        extent=[y_range[0], y_range[1], v_range[0], v_range[1]], 
        origin='lower', cmap='viridis', vmin=0.8, vmax=1.0 
    )
    axes[2].set_title("(c) Cosine Similarity")
    axes[2].set_xlabel("$y$")
    overlay_trajs(axes[2])

    # Colorbar for Similarity
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im, cax=cax2)
    # cbar2.set_label("Sim: $\mathbf{f}_{gt} \cdot \mathbf{f}_{pred}$", fontsize=7)

    # Final Polish
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0) 
        # plt.savefig(save_path, dpi=300)
    plt.show()

    # Return Metrics
    metrics = {
        "cos_sim_mean": cos_sim.mean(),
        "l2_err_mean": (F_gt - F_pred).norm(dim=1).mean().item()
    }
    return metrics