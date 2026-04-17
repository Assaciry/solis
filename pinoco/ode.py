"""
PINNKit ODE — a tiny, clean scaffold for prototyping PINNs on ODEs in control.

Features
--------
- Declare ODEs (possibly higher-order) with a tiny DSL using SymPy under the hood.
- Auto-convert higher-order scalar/vector ODEs to first-order state-space.
- Numerics via SciPy `solve_ivp`.
- Torch dataset support.
- NEW: Supports exogenous input derivatives D(u), D(u,2), ...

Quick taste
-----------
from pinnkit_ode import ODEEquation, TorchPINNMediator
import numpy as np, torch

# y' + y = u' + u  with u(t) = sin t
ode = ODEEquation(
    eqs=["Eq(D(y) + y(t), D(u) + u(t))"],
    dependent_variables=["y"],
    exogenous_functions=["u"],
)
sol = ode.solve_ivp((0.0, 6.0), y0=[0.0], exogenous={"u": lambda t: np.sin(t)})
print(sol.t.shape, sol.y.shape)   # (N,), (state_dim, N)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import re

import torch

import sympy as sp
from sympy import Eq, Derivative
from scipy.integrate import solve_ivp as scipy_solve_ivp

Number = Union[int, float, np.ndarray]
ArrayLike = Union[Sequence[float], np.ndarray]

# ----------------------------
# Helpers: tiny DSL for ODEs
# ----------------------------

def _build_sympy_symbols(
    dependent_variables: List[str],
    exogenous_functions: List[str],
    params: Dict[str, float],
    independent_var: str = "t",
):
    """Create SymPy objects used in parsing/compilation.

    Returns a dict suitable for `sympify(locals=...)` and key objects.
    """
    t = sp.Symbol(independent_var, real=True)
    dep_funcs = {name: sp.Function(name) for name in dependent_variables}
    exo_funcs = {name: sp.Function(name) for name in exogenous_functions}
    param_symbols = {name: sp.Symbol(name, real=True) for name in params.keys()}

    # Tiny DSL: D(f, n) -> Derivative(f(t), (t,n)) ; D(f) -> first derivative
    def D(func, n: int = 1):
        """Derivative helper accepting a Function (y) or an applied expr (y(t))."""
        if isinstance(func, sp.FunctionClass):
            expr = func(t)
        else:  # already something like y(t)
            expr = func
        return sp.Derivative(expr, (t, int(n)))

    # Locals map for sympify
    dsl_locals = {
        "t": t,
        "Eq": sp.Eq,
        "D": D,
        **dep_funcs,
        **exo_funcs,
        **param_symbols,
    }
    return t, dep_funcs, exo_funcs, param_symbols, dsl_locals


# ----------------------------
# Finite-difference helpers
# ----------------------------

def _numeric_step_from_t_scalar(t: float) -> float:
    # Scales with |t| to avoid subtractive cancellation; bounded below.
    return max(1e-6, 1e-6 * (1.0 + abs(float(t))))

def _central_diff_numpy(f: Callable[[float], float], t: float, order: int, h: Optional[float] = None) -> float:
    """Centered finite differences for orders 1..3."""
    if order <= 0:
        return float(f(t))
    if h is None:
        h = _numeric_step_from_t_scalar(t)
    if order == 1:
        return float((f(t + h) - f(t - h)) / (2.0 * h))
    if order == 2:
        return float((f(t + h) - 2.0 * f(t) + f(t - h)) / (h * h))
    if order == 3:
        # 3rd order accurate central stencil for 3rd derivative
        return float((f(t + 2*h) - 2*f(t + h) + 2*f(t - h) - f(t - 2*h)) / (2.0 * (h**3)))
    raise ValueError("Finite-difference derivatives supported up to order 3.")

def _central_diff_torch(f: Callable[["torch.Tensor"], "torch.Tensor"], t: "torch.Tensor", order: int, h: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    """Torch finite differences for orders 1..3; works batched on (N,1)."""
    if order <= 0:
        return f(t)
    if h is None:
        # Per-sample step
        h = (1e-6 * (1.0 + torch.abs(t))).to(t.dtype)
    if h.ndim == 0:
        h = h.reshape(1, 1).expand_as(t)
    if order == 1:
        return (f(t + h) - f(t - h)) / (2.0 * h)
    if order == 2:
        return (f(t + h) - 2.0 * f(t) + f(t - h)) / (h * h)
    if order == 3:
        return (f(t + 2*h) - 2*f(t + h) + 2*f(t - h) - f(t - 2*h)) / (2.0 * (h**3))
    raise ValueError("Finite-difference derivatives supported up to order 3.")

def _autograd_nth_derivative(f: Callable[["torch.Tensor"], "torch.Tensor"], t: "torch.Tensor", n: int) -> "torch.Tensor":
    """Try autograd to compute nth derivative; returns tensor or raises RuntimeError."""
    # Ensure t requires grad
    if not t.requires_grad:
        t.requires_grad_(True)
    y = f(t)
    for k in range(n):
        grad = torch.autograd.grad(
            y, t, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True
        )[0]
        if grad is None:
            raise RuntimeError("autograd returned None (non-differentiable exogenous?).")
        y = grad
    return y


@dataclass
class CompiledFirstOrder:
    """Holds compiled first-order dynamics and metadata."""
    t: sp.Symbol
    state_syms: List[sp.Symbol]  # x0, x1, ...
    xdot_exprs: List[sp.Expr]    # same length as state_syms
    param_syms: Dict[str, sp.Symbol]
    # Exogenous value symbols:
    exo_value_syms: Dict[str, sp.Symbol]  # u_val (order 0)
    # NEW: exogenous derivative symbol table and orders
    exo_deriv_value_syms: Dict[Tuple[str, int], sp.Symbol]  # (name,k) -> u_dk_val
    exo_max_order: Dict[str, int]  # required derivative order per exogenous
    dep_decomp: Dict[str, List[sp.Symbol]]  # per dependent var: [y, y', ..., y^(n-1)] states

    # Numeric functions (NumPy + torch-native)
    f_numpy: Callable[[float, np.ndarray, Dict[str, float], Dict[str, Callable[[float], Number]]], np.ndarray]
    f_torch: Optional[Callable[["torch.Tensor", "torch.Tensor", Dict[str, Callable[["torch.Tensor"], "torch.Tensor"]], Optional[Dict[str, float]]], "torch.Tensor"]] = None


@dataclass
class ODEEquation:
    """Symbolic ODE definition + numeric compilation.

    Define ODEs with a small DSL and automatically compile to first-order state-space.

    Examples
    --------
    # y'' + y = k*u(t)
    node = ODEEquation(
        eqs=["Eq(D(y,2) + y(t), k*u(t))"],
        params={"k": 2.0},
        dependent_variables=["y"],
        exogenous_functions=["u"],
    )
    sol = node.solve_ivp(t_span=(0,10), y0=[0,0], exogenous={"u": lambda t: np.sin(t)})
    """

    independent_var: str = "t"
    eqs: List[Eq] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    dependent_variables: List[str] = field(default_factory=lambda: ["y"])  # names
    exogenous_functions: List[str] = field(default_factory=list)

    compiled: Optional[CompiledFirstOrder] = None

    # ----------------------
    # Construction methods
    # ----------------------
    def __init__(
        self,
        eqs: Iterable[Union[str, Eq]],
        params: Optional[Dict[str, float]] = None,
        dependent_variables: Optional[Iterable[str]] = None,
        exogenous_functions: Optional[Iterable[str]] = None,
        name : Optional[str] = ""
    ) -> None:
        """Construct the ODEEquation and Assign the ODEs in symbolic form.

        Parameters
        ----------
        eqs : iterable[str|sympy.Eq]
            Differential equations. Use the DSL: `Eq(LHS,RHS)` or `LHS`  
            
            Examples:
                - `Eq(D(y,2) + y, k*u)`
                - 'D(y,2) + b*D(y) + y' 
            - `D(y)` is dy/dt, `D(y,2)` is d^2 y / dt^2, etc.
            - Explicit multiplication only (`k*u`, not `ku`).
        params : dict
            Parameter values `{name: float}`. Can be updated later.
        dependent_variables : iterable[str]
            Names of dependent functions, e.g., ["y", "z"].
        exogenous_functions : iterable[str]
            Names of known exogenous functions of time, e.g., ["u"] used as `u` in strings.
        """
        if params is not None:
            self.params = dict(params)
        if dependent_variables is not None:
            self.dependent_variables = list(dependent_variables)
        if exogenous_functions is not None:
            self.exogenous_functions = list(exogenous_functions)

        t, dep_funcs, exo_funcs, param_syms, dsl_locals = _build_sympy_symbols(
            self.dependent_variables, self.exogenous_functions, self.params, self.independent_var
        )

        self.name = name

        parsed_eqs: List[Eq] = []
        for e in eqs:
            if isinstance(e, str):
                e_str = e
                # Expand bare dependent/exogenous names to applied form name(t),
                # but do NOT touch those already followed by '('.
                for name in self.dependent_variables:
                    e_str = re.sub(rf"\b{name}\b(?!\s*\()", rf"{name}(t)", e_str)
                for name in self.exogenous_functions:
                    e_str = re.sub(rf"\b{name}\b(?!\s*\()", rf"{name}(t)", e_str)
                expr = sp.sympify(e_str, locals=dsl_locals)
                if isinstance(expr, Eq):
                    parsed_eqs.append(expr)
                else:
                    # If user wrote an expression, assume == 0
                    parsed_eqs.append(sp.Eq(expr, 0))
            elif isinstance(e, Eq):
                parsed_eqs.append(e)
            else:
                raise TypeError("Each equation must be a str or sympy.Eq")

        self.eqs = parsed_eqs
        self.orders = [max((d.derivative_count for d in eq.atoms(Derivative)), default=0) for eq in self.eqs]
        self.compiled = None  # reset compilation

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        name_part = f"{self.name}" if self.name else ""
        eqs_repr = "\n\t".join(repr(eq) for eq in self.eqs)
        return f"[{cls_name}] {name_part}\nEquations:\n\t{eqs_repr}"

    # ----------------------
    # Compilation
    # ----------------------
    def _compile_first_order(self) -> CompiledFirstOrder:
        if not self.eqs:
            raise ValueError("No equations assigned.")

        t, dep_funcs, exo_funcs, param_syms, _ = _build_sympy_symbols(
            self.dependent_variables, self.exogenous_functions, self.params, self.independent_var
        )

        # Determine highest derivative order for each dependent variable
        max_order: Dict[str, int] = {name: 0 for name in self.dependent_variables}
        for eq in self.eqs:
            for name, f in dep_funcs.items():
                for d in eq.atoms(sp.Derivative):
                    if d.expr == f(t):
                        # d.variables is a tuple like (t, t, ...) or ((t, n),)
                        if len(d.variables) == 1 and isinstance(d.variables[0], tuple):
                            order = int(d.variables[0][1])
                        else:
                            order = sum(1 for v in d.variables if v == t)
                        if order > max_order[name]:
                            max_order[name] = order
            # If variable appears without derivative, consider order >= 0
            for name, f in dep_funcs.items():
                if eq.has(f(t)) and max_order[name] == 0:
                    max_order[name] = 1  # at least first-order state to carry y

        # Ensure each dependent has at least order 1 (first-order state)
        for k, v in list(max_order.items()):
            if v <= 0:
                max_order[k] = 1

        # Unknowns to solve for: highest derivatives y^(n)
        highest_derivs = []
        for name, f in dep_funcs.items():
            n = max_order[name]
            highest_derivs.append(sp.Derivative(f(t), (t, n)))

        # Solve equations for highest derivatives
        sol = sp.solve(self.eqs, highest_derivs, dict=True)
        if not sol:
            raise ValueError("Could not solve equations for highest derivatives. Check well-posedness.")
        sol = sol[0]  # pick first solution

        # Build state vector symbols x0, x1, ... and mapping per dependent var to its states
        state_syms: List[sp.Symbol] = []
        dep_decomp: Dict[str, List[sp.Symbol]] = {}
        for name, f in dep_funcs.items():
            n = max_order[name]
            states = []
            for k in range(n):
                sym = sp.Symbol(f"{name}_d{k}", real=True)  # y_d0 ~ y, y_d1 ~ y', ...
                states.append(sym)
                state_syms.append(sym)
            dep_decomp[name] = states

        # Build expressions for xdot for each state
        xdot_exprs: List[sp.Expr] = []

        # --- NEW: exogenous derivative requirements scan ---
        exo_max_order: Dict[str, int] = {name: 0 for name in self.exogenous_functions}
        for eq in self.eqs:
            for name, f in exo_funcs.items():
                # Direct presence of u(t) means at least order 0; derivatives bump order
                if eq.has(f(t)):
                    exo_max_order[name] = max(exo_max_order[name], 0)
                for d in eq.atoms(sp.Derivative):
                    if d.expr == f(t):
                        if len(d.variables) == 1 and isinstance(d.variables[0], tuple):
                            order = int(d.variables[0][1])
                        else:
                            order = sum(1 for v in d.variables if v == t)
                        exo_max_order[name] = max(exo_max_order[name], order)

        # Replace exogenous functions and their derivatives with standalone symbols
        exo_value_syms: Dict[str, sp.Symbol] = {name: sp.Symbol(f"{name}_val", real=True) for name in self.exogenous_functions}
        exo_deriv_value_syms: Dict[Tuple[str, int], sp.Symbol] = {}
        for name in self.exogenous_functions:
            for k in range(1, exo_max_order[name] + 1):
                exo_deriv_value_syms[(name, k)] = sp.Symbol(f"{name}_d{k}_val", real=True)

        # Substitution dicts for exogenous
        subs_exo: Dict[sp.Expr, sp.Symbol] = {exo_funcs[name](t): exo_value_syms[name] for name in self.exogenous_functions}
        subs_exo_derivs: Dict[sp.Expr, sp.Symbol] = {}
        for name in self.exogenous_functions:
            for k in range(1, exo_max_order[name] + 1):
                subs_exo_derivs[sp.Derivative(exo_funcs[name](t), (t, k))] = exo_deriv_value_syms[(name, k)]

        # Helper: express any derivative y^(k) in terms of state symbols and exogenous symbols
        def _lower_deriv(expr: sp.Expr) -> sp.Expr:
            # Replace exogenous derivatives first, then base exogenous, then dependent derivatives.
            expr = expr.xreplace(subs_exo_derivs).xreplace(subs_exo)
            for name, f in dep_funcs.items():
                states = dep_decomp[name]
                # map y'(t)-> y_d1, y''(t)-> y_d2, ... (do higher orders first)
                for k in range(len(states) - 1, 0, -1):
                    expr = expr.xreplace({sp.Derivative(f(t), (t, k)): states[k]})
                # now map y(t) -> y_d0
                expr = expr.xreplace({f(t): states[0]})
            return expr

        # First, precompute highest derivative expressions lowered to states
        highest_exprs_lowered: Dict[sp.Derivative, sp.Expr] = {}
        for name, f in dep_funcs.items():
            n = max_order[name]
            hd = sp.Derivative(f(t), (t, n))
            if hd not in sol:
                raise ValueError(f"Equation for highest derivative {name}^{n} not found.")
            highest_exprs_lowered[hd] = _lower_deriv(sol[hd])

        # Now assemble xdot
        for name, states in dep_decomp.items():
            n = len(states)
            if n == 1:
                # Only y: y' = highest_exprs_lowered[y^(1)]
                hd = sp.Derivative(sp.Function(name)(t), (t, 1))
                xdot_exprs.append(highest_exprs_lowered[hd])
            else:
                # chain: y_d0' = y_d1, y_d1' = y_d2, ..., y_d{n-2}' = y_d{n-1}
                for k in range(n - 1):
                    xdot_exprs.append(states[k + 1])
                # last one uses highest derivative
                hd = sp.Derivative(sp.Function(name)(t), (t, n))
                xdot_exprs.append(highest_exprs_lowered[hd])

        # Build NumPy function f_numpy(t, x, params, exogenous)
        # Prepare lambdify variables: [t, *state_syms, *param_syms, *exo_syms (val + derivatives)]
        # Order exogenous symbols deterministically by self.exogenous_functions and increasing derivative order
        exo_symbol_sequence: List[sp.Symbol] = []
        for name in self.exogenous_functions:
            exo_symbol_sequence.append(exo_value_syms[name])
            for k in range(1, exo_max_order[name] + 1):
                exo_symbol_sequence.append(exo_deriv_value_syms[(name, k)])

        lamb_syms = [t] + state_syms + list(param_syms.values()) + exo_symbol_sequence
        xdot_numpy_fns = [sp.lambdify(lamb_syms, expr, modules="numpy") for expr in xdot_exprs]

        def f_numpy(
            t_num: float,
            x_num: np.ndarray,
            params_num: Dict[str, float],
            exogenous: Dict[str, Union[Callable[[float], Number], Dict[Union[int,str], Callable[[float], Number]], Sequence[Callable]]] | None = None,
        ) -> np.ndarray:
            vals = [t_num]
            vals.extend([float(v) for v in np.asarray(x_num).ravel()])
            for name, sym in param_syms.items():
                vals.append(float(params_num.get(name, float(self.params.get(name, 0.0)))))

            # Pack exogenous values
            if exogenous is None:
                # zeros for all required symbols
                for _ in exo_symbol_sequence:
                    vals.append(0.0)
            else:
                for name in self.exogenous_functions:
                    entry = exogenous.get(name, None)
                    # Determine value function and optional derivative functions
                    val_func = None
                    deriv_funcs: Dict[int, Callable[[float], Number]] = {}
                    if entry is None:
                        # all zeros
                        vals.append(0.0)
                        for k in range(1, exo_max_order[name] + 1):
                            vals.append(0.0)
                        continue
                    if callable(entry):
                        val_func = entry
                    elif isinstance(entry, (list, tuple)):
                        if len(entry) >= 1 and callable(entry[0]):
                            val_func = entry[0]
                        for k in range(1, min(len(entry), exo_max_order[name]) + 1):
                            if callable(entry[k]):
                                deriv_funcs[k] = entry[k]
                    elif isinstance(entry, dict):
                        # keys may be "val"/0 for value, and integers for derivatives
                        if "val" in entry and callable(entry["val"]):
                            val_func = entry["val"]
                        elif 0 in entry and callable(entry[0]):
                            val_func = entry[0]
                        for k in range(1, exo_max_order[name] + 1):
                            fnk = entry.get(k, None)
                            if callable(fnk):
                                deriv_funcs[k] = fnk
                    else:
                        raise TypeError("Unsupported exogenous entry type.")

                    if val_func is None:
                        # default 0 if value not provided
                        base_val = 0.0
                        vals.append(base_val)
                        for k in range(1, exo_max_order[name] + 1):
                            vals.append(0.0)
                        continue

                    base_val = float(val_func(t_num))
                    vals.append(base_val)

                    # Derivatives
                    for k in range(1, exo_max_order[name] + 1):
                        if k in deriv_funcs:
                            vals.append(float(deriv_funcs[k](t_num)))
                        else:
                            vals.append(_central_diff_numpy(val_func, t_num, k))

            out = [fn(*vals) for fn in xdot_numpy_fns]
            return np.asarray(out, dtype=float)

        # Torch function (optional)
        f_torch = None
        if torch is not None:
            # Minimal mapping of SymPy functions to torch ops
            torch_modules = {
                "sin": torch.sin,
                "cos": torch.cos,
                "tan": torch.tan,
                "asin": torch.arcsin,
                "acos": torch.arccos,
                "atan": torch.arctan,
                "sinh": torch.sinh,
                "cosh": torch.cosh,
                "tanh": torch.tanh,
                "exp": torch.exp,
                "log": torch.log,
                "sqrt": torch.sqrt,
                "Abs": torch.abs,
                "sign": torch.sign,
                "Piecewise": sp.lambdify,  # unsupported directly; discourage in ODEs
            }
            xdot_torch_fns = [sp.lambdify(lamb_syms, expr, modules=[torch_modules, "math"]) for expr in xdot_exprs]

            def f_torch(
                t_tensor: "torch.Tensor",
                x_tensor: "torch.Tensor",
                exogenous: Dict[str, Union[Callable[["torch.Tensor"], "torch.Tensor"], Dict[Union[int,str], Callable], Sequence[Callable]]] | None = None,
                params_override: Optional[Dict[str, float|torch.Tensor]] = None,
            ) -> "torch.Tensor":
                # t_tensor: (N,1) or (N,) ; x_tensor: (N, state_dim)
                t_flat = t_tensor.reshape(-1, 1)
                x_flat = x_tensor.reshape(t_flat.shape[0], -1)
                # Build param vector (scalar constants broadcasted)
                pvals = []
                # psrc = params_override or self.params
                _param_names = list(param_syms.keys())
                for name in _param_names:
                    val = (params_override or self.params).get(name, 0.0)
                    if torch.is_tensor(val):
                        pv = val.to(dtype=t_flat.dtype, device=t_flat.device)
                        if pv.ndim == 0:
                            pv = pv.reshape(1, 1)
                        pv = pv.expand_as(t_flat)              # keep grad!
                    else:
                        pv = torch.as_tensor(float(val), dtype=t_flat.dtype, device=t_flat.device).expand_as(t_flat)
                    pvals.append(pv)


                # Collect exogenous values (val + derivatives)
                exo_vals_tensors: List["torch.Tensor"] = []
                need_any_derivs = any(exo_max_order[n] > 0 for n in self.exogenous_functions)

                # Ensure we *can* differentiate w.r.t. t if needed
                if need_any_derivs and not t_flat.requires_grad:
                    t_flat.requires_grad_(True)

                if exogenous is None or len(self.exogenous_functions) == 0:
                    for name in self.exogenous_functions:
                        exo_vals_tensors.append(torch.zeros_like(t_flat))  # val
                        for k in range(1, exo_max_order[name] + 1):
                            exo_vals_tensors.append(torch.zeros_like(t_flat))
                else:
                    for name in self.exogenous_functions:
                        entry = exogenous.get(name, None)
                        def _resolve_entry_callable_k(entry, k: int) -> Optional[Callable]:
                            if callable(entry):
                                return entry if k == 0 else None
                            if isinstance(entry, (list, tuple)):
                                if k < len(entry) and callable(entry[k]):
                                    return entry[k]
                                return None
                            if isinstance(entry, dict):
                                if k == 0:
                                    fn = entry.get("val", entry.get(0, None))
                                else:
                                    fn = entry.get(k, None)
                                return fn if callable(fn) else None
                            return None

                        # value (k=0)
                        val_fn = _resolve_entry_callable_k(entry, 0)
                        if val_fn is None:
                            v = torch.zeros_like(t_flat)
                        else:
                            v = val_fn(t_flat)
                            if v.ndim == 1:
                                v = v.reshape(-1, 1)
                        exo_vals_tensors.append(v)

                        # derivatives
                        for k in range(1, exo_max_order[name] + 1):
                            dfn = _resolve_entry_callable_k(entry, k)
                            if dfn is not None:
                                dv = dfn(t_flat)
                                if dv.ndim == 1:
                                    dv = dv.reshape(-1, 1)
                                exo_vals_tensors.append(dv)
                                continue
                            # Try autograd
                            if val_fn is not None:
                                try:
                                    dv = _autograd_nth_derivative(val_fn, t_flat, k)
                                    if dv.ndim == 1:
                                        dv = dv.reshape(-1, 1)
                                    exo_vals_tensors.append(dv)
                                    continue
                                except Exception:
                                    pass
                            # Fallback: centered finite difference (torch)
                            if val_fn is not None:
                                dv = _central_diff_torch(val_fn, t_flat, k)
                            else:
                                dv = torch.zeros_like(t_flat)
                            if dv.ndim == 1:
                                dv = dv.reshape(-1, 1)
                            exo_vals_tensors.append(dv)

                outs = []
                for i, fn in enumerate(xdot_torch_fns):
                    vals = [t_flat] + [x_flat[:, j:j+1] for j in range(x_flat.shape[1])]
                    vals += pvals
                    vals += exo_vals_tensors
                    out_i = fn(*vals)
                    # Ensure output shape (N,1)
                    if not torch.is_tensor(out_i):
                        out_i = torch.as_tensor(out_i, dtype=t_flat.dtype, device=t_flat.device)
                    # out_i = out_i.reshape(-1, 1)
                    out_i = _to_N1(out_i, t_flat)
                    outs.append(out_i)
                return torch.hstack(outs)

        return CompiledFirstOrder(
            t=t,
            state_syms=state_syms,
            xdot_exprs=xdot_exprs,
            param_syms=param_syms,
            exo_value_syms=exo_value_syms,
            exo_deriv_value_syms=exo_deriv_value_syms,
            exo_max_order=exo_max_order,
            dep_decomp=dep_decomp,
            f_numpy=f_numpy,
            f_torch=f_torch,
        )

    @property
    def total_state_dim(self) -> int:
        if self.compiled is None:
            # attempt compile minimally to know dim
            compiled = self._compile_first_order()
            self.compiled = compiled
        return len(self.compiled.state_syms)


    # ----------------------
    # IVP solve (SciPy)
    # ----------------------
    def solve_ivp(
        self,
        t_span: Tuple[float, float],
        y0: ArrayLike,
        exogenous: Optional[Dict[str, Union[
            Callable[[float], Number],
            Dict[Union[int,str], Callable[[float], Number]],
            Sequence[Callable[[float], Number]]
        ]]] = None,
        params_override: Optional[Dict[str, float]] = None,
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **kwargs,
    ):
        """Solve the compiled ODE as a first-order IVP.

        Parameters
        ----------
        y0 : array-like
            Initial condition vector in *state* coordinates. For y'' + y = ...,
            y0 = [y(0), y'(0)].
        exogenous : dict[name -> callable or dict/sequence]
            For each exogenous "u", you can pass:
              - callable(t)->u(t)                # derivatives are auto-approximated
              - [u, du, d2u, ...]                # sequence of callables
              - {"val": u, 1: du, 2: d2u, ...}   # dict with explicit derivatives
        params_override : dict
            Override parameter values for this solve.
        Returns
        -------
        OdeResult from SciPy, with `y` shaped (state_dim, len(t)).
        """
        if self.compiled is None:
            self.compiled = self._compile_first_order()
        comp = self.compiled

        if params_override is None:
            params_override = self.params

        y0 = np.asarray(y0, dtype=float).reshape(-1)
        if y0.shape[0] != len(comp.state_syms):
            raise ValueError(f"y0 has dim {y0.shape[0]}, expected {len(comp.state_syms)}")

        def rhs(t_num, x_num):
            return comp.f_numpy(t_num, x_num, params_override, exogenous)

        return scipy_solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    def simulate_one_step(
        self,
        t,
        x,
        dt,
        exogenous: Optional[Dict[str, Union[
            Callable[["torch.Tensor"], "torch.Tensor"],
            Dict[Union[int,str], Callable],
            Sequence[Callable]
        ]]] = None,
        params_override: Optional[Dict[str, float]] = None,
        method: str = "rk4",
    ):
        """
        Advance one integration step for the compiled first-order ODE.

        Works with NumPy arrays (float) *and* PyTorch tensors (keeps autograd graph).
        If `x` is a torch.Tensor, the step is differentiable w.r.t. x, t, dt, and any
        parameters used inside the symbolic RHS.

        Parameters
        ----------
        t : float or torch.Tensor
            Current time. If torch, may be scalar (), shape (1,1), or batched (N,1)/(N,).
        x : np.ndarray or torch.Tensor
            Current state. Shape (state_dim,) or batched (N, state_dim).
        dt : float or torch.Tensor
            Step size. If torch, may be scalar or broadcastable to (N,1).
        exogenous : dict[str -> callable/dict/sequence], optional
            Functions of time for exogenous inputs (and optionally their derivatives).
            - NumPy mode: callables take float t -> float.
            - Torch mode: callables take tensor t (N,1) -> tensor (N,1).
        params_override : dict, optional
            Parameter overrides for this step (else defaults to self.params).
        method : {"euler", "rk4"}
            Integration scheme.

        Returns
        -------
        Next state with the same array/tensor type and shape as `x`.
        """
        if self.compiled is None:
            self.compiled = self._compile_first_order()
        comp = self.compiled

        method = method.lower()
        if method not in ("euler", "rk4"):
            raise ValueError("method must be 'euler' or 'rk4'")

        # ----- Torch pathway -----
        if (torch is not None) and isinstance(x, torch.Tensor):
            if comp.f_torch is None:
                raise RuntimeError("Torch RHS not available; check your equation or install torch.")
            f = comp.f_torch

            # Normalize shapes
            x_in = x
            batched = x_in.ndim == 2
            if not batched:
                x_b = x_in.reshape(1, -1)
            else:
                x_b = x_in

            N = x_b.shape[0]
            # time tensor (N,1)
            if not torch.is_tensor(t):
                t_t = torch.as_tensor(t, dtype=x_b.dtype, device=x_b.device)
            else:
                t_t = t.to(dtype=x_b.dtype, device=x_b.device)
            if t_t.ndim == 0:
                t_t = t_t.reshape(1, 1).expand(N, 1)
            elif t_t.ndim == 1:
                t_t = t_t.reshape(-1, 1)
            elif t_t.ndim == 2 and t_t.shape[1] == 1:
                pass
            else:
                raise ValueError("t tensor must be scalar, (N,), or (N,1)")

            # dt tensor broadcastable to (N,1)
            if not torch.is_tensor(dt):
                dt_t = torch.as_tensor(dt, dtype=x_b.dtype, device=x_b.device)
            else:
                dt_t = dt.to(dtype=x_b.dtype, device=x_b.device)
            if dt_t.ndim == 0:
                dt_t = dt_t.reshape(1, 1).expand_as(t_t)
            elif dt_t.ndim == 1:
                dt_t = dt_t.reshape(-1, 1)
            elif dt_t.ndim == 2 and dt_t.shape[1] == 1:
                # if different N, try to expand
                if dt_t.shape[0] == 1 and t_t.shape[0] > 1:
                    dt_t = dt_t.expand_as(t_t)
            else:
                raise ValueError("dt tensor must be scalar, (N,), or (N,1)")

            if method == "euler":
                k1 = f(t_t, x_b, exogenous, params_override)  # (N, D)
                x_next = x_b + dt_t * k1
            else:  # RK4
                h = dt_t
                half = 0.5 * h
                k1 = f(t_t,         x_b,             exogenous, params_override)
                k2 = f(t_t + half,  x_b + half * k1, exogenous, params_override)
                k3 = f(t_t + half,  x_b + half * k2, exogenous, params_override)
                k4 = f(t_t + h,     x_b + h * k3,    exogenous, params_override)
                x_next = x_b + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            return x_next.reshape(x_in.shape)

        else:
            raise ValueError("Inputs must be torch tensors.")

def _to_N1(out, ref_t):
    # ref_t is (N,1)
    if not torch.is_tensor(out):
        out = torch.as_tensor(out, dtype=ref_t.dtype, device=ref_t.device)

    # Convert scalars / (1,) / (1,1) to (N,1)
    if out.ndim == 0:
        out = out.reshape(1, 1).expand_as(ref_t)
    elif out.ndim == 1:
        if out.numel() == 1:
            out = out.reshape(1, 1).expand_as(ref_t)
        else:
            out = out.reshape(-1, 1)
    elif out.ndim == 2:
        if out.shape == (1, 1):
            out = out.expand_as(ref_t)
        elif out.shape[1] != 1:
            out = out.reshape(-1, 1)
    else:
        out = out.reshape(-1, 1)

    # Final guard
    if out.shape != ref_t.shape:
        # If it's still broadcastable (e.g., (N,) -> (N,1)), make it so.
        if out.numel() == 1:
            out = out.reshape(1, 1).expand_as(ref_t)
        else:
            raise RuntimeError(f"RHS produced shape {tuple(out.shape)}; expected {tuple(ref_t.shape)}")
    return out
