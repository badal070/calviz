"""
calculus.py
Handles safe parsing, symbolic differentiation, integration, and lambdification.
"""

import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify

# ── Symbols ────────────────────────────────────────────────────────────────────
x, y = sp.symbols("x y", real=True)
ALLOWED_SYMBOLS = {"x": x, "y": y}

# Safe namespace for eval – only maths, no builtins abuse
SAFE_NAMESPACE = {
    "x": x, "y": y,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
    "pi": sp.pi, "E": sp.E, "e": sp.E,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "ceiling": sp.ceiling, "floor": sp.floor,
    "__builtins__": {},
}


class CalcEngine:
    """Encapsulates all symbolic operations for a given f(x,y)."""

    def __init__(self):
        self.expr = None          # sympy expression
        self.expr_str = ""
        self.df_dx = None
        self.df_dy = None
        self.integral_x = None    # ∫f dx (antiderivative)
        self.integral_y = None    # ∫f dy

    # ── Parsing ───────────────────────────────────────────────────────────────
    def parse(self, expr_str: str) -> tuple[bool, str]:
        """
        Safely parse a user-supplied expression string.
        Returns (success, message).
        """
        try:
            expr_str = expr_str.strip()
            parsed = eval(expr_str, SAFE_NAMESPACE)   # noqa: S307
            if not isinstance(parsed, (sp.Basic, int, float)):
                return False, "Expression must produce a mathematical value."
            self.expr = sp.sympify(parsed)
            self.expr_str = expr_str
            # Pre-compute derivatives and integrals
            self._differentiate()
            self._integrate()
            return True, "OK"
        except Exception as exc:
            return False, f"Parse error: {exc}"

    # ── Differentiation ───────────────────────────────────────────────────────
    def _differentiate(self):
        self.df_dx = sp.diff(self.expr, x)
        self.df_dy = sp.diff(self.expr, y)

    def get_partial_x_str(self) -> str:
        return sp.pretty(self.df_dx, use_unicode=True) if self.df_dx is not None else ""

    def get_partial_y_str(self) -> str:
        return sp.pretty(self.df_dy, use_unicode=True) if self.df_dy is not None else ""

    # ── Integration ───────────────────────────────────────────────────────────
    def _integrate(self):
        try:
            self.integral_x = sp.integrate(self.expr, x)
        except Exception:
            self.integral_x = None
        try:
            self.integral_y = sp.integrate(self.expr, y)
        except Exception:
            self.integral_y = None

    # ── Lambdification ────────────────────────────────────────────────────────
    def _make_callable(self, sym_expr):
        """Return a numpy-compatible callable from a sympy expression."""
        if sym_expr is None:
            return None
        try:
            fn = lambdify((x, y), sym_expr, modules=["numpy"])
            return fn
        except Exception:
            return None

    def get_surface_fn(self):
        return self._make_callable(self.expr)

    def get_dfdx_fn(self):
        return self._make_callable(self.df_dx)

    def get_dfdy_fn(self):
        return self._make_callable(self.df_dy)

    def get_integral_x_fn(self):
        return self._make_callable(self.integral_x)

    def get_integral_y_fn(self):
        return self._make_callable(self.integral_y)

    # ── Safe evaluation on meshgrid ───────────────────────────────────────────
    @staticmethod
    def safe_eval(fn, X, Y, clip=200.0):
        """
        Evaluate fn(X, Y) on a meshgrid, clipping extreme values and
        replacing NaN/Inf with NaN so Matplotlib skips them.
        """
        if fn is None:
            return np.full_like(X, np.nan, dtype=float)
        try:
            Z = fn(X, Y)
            # Handle scalar return (constant functions)
            if np.ndim(Z) == 0:
                Z = np.full_like(X, float(Z), dtype=float)
            Z = np.where(np.isfinite(Z), Z, np.nan)
            Z = np.clip(Z, -clip, clip)
            return Z.astype(float)
        except Exception:
            return np.full_like(X, np.nan, dtype=float)

    # ── Gradient magnitude (for curvature colormap) ────────────────────────────
    def gradient_magnitude(self, X, Y):
        fx = self.safe_eval(self.get_dfdx_fn(), X, Y)
        fy = self.safe_eval(self.get_dfdy_fn(), X, Y)
        return np.sqrt(fx**2 + fy**2)

    # ── LaTeX strings ─────────────────────────────────────────────────────────
    def latex(self) -> dict:
        out = {}
        if self.expr is not None:
            out["f"] = f"$f(x,y) = {sp.latex(self.expr)}$"
            out["dfdx"] = f"$\\partial f/\\partial x = {sp.latex(self.df_dx)}$"
            out["dfdy"] = f"$\\partial f/\\partial y = {sp.latex(self.df_dy)}$"
            if self.integral_x is not None:
                out["int_x"] = f"$\\int f\\,dx = {sp.latex(self.integral_x)} + C$"
            if self.integral_y is not None:
                out["int_y"] = f"$\\int f\\,dy = {sp.latex(self.integral_y)} + C$"
        return out
