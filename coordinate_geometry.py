"""
coordinate_geometry.py
Handles safe parsing and symbolic analysis for implicit coordinate-geometry
relations F(x, y) = 0.
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

from calculus import SAFE_NAMESPACE, x, y


class CoordinateGeometryEngine:
    """Encapsulates symbolic operations for an implicit relation F(x, y) = 0."""

    def __init__(self):
        self.relation = None
        self.relation_str = ""
        self.fx = None
        self.fy = None
        self.slope = None
        self.classification = "Unclassified"
        self.x_intercepts = []
        self.y_intercepts = []

    def parse(self, expr_str: str) -> tuple[bool, str]:
        """
        Safely parse a user-supplied coordinate-geometry relation.
        Supports either:
        - x**2 + y**2 = 9
        - x**2 + y**2 - 9
        """
        try:
            expr_str = expr_str.strip()
            if not expr_str:
                return False, "Please enter a relation."

            if "=" in expr_str:
                lhs_str, rhs_str = expr_str.split("=", 1)
                lhs = eval(lhs_str.strip(), SAFE_NAMESPACE)  # noqa: S307
                rhs = eval(rhs_str.strip(), SAFE_NAMESPACE)  # noqa: S307
                relation = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                parsed = eval(expr_str, SAFE_NAMESPACE)  # noqa: S307
                relation = sp.sympify(parsed)

            if not isinstance(relation, (sp.Basic, int, float)):
                return False, "Relation must produce a mathematical value."

            self.relation = sp.expand(sp.sympify(relation))
            self.relation_str = expr_str
            self._differentiate()
            self._classify_relation()
            self._find_intercepts()
            return True, "OK"
        except Exception as exc:
            return False, f"Parse error: {exc}"

    def _differentiate(self):
        self.fx = sp.diff(self.relation, x)
        self.fy = sp.diff(self.relation, y)
        try:
            self.slope = sp.simplify(-self.fx / self.fy)
        except Exception:
            self.slope = None

    def _classify_relation(self):
        self.classification = "General implicit relation"
        try:
            poly = sp.Poly(self.relation, x, y)
        except sp.PolynomialError:
            return

        degree = poly.total_degree()
        if degree <= 0:
            self.classification = "Degenerate relation"
            return
        if degree == 1:
            self.classification = "Line"
            return
        if degree > 2:
            self.classification = "Higher-order curve"
            return

        a = poly.coeff_monomial(x**2)
        b = poly.coeff_monomial(x * y)
        c = poly.coeff_monomial(y**2)
        disc = sp.simplify(b**2 - 4 * a * c)

        if disc.is_zero:
            self.classification = "Parabola"
        elif disc.is_positive:
            self.classification = "Hyperbola"
        elif disc.is_negative:
            if sp.simplify(a - c).is_zero and sp.simplify(b).is_zero:
                self.classification = "Circle"
            else:
                self.classification = "Ellipse"
        else:
            self.classification = "Conic section"

    def _find_intercepts(self):
        self.x_intercepts = self._real_solutions(self.relation.subs(y, 0), x)
        self.y_intercepts = self._real_solutions(self.relation.subs(x, 0), y)

    def _real_solutions(self, expr, symbol):
        try:
            solset = sp.solveset(sp.Eq(sp.simplify(expr), 0), symbol, domain=sp.S.Reals)
            if isinstance(solset, sp.FiniteSet):
                candidates = list(solset)
            else:
                candidates = list(sp.solve(sp.Eq(sp.simplify(expr), 0), symbol))
        except Exception:
            return []

        values = []
        for candidate in candidates:
            numeric = sp.N(candidate)
            if getattr(numeric, "is_real", False):
                try:
                    values.append(float(numeric))
                except Exception:
                    continue

        deduped = []
        seen = set()
        for value in values:
            rounded = round(value, 8)
            if rounded not in seen:
                seen.add(rounded)
                deduped.append(value)
        return deduped[:12]

    def _make_callable(self, sym_expr):
        if sym_expr is None:
            return None
        try:
            return lambdify((x, y), sym_expr, modules=["numpy"])
        except Exception:
            return None

    def get_relation_fn(self):
        return self._make_callable(self.relation)

    def get_fx_fn(self):
        return self._make_callable(self.fx)

    def get_fy_fn(self):
        return self._make_callable(self.fy)

    def get_slope_fn(self):
        return self._make_callable(self.slope)

    @staticmethod
    def safe_eval(fn, X, Y, clip=200.0):
        if fn is None:
            return np.full_like(X, np.nan, dtype=float)
        try:
            Z = fn(X, Y)
            if np.ndim(Z) == 0:
                Z = np.full_like(X, float(Z), dtype=float)
            Z = np.where(np.isfinite(Z), Z, np.nan)
            Z = np.clip(Z, -clip, clip)
            return Z.astype(float)
        except Exception:
            return np.full_like(X, np.nan, dtype=float)

    def describe(self) -> dict:
        return {
            "relation": self.relation,
            "fx": self.fx,
            "fy": self.fy,
            "slope": self.slope,
            "classification": self.classification,
            "x_intercepts": self.x_intercepts,
            "y_intercepts": self.y_intercepts,
        }
