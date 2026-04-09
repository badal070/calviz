"""
surfaces.py
Generates X/Y/Z meshgrids and plots 3-D surfaces into a given Matplotlib axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource


# ── Colour palette (dark theme) ───────────────────────────────────────────────
THEMES = {
    "plasma":    cm.plasma,
    "viridis":   cm.viridis,
    "coolwarm":  cm.coolwarm,
    "magma":     cm.magma,
    "turbo":     cm.turbo,
    "inferno":   cm.inferno,
}

BG_COLOR     = "#0f0f1a"
AXES_COLOR   = "#1a1a2e"
GRID_COLOR   = "#2a2a4a"
TEXT_COLOR   = "#e0e0ff"
ACCENT       = "#7b68ee"


def make_grid(x_min: float, x_max: float,
              y_min: float, y_max: float,
              step: float) -> tuple:
    """Return (X, Y) numpy meshgrid."""
    step = max(step, 0.01)
    xs = np.arange(x_min, x_max + step, step)
    ys = np.arange(y_min, y_max + step, step)
    # Cap grid size to avoid memory explosion
    xs = xs[:300]
    ys = ys[:300]
    return np.meshgrid(xs, ys)


def _style_axes(ax, title: str, formula: str = ""):
    """Apply dark-theme styling to a 3-D axes object."""
    ax.set_facecolor(AXES_COLOR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.zaxis.label.set_color(TEXT_COLOR)
    ax.set_xlabel("x", labelpad=5)
    ax.set_ylabel("y", labelpad=5)
    ax.set_zlabel("z", labelpad=5)
    ax.set_title(title, color=TEXT_COLOR, pad=8, fontsize=9, fontweight="bold")
    if formula:
        ax.text2D(0.05, 0.97, formula, transform=ax.transAxes,
                  fontsize=7, color=ACCENT, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, alpha=0.7))
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _colormap_by_z(Z, cmap_name: str = "plasma"):
    """Return face colours scaled by Z values."""
    cmap = THEMES.get(cmap_name, cm.plasma)
    Z_finite = np.where(np.isfinite(Z), Z, np.nan)
    z_min, z_max = np.nanmin(Z_finite), np.nanmax(Z_finite)
    if z_min == z_max:
        z_min, z_max = z_min - 1, z_max + 1
    norm = Normalize(vmin=z_min, vmax=z_max)
    return cmap(norm(Z_finite)), cmap, norm


def _colormap_by_curvature(Z_grad, cmap_name: str = "coolwarm"):
    """Return face colours scaled by gradient magnitude."""
    cmap = THEMES.get(cmap_name, cm.coolwarm)
    g = np.where(np.isfinite(Z_grad), Z_grad, 0.0)
    norm = Normalize(vmin=g.min(), vmax=max(g.max(), 1e-9))
    return cmap(norm(g)), cmap, norm


# ── Primary plot functions ────────────────────────────────────────────────────

def plot_surface(ax, X, Y, Z,
                 title="f(x, y)", formula="",
                 cmap_name="plasma", alpha=0.85,
                 colorby="z"):
    """Plot a 3-D surface on *ax*."""
    ax.clear()
    _style_axes(ax, title, formula)

    valid = np.any(np.isfinite(Z))
    if not valid:
        ax.text(0, 0, 0, "No valid data\n(check expression & range)",
                color="red", ha="center")
        return

    if colorby == "z":
        facecolors, cmap, norm = _colormap_by_z(Z, cmap_name)
    else:
        facecolors, cmap, norm = facecolors, cmap, norm  # fallback

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        linewidth=0, antialiased=True,
        alpha=alpha, shade=True,
    )
    # Colorbar
    try:
        cmap_obj = THEMES.get(cmap_name, cm.plasma)
        z_finite = np.where(np.isfinite(Z), Z, np.nan)
        z_min, z_max = np.nanmin(z_finite), np.nanmax(z_finite)
        sm = plt.cm.ScalarMappable(
            cmap=cmap_obj,
            norm=Normalize(vmin=z_min, vmax=z_max)
        )
        sm.set_array([])
        cb = ax.get_figure().colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        cb.outline.set_edgecolor(GRID_COLOR)
    except Exception:
        pass

    return surf


def plot_wireframe(ax, X, Y, Z, title="Wireframe", formula="", color=ACCENT):
    """Plot a wireframe (no fill) – good for overlay."""
    ax.clear()
    _style_axes(ax, title, formula)
    ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.5, alpha=0.6,
                      rstride=2, cstride=2)


def plot_contour_projection(ax, X, Y, Z, cmap_name="plasma"):
    """Add contour lines projected onto the bottom plane."""
    try:
        z_floor = np.nanmin(Z) - abs(np.nanmin(Z)) * 0.15 - 0.5
        ax.contourf(X, Y, Z, zdir="z", offset=z_floor,
                    cmap=THEMES.get(cmap_name, cm.plasma),
                    alpha=0.4, levels=15)
    except Exception:
        pass


def plot_gradient_arrows(ax, X, Y, Z, dfdx_fn, dfdy_fn, safe_eval_fn,
                          skip=5, scale=1.0):
    """
    Overlay gradient arrows on the surface using quiver.
    *skip* controls density (every N-th grid point).
    """
    try:
        Xs = X[::skip, ::skip]
        Ys = Y[::skip, ::skip]
        Zs = Z[::skip, ::skip]
        Ux = safe_eval_fn(dfdx_fn, Xs, Ys)
        Uy = safe_eval_fn(dfdy_fn, Xs, Ys)
        Uz = np.zeros_like(Ux)
        mask = np.isfinite(Zs) & np.isfinite(Ux) & np.isfinite(Uy)
        ax.quiver(Xs[mask], Ys[mask], Zs[mask],
                  Ux[mask] * scale, Uy[mask] * scale, Uz[mask],
                  length=0.3, color="#ff6b6b", linewidth=0.8,
                  arrow_length_ratio=0.3, alpha=0.7)
    except Exception:
        pass


def build_animation_frames(X, Y, fn_original, fn_target, n_frames=40):
    """
    Linearly interpolate Z-values from fn_original to fn_target
    across n_frames. Returns list of Z arrays.
    """
    from calculus import CalcEngine
    engine = CalcEngine()

    Z0 = engine.safe_eval(fn_original, X, Y)
    Z1 = engine.safe_eval(fn_target, X, Y)

    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        Zt = (1 - t) * Z0 + t * Z1
        frames.append(Zt)
    return frames
