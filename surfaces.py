"""
surfaces.py
Generates grids and plots either 3-D calculus surfaces or 2-D coordinate-
geometry views into a Matplotlib axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


THEMES = {
    "plasma": cm.plasma,
    "viridis": cm.viridis,
    "coolwarm": cm.coolwarm,
    "magma": cm.magma,
    "turbo": cm.turbo,
    "inferno": cm.inferno,
}

BG_COLOR = "#0f0f1a"
AXES_COLOR = "#1a1a2e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0ff"
ACCENT = "#7b68ee"
WARNING = "#ffd460"
SUCCESS = "#4ecca3"
ERROR = "#ff6b6b"


def make_grid(x_min: float, x_max: float,
              y_min: float, y_max: float,
              step: float) -> tuple:
    """Return (X, Y) numpy meshgrid."""
    step = max(step, 0.01)
    xs = np.arange(x_min, x_max + step, step)
    ys = np.arange(y_min, y_max + step, step)
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
        ax.text2D(
            0.05,
            0.97,
            formula,
            transform=ax.transAxes,
            fontsize=7,
            color=ACCENT,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, alpha=0.7),
        )
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _style_plane_axes(ax, title: str, formula: str = ""):
    """Apply dark-theme styling to a 2-D axes object."""
    ax.set_facecolor(AXES_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, color=TEXT_COLOR, pad=8, fontsize=10, fontweight="bold")
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    if formula:
        ax.text(
            0.02,
            0.98,
            formula,
            transform=ax.transAxes,
            fontsize=8,
            color=ACCENT,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, alpha=0.7),
        )


def _colormap_by_z(Z, cmap_name: str = "plasma"):
    cmap = THEMES.get(cmap_name, cm.plasma)
    Z_finite = np.where(np.isfinite(Z), Z, np.nan)
    z_min, z_max = np.nanmin(Z_finite), np.nanmax(Z_finite)
    if z_min == z_max:
        z_min, z_max = z_min - 1, z_max + 1
    norm = Normalize(vmin=z_min, vmax=z_max)
    return cmap(norm(Z_finite)), cmap, norm


def _add_scalarbar(ax, cmap_name: str, vmin: float, vmax: float, label: str):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if vmin == vmax:
        vmin -= 1
        vmax += 1

    scalar = plt.cm.ScalarMappable(
        cmap=THEMES.get(cmap_name, cm.plasma),
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    scalar.set_array([])
    cb = ax.get_figure().colorbar(scalar, ax=ax, shrink=0.75, pad=0.06)
    cb.set_label(label, color=TEXT_COLOR, fontsize=8)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb.outline.set_edgecolor(GRID_COLOR)


def plot_surface(ax, X, Y, Z,
                 title="f(x, y)", formula="",
                 cmap_name="plasma", alpha=0.85,
                 show_colorbar=True):
    """Plot a 3-D surface on *ax*."""
    ax.clear()
    _style_axes(ax, title, formula)

    if not np.any(np.isfinite(Z)):
        ax.text(0, 0, 0, "No valid data\n(check expression & range)", color=ERROR, ha="center")
        return

    facecolors, _, _ = _colormap_by_z(Z, cmap_name)
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
        shade=True,
    )

    if show_colorbar:
        z_finite = np.where(np.isfinite(Z), Z, np.nan)
        _add_scalarbar(ax, cmap_name, np.nanmin(z_finite), np.nanmax(z_finite), "z")


def plot_wireframe(ax, X, Y, Z, title="Wireframe", formula="", color=ACCENT):
    """Plot a wireframe (no fill)."""
    ax.clear()
    _style_axes(ax, title, formula)
    ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.5, alpha=0.6, rstride=2, cstride=2)


def plot_contour_projection(ax, X, Y, Z, cmap_name="plasma"):
    """Add contour lines projected onto the bottom plane."""
    try:
        z_floor = np.nanmin(Z) - abs(np.nanmin(Z)) * 0.15 - 0.5
        ax.contourf(
            X,
            Y,
            Z,
            zdir="z",
            offset=z_floor,
            cmap=THEMES.get(cmap_name, cm.plasma),
            alpha=0.4,
            levels=15,
        )
    except Exception:
        pass


def plot_gradient_arrows(ax, X, Y, Z, dfdx_fn, dfdy_fn, safe_eval_fn, skip=5, scale=1.0):
    """Overlay gradient arrows on the surface using quiver."""
    try:
        Xs = X[::skip, ::skip]
        Ys = Y[::skip, ::skip]
        Zs = Z[::skip, ::skip]
        Ux = safe_eval_fn(dfdx_fn, Xs, Ys)
        Uy = safe_eval_fn(dfdy_fn, Xs, Ys)
        Uz = np.zeros_like(Ux)
        mask = np.isfinite(Zs) & np.isfinite(Ux) & np.isfinite(Uy)
        ax.quiver(
            Xs[mask],
            Ys[mask],
            Zs[mask],
            Ux[mask] * scale,
            Uy[mask] * scale,
            Uz[mask],
            length=0.3,
            color=ERROR,
            linewidth=0.8,
            arrow_length_ratio=0.3,
            alpha=0.7,
        )
    except Exception:
        pass


def _draw_zero_locus(ax, X, Y, F):
    try:
        contour = ax.contour(X, Y, F, levels=[0], colors=[WARNING], linewidths=2.4)
        return contour
    except Exception:
        return None


def plot_zero_locus(ax, X, Y, F, title="F(x, y) = 0", formula=""):
    """Plot the implicit curve F(x, y) = 0 on the plane."""
    ax.clear()
    _style_plane_axes(ax, title, formula)
    contour = _draw_zero_locus(ax, X, Y, F)
    if contour is None or not contour.allsegs[0]:
        ax.text(0.5, 0.5, "No curve in current window", color=ERROR, ha="center", va="center",
                transform=ax.transAxes)


def plot_residual_map(ax, X, Y, F, title="Residual Map", formula="", cmap_name="coolwarm"):
    """Plot a heatmap of F(x, y) with the zero locus overlaid."""
    ax.clear()
    _style_plane_axes(ax, title, formula)

    finite = np.where(np.isfinite(F), F, np.nan)
    if not np.any(np.isfinite(finite)):
        ax.text(0.5, 0.5, "No valid data in current window", color=ERROR, ha="center", va="center",
                transform=ax.transAxes)
        return

    span = np.nanpercentile(np.abs(finite), 95)
    span = max(float(span), 1.0)
    clipped = np.clip(finite, -span, span)
    ax.contourf(
        X,
        Y,
        clipped,
        levels=24,
        cmap=THEMES.get(cmap_name, cm.coolwarm),
        alpha=0.82,
    )
    _draw_zero_locus(ax, X, Y, F)
    _add_scalarbar(ax, cmap_name, -span, span, "F(x, y)")


def plot_slope_map(ax, X, Y, slope, F, title="Implicit Slope  dy/dx", formula="", cmap_name="plasma"):
    """Plot dy/dx over the plane with the implicit curve overlaid."""
    ax.clear()
    _style_plane_axes(ax, title, formula)

    finite = np.where(np.isfinite(slope), slope, np.nan)
    if not np.any(np.isfinite(finite)):
        ax.text(0.5, 0.5, "Slope undefined in current window", color=ERROR, ha="center", va="center",
                transform=ax.transAxes)
        return

    span = np.nanpercentile(np.abs(finite), 90)
    span = max(float(span), 1.0)
    clipped = np.clip(finite, -span, span)
    ax.contourf(
        X,
        Y,
        clipped,
        levels=24,
        cmap=THEMES.get(cmap_name, cm.plasma),
        alpha=0.82,
    )
    _draw_zero_locus(ax, X, Y, F)
    _add_scalarbar(ax, cmap_name, -span, span, "dy/dx")


def _sample_curve_points(contour, step=18):
    if contour is None or not contour.allsegs or not contour.allsegs[0]:
        return np.empty((0, 2))

    points = []
    for segment in contour.allsegs[0]:
        if len(segment) == 0:
            continue
        stride = max(len(segment) // step, 1)
        points.append(segment[::stride])

    if not points:
        return np.empty((0, 2))
    return np.vstack(points)


def plot_curve_vectors(ax, X, Y, F, fx, fy, title="Vector Overlay", formula="", mode="tangent"):
    """Plot tangent or normal vectors sampled along the implicit curve."""
    ax.clear()
    _style_plane_axes(ax, title, formula)
    contour = _draw_zero_locus(ax, X, Y, F)

    points = _sample_curve_points(contour)
    if len(points) == 0:
        ax.text(0.5, 0.5, "No curve points available for vectors", color=ERROR, ha="center", va="center",
                transform=ax.transAxes)
        return

    U = np.array(fx(points[:, 0], points[:, 1]), dtype=float)
    V = np.array(fy(points[:, 0], points[:, 1]), dtype=float)
    if U.ndim == 0:
        U = np.full(len(points), float(U), dtype=float)
    if V.ndim == 0:
        V = np.full(len(points), float(V), dtype=float)
    if mode == "tangent":
        U, V = -V, U

    mag = np.hypot(U, V)
    mask = np.isfinite(U) & np.isfinite(V) & (mag > 1e-9)
    if not np.any(mask):
        ax.text(0.5, 0.5, "Vectors undefined on the current curve", color=ERROR, ha="center", va="center",
                transform=ax.transAxes)
        return

    U = U[mask] / mag[mask]
    V = V[mask] / mag[mask]
    pts = points[mask]
    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=2.8,
        color=SUCCESS if mode == "tangent" else ERROR,
        width=0.005,
        alpha=0.9,
    )


def plot_intercepts(ax, X, Y, F, x_intercepts, y_intercepts, title="Axis Intercepts", formula=""):
    """Plot the implicit curve and mark axis intercepts."""
    ax.clear()
    _style_plane_axes(ax, title, formula)
    _draw_zero_locus(ax, X, Y, F)

    if x_intercepts:
        xs = np.array(x_intercepts, dtype=float)
        ax.scatter(xs, np.zeros_like(xs), color=SUCCESS, s=40, label="x-intercepts", zorder=5)
        for value in xs:
            ax.annotate(f"({value:.2f}, 0)", (value, 0), color=SUCCESS, fontsize=7, xytext=(4, 4),
                        textcoords="offset points")

    if y_intercepts:
        ys = np.array(y_intercepts, dtype=float)
        ax.scatter(np.zeros_like(ys), ys, color=ERROR, s=40, label="y-intercepts", zorder=5)
        for value in ys:
            ax.annotate(f"(0, {value:.2f})", (0, value), color=ERROR, fontsize=7, xytext=(4, 4),
                        textcoords="offset points")

    if x_intercepts or y_intercepts:
        legend = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=8)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    else:
        ax.text(0.5, 0.5, "No axis intercepts found in symbolic analysis", color=WARNING,
                ha="center", va="center", transform=ax.transAxes)


def build_animation_frames(X, Y, fn_original, fn_target, safe_eval_fn, n_frames=40):
    """Interpolate Z-values from fn_original to fn_target across n_frames."""
    Z0 = safe_eval_fn(fn_original, X, Y)
    Z1 = safe_eval_fn(fn_target, X, Y)

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frames.append((1 - t) * Z0 + t * Z1)
    return frames
