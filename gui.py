"""
gui.py
Full Tkinter GUI: equation input, sliders, view switcher, animation, export.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
import threading

from calculus import CalcEngine
import surfaces as sf

# ── Colour constants (mirrored from surfaces.py) ──────────────────────────────
BG       = "#0f0f1a"
PANEL    = "#1a1a2e"
ACCENT   = "#7b68ee"
ACCENT2  = "#ff6b6b"
BTN_BG   = "#16213e"
BTN_ACT  = "#0f3460"
TEXT     = "#e0e0ff"
SUBTEXT  = "#9090bb"
SUCCESS  = "#4ecca3"
WARNING  = "#ffd460"
BORDER   = "#2a2a4a"

FONT_H   = ("Segoe UI", 11, "bold")
FONT_B   = ("Segoe UI", 9, "bold")
FONT_N   = ("Segoe UI", 9)
FONT_S   = ("Segoe UI", 8)
FONT_MONO= ("Consolas", 9)

PRESETS = [
    ("Paraboloid",    "x**2 + y**2"),
    ("Saddle",        "x**2 - y**2"),
    ("Sine Wave",     "sin(x) * cos(y)"),
    ("Ripple",        "sin(sqrt(x**2 + y**2) + 1e-9) / (sqrt(x**2 + y**2) + 1e-9)"),
    ("Gaussian",      "exp(-(x**2 + y**2)/4)"),
    ("Monkey Saddle",  "x**3 - 3*x*y**2"),
    ("Twisted Plane", "x * sin(y)"),
    ("Double Well",   "(x**2 - 1)**2 + y**2"),
    ("Peaks",         "3*(1-x)**2*exp(-x**2-(y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - (1/3)*exp(-(x+1)**2-y**2)"),
]

VIEW_LABELS = ["f(x,y)  Surface", "∂f/∂x  Partial x", "∂f/∂y  Partial y",
               "∫f dx  Integral", "Gradient Overlay", "Wireframe"]
VIEW_KEYS   = ["surface", "dfdx", "dfdy", "integral", "gradient", "wire"]


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("3D Calculus Visualizer")
        self.configure(bg=BG)
        self.minsize(1100, 720)
        self.resizable(True, True)

        # State
        self.engine      = CalcEngine()
        self.current_view = "surface"
        self._anim        = None
        self._anim_running = False
        self._frame_data  = []

        self._build_ui()
        self._apply_preset(PRESETS[0][1])    # default: paraboloid

    # ═══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, height=54)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(top, text="3D Calculus Visualizer", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=18, pady=10)

        tk.Label(top, text="▸ Plot • Differentiate • Integrate • Animate",
                 bg=PANEL, fg=SUBTEXT, font=FONT_S).pack(side="left", pady=10)

        # ── Main layout ───────────────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        left  = tk.Frame(main, bg=PANEL, width=310)
        left.pack(fill="y", side="left", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(main, bg=BG)
        right.pack(fill="both", expand=True, side="left", padx=(4, 8), pady=8)

        self._build_left_panel(left)
        self._build_right_panel(right)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        pad = dict(padx=12, pady=4)

        # Section: Equation
        self._section(parent, "⟨ f(x, y) ⟩  EQUATION")
        self.eq_var = tk.StringVar(value="x**2 + y**2")
        eq_entry = tk.Entry(parent, textvariable=self.eq_var, bg=BTN_BG,
                            fg=SUCCESS, insertbackground=SUCCESS,
                            font=FONT_MONO, relief="flat", bd=4)
        eq_entry.pack(fill="x", **pad)
        eq_entry.bind("<Return>", lambda e: self._plot())

        # Presets
        self._section(parent, "PRESETS")
        preset_frame = tk.Frame(parent, bg=PANEL)
        preset_frame.pack(fill="x", padx=12, pady=2)
        for i, (name, expr) in enumerate(PRESETS):
            btn = tk.Button(preset_frame, text=name, font=FONT_S,
                            bg=BTN_BG, fg=SUBTEXT, activebackground=BTN_ACT,
                            activeforeground=TEXT, relief="flat", bd=0,
                            padx=6, pady=3, cursor="hand2",
                            command=lambda e=expr: self._apply_preset(e))
            btn.grid(row=i//2, column=i%2, sticky="ew", padx=2, pady=2)
        preset_frame.columnconfigure(0, weight=1)
        preset_frame.columnconfigure(1, weight=1)

        # Section: Range & step
        self._section(parent, "DOMAIN & RESOLUTION")
        sliders_cfg = [
            ("X min", "x_min", -6.0, -10.0, 0.0, 0.5),
            ("X max", "x_max",  6.0,  0.0, 10.0, 0.5),
            ("Y min", "y_min", -6.0, -10.0, 0.0, 0.5),
            ("Y max", "y_max",  6.0,  0.0, 10.0, 0.5),
            ("Step",  "step",   0.2,  0.05, 1.0, 0.05),
        ]
        self._sliders = {}
        for label, key, default, lo, hi, res in sliders_cfg:
            self._build_slider(parent, label, key, default, lo, hi, res)

        # Transparency
        self._section(parent, "APPEARANCE")
        self._build_slider(parent, "Transparency α", "alpha", 0.85, 0.1, 1.0, 0.05)

        # Colourmap
        cm_row = tk.Frame(parent, bg=PANEL)
        cm_row.pack(fill="x", padx=12, pady=(2, 6))
        tk.Label(cm_row, text="Colourmap", bg=PANEL, fg=SUBTEXT, font=FONT_S,
                 width=13, anchor="w").pack(side="left")
        self.cmap_var = tk.StringVar(value="plasma")
        cm_menu = ttk.Combobox(cm_row, textvariable=self.cmap_var,
                                values=list(sf.THEMES.keys()),
                                font=FONT_S, width=10, state="readonly")
        cm_menu.pack(side="left", padx=4)
        cm_menu.bind("<<ComboboxSelected>>", lambda e: self._plot())

        # Section: Formulas
        self._section(parent, "SYMBOLIC RESULTS")
        self.formula_box = tk.Text(parent, bg=BTN_BG, fg=SUCCESS,
                                   font=FONT_S, height=7, relief="flat",
                                   wrap="word", state="disabled", bd=4)
        self.formula_box.pack(fill="x", padx=12, pady=4)

        # ── Action buttons ────────────────────────────────────────────────────
        self._section(parent, "ACTIONS")
        btn_data = [
            ("▶  Plot / Refresh",   self._plot,          ACCENT,  BG),
            ("🎬  Animate Morph",   self._start_animate, WARNING, "#2a1a00"),
            ("⏹  Stop Animation",  self._stop_animate,  ACCENT2, "#2a0010"),
            ("💾  Save Image",      self._save_image,    SUCCESS, "#002a1a"),
            ("📋  Copy Formulas",   self._copy_formulas, SUBTEXT, BTN_BG),
        ]
        for txt, cmd, fg, bg in btn_data:
            tk.Button(parent, text=txt, font=FONT_B, fg=fg, bg=bg,
                      activebackground=BTN_ACT, activeforeground=TEXT,
                      relief="flat", bd=0, pady=7, cursor="hand2",
                      command=cmd).pack(fill="x", padx=12, pady=2)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(parent, textvariable=self.status_var, bg=PANEL, fg=SUBTEXT,
                 font=FONT_S, wraplength=270, justify="left").pack(
                     fill="x", padx=12, pady=(8, 4))

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        # View selector tabs
        tab_row = tk.Frame(parent, bg=BG)
        tab_row.pack(fill="x", pady=(0, 6))
        self._view_btns = {}
        for key, label in zip(VIEW_KEYS, VIEW_LABELS):
            btn = tk.Button(tab_row, text=label, font=FONT_S,
                            bg=BTN_BG, fg=SUBTEXT, relief="flat", bd=0,
                            padx=10, pady=5, cursor="hand2",
                            command=lambda k=key: self._switch_view(k))
            btn.pack(side="left", padx=2)
            self._view_btns[key] = btn
        self._highlight_tab("surface")

        # Matplotlib figure
        self.fig = plt.Figure(figsize=(9, 6.5), facecolor=sf.BG_COLOR)
        self.ax  = self.fig.add_subplot(111, projection="3d",
                                         facecolor=sf.AXES_COLOR)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(parent, bg=PANEL)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=PANEL)
        for child in toolbar.winfo_children():
            try:
                child.config(bg=PANEL, fg=TEXT)
            except Exception:
                pass
        toolbar.update()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(
                     anchor="w", padx=12, pady=(10, 2))

    def _build_slider(self, parent, label, key, default, lo, hi, res):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=12, pady=1)
        var = tk.DoubleVar(value=default)
        self._sliders[key] = var
        lbl = tk.Label(row, text=label, bg=PANEL, fg=SUBTEXT,
                        font=FONT_S, width=13, anchor="w")
        lbl.pack(side="left")
        val_lbl = tk.Label(row, textvariable=var, bg=PANEL, fg=TEXT,
                            font=FONT_S, width=6)
        val_lbl.pack(side="right")
        sld = tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                        orient="horizontal", bg=PANEL, fg=TEXT,
                        troughcolor=BTN_BG, activebackground=ACCENT,
                        highlightthickness=0, showvalue=False, bd=0,
                        command=lambda _: None)
        sld.pack(side="left", fill="x", expand=True, padx=4)

    def _highlight_tab(self, key):
        for k, btn in self._view_btns.items():
            btn.configure(bg=ACCENT if k == key else BTN_BG,
                           fg=BG      if k == key else SUBTEXT)

    def _sv(self, key):
        return self._sliders[key].get()

    # ═══════════════════════════════════════════════════════════════════════════
    # Core logic
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_preset(self, expr: str):
        self.eq_var.set(expr)
        self._plot()

    def _parse(self) -> bool:
        expr = self.eq_var.get().strip()
        ok, msg = self.engine.parse(expr)
        if not ok:
            self._status(f"⚠ {msg}", error=True)
            messagebox.showerror("Parse Error", msg)
            return False
        self._update_formula_box()
        return True

    def _update_formula_box(self):
        latex_map = self.engine.latex()
        lines = []
        for key in ("f", "dfdx", "dfdy", "int_x", "int_y"):
            if key in latex_map:
                label = {"f": "f(x,y)", "dfdx": "∂f/∂x", "dfdy": "∂f/∂y",
                         "int_x": "∫f dx", "int_y": "∫f dy"}[key]
                # Use sympy pretty-print strings instead of LaTeX in the text box
                import sympy as sp
                sym_map = {"f": self.engine.expr,
                           "dfdx": self.engine.df_dx,
                           "dfdy": self.engine.df_dy,
                           "int_x": self.engine.integral_x,
                           "int_y": self.engine.integral_y}
                sym = sym_map.get(key)
                txt = sp.pretty(sym, use_unicode=True) if sym is not None else "—"
                lines.append(f"[{label}]\n{txt}\n")

        content = "\n".join(lines)
        self.formula_box.configure(state="normal")
        self.formula_box.delete("1.0", "end")
        self.formula_box.insert("1.0", content)
        self.formula_box.configure(state="disabled")

    def _get_grid(self):
        return sf.make_grid(self._sv("x_min"), self._sv("x_max"),
                            self._sv("y_min"), self._sv("y_max"),
                            self._sv("step"))

    def _plot(self):
        self._stop_animate()
        if not self._parse():
            return
        self._draw_view(self.current_view)

    def _switch_view(self, key: str):
        self._stop_animate()
        self.current_view = key
        self._highlight_tab(key)
        if not self._parse():
            return
        self._draw_view(key)

    def _draw_view(self, key: str):
        X, Y = self._get_grid()
        eng  = self.engine
        cmap = self.cmap_var.get()
        alpha = self._sv("alpha")

        ax = self.ax
        ax.clear()

        try:
            if key == "surface":
                Z = CalcEngine.safe_eval(eng.get_surface_fn(), X, Y)
                sf.plot_surface(ax, X, Y, Z,
                                title="Original Surface  f(x, y)",
                                formula=f"f = {self.eq_var.get()}",
                                cmap_name=cmap, alpha=alpha)
                sf.plot_contour_projection(ax, X, Y, Z, cmap_name=cmap)

            elif key == "dfdx":
                Z = CalcEngine.safe_eval(eng.get_dfdx_fn(), X, Y)
                import sympy as sp
                sf.plot_surface(ax, X, Y, Z,
                                title="Partial Derivative  ∂f/∂x",
                                formula=f"∂f/∂x = {sp.pretty(eng.df_dx, use_unicode=True)}",
                                cmap_name=cmap, alpha=alpha)

            elif key == "dfdy":
                Z = CalcEngine.safe_eval(eng.get_dfdy_fn(), X, Y)
                import sympy as sp
                sf.plot_surface(ax, X, Y, Z,
                                title="Partial Derivative  ∂f/∂y",
                                formula=f"∂f/∂y = {sp.pretty(eng.df_dy, use_unicode=True)}",
                                cmap_name=cmap, alpha=alpha)

            elif key == "integral":
                fn = eng.get_integral_x_fn()
                if fn is None:
                    self._status("Integral could not be computed symbolically.", error=True)
                    return
                Z = CalcEngine.safe_eval(fn, X, Y)
                sf.plot_surface(ax, X, Y, Z,
                                title="Antiderivative  ∫f dx  (+ C)",
                                cmap_name=cmap, alpha=alpha)

            elif key == "gradient":
                Z  = CalcEngine.safe_eval(eng.get_surface_fn(), X, Y)
                sf.plot_surface(ax, X, Y, Z,
                                title="Gradient Overlay  f(x,y) + ∇f arrows",
                                cmap_name=cmap, alpha=0.6)
                sf.plot_gradient_arrows(ax, X, Y, Z,
                                        eng.get_dfdx_fn(), eng.get_dfdy_fn(),
                                        CalcEngine.safe_eval)

            elif key == "wire":
                Z = CalcEngine.safe_eval(eng.get_surface_fn(), X, Y)
                sf.plot_wireframe(ax, X, Y, Z,
                                  title="Wireframe  f(x, y)",
                                  formula=f"f = {self.eq_var.get()}")

        except Exception as exc:
            self._status(f"Plot error: {exc}", error=True)
            return

        self.canvas.draw()
        self._status(f"Plotted: {key}  |  grid {X.shape[1]}×{X.shape[0]}")

    # ── Animation ─────────────────────────────────────────────────────────────
    def _start_animate(self):
        if not self._parse():
            return
        self._stop_animate()

        X, Y = self._get_grid()
        eng  = self.engine
        fn0  = eng.get_surface_fn()
        fn1  = eng.get_dfdx_fn()
        cmap = self.cmap_var.get()
        alpha = self._sv("alpha")

        Z0 = CalcEngine.safe_eval(fn0, X, Y)
        Z1 = CalcEngine.safe_eval(fn1, X, Y)

        N = 50
        self._frame_data = [(1-t/(N-1))*Z0 + (t/(N-1))*Z1 for t in range(N)]
        self._frame_data += self._frame_data[::-1]    # ping-pong

        ax = self.ax

        def update(frame_idx):
            ax.clear()
            Zt = self._frame_data[frame_idx]
            sf.plot_surface(ax, X, Y, Zt,
                            title=f"Morphing: f → ∂f/∂x  [{frame_idx+1}/{len(self._frame_data)}]",
                            cmap_name=cmap, alpha=alpha)
            self.canvas.draw()

        self._anim_running = True
        self._status("Animation running…")
        self._anim = FuncAnimation(self.fig, update,
                                   frames=len(self._frame_data),
                                   interval=50, repeat=True)
        self.canvas.draw()

    def _stop_animate(self):
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass
            self._anim = None
        self._anim_running = False

    # ── Export ────────────────────────────────────────────────────────────────
    def _save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"),
                       ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Save Plot As…"
        )
        if path:
            self.fig.savefig(path, dpi=180, bbox_inches="tight",
                              facecolor=sf.BG_COLOR)
            self._status(f"Saved → {path}")

    def _copy_formulas(self):
        self.clipboard_clear()
        content = self.formula_box.get("1.0", "end")
        self.clipboard_append(content)
        self._status("Formulas copied to clipboard.")

    def _status(self, msg: str, error: bool = False):
        self.status_var.set(msg)
        # Flash colour
        colour = ACCENT2 if error else SUCCESS
        lbl_widget = None
        for w in self.winfo_children():
            pass   # colour flash via after() is sufficient
        self.after(100, lambda: None)   # let event loop breathe

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def on_close(self):
        self._stop_animate()
        plt.close("all")
        self.destroy()
