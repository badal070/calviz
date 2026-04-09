"""
gui.py
Full Tkinter GUI for switching between calculus and coordinate geometry views.
"""

import queue
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sympy as sp

from chat_panel import ChatPanel
from calculus import CalcEngine
from coordinate_geometry import CoordinateGeometryEngine
import surfaces as sf
import viz_bus


BG = "#0f0f1a"
PANEL = "#1a1a2e"
ACCENT = "#7b68ee"
ACCENT2 = "#ff6b6b"
BTN_BG = "#16213e"
BTN_ACT = "#0f3460"
TEXT = "#e0e0ff"
SUBTEXT = "#9090bb"
SUCCESS = "#4ecca3"
WARNING = "#ffd460"

FONT_B = ("Segoe UI", 9, "bold")
FONT_S = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 9)

CALCULUS_PRESETS = [
    ("Paraboloid", "x**2 + y**2"),
    ("Saddle", "x**2 - y**2"),
    ("Sine Wave", "sin(x) * cos(y)"),
    ("Ripple", "sin(sqrt(x**2 + y**2) + 1e-9) / (sqrt(x**2 + y**2) + 1e-9)"),
    ("Gaussian", "exp(-(x**2 + y**2)/4)"),
    ("Monkey Saddle", "x**3 - 3*x*y**2"),
    ("Twisted Plane", "x * sin(y)"),
    ("Double Well", "(x**2 - 1)**2 + y**2"),
    ("Peaks", "3*(1-x)**2*exp(-x**2-(y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - (1/3)*exp(-(x+1)**2-y**2)"),
]

GEOMETRY_PRESETS = [
    ("Line", "2*x - y - 3 = 0"),
    ("Circle", "x**2 + y**2 = 9"),
    ("Ellipse", "x**2/9 + y**2/4 = 1"),
    ("Parabola", "y - x**2 = 0"),
    ("Hyperbola", "x**2/9 - y**2/4 = 1"),
    ("Pair of Lines", "x**2 - y**2 = 0"),
    ("Shifted Circle", "(x-2)**2 + (y+1)**2 = 16"),
    ("Rotated Conic", "x**2 + x*y + y**2 - 6 = 0"),
]

CALCULUS_VIEWS = [
    ("surface", "f(x,y) Surface"),
    ("dfdx", "∂f/∂x"),
    ("dfdy", "∂f/∂y"),
    ("integral", "∫f dx"),
    ("gradient", "Gradient"),
    ("wire", "Wireframe"),
]

GEOMETRY_VIEWS = [
    ("relation_surface", "F(x,y) Surface"),
    ("curve", "F(x,y)=0 Curve"),
    ("slope", "dy/dx Slope"),
    ("tangent", "Tangent"),
    ("normal", "Normal"),
    ("intercepts", "Intercepts"),
]

DOMAIN_CONFIG = {
    "calculus": {
        "window_title": "3D Calculus Visualizer",
        "tagline": "Plot • Differentiate • Integrate • Animate",
        "equation_label": "⟨ f(x, y) ⟩  EQUATION",
        "default_expr": CALCULUS_PRESETS[0][1],
        "presets": CALCULUS_PRESETS,
        "views": CALCULUS_VIEWS,
        "default_view": "surface",
    },
    "geometry": {
        "window_title": "Coordinate Geometry Visualizer",
        "tagline": "Plot • Classify • Analyze • Compare",
        "equation_label": "⟨ F(x, y) = 0 ⟩  RELATION",
        "default_expr": GEOMETRY_PRESETS[1][1],
        "presets": GEOMETRY_PRESETS,
        "views": GEOMETRY_VIEWS,
        "default_view": "curve",
    },
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg=BG)
        self.minsize(1440, 720)
        self.resizable(True, True)

        self.engines = {
            "calculus": CalcEngine(),
            "geometry": CoordinateGeometryEngine(),
        }
        self.current_domain = "calculus"
        self.current_views = {
            key: cfg["default_view"] for key, cfg in DOMAIN_CONFIG.items()
        }
        self.domain_exprs = {
            key: cfg["default_expr"] for key, cfg in DOMAIN_CONFIG.items()
        }
        self.current_view = self.current_views[self.current_domain]
        self._anim = None
        self._frame_data = []
        self._viz_queue = viz_bus.subscribe()

        self._build_ui()
        self._set_domain("calculus", initial=True)
        self._poll_bus()

    def _build_ui(self):
        top = tk.Frame(self, bg=PANEL, height=54)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        title_wrap = tk.Frame(top, bg=PANEL)
        title_wrap.pack(side="left", padx=18, pady=10)
        self.top_title = tk.Label(title_wrap, bg=PANEL, fg=ACCENT, font=("Segoe UI", 14, "bold"))
        self.top_title.pack(side="left")
        self.top_subtitle = tk.Label(title_wrap, bg=PANEL, fg=SUBTEXT, font=FONT_S)
        self.top_subtitle.pack(side="left", padx=(12, 0))

        domain_wrap = tk.Frame(top, bg=PANEL)
        domain_wrap.pack(side="right", padx=18, pady=10)
        tk.Label(domain_wrap, text="Domain", bg=PANEL, fg=SUBTEXT, font=FONT_S).pack(side="left", padx=(0, 6))
        self._domain_btns = {}
        for key, label in (("calculus", "Calculus"), ("geometry", "Coordinate Geometry")):
            btn = tk.Button(
                domain_wrap,
                text=label,
                font=FONT_S,
                bg=BTN_BG,
                fg=SUBTEXT,
                activebackground=BTN_ACT,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                padx=10,
                pady=5,
                cursor="hand2",
                command=lambda domain=key: self._set_domain(domain),
            )
            btn.pack(side="left", padx=2)
            self._domain_btns[key] = btn

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        chat_col = tk.Frame(main, bg=PANEL, width=330)
        chat_col.pack(fill="y", side="left", padx=(8, 4), pady=8)
        chat_col.pack_propagate(False)
        self.chat_panel = ChatPanel(
            chat_col,
            on_status=self._status,
            on_error=lambda msg: self._status(msg, error=True),
        )
        self.chat_panel.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=PANEL, width=310)
        left.pack(fill="y", side="left", padx=(4, 4), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(main, bg=BG)
        right.pack(fill="both", expand=True, side="left", padx=(4, 8), pady=8)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        self.eq_section_label = self._section(parent, "")
        self.eq_var = tk.StringVar(value=DOMAIN_CONFIG["calculus"]["default_expr"])
        eq_entry = tk.Entry(
            parent,
            textvariable=self.eq_var,
            bg=BTN_BG,
            fg=SUCCESS,
            insertbackground=SUCCESS,
            font=FONT_MONO,
            relief="flat",
            bd=4,
        )
        eq_entry.pack(fill="x", padx=12, pady=4)
        eq_entry.bind("<Return>", lambda _event: self._plot())

        self._section(parent, "PRESETS")
        self.preset_frame = tk.Frame(parent, bg=PANEL)
        self.preset_frame.pack(fill="x", padx=12, pady=2)

        self._section(parent, "DOMAIN & RESOLUTION")
        sliders_cfg = [
            ("X min", "x_min", -6.0, -10.0, 0.0, 0.5),
            ("X max", "x_max", 6.0, 0.0, 10.0, 0.5),
            ("Y min", "y_min", -6.0, -10.0, 0.0, 0.5),
            ("Y max", "y_max", 6.0, 0.0, 10.0, 0.5),
            ("Step", "step", 0.2, 0.05, 1.0, 0.05),
        ]
        self._sliders = {}
        for label, key, default, lo, hi, res in sliders_cfg:
            self._build_slider(parent, label, key, default, lo, hi, res)

        self._section(parent, "APPEARANCE")
        self._build_slider(parent, "Transparency α", "alpha", 0.85, 0.1, 1.0, 0.05)

        cm_row = tk.Frame(parent, bg=PANEL)
        cm_row.pack(fill="x", padx=12, pady=(2, 6))
        tk.Label(cm_row, text="Colourmap", bg=PANEL, fg=SUBTEXT, font=FONT_S, width=13, anchor="w").pack(side="left")
        self.cmap_var = tk.StringVar(value="plasma")
        cmap_menu = tk.OptionMenu(cm_row, self.cmap_var, *sf.THEMES.keys(), command=lambda _value: self._plot())
        cmap_menu.config(bg=BTN_BG, fg=TEXT, activebackground=BTN_ACT, activeforeground=TEXT, relief="flat")
        cmap_menu["menu"].config(bg=BTN_BG, fg=TEXT, activebackground=BTN_ACT, activeforeground=TEXT)
        cmap_menu.pack(side="left", padx=4)

        self._section(parent, "SYMBOLIC RESULTS")
        self.formula_box = tk.Text(
            parent,
            bg=BTN_BG,
            fg=SUCCESS,
            font=FONT_S,
            height=9,
            relief="flat",
            wrap="word",
            state="disabled",
            bd=4,
        )
        self.formula_box.pack(fill="x", padx=12, pady=4)

        self._section(parent, "ACTIONS")
        btn_data = [
            ("▶  Plot / Refresh", self._plot, ACCENT, BG),
            ("🎬  Animate Morph", self._start_animate, WARNING, "#2a1a00"),
            ("⏹  Stop Animation", self._stop_animate, ACCENT2, "#2a0010"),
            ("💾  Save Image", self._save_image, SUCCESS, "#002a1a"),
            ("📋  Copy Formulas", self._copy_formulas, SUBTEXT, BTN_BG),
        ]
        for text, command, fg, bg in btn_data:
            tk.Button(
                parent,
                text=text,
                font=FONT_B,
                fg=fg,
                bg=bg,
                activebackground=BTN_ACT,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                pady=7,
                cursor="hand2",
                command=command,
            ).pack(fill="x", padx=12, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            parent,
            textvariable=self.status_var,
            bg=PANEL,
            fg=SUBTEXT,
            font=FONT_S,
            wraplength=270,
            justify="left",
        ).pack(fill="x", padx=12, pady=(8, 4))

    def _build_right_panel(self, parent):
        self.tab_row = tk.Frame(parent, bg=BG)
        self.tab_row.pack(fill="x", pady=(0, 6))
        self._view_btns = {}

        self.fig = plt.Figure(figsize=(9, 6.5), facecolor=sf.BG_COLOR)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor=sf.AXES_COLOR)
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

    def _section(self, parent, title):
        label = tk.Label(parent, text=title, bg=PANEL, fg=ACCENT, font=("Segoe UI", 8, "bold"))
        label.pack(anchor="w", padx=12, pady=(10, 2))
        return label

    def _build_slider(self, parent, label, key, default, lo, hi, res):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=12, pady=1)
        var = tk.DoubleVar(value=default)
        self._sliders[key] = var
        tk.Label(row, text=label, bg=PANEL, fg=SUBTEXT, font=FONT_S, width=13, anchor="w").pack(side="left")
        tk.Label(row, textvariable=var, bg=PANEL, fg=TEXT, font=FONT_S, width=6).pack(side="right")
        scale = tk.Scale(
            row,
            variable=var,
            from_=lo,
            to=hi,
            resolution=res,
            orient="horizontal",
            bg=PANEL,
            fg=TEXT,
            troughcolor=BTN_BG,
            activebackground=ACCENT,
            highlightthickness=0,
            showvalue=False,
            bd=0,
            command=lambda _value: None,
        )
        scale.pack(side="left", fill="x", expand=True, padx=4)

    def _sv(self, key):
        return self._sliders[key].get()

    def _current_engine(self):
        return self.engines[self.current_domain]

    def _set_domain(self, domain_key: str, initial: bool = False):
        if not initial and domain_key == self.current_domain:
            return

        if not initial:
            self.domain_exprs[self.current_domain] = self.eq_var.get().strip()

        self.current_domain = domain_key
        self.current_view = self.current_views[domain_key]
        cfg = DOMAIN_CONFIG[domain_key]

        self.title(cfg["window_title"])
        self.top_title.configure(text=cfg["window_title"])
        self.top_subtitle.configure(text=cfg["tagline"])
        self.eq_section_label.configure(text=cfg["equation_label"])
        self.eq_var.set(self.domain_exprs.get(domain_key, cfg["default_expr"]))

        self._highlight_domain_buttons()
        self._rebuild_presets()
        self._rebuild_view_tabs()
        self._highlight_tab(self.current_view)
        self._plot()

    def _highlight_domain_buttons(self):
        for key, btn in self._domain_btns.items():
            btn.configure(bg=ACCENT if key == self.current_domain else BTN_BG, fg=BG if key == self.current_domain else SUBTEXT)

    def _rebuild_presets(self):
        for child in self.preset_frame.winfo_children():
            child.destroy()

        presets = DOMAIN_CONFIG[self.current_domain]["presets"]
        for index, (name, expr) in enumerate(presets):
            btn = tk.Button(
                self.preset_frame,
                text=name,
                font=FONT_S,
                bg=BTN_BG,
                fg=SUBTEXT,
                activebackground=BTN_ACT,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                padx=6,
                pady=3,
                cursor="hand2",
                command=lambda preset_expr=expr: self._apply_preset(preset_expr),
            )
            btn.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)

        self.preset_frame.columnconfigure(0, weight=1)
        self.preset_frame.columnconfigure(1, weight=1)

    def _rebuild_view_tabs(self):
        for child in self.tab_row.winfo_children():
            child.destroy()

        self._view_btns = {}
        for key, label in DOMAIN_CONFIG[self.current_domain]["views"]:
            btn = tk.Button(
                self.tab_row,
                text=label,
                font=FONT_S,
                bg=BTN_BG,
                fg=SUBTEXT,
                relief="flat",
                bd=0,
                padx=10,
                pady=5,
                cursor="hand2",
                command=lambda view_key=key: self._switch_view(view_key),
            )
            btn.pack(side="left", padx=2)
            self._view_btns[key] = btn

    def _highlight_tab(self, key):
        for view_key, btn in self._view_btns.items():
            btn.configure(bg=ACCENT if view_key == key else BTN_BG, fg=BG if view_key == key else SUBTEXT)

    def _apply_preset(self, expr: str):
        self.eq_var.set(expr)
        self.domain_exprs[self.current_domain] = expr
        self._plot()

    def _poll_bus(self):
        try:
            payload = self._viz_queue.get_nowait()
            self._apply_llm_payload(payload)
        except queue.Empty:
            pass
        self.after(200, self._poll_bus)

    def _apply_llm_payload(self, payload):
        """Load a Colab-emitted payload into the visualizer."""
        if self.current_domain != "calculus":
            self._set_domain("calculus")

        self.eq_var.set(payload.equation)
        self.domain_exprs["calculus"] = payload.equation
        self._sliders["x_min"].set(payload.x_range[0])
        self._sliders["x_max"].set(payload.x_range[1])
        self._sliders["y_min"].set(payload.y_range[0])
        self._sliders["y_max"].set(payload.y_range[1])
        self._sliders["step"].set(payload.step)
        self.current_view = "surface"
        self.current_views["calculus"] = "surface"
        self._highlight_tab("surface")
        self._plot()
        self._status(f"Loaded: {payload.concept_title}")
        self.chat_panel.render_explanation(payload)

    def _parse(self) -> bool:
        expr = self.eq_var.get().strip()
        ok, msg = self._current_engine().parse(expr)
        if not ok:
            self._status(f"⚠ {msg}", error=True)
            messagebox.showerror("Parse Error", msg)
            return False

        self.domain_exprs[self.current_domain] = expr
        self._update_formula_box()
        return True

    def _update_formula_box(self):
        engine = self._current_engine()
        lines = []

        if self.current_domain == "calculus":
            sym_map = {
                "f(x,y)": engine.expr,
                "∂f/∂x": engine.df_dx,
                "∂f/∂y": engine.df_dy,
                "∫f dx": engine.integral_x,
                "∫f dy": engine.integral_y,
            }
            for label, sym_expr in sym_map.items():
                if sym_expr is None:
                    continue
                lines.append(f"[{label}]\n{sp.pretty(sym_expr, use_unicode=True)}\n")
        else:
            x_ints = ", ".join(f"{value:.4g}" for value in engine.x_intercepts) or "None"
            y_ints = ", ".join(f"{value:.4g}" for value in engine.y_intercepts) or "None"
            lines.extend(
                [
                    f"[F(x,y)]\n{sp.pretty(engine.relation, use_unicode=True)} = 0\n",
                    f"[Type]\n{engine.classification}\n",
                    f"[∂F/∂x]\n{sp.pretty(engine.fx, use_unicode=True)}\n",
                    f"[∂F/∂y]\n{sp.pretty(engine.fy, use_unicode=True)}\n",
                    f"[dy/dx]\n{sp.pretty(engine.slope, use_unicode=True) if engine.slope is not None else 'Undefined'}\n",
                    f"[x-intercepts]\n{x_ints}\n",
                    f"[y-intercepts]\n{y_ints}\n",
                ]
            )

        self.formula_box.configure(state="normal")
        self.formula_box.delete("1.0", "end")
        self.formula_box.insert("1.0", "\n".join(lines))
        self.formula_box.configure(state="disabled")

    def _get_grid(self):
        return sf.make_grid(
            self._sv("x_min"),
            self._sv("x_max"),
            self._sv("y_min"),
            self._sv("y_max"),
            self._sv("step"),
        )

    def _set_axes(self, projection: str):
        self.fig.clear()
        if projection == "3d":
            self.ax = self.fig.add_subplot(111, projection="3d", facecolor=sf.AXES_COLOR)
        else:
            self.ax = self.fig.add_subplot(111, facecolor=sf.AXES_COLOR)

    def _plot(self):
        self._stop_animate()
        if not self._parse():
            return
        self._draw_view(self.current_view)

    def _switch_view(self, key: str):
        self._stop_animate()
        self.current_view = key
        self.current_views[self.current_domain] = key
        self._highlight_tab(key)
        if not self._parse():
            return
        self._draw_view(key)

    def _draw_view(self, key: str):
        if self.current_domain == "calculus":
            self._draw_calculus_view(key)
        else:
            self._draw_geometry_view(key)

    def _draw_calculus_view(self, key: str):
        self._set_axes("3d")
        X, Y = self._get_grid()
        engine = self._current_engine()
        cmap = self.cmap_var.get()
        alpha = self._sv("alpha")

        try:
            if key == "surface":
                Z = CalcEngine.safe_eval(engine.get_surface_fn(), X, Y)
                sf.plot_surface(self.ax, X, Y, Z, title="Original Surface  f(x, y)", formula=f"f = {self.eq_var.get()}", cmap_name=cmap, alpha=alpha)
                sf.plot_contour_projection(self.ax, X, Y, Z, cmap_name=cmap)
            elif key == "dfdx":
                Z = CalcEngine.safe_eval(engine.get_dfdx_fn(), X, Y)
                sf.plot_surface(self.ax, X, Y, Z, title="Partial Derivative  ∂f/∂x", formula=f"∂f/∂x = {sp.pretty(engine.df_dx, use_unicode=True)}", cmap_name=cmap, alpha=alpha)
            elif key == "dfdy":
                Z = CalcEngine.safe_eval(engine.get_dfdy_fn(), X, Y)
                sf.plot_surface(self.ax, X, Y, Z, title="Partial Derivative  ∂f/∂y", formula=f"∂f/∂y = {sp.pretty(engine.df_dy, use_unicode=True)}", cmap_name=cmap, alpha=alpha)
            elif key == "integral":
                fn = engine.get_integral_x_fn()
                if fn is None:
                    self._status("Integral could not be computed symbolically.", error=True)
                    return
                Z = CalcEngine.safe_eval(fn, X, Y)
                sf.plot_surface(self.ax, X, Y, Z, title="Antiderivative  ∫f dx  (+ C)", formula="Integral with respect to x", cmap_name=cmap, alpha=alpha)
            elif key == "gradient":
                Z = CalcEngine.safe_eval(engine.get_surface_fn(), X, Y)
                sf.plot_surface(self.ax, X, Y, Z, title="Gradient Overlay  f(x,y) + ∇f arrows", formula=f"f = {self.eq_var.get()}", cmap_name=cmap, alpha=0.6)
                sf.plot_gradient_arrows(self.ax, X, Y, Z, engine.get_dfdx_fn(), engine.get_dfdy_fn(), CalcEngine.safe_eval)
            elif key == "wire":
                Z = CalcEngine.safe_eval(engine.get_surface_fn(), X, Y)
                sf.plot_wireframe(self.ax, X, Y, Z, title="Wireframe  f(x, y)", formula=f"f = {self.eq_var.get()}")
        except Exception as exc:
            self._status(f"Plot error: {exc}", error=True)
            return

        self.canvas.draw()
        self._highlight_tab(key)
        self._status(f"Plotted calculus view: {key}  |  grid {X.shape[1]}×{X.shape[0]}")

    def _draw_geometry_view(self, key: str):
        projection = "3d" if key == "relation_surface" else "2d"
        self._set_axes(projection)

        X, Y = self._get_grid()
        engine = self._current_engine()
        cmap = self.cmap_var.get()
        alpha = self._sv("alpha")
        F = CoordinateGeometryEngine.safe_eval(engine.get_relation_fn(), X, Y)
        relation_text = self.eq_var.get()

        try:
            if key == "relation_surface":
                sf.plot_surface(self.ax, X, Y, F, title="Residual Surface  z = F(x, y)", formula=relation_text, cmap_name=cmap, alpha=alpha)
                sf.plot_contour_projection(self.ax, X, Y, F, cmap_name=cmap)
            elif key == "curve":
                sf.plot_zero_locus(self.ax, X, Y, F, title=f"Zero Locus  ({engine.classification})", formula=relation_text)
            elif key == "slope":
                slope = CoordinateGeometryEngine.safe_eval(engine.get_slope_fn(), X, Y)
                sf.plot_slope_map(self.ax, X, Y, slope, F, title="Implicit Slope  dy/dx", formula=relation_text, cmap_name=cmap)
            elif key == "tangent":
                fx_fn = engine.get_fx_fn()
                fy_fn = engine.get_fy_fn()
                if fx_fn is None or fy_fn is None:
                    self._status("Tangent field could not be computed.", error=True)
                    return
                sf.plot_curve_vectors(self.ax, X, Y, F, fx_fn, fy_fn, title="Tangent Field Along  F(x,y)=0", formula=relation_text, mode="tangent")
            elif key == "normal":
                fx_fn = engine.get_fx_fn()
                fy_fn = engine.get_fy_fn()
                if fx_fn is None or fy_fn is None:
                    self._status("Normal field could not be computed.", error=True)
                    return
                sf.plot_curve_vectors(self.ax, X, Y, F, fx_fn, fy_fn, title="Normal Field Along  F(x,y)=0", formula=relation_text, mode="normal")
            elif key == "intercepts":
                sf.plot_intercepts(self.ax, X, Y, F, engine.x_intercepts, engine.y_intercepts, title="Axis Intercepts", formula=relation_text)
        except Exception as exc:
            self._status(f"Plot error: {exc}", error=True)
            return

        self.canvas.draw()
        self._highlight_tab(key)
        self._status(f"Plotted geometry view: {key}  |  grid {X.shape[1]}×{X.shape[0]}")

    def _start_animate(self):
        if not self._parse():
            return
        self._stop_animate()

        X, Y = self._get_grid()
        cmap = self.cmap_var.get()
        alpha = self._sv("alpha")

        if self.current_domain == "calculus":
            engine = self._current_engine()
            fn0 = engine.get_surface_fn()
            fn1 = engine.get_dfdx_fn()
            if fn0 is None or fn1 is None:
                self._status("Animation source could not be prepared.", error=True)
                return
            safe_eval_fn = CalcEngine.safe_eval
            title_prefix = "Morphing: f → ∂f/∂x"
        else:
            engine = self._current_engine()
            fn0 = engine.get_relation_fn()
            fn1 = engine.get_slope_fn()
            if fn0 is None or fn1 is None:
                self._status("Animation source could not be prepared.", error=True)
                return
            safe_eval_fn = CoordinateGeometryEngine.safe_eval
            title_prefix = "Morphing: F → dy/dx"

        self._set_axes("3d")
        self._frame_data = sf.build_animation_frames(X, Y, fn0, fn1, safe_eval_fn, n_frames=50)
        self._frame_data += self._frame_data[::-1]
        ax = self.ax

        def update(frame_idx):
            Zt = self._frame_data[frame_idx]
            sf.plot_surface(ax, X, Y, Zt, title=f"{title_prefix}  [{frame_idx + 1}/{len(self._frame_data)}]", cmap_name=cmap, alpha=alpha, show_colorbar=False)
            self.canvas.draw()

        self._status("Animation running…")
        self._anim = FuncAnimation(self.fig, update, frames=len(self._frame_data), interval=50, repeat=True)
        self.canvas.draw()

    def _stop_animate(self):
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass
            self._anim = None

    def _save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Save Plot As…",
        )
        if path:
            self.fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=sf.BG_COLOR)
            self._status(f"Saved → {path}")

    def _copy_formulas(self):
        self.clipboard_clear()
        self.clipboard_append(self.formula_box.get("1.0", "end"))
        self._status("Formulas copied to clipboard.")

    def _status(self, msg: str, error: bool = False):
        self.status_var.set(msg)
        self.after(100, lambda: None)

    def on_close(self):
        self._stop_animate()
        plt.close("all")
        self.destroy()
