# 3D Calculus Visualizer

An interactive desktop application for visualising multivariable functions **f(x, y)** — their surfaces, partial derivatives, antiderivatives, and gradient fields — with a smooth dark-theme GUI.

---

## Features

| Feature | Details |
|---|---|
| **Any f(x,y)** | Type any expression using `x`, `y`, `sin`, `cos`, `exp`, `log`, `sqrt`, … |
| **6 view modes** | Original surface, ∂f/∂x, ∂f/∂y, ∫f dx, Gradient overlay, Wireframe |
| **9 presets** | Paraboloid, Saddle, Gaussian, Monkey Saddle, Peaks, and more |
| **Symbolic engine** | SymPy computes exact derivatives and antiderivatives |
| **Interactive sliders** | Adjust X/Y domain, step size, and transparency live |
| **6 colourmaps** | plasma, viridis, coolwarm, magma, turbo, inferno |
| **Gradient arrows** | Quiver plot showing ∇f on the surface |
| **Contour projection** | Contour lines projected onto the floor plane |
| **Ping-pong animation** | Morphs surface smoothly from f → ∂f/∂x and back |
| **Export** | Save as PNG, SVG, or PDF at 180 dpi |
| **Copy formulas** | Copy symbolic results to clipboard |

---

## Quick Start

```bash
# 1. Clone / download the project
cd 3D_Calculus_Visualizer

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run
python main.py
```

---

## Project Structure

```
3D_Calculus_Visualizer/
├── main.py          # Entry point
├── calculus.py      # SymPy parsing, differentiation, integration, lambdify
├── surfaces.py      # Meshgrid generation, plot helpers, animation frames
├── gui.py           # Tkinter GUI + embedded Matplotlib canvas
├── requirements.txt
└── README.md
```

---

## Expression Syntax

Use standard Python / SymPy math syntax:

| Math | Type |
|---|---|
| x² + y² | `x**2 + y**2` |
| sin(x)·cos(y) | `sin(x) * cos(y)` |
| e^(−x²) | `exp(-x**2)` |
| √(x²+y²) | `sqrt(x**2 + y**2)` |
| ln(x+y) | `log(x + y)` |

Supported functions: `sin cos tan asin acos atan exp log ln sqrt Abs sinh cosh tanh ceiling floor`  
Constants: `pi`, `e` / `E`

---

## Controls

- **Sliders** – drag to update domain or transparency; click **Plot / Refresh** to redraw
- **View tabs** – click any tab to switch the active surface without re-parsing
- **Presets** – click any preset name to instantly load a famous surface
- **Animate Morph** – watch the surface interpolate from f to ∂f/∂x in a ping-pong loop
- **Stop Animation** – halts the animation and restores interactive rotation
- **Save Image** – opens a file dialog; supports PNG, SVG, PDF
- **Matplotlib toolbar** – pan, zoom, home, and rotate via the toolbar below the canvas

---

## Requirements

- Python 3.10 or newer
- numpy ≥ 1.24
- matplotlib ≥ 3.7
- sympy ≥ 1.12

Tkinter ships with the standard Python installer on Windows and macOS.  
On Linux: `sudo apt install python3-tk`

---

## License

MIT — free for personal and educational use.
