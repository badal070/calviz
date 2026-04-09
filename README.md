# Math Visualizer

An interactive desktop application for exploring both **calculus** and **coordinate geometry** concepts with a smooth dark-theme GUI.

---

## Features

| Feature | Details |
|---|---|
| **Two math domains** | Switch between Calculus and Coordinate Geometry from the top bar |
| **Calculus mode** | Plot `f(x,y)`, partial derivatives, antiderivatives, gradients, and wireframes |
| **Coordinate Geometry mode** | Explore implicit relations `F(x,y)=0`, slope maps, tangent fields, normal fields, and intercepts |
| **Rich presets** | Includes classic surfaces plus lines, circles, ellipses, parabolas, hyperbolas, and more |
| **Symbolic engine** | SymPy computes derivatives, antiderivatives, implicit slope, relation type, and intercepts |
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
cd calviz

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run
python main.py
```

---

## Project Structure

```
calviz/
├── main.py          # Entry point
├── calculus.py      # SymPy parsing, differentiation, integration, lambdify
├── coordinate_geometry.py  # Implicit relation parsing + conic analysis
├── surfaces.py      # 2-D/3-D plot helpers and animation frames
├── gui.py           # Tkinter GUI + domain switcher + embedded Matplotlib canvas
├── requirements.txt
└── README.md
```

---

## Expression Syntax

Use standard Python / SymPy math syntax.

Calculus examples:

| Math | Type |
|---|---|
| x² + y² | `x**2 + y**2` |
| sin(x)·cos(y) | `sin(x) * cos(y)` |
| e^(−x²) | `exp(-x**2)` |
| √(x²+y²) | `sqrt(x**2 + y**2)` |
| ln(x+y) | `log(x + y)` |

Supported functions: `sin cos tan asin acos atan exp log ln sqrt Abs sinh cosh tanh ceiling floor`  
Constants: `pi`, `e` / `E`

Coordinate geometry examples:

| Relation | Type |
|---|---|
| x² + y² = 9 | `x**2 + y**2 = 9` |
| y = x² | `y - x**2 = 0` |
| x²/9 + y²/4 = 1 | `x**2/9 + y**2/4 = 1` |
| 2x - y - 3 = 0 | `2*x - y - 3 = 0` |

---

## Controls

- **Sliders** – drag to update domain or transparency; click **Plot / Refresh** to redraw
- **Domain switch** – toggle between Calculus and Coordinate Geometry from the top bar
- **View tabs** – click any tab to switch the active view within the current domain
- **Presets** – click any preset name to instantly load a relevant example for the current domain
- **Animate Morph** – watch the plot interpolate between two symbolic views of the active domain
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
