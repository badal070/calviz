"""
main.py
Entry point for the 3D Calculus Visualizer.

Run:
    python main.py
"""

import sys

def main():
    try:
        from gui import App
    except ImportError as exc:
        print(f"[Error] Missing dependency: {exc}")
        print("Please run:  pip install -r requirements.txt")
        sys.exit(1)

    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
