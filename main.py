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
        from llm_bridge import check_connection
    except ImportError as exc:
        print(f"[Error] Missing dependency: {exc}")
        print("Please run:  pip install -r requirements.txt")
        sys.exit(1)

    ok, msg = check_connection()
    if not ok:
        print(f"[Warning] Colab server unreachable: {msg}")
        print("The visualizer will still open. Set the URL in llm_bridge.py")
        print("or use the manual equation entry mode until the bridge is live.")

    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
