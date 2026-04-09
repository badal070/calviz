"""
chat_panel.py
Chat sidebar that submits user questions to the LLM bridge and renders replies.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, Callable

import llm_bridge

if TYPE_CHECKING:
    from viz_bus import VisualizationPayload


PANEL = "#1a1a2e"
BG = "#0f0f1a"
BTN_BG = "#16213e"
BTN_ACT = "#0f3460"
TEXT = "#e0e0ff"
SUBTEXT = "#9090bb"
ACCENT = "#7b68ee"
SUCCESS = "#4ecca3"
ERROR = "#ff8a8a"
USER_BUBBLE = "#243b6b"
TUTOR_BUBBLE = "#20243a"

FONT_BODY = ("Segoe UI", 9)
FONT_CAPTION = ("Segoe UI", 8, "bold")


class ChatPanel(tk.Frame):
    def __init__(
        self,
        parent,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        super().__init__(parent, bg=PANEL)
        self._on_status = on_status or (lambda _msg: None)
        self._on_error = on_error or (lambda _msg: None)
        self._pending = False
        self._thinking_bubble: tk.Label | None = None
        self._thinking_caption: tk.Label | None = None

        self._build_ui()
        self._add_message(
            speaker="Tutor",
            text="Ask about a multivariable calculus surface and I will load the equation into the visualizer when the Colab response arrives.",
            role="tutor",
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x", padx=10, pady=(10, 8))

        tk.Label(header, text="LLM Tutor", bg=PANEL, fg=ACCENT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            header,
            text="Questions are sent to the Colab bridge and rendered here after validation.",
            bg=PANEL,
            fg=SUBTEXT,
            font=("Segoe UI", 8),
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(self, bg=PANEL)
        body.pack(fill="both", expand=True, padx=10)

        self._canvas = tk.Canvas(body, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._messages = tk.Frame(self._canvas, bg=BG)
        self._messages_window = self._canvas.create_window((0, 0), window=self._messages, anchor="nw")
        self._messages.bind("<Configure>", self._on_messages_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        entry_wrap = tk.Frame(self, bg=PANEL)
        entry_wrap.pack(fill="x", padx=10, pady=10)

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(
            entry_wrap,
            textvariable=self.input_var,
            bg=BTN_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=6,
            font=FONT_BODY,
        )
        self.entry.pack(fill="x", side="left", expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda _event: self._send())

        self.send_btn = tk.Button(
            entry_wrap,
            text="Send",
            bg=ACCENT,
            fg=BG,
            activebackground=BTN_ACT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=14,
            pady=7,
            cursor="hand2",
            command=self._send,
        )
        self.send_btn.pack(side="right")

    def _on_messages_configure(self, _event=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._scroll_to_bottom()

    def _on_canvas_configure(self, event) -> None:
        self._canvas.itemconfigure(self._messages_window, width=event.width)

    def _scroll_to_bottom(self) -> None:
        self.after_idle(lambda: self._canvas.yview_moveto(1.0))

    def _set_pending(self, pending: bool) -> None:
        self._pending = pending
        state = "disabled" if pending else "normal"
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)
        if not pending:
            self.entry.focus_set()

    def _add_message(self, speaker: str, text: str, role: str) -> tuple[tk.Label, tk.Label]:
        row = tk.Frame(self._messages, bg=BG)
        row.pack(fill="x", pady=6)

        anchor = "e" if role == "user" else "w"
        caption_fg = SUCCESS if role == "user" else ACCENT
        bubble_bg = USER_BUBBLE if role == "user" else TUTOR_BUBBLE

        caption = tk.Label(
            row,
            text=speaker,
            bg=BG,
            fg=caption_fg,
            font=FONT_CAPTION,
        )
        caption.pack(anchor=anchor, padx=8, pady=(0, 2))

        bubble = tk.Label(
            row,
            text=text,
            bg=bubble_bg,
            fg=TEXT,
            font=FONT_BODY,
            justify="left",
            wraplength=260,
            padx=12,
            pady=10,
        )
        bubble.pack(anchor=anchor, padx=8)
        self._scroll_to_bottom()
        return caption, bubble

    def _send(self) -> None:
        user_text = self.input_var.get().strip()
        if not user_text:
            return
        if self._pending:
            self._safe_status("A Colab request is already in progress.")
            return

        self.input_var.set("")
        self._add_message("You", user_text, role="user")
        self._thinking_caption, self._thinking_bubble = self._add_message("Tutor", "Thinking...", role="tutor")
        self._set_pending(True)

        llm_bridge.ask(
            user_text,
            on_status=self._bridge_status,
            on_error=self._bridge_error,
        )

    def _dispatch_ui(self, callback) -> None:
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except tk.TclError:
            pass

    def _bridge_status(self, msg: str) -> None:
        self._dispatch_ui(lambda: self._handle_bridge_status(msg))

    def _bridge_error(self, msg: str) -> None:
        self._dispatch_ui(lambda: self._handle_bridge_error(msg))

    def _handle_bridge_status(self, msg: str) -> None:
        self._safe_status(msg)
        if self._thinking_bubble is not None and msg in {"Sending question to Colab...", "LLM is thinking..."}:
            self._thinking_bubble.configure(text=msg)

    def _handle_bridge_error(self, msg: str) -> None:
        self._safe_error(msg)
        if self._thinking_caption is not None:
            self._thinking_caption.configure(fg=ERROR)
        if self._thinking_bubble is not None:
            self._thinking_bubble.configure(text=msg, fg=ERROR)
        else:
            self._add_message("Tutor", msg, role="tutor")
        self._thinking_bubble = None
        self._thinking_caption = None
        self._set_pending(False)

    def _safe_status(self, msg: str) -> None:
        self._on_status(msg)

    def _safe_error(self, msg: str) -> None:
        self._on_error(msg)

    def render_explanation(self, payload: "VisualizationPayload") -> None:
        title = payload.concept_title.strip() or "Untitled Concept"
        text = f"{title}\nEquation: {payload.equation}\n\n{payload.explanation.strip()}"

        if self._thinking_bubble is not None:
            self._thinking_bubble.configure(text=text, fg=TEXT)
        else:
            self._add_message("Tutor", text, role="tutor")

        if self._thinking_caption is not None:
            self._thinking_caption.configure(text="Tutor", fg=ACCENT)

        self._thinking_bubble = None
        self._thinking_caption = None
        self._set_pending(False)
