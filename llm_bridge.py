"""
llm_bridge.py
HTTP bridge between the local app and the Colab-hosted LLM service.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

import requests

import viz_bus as bus
from viz_bus import VisualizationPayload


COLAB_BASE_URL = os.getenv("COLAB_BASE_URL", "https://xxxx.ngrok-free.app").rstrip("/")
POLL_INTERVAL = 1.5
POLL_TIMEOUT = 120


def _validate_pair(name: str, value) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two numeric values.") from exc

    if len(pair) != 2 or pair[0] >= pair[1]:
        raise ValueError(f"Invalid {name}: {pair}")
    return pair


def _validate_payload(data: dict) -> VisualizationPayload:
    """Strict field validation before the visualizer ever sees the data."""
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")

    required = {"concept_title", "equation", "x_range", "y_range", "step", "explanation"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"LLM response missing fields: {sorted(missing)}")

    xr = _validate_pair("x_range", data["x_range"])
    yr = _validate_pair("y_range", data["y_range"])
    step = float(data["step"])
    if not (0.01 <= step <= 1.0):
        raise ValueError(f"Step out of bounds: {step}")

    return VisualizationPayload(
        concept_title=str(data["concept_title"]).strip(),
        equation=str(data["equation"]).strip(),
        x_range=xr,
        y_range=yr,
        step=step,
        explanation=str(data["explanation"]).strip(),
    )


def ask(
    user_text: str,
    on_status: Callable[[str], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> None:
    """
    Non-blocking. Sends the question to Colab, polls until a result arrives,
    validates it, then publishes it to viz_bus.
    """

    def _worker():
        def _status(msg: str) -> None:
            if on_status:
                on_status(msg)

        def _error(msg: str) -> None:
            if on_error:
                on_error(msg)

        try:
            _status("Sending question to Colab...")
            resp = requests.post(
                f"{COLAB_BASE_URL}/ask",
                json={"text": user_text},
                timeout=15,
            )
            resp.raise_for_status()

            _status("LLM is thinking...")
            deadline = time.time() + POLL_TIMEOUT
            while time.time() < deadline:
                time.sleep(POLL_INTERVAL)
                poll_resp = requests.get(f"{COLAB_BASE_URL}/payload", timeout=10)
                poll_resp.raise_for_status()
                poll = poll_resp.json()

                status = poll.get("status")
                if status == "pending":
                    continue
                if status == "error":
                    _error(f"LLM error: {poll.get('error', 'unknown')}")
                    return
                if status == "ok":
                    payload = _validate_payload(poll.get("data"))
                    bus.publish(payload)
                    _status(f"Ready: {payload.concept_title}")
                    return

                _error(f"Unexpected poll status: {status!r}")
                return

            _error("Timeout: LLM did not respond in time.")
        except requests.ConnectionError:
            _error("Cannot reach Colab. Is the server cell still running?")
        except requests.RequestException as exc:
            _error(f"Bridge request failed: {exc}")
        except Exception as exc:
            _error(f"Bridge error: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def check_connection() -> tuple[bool, str]:
    """Synchronous health-check. Call once at startup."""
    try:
        response = requests.get(f"{COLAB_BASE_URL}/health", timeout=8)
        response.raise_for_status()
        if response.json().get("status") == "alive":
            return True, "Colab server reachable."
        return False, "Unexpected health response."
    except Exception as exc:
        return False, str(exc)
