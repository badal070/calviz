"""
viz_bus.py
Thread-safe queue used to hand validated visualization payloads to the GUI.
"""

from dataclasses import dataclass
import queue


@dataclass(slots=True)
class VisualizationPayload:
    concept_title: str
    equation: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    step: float
    explanation: str


_bus: "queue.Queue[VisualizationPayload]" = queue.Queue()


def publish(payload: VisualizationPayload) -> None:
    _bus.put(payload)


def subscribe() -> "queue.Queue[VisualizationPayload]":
    return _bus
