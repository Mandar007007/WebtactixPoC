"""Request/response schemas for the bridge API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RunRequest:
    """Payload to start a goal-driven agent run."""

    goal: str
    cdp_url: str = "http://localhost:9222"
    dataset: str = "vision_recorder"
    task_id: int = 0


@dataclass
class RunState:
    """Tracks an in-flight agent run."""

    run_id: str
    goal: str
    cdp_url: str
    page_url: str = ""
    status: str = "pending"
    steps_count: int = 0
    error: Optional[str] = None
    cancel_requested: bool = False
    subscribers: list = field(default_factory=list)


def step_event(step: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a VR step for WebSocket broadcast."""
    return {"type": "step", "step": step}


def done_event(
    *,
    status: str,
    steps_count: int,
    answer: str = "",
) -> Dict[str, Any]:
    """Wrap a completion event."""
    return {
        "type": "done",
        "status": status,
        "steps_count": steps_count,
        "answer": answer,
    }


def error_event(message: str) -> Dict[str, Any]:
    """Wrap an error event."""
    return {"type": "error", "message": message}


def progress_event(round_num: int, status: str) -> Dict[str, Any]:
    """Wrap a progress event."""
    return {
        "type": "progress",
        "round": round_num,
        "status": status,
    }
