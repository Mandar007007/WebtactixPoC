"""Result types for Playwright step execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Page


@dataclass
class ScrollEvent:
    """Auto-scroll performed before a click or input action."""

    direction: str
    container_x: int
    container_y: int
    target_x: int
    target_y: int
    screenshot_before_bytes: bytes
    screenshot_after_bytes: bytes
    viewport_width: int = 1920
    viewport_height: int = 1080


@dataclass
class ApplyStepOutcome:
    """Outcome of applying one ActionStep."""

    page: Page
    scroll: Optional[ScrollEvent] = None
    pre_click_screenshot: bytes = field(default_factory=bytes)
    click_x: int = 0
    click_y: int = 0
    viewport_width: int = 0
    viewport_height: int = 0
