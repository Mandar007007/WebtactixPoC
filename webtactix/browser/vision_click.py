"""Vision-click fallback: screenshot + Claude vision -> coordinates.

When the actree fails to index a visible element (popups, custom
widgets, etc.), this module takes a screenshot, sends it to Claude
with a description of what to click, and returns pixel coordinates.
The caller then clicks at those coordinates directly, bypassing
the fragile actree index entirely.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional, Tuple

import anthropic
from playwright.async_api import Page

from webtactix.llm.anthropic_client import _load_api_key_from_xlm_core

_SYSTEM = (
    "You are a visual UI element locator. You receive a "
    "screenshot of a web page and a description of an "
    "element to click. Return the EXACT pixel coordinates "
    "of the CENTER of that element.\n\n"
    "Return JSON only: {\"x\": int, \"y\": int, "
    "\"confidence\": float, \"reasoning\": string}\n\n"
    "Rules:\n"
    "- x,y are pixel coordinates relative to the "
    "screenshot's top-left corner.\n"
    "- Target the CENTER of the described element.\n"
    "- confidence is 0.0-1.0 for how sure you are.\n"
    "- If the element is not visible, return "
    "{\"x\": -1, \"y\": -1, \"confidence\": 0.0, "
    "\"reasoning\": \"element not found\"}.\n"
)


async def vision_locate(
    screenshot_bytes: bytes,
    element_description: str,
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    """Ask Claude to find an element's coordinates in a screenshot.

    Args:
        screenshot_bytes: PNG screenshot of the page.
        element_description: What to find (e.g.
            "Governance and Procedure > SOP SOP" in
            a dropdown popup).
        api_key: Anthropic API key. Falls back to env/xlm_core.
        model: Claude model to use.

    Returns:
        Dict with x, y, confidence, reasoning.
    """
    key = api_key or _load_api_key_from_xlm_core()
    client = anthropic.AsyncAnthropic(
        api_key=key, timeout=30.0,
    )

    img_b64 = base64.standard_b64encode(
        screenshot_bytes,
    ).decode("ascii")

    user_text = (
        f"Find and return the pixel coordinates of the "
        f"CENTER of this element on the screenshot:\n\n"
        f"Element: {element_description}\n\n"
        f"Return JSON only."
    )

    resp = await client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0.0,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }],
    )

    raw = ""
    for block in resp.content:
        if block.type == "text":
            raw += block.text

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "x": -1, "y": -1,
            "confidence": 0.0,
            "reasoning": f"JSON parse error: {raw[:200]}",
        }


async def vision_click(
    page: Page,
    element_description: str,
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> Tuple[bool, dict]:
    """Screenshot the page, locate element via vision, click it.

    Args:
        page: Playwright page to screenshot and click.
        element_description: What to click.
        api_key: Anthropic API key.
        model: Claude model.

    Returns:
        Tuple of (success, location_result).
    """
    screenshot = await page.screenshot(type="png")

    result = await vision_locate(
        screenshot,
        element_description,
        api_key=api_key,
        model=model,
    )

    x = result.get("x", -1)
    y = result.get("y", -1)
    confidence = result.get("confidence", 0.0)

    print(
        f"[VISION-CLICK] element={element_description!r} "
        f"coords=({x},{y}) conf={confidence:.2f} "
        f"reason={result.get('reasoning', '')!r}"
    )

    if x < 0 or y < 0 or confidence < 0.1:
        print("[VISION-CLICK] FAILED: element not found")
        return False, result

    await page.mouse.click(x, y)
    print(f"[VISION-CLICK] Clicked at ({x}, {y})")
    return True, result
