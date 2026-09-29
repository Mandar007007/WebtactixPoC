"""HTTP client that delegates descriptor generation to Vision Recorder backend.

Instead of running a local Anthropic call, this client POSTs step data to
VR backend's ``/api/generate-descriptor`` endpoint and returns the response
in the same ``ElementDescriptorResponse`` shape that VR's UI expects.

Falls back to the local ``DescriptorAgent`` when the VR backend is
unreachable.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

import httpx

from webtactix.core.schemas import ActionStep, ActionType

logger = logging.getLogger(__name__)

_DEFAULT_VR_BACKEND_URL = (
    "https://manava-dev-vision-recorder.azurewebsites.net"
)
_DESCRIPTOR_PATH = "/generate-descriptor"
_REQUEST_TIMEOUT_SECONDS = 60.0


def _get_vr_backend_url() -> str:
    """Resolve the VR backend URL from env or use the default."""
    return os.environ.get(
        "VR_BACKEND_URL",
        _DEFAULT_VR_BACKEND_URL,
    )


def _map_action_to_vr(action: str) -> str:
    """Map WebTactix action names to Vision Recorder actions."""
    mapping = {
        "click": "click",
        "input": "type",
        "select": "click",
        "press_enter": "presskey",
        "goto": "click",
    }
    return mapping.get(action, action)


def _encode_screenshot(screenshot_bytes: bytes) -> str:
    """Base64-encode raw PNG bytes for the VR API payload."""
    return base64.standard_b64encode(screenshot_bytes).decode("ascii")


class VRDescriptorClient:
    """Generates descriptors by calling Vision Recorder's backend API.

    Attributes:
        base_url: The VR backend base URL.
        fallback_agent: Optional local DescriptorAgent for resilience.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        fallback_api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or _get_vr_backend_url()).rstrip("/")
        self._endpoint = f"{self._base_url}{_DESCRIPTOR_PATH}"
        self._client = httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

        self._fallback_agent = None
        if fallback_api_key:
            from webtactix.agents.descriptor_agent import DescriptorAgent
            self._fallback_agent = DescriptorAgent(
                api_key=fallback_api_key,
            )

        logger.info(
            "VRDescriptorClient targeting %s",
            self._endpoint,
        )

    async def generate(
        self,
        *,
        screenshot_bytes: bytes,
        step: ActionStep,
        plan_goal: str,
        url: str,
        x: int = 0,
        y: int = 0,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Dict[str, Any]:
        """Generate a descriptor for one action step via VR backend.

        Args:
            screenshot_bytes: PNG image of the page before the action.
            step: The ActionStep that was executed.
            plan_goal: Human-readable goal of the current plan.
            url: Page URL at the time of the action.
            x: Viewport X coordinate of the element center.
            y: Viewport Y coordinate of the element center.
            screen_width: Browser viewport width in pixels.
            screen_height: Browser viewport height in pixels.

        Returns:
            Descriptor dict in VR's ElementDescriptorResponse shape.
        """
        wt_action = (
            step.action.value
            if isinstance(step.action, ActionType)
            else str(step.action)
        )
        vr_action = _map_action_to_vr(wt_action)
        screenshot_b64 = _encode_screenshot(screenshot_bytes)

        vr_step: Dict[str, Any] = {
            "action": vr_action,
            "screenshot": screenshot_b64,
            "zoomedScreenshot": screenshot_b64,
            "x": x,
            "y": y,
            "screenWidth": screen_width,
            "screenHeight": screen_height,
        }

        if step.text and vr_action == "type":
            vr_step["value"] = step.text

        payload: Dict[str, Any] = {"steps": [vr_step]}

        try:
            resp = await self._client.post(
                self._endpoint,
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            logger.warning(
                "VR backend call failed (%s), falling back to local agent",
                exc,
            )
            return await self._local_fallback(
                screenshot_bytes=screenshot_bytes,
                step=step,
                plan_goal=plan_goal,
                url=url,
                x=x,
                y=y,
                screen_width=screen_width,
                screen_height=screen_height,
            )

        return self._extract_descriptor(body, step=step)

    async def generate_scroll(
        self,
        *,
        screenshot_before_bytes: bytes,
        screenshot_after_bytes: bytes,
        direction: str,
        plan_goal: str,
        url: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        container_x: int = 0,
        container_y: int = 0,
        target_x: int = 0,
        target_y: int = 0,
    ) -> Dict[str, Any]:
        """Generate a scroll descriptor via VR backend.

        Args:
            screenshot_before_bytes: PNG before the scroll.
            screenshot_after_bytes: PNG after the scroll.
            direction: Scroll direction (up/down/left/right).
            plan_goal: Human-readable goal.
            url: Page URL.
            viewport_width: Browser viewport width.
            viewport_height: Browser viewport height.
            container_x: Scroll container X coordinate.
            container_y: Scroll container Y coordinate.
            target_x: Target element X coordinate.
            target_y: Target element Y coordinate.

        Returns:
            Descriptor dict in VR's ScrollDescriptorResponse shape.
        """
        before_b64 = _encode_screenshot(screenshot_before_bytes)
        after_b64 = _encode_screenshot(screenshot_after_bytes)

        vr_step: Dict[str, Any] = {
            "action": "scroll",
            "scrollDirection": direction,
            "x": container_x,
            "y": container_y,
            "screenWidth": viewport_width,
            "screenHeight": viewport_height,
            "scrollContainer": {
                "screenshot": before_b64,
                "zoomedScreenshot": before_b64,
                "x": container_x,
                "y": container_y,
            },
            "targetElement": {
                "screenshot": after_b64,
                "zoomedScreenshot": after_b64,
                "x": target_x,
                "y": target_y,
            },
        }

        payload: Dict[str, Any] = {"steps": [vr_step]}

        try:
            resp = await self._client.post(
                self._endpoint,
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            logger.warning(
                "VR backend scroll call failed (%s), "
                "returning fallback descriptor",
                exc,
            )
            return {
                "action": "scroll",
                "instruction": (
                    f"Scroll {direction} to bring target into view "
                    f"for: {plan_goal}"
                ),
                "scroll_direction": direction,
                "scroll_container_element": {
                    "purpose": f"Scroll to reach target for {plan_goal}",
                    "description": "Main page scrollable area",
                    "spatial_anchors": [],
                    "visual_hints": "",
                    "confidence": 0.5,
                },
                "target_element": {
                    "purpose": plan_goal,
                    "description": "Target element after scroll",
                    "spatial_anchors": [],
                    "visual_hints": "",
                    "confidence": 0.5,
                },
            }

        return self._extract_scroll_descriptor(body)

    def _extract_descriptor(
        self,
        body: Dict[str, Any],
        *,
        step: ActionStep,
    ) -> Dict[str, Any]:
        """Pull the first descriptor from the VR API response.

        VR backend returns::

            {
                "success": true,
                "data": {
                    "descriptors": [{ ... }],
                    ...
                }
            }

        The descriptor is already in ``ElementDescriptorResponse`` shape.
        """
        descriptors = (
            body.get("data", {}).get("descriptors", [])
        )
        if not descriptors:
            logger.warning("VR backend returned empty descriptors list")
            wt_action = (
                step.action.value
                if isinstance(step.action, ActionType)
                else str(step.action)
            )
            return {
                "action": _map_action_to_vr(wt_action),
                "instruction": f"{wt_action} {step.role} '{step.name}'",
                "element_descriptor": None,
            }

        desc = descriptors[0]

        if step.text and _map_action_to_vr(
            step.action.value
            if isinstance(step.action, ActionType)
            else str(step.action)
        ) == "type":
            desc["text"] = step.text

        return desc

    def _extract_scroll_descriptor(
        self,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pull the first scroll descriptor from the VR API response."""
        descriptors = (
            body.get("data", {}).get("descriptors", [])
        )
        if not descriptors:
            return {
                "action": "scroll",
                "instruction": "Scroll to bring target into view",
                "scroll_direction": "down",
                "scroll_container_element": {},
                "target_element": {},
            }
        return descriptors[0]

    async def _local_fallback(
        self,
        *,
        screenshot_bytes: bytes,
        step: ActionStep,
        plan_goal: str,
        url: str,
        x: int = 0,
        y: int = 0,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> Dict[str, Any]:
        """Use the local DescriptorAgent as a fallback."""
        if self._fallback_agent is None:
            wt_action = (
                step.action.value
                if isinstance(step.action, ActionType)
                else str(step.action)
            )
            vr_action = _map_action_to_vr(wt_action)
            desc: Dict[str, Any] = {
                "action": vr_action,
                "instruction": f"{wt_action} {step.role} '{step.name}'",
                "element_descriptor": None,
            }
            if step.text and vr_action == "type":
                desc["text"] = step.text
            return desc

        from webtactix.agents.descriptor_agent import (
            format_for_vision_recorder,
        )
        raw_desc = await self._fallback_agent.generate(
            screenshot_bytes=screenshot_bytes,
            step=step,
            plan_goal=plan_goal,
            url=url,
        )
        vr_step = format_for_vision_recorder(
            raw_desc,
            screenshot_b64=_encode_screenshot(screenshot_bytes),
            url=url,
            action_step=step,
            x=x,
            y=y,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        return vr_step.get("descriptor", raw_desc)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
