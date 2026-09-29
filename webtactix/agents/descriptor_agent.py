# webtactix/agents/descriptor_agent
"""Generates visual element descriptors for test step recording.

After each successful browser action, this agent receives a screenshot
(taken before the action) plus the ActionStep metadata and produces a
rich JSON descriptor suitable for automated test step generation.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from webtactix.core.schemas import ActionStep, ActionType


DESCRIPTOR_SYSTEM_PROMPT = (
    "You are a test-step descriptor generator. You receive a "
    "screenshot of a web page and metadata about an action that "
    "was performed on it. Generate a precise visual descriptor "
    "of the target element.\n\n"
    "Return JSON only, no markdown fences."
)


def _build_user_prompt(
    *,
    step: ActionStep,
    plan_goal: str,
    url: str,
) -> str:
    """Build the text portion of the vision prompt."""
    action_name = (
        step.action.value
        if isinstance(step.action, ActionType)
        else str(step.action)
    )
    text_value = f', text="{step.text}"' if step.text else ""

    return (
        f"Page URL: {url}\n"
        f"Action performed: {action_name}\n"
        f"Element role: {step.role}\n"
        f"Element name/label: {step.name or '(none)'}\n"
        f"Input text: {step.text or '(none)'}\n"
        f"Plan goal: {plan_goal}\n\n"
        "Generate a JSON descriptor for this action in EXACTLY "
        "this schema:\n"
        "{\n"
        '  "action": "<click|input|select|goto|press_enter>",\n'
        '  "input_value": "<text typed, or null>",\n'
        '  "instruction": "<human-readable step description>",\n'
        '  "element_descriptor": {\n'
        '    "purpose": "<why this action is performed>",\n'
        '    "description": "<what the element looks like and '
        'where it is on the page>",\n'
        '    "spatial_anchors": [\n'
        "      {\n"
        '        "anchor_text": "<nearby visible text>",\n'
        '        "relationship": "<above|below|left-of|'
        'right-of|within>",\n'
        '        "distance": "<none|extra-small|small|medium|'
        'large|extra-large>"\n'
        "      }\n"
        "    ],\n"
        '    "visual_hints": "<color, shape, icon description>",\n'
        '    "confidence": <0.0-1.0>\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- The accessibility metadata (role, name, plan goal) is "
        "AUTHORITATIVE for which element was acted on.\n"
        "- When the element name is generic (e.g. 'Select row', "
        "'Select', 'Open Menu'), derive the instruction from "
        "plan goal — do NOT pick a different row/label from the "
        "screenshot.\n"
        "- Look at the screenshot to describe appearance and "
        "spatial anchors for the element identified by metadata.\n"
        "- Provide 2-4 spatial_anchors using nearby visible text.\n"
        "- instruction must be a clear, human-readable sentence.\n"
        "- For input/goto actions, include the typed value in "
        "instruction.\n"
        "- confidence reflects how certain you are about the "
        "element identification.\n"
    )


_GENERIC_ELEMENT_NAMES = frozenset({
    "select row",
    "select",
    "open menu",
    "menu",
    "more",
    "actions",
    "",
})


def _instruction_from_metadata(
    *,
    step: ActionStep,
    plan_goal: str,
) -> str:
    """Build a reliable instruction from a11y metadata and plan goal."""
    action_name = (
        step.action.value
        if isinstance(step.action, ActionType)
        else str(step.action)
    )
    name = (step.name or "").strip()
    name_lower = name.lower()

    if plan_goal and name_lower in _GENERIC_ELEMENT_NAMES:
        goal = plan_goal.strip().rstrip(".")
        if action_name == "click":
            return f"{goal}."
        return f"{goal} ({action_name})."

    if name:
        return f"{action_name.capitalize()} {step.role} '{name}'."

    if plan_goal:
        return f"{plan_goal.strip().rstrip('.')}."

    return f"{action_name} {step.role}."


def reconcile_descriptor_with_metadata(
    descriptor: Dict[str, Any],
    *,
    step: ActionStep,
    plan_goal: str,
) -> Dict[str, Any]:
    """Anchor vision output to accessibility metadata and plan goal."""
    anchored = _instruction_from_metadata(step=step, plan_goal=plan_goal)
    name = (step.name or "").strip()
    name_lower = name.lower()

    descriptor["instruction"] = anchored
    element_desc = dict(descriptor.get("element_descriptor") or {})
    if plan_goal:
        element_desc["purpose"] = plan_goal
    if name and name_lower not in _GENERIC_ELEMENT_NAMES:
        element_desc["description"] = (
            element_desc.get("description")
            or f"{step.role} labeled '{name}'"
        )
    descriptor["element_descriptor"] = element_desc
    return descriptor


SCROLL_DESCRIPTOR_SYSTEM = (
    "You are a test-step descriptor generator for scroll actions. "
    "Return JSON only, no markdown fences."
)


def _build_scroll_prompt(
    *,
    plan_goal: str,
    direction: str,
    url: str,
) -> str:
    """Build prompt for auto-scroll descriptor generation."""
    return (
        f"Page URL: {url}\n"
        f"Scroll direction: {direction}\n"
        f"Reason for scroll: {plan_goal}\n\n"
        "Two screenshots are provided:\n"
        "1. BEFORE scroll — page state before bringing target into view\n"
        "2. AFTER scroll — target element now visible\n\n"
        "Return JSON in EXACTLY this schema:\n"
        "{\n"
        '  "instruction": "<human-readable scroll step>",\n'
        '  "scroll_container_element": {\n'
        '    "purpose": "<why scroll this container>",\n'
        '    "description": "<scrollable region description>",\n'
        '    "spatial_anchors": [],\n'
        '    "visual_hints": "<scrollbar/container hints>",\n'
        '    "confidence": 0.0-1.0\n'
        "  },\n"
        '  "target_element": {\n'
        '    "purpose": "<why this target matters>",\n'
        '    "description": "<target element appearance>",\n'
        '    "spatial_anchors": [],\n'
        '    "visual_hints": "<visual hints>",\n'
        '    "confidence": 0.0-1.0\n'
        "  }\n"
        "}\n"
    )


def _encode_image(image_bytes: bytes) -> str:
    """Base64-encode raw PNG bytes."""
    return base64.standard_b64encode(image_bytes).decode("ascii")


@dataclass
class StepDescriptor:
    """One recorded test step with its visual descriptor."""

    step_index: int
    node_id: str
    url: str
    action_step: ActionStep
    plan_goal: str
    descriptor: Dict[str, Any]
    screenshot_path: str = ""


@dataclass
class StepDescriptorCollector:
    """Accumulates descriptors across a full task run."""

    steps: List[StepDescriptor] = field(default_factory=list)

    def add(self, desc: StepDescriptor) -> None:
        """Append a step descriptor."""
        self.steps.append(desc)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialise all steps for JSON output."""
        out: List[Dict[str, Any]] = []
        for s in self.steps:
            entry = dict(s.descriptor)
            entry["_meta"] = {
                "step_index": s.step_index,
                "node_id": s.node_id,
                "url": s.url,
                "role": s.action_step.role,
                "name": s.action_step.name,
                "plan_goal": s.plan_goal,
                "screenshot": s.screenshot_path,
            }
            out.append(entry)
        return out

    def save(self, path: Path) -> None:
        """Write all descriptors to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_list(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"[DESCRIPTOR] Saved {len(self.steps)} test step "
            f"descriptors to {path}"
        )


class DescriptorAgent:
    """Generates visual element descriptors using Claude vision.

    Takes a pre-action screenshot + ActionStep metadata and asks
    Claude to produce a rich descriptor identifying the element
    by its visual appearance and spatial context.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=60.0,
        )
        self._model = model

    async def generate(
        self,
        *,
        screenshot_bytes: bytes,
        step: ActionStep,
        plan_goal: str,
        url: str,
    ) -> Dict[str, Any]:
        """Generate a descriptor for one action step.

        Args:
            screenshot_bytes: PNG image of the page before action.
            step: The ActionStep that was executed.
            plan_goal: Human-readable goal of the plan.
            url: Page URL at the time of the action.

        Returns:
            Parsed JSON descriptor dict.
        """
        img_b64 = _encode_image(screenshot_bytes)
        user_text = _build_user_prompt(
            step=step, plan_goal=plan_goal, url=url,
        )

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=DESCRIPTOR_SYSTEM_PROMPT,
            messages=[
                {
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
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                }
            ],
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
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = {
                "action": step.action.value,
                "instruction": _instruction_from_metadata(
                    step=step, plan_goal=plan_goal,
                ),
                "element_descriptor": {
                    "purpose": plan_goal,
                    "description": raw[:200],
                    "spatial_anchors": [],
                    "visual_hints": "",
                    "confidence": 0.5,
                },
                "_raw": raw,
            }

        return reconcile_descriptor_with_metadata(
            obj, step=step, plan_goal=plan_goal,
        )

    async def generate_scroll(
        self,
        *,
        screenshot_before_bytes: bytes,
        screenshot_after_bytes: bytes,
        direction: str,
        plan_goal: str,
        url: str,
    ) -> Dict[str, Any]:
        """Generate descriptors for an auto-scroll before a click."""
        before_b64 = _encode_image(screenshot_before_bytes)
        after_b64 = _encode_image(screenshot_after_bytes)
        user_text = _build_scroll_prompt(
            plan_goal=plan_goal,
            direction=direction,
            url=url,
        )

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=SCROLL_DESCRIPTOR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": before_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": after_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
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
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = {
                "instruction": (
                    f"Scroll {direction} to bring target into view "
                    f"for: {plan_goal}"
                ),
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

        if not obj.get("instruction"):
            obj["instruction"] = (
                f"Scroll {direction} to bring target into view "
                f"for: {plan_goal}"
            )
        return obj


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


def format_for_vision_recorder(
    descriptor: Dict[str, Any],
    *,
    screenshot_b64: str,
    url: str,
    action_step: ActionStep,
    screen_width: int = 1920,
    screen_height: int = 1080,
    x: int = 0,
    y: int = 0,
) -> Dict[str, Any]:
    """Convert a WebTactix descriptor to Vision Recorder step shape."""
    wt_action = (
        action_step.action.value
        if isinstance(action_step.action, ActionType)
        else str(descriptor.get("action", "click"))
    )
    vr_action = _map_action_to_vr(wt_action)

    instruction = descriptor.get("instruction", "")
    element_desc = descriptor.get("element_descriptor", {})

    vr_descriptor = {
        "action": vr_action,
        "instruction": instruction,
        "element_descriptor": element_desc,
    }
    if vr_action == "type" and action_step.text:
        vr_descriptor["text"] = action_step.text
    elif action_step.text:
        vr_descriptor["value"] = action_step.text

    step: Dict[str, Any] = {
        "id": int(time.time() * 1000),
        "action": vr_action,
        "x": x,
        "y": y,
        "screenWidth": screen_width,
        "screenHeight": screen_height,
        "screenshot": screenshot_b64,
        "zoomedScreenshot": screenshot_b64,
        "descriptor": vr_descriptor,
        "source": "agent",
        "url": url,
    }

    if vr_action == "type" and action_step.text:
        step["value"] = action_step.text
    elif vr_action == "presskey":
        step["value"] = "Enter"

    return step


def format_scroll_for_vision_recorder(
    descriptor: Dict[str, Any],
    *,
    scroll_direction: str,
    container_x: int,
    container_y: int,
    target_x: int,
    target_y: int,
    container_screenshot_b64: str,
    target_screenshot_b64: str,
    url: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> Dict[str, Any]:
    """Convert a scroll descriptor to Vision Recorder step shape."""
    instruction = descriptor.get(
        "instruction",
        f"Scroll {scroll_direction}",
    )
    container_el = descriptor.get("scroll_container_element", {})
    target_el = descriptor.get("target_element", {})

    vr_descriptor = {
        "action": "scroll",
        "instruction": instruction,
        "scroll_direction": scroll_direction,
        "scroll_container_element": container_el,
        "target_element": target_el,
    }

    return {
        "id": int(time.time() * 1000),
        "action": "scroll",
        "scrollDirection": scroll_direction,
        "x": container_x,
        "y": container_y,
        "screenWidth": screen_width,
        "screenHeight": screen_height,
        "scrollContainer": {
            "x": container_x,
            "y": container_y,
            "screenshot": container_screenshot_b64,
            "zoomedScreenshot": container_screenshot_b64,
        },
        "targetElement": {
            "x": target_x,
            "y": target_y,
            "screenshot": target_screenshot_b64,
            "zoomedScreenshot": target_screenshot_b64,
        },
        "descriptor": vr_descriptor,
        "source": "agent",
        "url": url,
    }
