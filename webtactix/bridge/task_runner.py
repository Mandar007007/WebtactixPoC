"""Runs WebTactix agent attached to Vision Recorder Chrome.

Steps are buffered during execution and only the winning-path
steps are emitted (with descriptors) after the run completes.
This prevents duplicate / exploratory-branch steps from reaching
Vision Recorder.
"""
from __future__ import annotations

from webtactix.bridge.stdio_utf8 import configure_utf8_stdio

configure_utf8_stdio()

import asyncio
import base64
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from webtactix.agents.constraint_agent import ConstraintAgent
from webtactix.agents.data_agent import DataExtractionAgent
from webtactix.agents.decision_agent import DecisionAgent
from webtactix.agents.descriptor_agent import (
    format_for_vision_recorder,
    format_scroll_for_vision_recorder,
)
from webtactix.agents.planner_agent import PlannerAgent
from webtactix.agents.vr_descriptor_client import VRDescriptorClient
from webtactix.bridge.schemas import (
    RunState,
    done_event,
    error_event,
    step_event,
)
from webtactix.bridge.step_stream import StepStreamHub
from webtactix.browser.auto_login import auto_login_if_needed
from webtactix.browser.playwright_session import (
    PlaywrightConfig,
    PlaywrightSession,
)
from webtactix.core.schemas import Edge, NodeId, TaskSpec
from webtactix.core.semantic_tree import SemanticTree
from webtactix.llm.anthropic_client import _load_api_key_from_xlm_core
from webtactix.llm.presets import preset_claude
from webtactix.runner.experiment_runner import (
    ExperimentRunner,
    RunnerConfig,
)
from webtactix.runner.recorder import Recorder
from webtactix.workflows.execute import Executor


def _action_signature(edge: Edge) -> str:
    """Compact signature for dedup: action(role name, text)."""
    parts: list[str] = []
    for s in edge.plan.steps:
        act = (
            s.action.value
            if hasattr(s.action, "value")
            else str(s.action)
        )
        txt = f", {s.text}" if s.text else ""
        parts.append(f"{act}({s.role} {s.name}{txt})")
    return " -> ".join(parts)


def _get_winning_path_edges(
    tree: SemanticTree,
    final_node_id: NodeId,
) -> List[Tuple[NodeId, NodeId, Edge]]:
    """Walk root -> final_node and return ``(src, dst, edge)``.

    Consecutive edges with identical action signatures are
    deduplicated (back-to-back repeats with no state change).
    """
    path = tree.path_to_root(final_node_id)
    if len(path) < 2:
        return []

    edge_map: Dict[Tuple[NodeId, NodeId], Edge] = {}
    for e in tree.edges:
        edge_map[(e.src, e.dst)] = e

    raw_edges: List[Tuple[NodeId, NodeId, Edge]] = []
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        if src == "virtual":
            continue
        edge = edge_map.get((src, dst))
        if edge is None:
            continue
        raw_edges.append((src, dst, edge))

    if not raw_edges:
        return []

    deduped: List[Tuple[NodeId, NodeId, Edge]] = []
    prev_sig: str = ""
    for src, dst, edge in raw_edges:
        sig = _action_signature(edge)
        if sig == prev_sig:
            print(
                f"[BRIDGE] Skipping duplicate edge "
                f"{src}->{dst}: {sig}"
            )
            continue
        deduped.append((src, dst, edge))
        prev_sig = sig

    return deduped


def _find_deepest_leaf(
    tree: SemanticTree,
) -> Optional[NodeId]:
    """Return the leaf node with the longest path from root.

    Used as a fallback ``final_node_id`` when the run crashes
    before producing a proper result.
    """
    best_leaf: Optional[NodeId] = None
    best_depth = 0
    for nid in tree.nodes:
        children = tree.children_map.get(nid)
        if children:
            continue
        path = tree.path_to_root(nid)
        if len(path) > best_depth:
            best_depth = len(path)
            best_leaf = nid
    return best_leaf


async def _generate_and_emit_scroll(
    *,
    evt: Dict[str, Any],
    vr_client: VRDescriptorClient,
    emit_fn,
) -> None:
    """Generate a scroll descriptor and emit a VR step."""
    scroll = evt["scroll"]
    plan = evt["plan"]
    url = evt["url"]

    before_b64 = base64.standard_b64encode(
        scroll.screenshot_before_bytes,
    ).decode("ascii")
    after_b64 = base64.standard_b64encode(
        scroll.screenshot_after_bytes,
    ).decode("ascii")

    try:
        desc = await vr_client.generate_scroll(
            screenshot_before_bytes=scroll.screenshot_before_bytes,
            screenshot_after_bytes=scroll.screenshot_after_bytes,
            direction=scroll.direction,
            plan_goal=plan.goal,
            url=url,
            viewport_width=scroll.viewport_width,
            viewport_height=scroll.viewport_height,
            container_x=scroll.container_x,
            container_y=scroll.container_y,
            target_x=scroll.target_x,
            target_y=scroll.target_y,
        )
    except Exception as exc:
        print(f"[BRIDGE] scroll descriptor error: {exc}")
        desc = {
            "instruction": (
                f"Scroll {scroll.direction} to reach target "
                f"for: {plan.goal}"
            ),
            "scroll_container_element": {},
            "target_element": {},
        }

    vr_step = format_scroll_for_vision_recorder(
        desc,
        scroll_direction=scroll.direction,
        container_x=scroll.container_x,
        container_y=scroll.container_y,
        target_x=scroll.target_x,
        target_y=scroll.target_y,
        container_screenshot_b64=before_b64,
        target_screenshot_b64=after_b64,
        url=url,
        screen_width=scroll.viewport_width,
        screen_height=scroll.viewport_height,
    )
    await emit_fn(vr_step)


async def _generate_and_emit_step(
    *,
    evt: Dict[str, Any],
    vr_client: VRDescriptorClient,
    emit_fn,
) -> None:
    """Generate a step descriptor and emit a VR step."""
    s = evt["step"]
    plan = evt["plan"]
    url = evt["url"]
    screenshot_bytes = evt["screenshot_bytes"]
    click_x = int(evt.get("click_x", 0) or 0)
    click_y = int(evt.get("click_y", 0) or 0)
    screen_width = int(evt.get("screen_width", 0) or 1920)
    screen_height = int(evt.get("screen_height", 0) or 1080)

    screenshot_b64 = ""
    if screenshot_bytes:
        screenshot_b64 = base64.standard_b64encode(
            screenshot_bytes,
        ).decode("ascii")

    if screenshot_bytes:
        try:
            desc = await vr_client.generate(
                screenshot_bytes=screenshot_bytes,
                step=s,
                plan_goal=plan.goal,
                url=url,
                x=click_x,
                y=click_y,
                screen_width=screen_width,
                screen_height=screen_height,
            )
        except Exception as exc:
            print(f"[BRIDGE] descriptor error: {exc}")
            desc = {
                "action": s.action.value,
                "instruction": (
                    f"{s.action.value} {s.role} '{s.name}'"
                ),
                "element_descriptor": {},
            }
    else:
        desc = {
            "action": s.action.value,
            "instruction": (
                f"{s.action.value} {s.role} '{s.name}'"
            ),
            "element_descriptor": {},
        }

    vr_step = format_for_vision_recorder(
        desc,
        screenshot_b64=screenshot_b64,
        url=url,
        action_step=s,
        x=click_x,
        y=click_y,
        screen_width=screen_width,
        screen_height=screen_height,
    )
    await emit_fn(vr_step)


async def run_goal_task(
    *,
    state: RunState,
    hub: StepStreamHub,
    start_url: str = "",
) -> None:
    """Execute a goal-driven task over CDP and stream VR steps.

    Steps are buffered during the agent's exploration.  After the
    run finishes, only the winning-path steps are emitted with
    AI-generated descriptors — exploratory dead-end branches are
    discarded so that Vision Recorder receives a clean sequence.
    """
    sess: Optional[PlaywrightSession] = None
    vr_client: Optional[VRDescriptorClient] = None
    try:
        print("[BRIDGE] run_goal_task starting...", flush=True)
        sess = PlaywrightSession(
            PlaywrightConfig(
                headless=False,
                cdp_url=state.cdp_url,
                single_tab=True,
                page_url=state.page_url or None,
            ),
        )
        tree = SemanticTree()
        runner_cfg = RunnerConfig(max_parallel=1)
        print("[BRIDGE] Loading LLM preset...", flush=True)
        llm = preset_claude()
        print("[BRIDGE] LLM ready", flush=True)

        goal = state.goal
        page_url = state.page_url or "about:blank"
        print(
            f"[BRIDGE] Connecting Playwright CDP: {state.cdp_url}",
            flush=True,
        )
        try:
            await asyncio.wait_for(sess.start(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Playwright CDP connection timed out after 30s"
            )
        print("[BRIDGE] Playwright connected", flush=True)
        page = await sess.get_attached_page()
        print(f"[BRIDGE] Attached page: {page.url}", flush=True)
        page, goal = await auto_login_if_needed(
            page, goal=goal,
        )
        if goal != state.goal:
            page_url = page.url

        task = TaskSpec(
            dataset="vision_recorder",
            task_id=0,
            intent=goal,
            start_url=page_url,
            require_login=False,
        )

        rec = Recorder(
            base_dir=Path("record"),
            task=task,
            model_name=runner_cfg.llm_type,
        )
        meta: Dict[str, Any] = asdict(runner_cfg)
        meta["dataset"] = task.dataset
        meta["cdp_url"] = state.cdp_url
        rec.write_meta(meta=meta)

        print("[BRIDGE] Running constraint agent...", flush=True)
        cons_agent = ConstraintAgent(llm=llm, task=task)
        constraints = await cons_agent.run()
        print("[BRIDGE] Constraints ready", flush=True)
        rec.write_task_info(task, constraints=constraints)

        vr_client = VRDescriptorClient(
            fallback_api_key=_load_api_key_from_xlm_core(),
        )
        step_counter = {"n": 0}

        event_buffer: Dict[int, List[Dict[str, Any]]] = {}

        async def _emit_vr_step(vr_step: Dict[str, Any]) -> None:
            step_counter["n"] += 1
            state.steps_count = step_counter["n"]
            action = vr_step.get("action", "?")
            instruction = (
                vr_step.get("descriptor", {}).get("instruction")
                or vr_step.get("instruction", "?")
            )
            print(
                f"[BRIDGE] emitting step {step_counter['n']}: "
                f"[{action}] {instruction}"
            )
            await hub.emit(state.run_id, step_event(vr_step))

        async def on_auto_scroll(event: Dict[str, Any]) -> None:
            """Buffer scroll events; descriptors generated post-run."""
            if state.cancel_requested:
                return
            plan = event["plan"]
            event_buffer.setdefault(id(plan), []).append({
                "kind": "scroll",
                "scroll": event["scroll"],
                "plan": plan,
                "url": event.get("url", ""),
            })

        async def on_step_after(event: Dict[str, Any]) -> None:
            """Buffer step events; descriptors generated post-run."""
            if state.cancel_requested:
                return
            plan = event["plan"]
            event_buffer.setdefault(id(plan), []).append({
                "kind": "step",
                "step": event["step"],
                "plan": plan,
                "url": event.get("url", ""),
                "screenshot_bytes": event.get(
                    "screenshot_bytes", b"",
                ),
                "click_x": event.get("click_x", 0),
                "click_y": event.get("click_y", 0),
                "screen_width": event.get("screen_width", 0),
                "screen_height": event.get("screen_height", 0),
            })

        extractor = DataExtractionAgent(
            task=task, llm=llm, tree=tree, sess=sess, rec=rec,
        )
        executor = Executor(
            sess=sess,
            tree=tree,
            rec=rec,
            data_agent=extractor,
            evaluator=None,
            max_parallel=1,
            on_step_after=on_step_after,
            on_auto_scroll=on_auto_scroll,
        )
        planner = PlannerAgent(
            llm=llm,
            q=task.intent,
            constraints=constraints,
            tree=tree,
            rec=rec,
        )
        decision = DecisionAgent(
            llm=llm,
            q=task.intent,
            constraints=constraints,
            executor=executor,
            tree=tree,
            sess=sess,
            rec=rec,
        )
        runner = ExperimentRunner(
            sess=sess,
            tree=tree,
            planner=planner,
            decision=decision,
            task=task,
            rec=rec,
            cfg=runner_cfg,
        )

        async def _emit_winning_path(
            final_node_id: NodeId,
        ) -> None:
            """Walk winning path, emit steps with descriptors."""
            edges = _get_winning_path_edges(
                tree, final_node_id,
            )
            total_buffered = sum(
                len(v) for v in event_buffer.values()
            )
            print(
                f"[BRIDGE] Winning path: {len(edges)} edges, "
                f"{total_buffered} total buffered events"
            )
            for src, dst, edge in edges:
                plan_key = id(edge.plan)
                buffered = event_buffer.get(plan_key, [])
                if not buffered:
                    print(
                        f"[BRIDGE] No buffered events for "
                        f"edge {src}->{dst}, skipping"
                    )
                    continue
                for evt in buffered:
                    if state.cancel_requested:
                        return
                    if evt["kind"] == "scroll":
                        await _generate_and_emit_scroll(
                            evt=evt,
                            vr_client=vr_client,
                            emit_fn=_emit_vr_step,
                        )
                    elif evt["kind"] == "step":
                        await _generate_and_emit_step(
                            evt=evt,
                            vr_client=vr_client,
                            emit_fn=_emit_vr_step,
                        )

        state.status = "running"
        print("[BRIDGE] Starting agent loop...")

        if state.cancel_requested:
            state.status = "cancelled"
            return

        root = await runner._bootstrap_root(page)
        result = await runner._run_loop(root)

        if result.final_node_id:
            await _emit_winning_path(result.final_node_id)
        else:
            print(
                "[BRIDGE] No final_node_id; "
                "no winning-path steps to emit"
            )

        state.status = result.status
        await hub.emit(
            state.run_id,
            done_event(
                status=result.status,
                steps_count=state.steps_count,
                answer=result.answer,
            ),
        )
    except Exception as exc:
        import traceback
        import sys
        traceback.print_exc()
        print(f"[BRIDGE] run error: {exc}", flush=True)
        sys.stderr.flush()
        state.status = "error"
        state.error = str(exc)
        await hub.emit(
            state.run_id, error_event(str(exc)),
        )
    finally:
        if sess is not None:
            await sess.close()
        if vr_client is not None:
            await vr_client.close()
