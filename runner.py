"""Standalone runner: launches Playwright, runs the webtactix agent, collects steps.

This is the glue between the server and the webtactix agent. It:
1. Launches a headed Playwright browser
2. Navigates to the user's URL
3. Runs the planner → decision → executor loop
4. Collects every step with screenshots, action descriptions, expected/actual results
5. Returns a list of StepRecord objects for the report
"""
from __future__ import annotations

import asyncio
import base64
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from webtactix.agents.constraint_agent import ConstraintAgent
from webtactix.agents.data_agent import DataExtractionAgent
from webtactix.agents.decision_agent import DecisionAgent
from webtactix.agents.planner_agent import PlannerAgent
from webtactix.browser.auto_login import auto_login_if_needed
from webtactix.browser.playwright_session import PlaywrightConfig, PlaywrightSession
from webtactix.core.schemas import TaskSpec
from webtactix.core.semantic_tree import SemanticTree
from webtactix.llm.anthropic_client import AnthropicClient, AnthropicConfig, _load_api_key_from_xlm_core
from webtactix.runner.experiment_runner import ExperimentRunner, RunnerConfig
from webtactix.runner.recorder import Recorder
from webtactix.workflows.execute import Executor

from report import StepRecord


@dataclass
class CollectedStep:
    """Raw step data collected during the agent run."""

    step_index: int
    action: str
    role: str
    name: str
    text: str
    plan_goal: str
    url: str
    ok: bool
    screenshot_b64: str = ""
    click_x: int = 0
    click_y: int = 0
    screen_width: int = 0
    screen_height: int = 0
    scroll_direction: str = ""


StepCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class RunConfig:
    """Configuration for a single PoC run."""

    goal: str
    start_url: str
    headless: bool = False
    max_rounds: int = 15
    max_parallel: int = 1


@dataclass
class RunOutput:
    """Result of a completed run."""

    steps: list[StepRecord] = field(default_factory=list)
    status: str = "incomplete"
    answer: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0


def _make_llm() -> AnthropicClient:
    """Create the Claude LLM client using Azure Foundry env vars."""
    api_key, base_url = _load_api_key_from_xlm_core()
    cfg = AnthropicConfig(
        api_key=api_key,
        model="claude-sonnet-4-6",
        base_url=base_url,
    )
    return AnthropicClient(cfg)


def _step_to_record(step: CollectedStep, index: int) -> StepRecord:
    """Convert a collected step into a TPE report record."""
    action_desc = step.action
    if step.role and step.name:
        action_desc = f"{step.action} {step.role} '{step.name}'"
    if step.text and step.action in ("input", "select", "goto"):
        action_desc += f" → '{step.text}'"

    expected = step.plan_goal or "Execute planned action"
    actual = "Completed successfully" if step.ok else "Action failed"

    return StepRecord(
        step_number=index,
        action=action_desc,
        expected_result=expected,
        actual_result=actual,
        screenshot_b64=step.screenshot_b64,
        url=step.url,
        status="pass" if step.ok else "fail",
    )


async def run_goal(
    config: RunConfig,
    on_step: StepCallback | None = None,
    on_progress: StepCallback | None = None,
) -> RunOutput:
    """Execute a goal against a URL and collect step records.

    Args:
        config: What to do and where.
        on_step: Called after each step is executed (for live streaming).
        on_progress: Called on round/status changes.

    Returns:
        RunOutput with step records ready for report generation.
    """
    output = RunOutput()
    output.started_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    t0 = time.monotonic()

    sess: PlaywrightSession | None = None
    collected: list[CollectedStep] = []
    step_counter = {"n": 0}

    try:
        print(f"[RUNNER] Starting run: goal={config.goal!r} url={config.start_url!r}")

        sess = PlaywrightSession(
            PlaywrightConfig(
                headless=config.headless,
                viewport={"width": 1280, "height": 720},
            )
        )

        tree = SemanticTree()
        runner_cfg = RunnerConfig(max_parallel=config.max_parallel)

        print("[RUNNER] Loading LLM...")
        llm = _make_llm()
        print("[RUNNER] LLM ready")

        await sess.start()
        page = await sess.new_page()
        await sess.goto(page, config.start_url)
        print(f"[RUNNER] Navigated to {config.start_url}")

        if on_progress:
            await on_progress({"phase": "navigated", "url": config.start_url})

        page, goal = await auto_login_if_needed(page, goal=config.goal)

        task = TaskSpec(
            dataset="webtactix_poc",
            task_id=0,
            intent=goal,
            start_url=page.url,
            require_login=False,
        )

        rec = Recorder(
            base_dir=Path("runs"),
            task=task,
            model_name="claude",
        )
        rec.write_meta({"goal": goal, "start_url": config.start_url})

        print("[RUNNER] Running constraint agent...")
        cons_agent = ConstraintAgent(llm=llm, task=task)
        constraints = await cons_agent.run()
        rec.write_task_info(task, constraints=constraints)

        if on_progress:
            await on_progress({"phase": "planning", "constraints": len(constraints)})

        async def _on_step_after(event: dict[str, Any]) -> None:
            """Collect step data after each action."""
            step_counter["n"] += 1
            s = event["step"]
            plan = event["plan"]

            screenshot_b64 = ""
            shot_bytes = event.get("screenshot_bytes", b"")
            if shot_bytes:
                screenshot_b64 = base64.standard_b64encode(shot_bytes).decode("ascii")

            cs = CollectedStep(
                step_index=step_counter["n"],
                action=s.action.value if hasattr(s.action, "value") else str(s.action),
                role=getattr(s, "role", ""),
                name=getattr(s, "name", ""),
                text=getattr(s, "text", "") or "",
                plan_goal=plan.goal,
                url=event.get("url", ""),
                ok=event.get("ok", True),
                screenshot_b64=screenshot_b64,
                click_x=int(event.get("click_x", 0) or 0),
                click_y=int(event.get("click_y", 0) or 0),
                screen_width=int(event.get("screen_width", 0) or 0),
                screen_height=int(event.get("screen_height", 0) or 0),
            )
            collected.append(cs)

            if on_step:
                await on_step({
                    "step_number": cs.step_index,
                    "action": cs.action,
                    "role": cs.role,
                    "name": cs.name,
                    "text": cs.text,
                    "plan_goal": cs.plan_goal,
                    "url": cs.url,
                    "ok": cs.ok,
                    "screenshot_b64": screenshot_b64[:100] + "..." if screenshot_b64 else "",
                })

        async def _on_auto_scroll(event: dict[str, Any]) -> None:
            """Collect scroll events."""
            step_counter["n"] += 1
            scroll = event["scroll"]
            plan = event["plan"]

            screenshot_b64 = ""
            if scroll.screenshot_after_bytes:
                screenshot_b64 = base64.standard_b64encode(
                    scroll.screenshot_after_bytes
                ).decode("ascii")

            cs = CollectedStep(
                step_index=step_counter["n"],
                action="scroll",
                role="",
                name=f"Scroll {scroll.direction}",
                text=f"Scroll {scroll.direction} to find target",
                plan_goal=plan.goal,
                url=event.get("url", ""),
                ok=True,
                screenshot_b64=screenshot_b64,
                scroll_direction=scroll.direction,
            )
            collected.append(cs)

            if on_step:
                await on_step({
                    "step_number": cs.step_index,
                    "action": "scroll",
                    "name": cs.name,
                    "plan_goal": cs.plan_goal,
                    "url": cs.url,
                    "ok": True,
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
            on_step_after=_on_step_after,
            on_auto_scroll=_on_auto_scroll,
        )
        planner = PlannerAgent(
            llm=llm, q=task.intent, constraints=constraints, tree=tree, rec=rec,
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

        if on_progress:
            await on_progress({"phase": "executing"})

        print("[RUNNER] Starting agent loop...")
        root = await runner._bootstrap_root(page)
        result = await runner._run_loop(root)

        output.status = result.status
        output.answer = result.answer
        print(f"[RUNNER] Agent finished: status={result.status} answer={result.answer}")

    except Exception as exc:
        traceback.print_exc()
        output.status = "error"
        output.error = f"{type(exc).__name__}: {exc}"
        print(f"[RUNNER] Error: {output.error}")

    finally:
        if sess is not None:
            try:
                await sess.close()
            except Exception:
                pass

    output.finished_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    output.duration_s = time.monotonic() - t0

    # Convert collected steps to report records
    output.steps = [_step_to_record(s, s.step_index) for s in collected]
    output.steps.sort(key=lambda s: s.step_number)

    if on_progress:
        await on_progress({"phase": "complete", "total_steps": len(output.steps)})

    return output
