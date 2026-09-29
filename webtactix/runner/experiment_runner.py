# webtactix/runner/experiment_runner
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import asyncio

from playwright.async_api import Page

from webtactix.agents.decision_agent import DecisionAgent, DecisionResult
from webtactix.agents.planner_agent import PlannerAgent, PlanningResult
from webtactix.browser.playwright_session import PlaywrightSession, wait_for_page_stable
from webtactix.core.schemas import NodeId, Plan
from webtactix.core.semantic_tree import SemanticTree
from webtactix.preprocess.observation_encoder import ObservationEncoder, ObservationEncoderConfig, EncodedObservation
from webtactix.core.priority_queue import PriorityQueue
from webtactix.runner.recorder import Recorder
from webtactix.core.schemas import TaskSpec
from webtactix.datasets.webarena_evaluator import EvalResult

@dataclass(frozen=True)
class RunnerConfig:
    max_rounds: int = 60
    max_parallel: int = 4
    table_max_rows: int = 10
    llm_type: str = "claude"

@dataclass
class RunnerResult:
    status: str  # finish | stopped | error
    answer: str = ""
    eval_result: EvalResult = None
    final_node_id: Optional[NodeId] = None

class ExperimentRunner:
    def __init__(
        self,
        *,
        sess: PlaywrightSession,
        tree: SemanticTree,
        planner: PlannerAgent,
        decision: DecisionAgent,
        task: TaskSpec,
        rec: Recorder,
        cfg: Optional[RunnerConfig] = None,
    ) -> None:
        self.sess = sess
        self.tree = tree
        self.planner = planner
        self.decision = decision
        self.cfg = cfg or RunnerConfig()
        self.q = PriorityQueue()
        self.task = task
        self.rec = rec
        self.encoder = ObservationEncoder(
            ObservationEncoderConfig(table_max_rows=self.cfg.table_max_rows)
        )

    def _print_planning_result(self, node_id: NodeId, r: PlanningResult) -> None:
        url = self.tree.get_url(node_id)

        print(f"\n[PLAN] node={node_id} url={url}")
        if r.page_summary is None:
            print(f"[PLAN] page_summary not exist")
            return
        if r.page_summary.strip():
            print(f"[PLAN] page_summary={r.page_summary.strip()}")
        if r.progress_analysis.strip():
            print(f"[PLAN] progress_analysis={r.progress_analysis.strip()}")

        for i, p in enumerate(r.plans, 1):
            print(f"[PLAN] plan#{i} name={p.name}")
            print(f"[PLAN] plan#{i} goal={p.goal}")
            if getattr(p, "steps", None):
                sig = self.tree._sig_from_plan(p)
                print(f"[PLAN] plan#{i} action_sig={sig}")

    async def _plan_frontier(self, node_ids: Sequence[NodeId], *, _round: int) -> Dict[NodeId, PlanningResult]:
        async def _one(nid: NodeId, _round: int) -> tuple[NodeId, PlanningResult]:
            r = await self.planner.run(nid, _round)
            return nid, r
        pairs = await asyncio.gather(*[_one(nid, _round) for nid in node_ids])
        return dict(pairs)

    async def _save_page_artifacts(self, *, page: Any, node_id: NodeId, enc: EncodedObservation) -> None:
        self.rec.save_actree(node_id=str(node_id), actree_text=enc.actree_yaml)
        try:
            await page.screenshot(path=self.rec.snapshot_path(node_id=str(node_id)))
        except:
            pass

    async def _bootstrap_root(self, page: Page) -> NodeId:
        """Capture snapshot from page and create semantic tree root."""
        single_tab = getattr(self.sess.cfg, "single_tab", False)
        await wait_for_page_stable(page, single_tab=single_tab)
        cur = await self.sess.get_snapshot(page)
        enc = self.encoder.encode(cur_snapshot=cur)
        if not (enc.actree_yaml or "").strip():
            raise RuntimeError(
                f"Could not read page content at {page.url!r}. "
                "Ensure the app tab is loaded in VR Chrome, then retry."
            )
        root = self.tree.add_root(url=page.url, enc=enc)
        self.rec.start_round(0, frontier=[], f_parent=[])
        await self._save_page_artifacts(page=page, node_id=root, enc=enc)
        return root

    def _inject_repeat_warning(
        self, frontier: List[NodeId],
    ) -> None:
        """Detect consecutive identical actions and warn the planner."""
        repeat_threshold = 3
        for nid in frontier:
            path = self.tree.path_to_root(nid)
            if len(path) < 2:
                continue
            last_sig = self.tree.get_incoming_action_sig(nid)
            if not last_sig:
                continue
            count = 0
            for pid in reversed(path):
                sig = self.tree.get_incoming_action_sig(pid)
                if sig == last_sig:
                    count += 1
                else:
                    break
            if count >= repeat_threshold:
                warning = (
                    f"CRITICAL: The action '{last_sig}' has been "
                    f"repeated {count} times in a row without any "
                    f"progress. You MUST use a COMPLETELY DIFFERENT "
                    f"action now. "
                    f"Tips: 'input' action clears pre-filled text "
                    f"and types new text. Use 'click(link Switch "
                    f"user)' to reset a login form. Use 'goto' to "
                    f"navigate directly to a URL."
                )
                st = self.tree.state.get(nid)
                if st is not None:
                    if not any(
                        "CRITICAL" in r for r in st.reflection
                    ):
                        st.reflection.append(warning)
                print(
                    f"[RUNNER] Repeat warning "
                    f"({count}x): {last_sig}"
                )

    async def _run_loop(self, root: NodeId) -> RunnerResult:
        """Main planner/decision loop after root is bootstrapped."""
        decision_result: DecisionResult
        frontier: List[NodeId] = [root]
        f_parent: NodeId = "virtual"
        for _round in range(self.cfg.max_rounds):
            self.rec.start_round(
                _round + 1, frontier=frontier, f_parent=f_parent,
            )
            if not frontier:
                frontier = copy.deepcopy([f_parent])
            self._inject_repeat_warning(frontier)
            plan_by_node = await self._plan_frontier(
                frontier, _round=_round,
            )
            if plan_by_node is None:
                print('[PLAN] all planners returned None, retrying')
                continue
            plan_by_node = {
                k: v for k, v in plan_by_node.items()
                if v is not None
            }
            if not plan_by_node:
                print('[PLAN] all planners returned None, retrying')
                continue
            for nid, r in plan_by_node.items():
                self._print_planning_result(nid, r)

            decision_result = await self.decision.run(f_parent)
            if decision_result.kind == "finish":
                eval_result: EvalResult = decision_result.eval_result
                self.rec.write_final(
                    status="finish",
                    answer=decision_result.reason,
                    final_node_id=getattr(
                        decision_result, "selected_node_id", None,
                    ),
                    reason="finish",
                    eval_result=eval_result,
                )
                return RunnerResult(
                    status="finish",
                    answer=decision_result.reason,
                    eval_result=eval_result,
                    final_node_id=decision_result.selected_node_id,
                )
            frontier = decision_result.new_child
            f_parent = decision_result.selected_node_id

        self.rec.write_final(
            status="stopped",
            answer="Max steps reached",
            final_node_id=getattr(
                decision_result, "selected_node_id", None,
            ),
        )
        return RunnerResult(
            status="stopped",
            answer="Max steps reached or frontier empty",
            final_node_id=getattr(
                decision_result, "selected_node_id", None,
            ),
        )

    async def run(
        self,
        *,
        start_url: str,
        storage_state: Optional[Path] = None,
        geolocation: Any = None,
    ) -> RunnerResult:
        await self.sess.start(
            storage_state=storage_state, geolocation=geolocation,
        )
        page = await self.sess.new_page()
        await self.sess.goto(page, start_url)
        root = await self._bootstrap_root(page)
        return await self._run_loop(root)

    async def run_over_cdp(
        self,
        *,
        cdp_url: str,
    ) -> RunnerResult:
        """Run agent attached to existing Chrome via CDP."""
        cfg = self.sess.cfg
        if not cfg.cdp_url:
            cfg.cdp_url = cdp_url
            cfg.single_tab = True
        await self.sess.start()
        page = await self.sess.get_attached_page()
        root = await self._bootstrap_root(page)
        return await self._run_loop(root)


