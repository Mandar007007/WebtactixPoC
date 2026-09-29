"""FastAPI bridge server for Vision Recorder."""
from __future__ import annotations

from webtactix.bridge.stdio_utf8 import configure_utf8_stdio

configure_utf8_stdio()

import asyncio
import uuid
from typing import Dict

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from webtactix.bridge.schemas import RunState
from webtactix.bridge.step_stream import StepStreamHub
from webtactix.bridge.task_runner import run_goal_task

app = FastAPI(title="WebTactix Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hub = StepStreamHub()
runs: Dict[str, RunState] = {}
tasks: Dict[str, asyncio.Task] = {}


class RunRequestBody(BaseModel):
    """HTTP body to start an agent run."""

    goal: str
    cdp_url: str = "http://localhost:9222"
    page_url: str = ""


@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/cdp-status")
async def cdp_status(
    cdp_url: str = "http://localhost:9222",
) -> dict:
    """Check if CDP endpoint is reachable."""
    from webtactix.browser.cdp_targets import resolve_preferred_page_url

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{cdp_url}/json/version")
            if resp.status_code == 200:
                data = resp.json()
                page_url = await resolve_preferred_page_url(cdp_url)
                return {
                    "ready": True,
                    "browser": data.get("Browser", ""),
                    "url": cdp_url,
                    "page_url": page_url or "",
                }
    except Exception as exc:
        return {"ready": False, "error": str(exc), "url": cdp_url}
    return {"ready": False, "url": cdp_url}


@app.post("/runs")
async def create_run(body: RunRequestBody) -> dict:
    """Register a run; agent starts when WebSocket client connects."""
    run_id = f"vr_{uuid.uuid4().hex[:8]}"
    state = RunState(
        run_id=run_id,
        goal=body.goal,
        cdp_url=body.cdp_url,
        page_url=body.page_url or "",
    )
    state.status = "pending"
    runs[run_id] = state
    return {"run_id": run_id, "status": "pending"}


async def _start_run_task(run_id: str) -> None:
    """Launch the agent task for a registered run."""
    import sys

    state = runs.get(run_id)
    if state is None or state.status not in {"pending", "waiting"}:
        print(
            f"[APP] _start_run_task skip: "
            f"state={state.status if state else 'None'}",
            flush=True,
        )
        return
    state.status = "running"
    print(f"[APP] Starting run task {run_id}", flush=True)

    async def _run() -> None:
        try:
            await run_goal_task(state=state, hub=hub)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[APP] Task {run_id} crashed: {exc}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            await hub.clear_buffer(run_id)

    task = asyncio.create_task(_run())
    tasks[run_id] = task


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    """Cancel an in-flight run."""
    state = runs.get(run_id)
    if state is None:
        return {"success": False, "error": "run not found"}
    state.cancel_requested = True
    task = tasks.get(run_id)
    if task and not task.done():
        task.cancel()
    return {"success": True, "run_id": run_id}


@app.websocket("/runs/{run_id}/steps")
async def ws_steps(ws: WebSocket, run_id: str) -> None:
    """Stream step events for a run; starts agent on connect."""
    if run_id not in runs:
        await ws.close(code=4004)
        return
    await hub.subscribe(run_id, ws)
    await _start_run_task(run_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(run_id, ws)


def create_app() -> FastAPI:
    """Return the FastAPI app instance."""
    return app
