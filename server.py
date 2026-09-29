"""FastAPI server for Webtactix PoC.

Serves the frontend, runs goals, streams steps over WebSocket,
and generates downloadable TPE Word reports.

    uv run uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load WebVoyager root .env first (has ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
# LLM_PROVIDER=foundry, and all Azure env vars), then any local overrides.
_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env)
    print(f"[SERVER] Loaded root .env: {_root_env}")
load_dotenv()  # local .env can override

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from report import RunSummary, StepRecord, generate_tpe_report
from runner import RunConfig, RunOutput, run_goal

app = FastAPI(title="Webtactix PoC", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# ---------- In-memory state ----------

@dataclass
class RunState:
    run_id: str
    goal: str
    start_url: str
    status: str = "pending"
    steps: list[dict] = field(default_factory=list)
    output: RunOutput | None = None
    report_path: str = ""
    error: str = ""
    ws_clients: list[WebSocket] = field(default_factory=list)


runs: dict[str, RunState] = {}
tasks: dict[str, asyncio.Task] = {}


# ---------- Request/Response models ----------

class RunRequest(BaseModel):
    goal: str
    url: str
    headless: bool = False


class RunResponse(BaseModel):
    run_id: str
    status: str


# ---------- Endpoints ----------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/run", response_model=RunResponse)
async def start_run(body: RunRequest):
    """Start a new goal execution run."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    state = RunState(
        run_id=run_id,
        goal=body.goal,
        start_url=body.url,
        status="pending",
    )
    runs[run_id] = state

    task = asyncio.create_task(_execute_run(state, body.headless))
    tasks[run_id] = task

    return RunResponse(run_id=run_id, status="pending")


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """Get the current status of a run."""
    state = runs.get(run_id)
    if state is None:
        return {"error": "run not found"}
    return {
        "run_id": state.run_id,
        "status": state.status,
        "steps_count": len(state.steps),
        "report_path": state.report_path,
        "error": state.error,
    }


@app.get("/api/runs/{run_id}/report")
async def download_report(run_id: str):
    """Download the generated TPE Word document."""
    state = runs.get(run_id)
    if state is None:
        return {"error": "run not found"}
    if not state.report_path:
        return {"error": "report not yet generated"}

    path = Path(state.report_path)
    if not path.exists():
        return {"error": "report file not found"}

    return FileResponse(
        path=str(path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=path.name,
    )


@app.websocket("/api/ws/{run_id}")
async def ws_stream(ws: WebSocket, run_id: str):
    """Stream step events for a run in real-time."""
    state = runs.get(run_id)
    if state is None:
        await ws.close(code=4004)
        return

    await ws.accept()
    state.ws_clients.append(ws)

    # Replay any steps already collected
    for step_data in state.steps:
        try:
            await ws.send_json({"type": "step", "data": step_data})
        except Exception:
            break

    # If already done, send completion
    if state.status in ("complete", "error", "finish"):
        try:
            await ws.send_json({
                "type": "done",
                "status": state.status,
                "steps_count": len(state.steps),
                "report_path": state.report_path,
                "error": state.error,
            })
        except Exception:
            pass

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


# ---------- Run execution ----------

async def _broadcast(state: RunState, event: dict[str, Any]) -> None:
    """Send an event to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in state.ws_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.remove(ws)


async def _execute_run(state: RunState, headless: bool) -> None:
    """Run the goal and generate the TPE report."""
    state.status = "running"
    await _broadcast(state, {"type": "status", "status": "running"})

    async def on_step(step_data: dict[str, Any]) -> None:
        state.steps.append(step_data)
        await _broadcast(state, {"type": "step", "data": step_data})

    async def on_progress(progress: dict[str, Any]) -> None:
        await _broadcast(state, {"type": "progress", "data": progress})

    try:
        config = RunConfig(
            goal=state.goal,
            start_url=state.start_url,
            headless=headless,
            max_rounds=15,
        )

        output = await run_goal(config, on_step=on_step, on_progress=on_progress)
        state.output = output
        state.status = output.status

        # Generate the TPE report
        if output.steps:
            report_name = f"TPE_{state.run_id}_{int(time.time())}.docx"
            report_path = REPORTS_DIR / report_name

            summary = RunSummary(
                goal=state.goal,
                start_url=state.start_url,
                total_steps=len(output.steps),
                status=output.status,
                started_at=output.started_at,
                finished_at=output.finished_at,
                duration_s=output.duration_s,
            )
            generate_tpe_report(summary, output.steps, report_path)
            state.report_path = str(report_path)
            print(f"[SERVER] Report generated: {report_path}")

        await _broadcast(state, {
            "type": "done",
            "status": state.status,
            "steps_count": len(state.steps),
            "report_path": state.report_path,
            "answer": output.answer,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        state.status = "error"
        state.error = str(exc)
        await _broadcast(state, {
            "type": "error",
            "message": str(exc),
        })


# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
