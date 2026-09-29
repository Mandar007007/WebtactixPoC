"""WebSocket step broadcast hub."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Set

from fastapi import WebSocket


class StepStreamHub:
    """Manages WebSocket subscribers per run."""

    def __init__(self) -> None:
        self._subs: Dict[str, Set[WebSocket]] = {}
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self, run_id: str, ws: WebSocket,
    ) -> None:
        """Add a WebSocket subscriber and replay buffered events."""
        await ws.accept()
        async with self._lock:
            self._subs.setdefault(run_id, set()).add(ws)
            buffered = list(self._buffer.get(run_id, []))
        for event in buffered:
            try:
                await ws.send_json(event)
            except Exception:
                break

    async def unsubscribe(
        self, run_id: str, ws: WebSocket,
    ) -> None:
        """Remove a WebSocket subscriber."""
        async with self._lock:
            subs = self._subs.get(run_id)
            if subs and ws in subs:
                subs.discard(ws)
            if subs is not None and not subs:
                del self._subs[run_id]

    async def emit(
        self, run_id: str, event: Dict[str, Any],
    ) -> None:
        """Broadcast an event; buffer if no subscribers yet."""
        async with self._lock:
            subs: List[WebSocket] = list(
                self._subs.get(run_id, set()),
            )
            if not subs:
                self._buffer.setdefault(run_id, []).append(event)
                return
        dead: List[WebSocket] = []
        for ws in subs:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unsubscribe(run_id, ws)

    async def clear_buffer(self, run_id: str) -> None:
        """Clear buffered events after run completes."""
        async with self._lock:
            self._buffer.pop(run_id, None)
