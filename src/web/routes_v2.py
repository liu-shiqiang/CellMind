"""API v2 routes that expose the new LangGraph agent with streaming."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.utils.langgraph_stream import run_agent_stream

from .deps import AgentRunRequest

router = APIRouter()


class _RunStreamState:
    def __init__(self, run_id: str, thread_id: str) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.done = asyncio.Event()
        self.task: Optional[asyncio.Task[None]] = None
        self.error_info: Optional[Any] = None
        self.final_message: Optional[Any] = None


_run_states: Dict[str, _RunStreamState] = {}
_run_states_lock = asyncio.Lock()


def _format_sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _execute_run(
    state: _RunStreamState,
    objective: str,
    input_files: Optional[List[str]],
) -> None:
    async def handler(event: Dict[str, Any]) -> None:
        await state.queue.put(event)

    try:
        final_message, error_info = await run_agent_stream(
            objective=objective,
            input_files=input_files,
            thread_id=state.thread_id,
            run_id=state.run_id,
            event_handler=handler,
        )
        state.final_message = final_message
        state.error_info = error_info
    except Exception as exc:  # pragma: no cover - defensive
        error_event = {
            "type": "error",
            "run_id": state.run_id,
            "thread_id": state.thread_id,
            "node": "runtime",
            "payload": {"detail": str(exc)},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        await state.queue.put(error_event)
        state.error_info = {"detail": str(exc)}
        end_event = {
            "type": "end",
            "run_id": state.run_id,
            "thread_id": state.thread_id,
            "node": "runtime",
            "payload": {
                "error": state.error_info,
                "execution_status": "failed",
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        await state.queue.put(end_event)
    finally:
        state.done.set()


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> Dict[str, str]:
    objective = request.objective
    input_files = request.normalized_files()

    run_id = str(uuid4())
    thread_id = request.thread_id or run_id
    state = _RunStreamState(run_id, thread_id)

    async with _run_states_lock:
        _run_states[run_id] = state

    state.task = asyncio.create_task(_execute_run(state, objective, input_files))

    return {"run_id": run_id, "thread_id": thread_id}


@router.get("/stream/{run_id}")
async def stream_events(run_id: str) -> StreamingResponse:
    async with _run_states_lock:
        state = _run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if state.done.is_set() and state.queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(state.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                yield _format_sse(event)
        finally:
            async with _run_states_lock:
                _run_states.pop(run_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
