"""API v2 routes that expose the new LangGraph agent with streaming.

Enhanced to support multiple streaming modes:
- updates: State updates with progress tracking (default)
- messages: LLM token-by-token streaming
- custom: Custom user events
- debug: Maximum verbosity
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
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
    """Format event as Server-Sent Event (SSE)."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _execute_run(
    state: _RunStreamState,
    objective: str,
    input_files: Optional[List[str]],
    stream_mode: str,
) -> None:
    """Execute agent run in background task."""
    async def handler(event: Dict[str, Any]) -> None:
        await state.queue.put(event)

    try:
        final_message, error_info = await run_agent_stream(
            objective=objective,
            input_files=input_files,
            thread_id=state.thread_id,
            run_id=state.run_id,
            event_handler=handler,
            stream_mode=stream_mode,
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
                "message": "❌ Execution failed",
            },
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        await state.queue.put(end_event)
    finally:
        state.done.set()


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> Dict[str, str]:
    """Start a new agent run.

    Returns a run_id that can be used to stream events.
    """
    objective = request.objective
    input_files = request.normalized_files()

    run_id = str(uuid4())
    thread_id = request.thread_id or run_id
    state = _RunStreamState(run_id, thread_id)

    async with _run_states_lock:
        _run_states[run_id] = state

    # Start background task
    state.task = asyncio.create_task(
        _execute_run(state, objective, input_files, stream_mode="updates")
    )

    return {"run_id": run_id, "thread_id": thread_id}


@router.get("/stream/{run_id}")
async def stream_events(
    run_id: str,
    mode: Optional[str] = Query(
        "updates",
        description="Streaming mode: updates, messages, custom, debug",
    ),
) -> StreamingResponse:
    """Stream agent events via Server-Sent Events (SSE).

    Args:
        run_id: The run identifier returned by /run endpoint
        mode: Streaming mode (only works for new runs, not existing ones)

    Returns:
        SSE stream with event data
    """
    async with _run_states_lock:
        state = _run_states.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the agent run."""
        try:
            while True:
                # Check if done and queue is empty
                if state.done.is_set() and state.queue.empty():
                    break

                try:
                    # Wait for events with timeout
                    event = await asyncio.wait_for(state.queue.get(), timeout=1.0)
                    yield _format_sse(event)

                    # Send heartbeat every 10 events for long-running tasks
                    if not state.queue.empty():
                        continue

                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat = {
                        "type": "heartbeat",
                        "run_id": run_id,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    yield _format_sse(heartbeat)

        finally:
            # Cleanup
            async with _run_states_lock:
                _run_states.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/run-and-stream")
async def run_and_stream_agent(
    request: AgentRunRequest,
    stream_mode: Optional[str] = Query(
        "updates",
        description="Streaming mode: updates, messages, custom, debug",
    ),
) -> StreamingResponse:
    """Convenience endpoint to run agent and stream events in one request.

    This is useful for clients that don't want to manage two separate calls.

    Args:
        request: Agent run request
        stream_mode: Streaming mode (updates, messages, custom, debug)

    Returns:
        SSE stream with event data
    """
    objective = request.objective
    input_files = request.normalized_files()

    run_id = str(uuid4())
    thread_id = request.thread_id or run_id

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate events from the agent run."""
        from src.utils.langgraph_stream import LangGraphEventAdapter

        adapter = LangGraphEventAdapter(run_id, thread_id)
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        async def handler(event: Dict[str, Any]) -> None:
            """Handle events from the agent stream."""
            await queue.put(event)

        try:
            # Send start event
            yield _format_sse(
                adapter.build_start_event(objective, input_files or [])
            )

            # Stream events in background task
            async def run_stream():
                try:
                    await run_agent_stream(
                        objective=objective,
                        input_files=input_files,
                        thread_id=thread_id,
                        run_id=run_id,
                        event_handler=handler,
                        stream_mode=stream_mode or "updates",
                    )
                finally:
                    await queue.put(None)  # Signal completion

            # Start background task
            stream_task = asyncio.create_task(run_stream())

            # Yield events as they arrive
            while True:
                event = await queue.get()
                if event is None:  # End signal
                    break
                yield _format_sse(event)

            # Wait for stream to complete
            await stream_task

            # Send end event
            yield _format_sse(adapter.build_end_event(None))

        except Exception as exc:
            # Send error event
            error_event = {
                "type": "error",
                "run_id": run_id,
                "thread_id": thread_id,
                "node": "runtime",
                "payload": {"detail": str(exc)},
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            yield _format_sse(error_event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
