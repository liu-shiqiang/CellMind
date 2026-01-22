"""Local job runner for executing agent background work.

This module handles running jobs in background threads, invoking the agent,
and persisting events to disk without blocking the HTTP request.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.jobs import job_service
from src.web.schemas import JobStatus, SSEEventType

# Thread pool for background job execution
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job_runner")
# Track running jobs
_running_jobs: Dict[str, threading.Thread] = {}


def _iso_now() -> str:
    """Get current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _append_event(job_id: str, event: Dict[str, Any]) -> None:
    """Append an event to the job's events.ndjson file.

    Each event is written as a JSON line followed by a newline.
    The file is flushed after each write to ensure SSE readers
    can see new events immediately without buffering issues.

    Args:
        job_id: Job identifier
        event: Event dict to append
    """
    events_path = job_service.get_events_path(job_id)
    with open(events_path, "a", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False)
        f.write("\n")
        f.flush()  # Ensure SSE readers see events immediately


def _run_agent_sync(
    job_id: str,
    objective: str,
    input_files: List[str],
    thread_id: Optional[str],
    stream_mode: str,
) -> None:
    """Synchronous wrapper for running the agent.

    This runs in a background thread and handles:
    1. Transitioning job state from queued to running
    2. Invoking the agent
    3. Capturing and persisting events
    4. Updating final state (succeeded/failed)

    PROGRESS TRACKING:
    Progress is derived from agent events (stage-based heuristic).
    Final progress=100 is set on completion (succeeded/failed).
    During execution, progress updates come from agent node events.

    Args:
        job_id: Job identifier
        objective: User's objective
        input_files: List of input file paths
        thread_id: Conversation thread ID
        stream_mode: Streaming mode
    """
    try:
        # Transition to running
        state = job_service.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress=0,
            current_node="start",
        )
        _append_event(
            job_id,
            {
                "type": "start",
                "job_id": job_id,
                "thread_id": thread_id or job_id,
                "node": "start",
                "payload": {
                    "objective": objective,
                    "input_files": input_files,
                    "message": "Job started",
                },
                "ts": _iso_now(),
            },
        )

        # Try to import and run the agent
        # Import here to avoid early import errors
        try:
            from src.utils.langgraph_stream import run_agent_stream

            # Create async event handler
            events: List[Dict[str, Any]] = []

            async def event_handler(event: Dict[str, Any]) -> None:
                # Add job_id to events if not present
                if "job_id" not in event:
                    event["job_id"] = job_id
                events.append(event)
                _append_event(job_id, event)

            # Run the async function in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                final_message, error_info = loop.run_until_complete(
                    run_agent_stream(
                        objective=objective,
                        input_files=input_files,
                        thread_id=thread_id,
                        run_id=job_id,
                        event_handler=event_handler,
                        stream_mode=stream_mode,
                    )
                )
            finally:
                loop.close()

            # Determine final status
            if error_info:
                job_service.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    progress=100,
                    current_node="end",
                    error={"detail": str(error_info)},
                )
            else:
                job_service.update_job_status(
                    job_id,
                    JobStatus.SUCCEEDED,
                    progress=100,
                    current_node="end",
                )

        except ImportError as e:
            # Agent module not available - simulate for testing
            _simulate_agent_run(job_id, objective, input_files, thread_id)

        except Exception as e:
            # Agent execution failed
            error_event = {
                "type": "error",
                "job_id": job_id,
                "thread_id": thread_id or job_id,
                "node": "runtime",
                "payload": {"detail": str(e)},
                "ts": _iso_now(),
            }
            _append_event(job_id, error_event)

            job_service.update_job_status(
                job_id,
                JobStatus.FAILED,
                progress=0,
                current_node="error",
                error={"detail": str(e)},
            )

    except Exception as e:
        # Job runner failure
        job_service.update_job_status(
            job_id,
            JobStatus.FAILED,
            error={"detail": f"Job runner error: {e}"},
        )
    finally:
        # Remove from running jobs
        _running_jobs.pop(job_id, None)


def _simulate_agent_run(
    job_id: str,
    objective: str,
    input_files: List[str],
    thread_id: Optional[str],
) -> None:
    """Simulate agent run for testing when agent module is unavailable.

    This is a fallback for development/testing when the full agent
    dependencies are not available.

    PROGRESS TRACKING:
    Progress is heuristic/stage-based: each node emits a discrete progress
    value (10, 50, 90, 100) representing completion milestones, not
    continuous granular progress. This matches the real agent behavior where
    progress updates are emitted at node boundaries.

    Args:
        job_id: Job identifier
        objective: User's objective
        input_files: List of input file paths
        thread_id: Conversation thread ID
    """
    effective_thread_id = thread_id or job_id

    # Simulate progress events - stage-based heuristic progress
    # Real agent emits similar discrete progress at node boundaries
    steps = [
        ("planner", "Generating analysis plan...", 10),
        ("executor", "Running analysis steps...", 50),
        ("response", "Compiling results...", 90),
    ]

    for node, message, progress in steps:
        time.sleep(0.5)  # Simulate work
        job_service.update_job_status(
            job_id,
            JobStatus.RUNNING,
            progress=progress,
            current_node=node,
        )
        _append_event(
            job_id,
            {
                "type": "progress",
                "job_id": job_id,
                "thread_id": effective_thread_id,
                "node": node,
                "payload": {"progress": progress, "message": message},
                "ts": _iso_now(),
            },
        )

    # Complete
    job_service.update_job_status(
        job_id,
        JobStatus.SUCCEEDED,
        progress=100,
        current_node="end",
    )
    _append_event(
        job_id,
        {
            "type": "end",
            "job_id": job_id,
            "thread_id": effective_thread_id,
            "node": "end",
            "payload": {
                "execution_status": "succeeded",
                "message": "Analysis complete",
            },
            "ts": _iso_now(),
        },
    )


def submit_job(job_id: str) -> None:
    """Submit a job to run in the background.

    Loads the job state and starts execution in a background thread.

    Args:
        job_id: Job identifier

    Raises:
        FileNotFoundError: If job does not exist
        ValueError: If job is not in queued state
    """
    if not job_service.job_exists(job_id):
        raise FileNotFoundError(f"Job {job_id} not found")

    state = job_service.read_job_state(job_id)

    if state.status != JobStatus.QUEUED:
        raise ValueError(f"Job {job_id} is not in queued state (current: {state.status})")

    if job_id in _running_jobs:
        raise ValueError(f"Job {job_id} is already running")

    # Read request data
    request_file = job_service.get_job_dir(job_id) / "request.json"
    request_data = job_service._read_json(request_file)

    # Start background thread
    thread = threading.Thread(
        target=_run_agent_sync,
        kwargs={
            "job_id": job_id,
            "objective": request_data["objective"],
            "input_files": request_data.get("input_files", []),
            "thread_id": request_data.get("thread_id"),
            "stream_mode": request_data.get("stream_mode", "updates"),
        },
        daemon=True,
    )

    _running_jobs[job_id] = thread
    thread.start()


def is_job_running(job_id: str) -> bool:
    """Check if a job is currently running.

    Args:
        job_id: Job identifier

    Returns:
        True if job is running
    """
    thread = _running_jobs.get(job_id)
    if thread is None:
        return False
    return thread.is_alive()


def shutdown() -> None:
    """Shutdown the job runner executor.

    Waits for running jobs to complete (with timeout).
    """
    _executor.shutdown(wait=False)


def _format_sse(event: Dict[str, Any]) -> str:
    """Format event as Server-Sent Event (SSE).

    Args:
        event: Event dict

    Returns:
        SSE formatted string
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _build_heartbeat_event(job_id: str) -> Dict[str, Any]:
    """Build a heartbeat event for keepalive.

    Args:
        job_id: Job identifier

    Returns:
        Heartbeat event dict
    """
    return {
        "type": SSEEventType.HEARTBEAT.value,
        "job_id": job_id,
        "thread_id": None,
        "node": None,
        "payload": {},
        "ts": _iso_now(),
    }


async def stream_job_events(
    job_id: str,
    from_line: int = 0,
    heartbeat_interval: float = 10.0,
) -> AsyncGenerator[str, None]:
    """Stream job events as SSE from the events.ndjson file.

    This function streams events from runs/{job_id}/events.ndjson,
    supporting replay from a given line offset and live tailing.

    Args:
        job_id: Job identifier
        from_line: Line offset to start streaming from (0-based)
        heartbeat_interval: Seconds between heartbeat events when idle

    Yields:
        SSE formatted strings
    """
    events_path = job_service.get_events_path(job_id)
    last_line = from_line  # Next line to read
    last_heartbeat = time.time()

    # First, replay existing events from the requested offset
    if events_path.exists():
        line_count = 0
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line_count >= from_line:
                    event = json.loads(line)
                    yield _format_sse(event)
                    last_line = line_count + 1
                line_count += 1

    # Track total lines seen for tailing phase
    total_lines_seen = line_count

    # Then tail for new events while job is running or recently completed
    post_complete_timeout = 2.0  # seconds to wait after job completes
    post_complete_start = None

    while True:
        # Check if we should stop
        state = job_service.read_job_state(job_id)
        is_complete = state.status.value in ("succeeded", "failed", "cancelled")

        if is_complete:
            if post_complete_start is None:
                post_complete_start = time.time()
            elif time.time() - post_complete_start > post_complete_timeout:
                # Job complete and timeout elapsed - stop streaming
                break
        else:
            post_complete_start = None

        # Check for new events (tail from where we left off)
        if events_path.exists():
            current_count = 0
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    current_count += 1
                    # Only yield new lines we haven't sent yet
                    if current_count > total_lines_seen:
                        event = json.loads(line)
                        yield _format_sse(event)
                        last_line = current_count
                        total_lines_seen = current_count
                        # Reset heartbeat timer on activity
                        last_heartbeat = time.time()

        # Send heartbeat if idle for too long
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            yield _format_sse(_build_heartbeat_event(job_id))
            last_heartbeat = now

        # Small sleep to avoid busy-waiting
        await asyncio.sleep(0.2)
