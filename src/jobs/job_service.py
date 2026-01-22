"""Job workspace management service.

This module provides functions to create and manage job workspaces under runs/.
It follows the workspace layout defined in docs/architecture/system_spec.md:

  runs/{job_id}/
    request.json  - objective, input_files, stream_mode, thread_id, timestamps
    state.json    - job state machine snapshot
    plan.json     - last accepted plan (if any)
    events.ndjson - append-only event log
    artifacts/    - generated outputs
    uploads/      - validated copies of user-uploaded files
    logs/agent.log - textual logs

All filesystem writes are contained within runs/{job_id}/.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from src.web.schemas import (
    JobCreateRequest,
    JobState,
    JobStatus,
    StreamMode,
)


# Root directory for all job workspaces
RUNS_ROOT = Path("runs")


def _get_job_dir(job_id: str) -> Path:
    """Get the workspace directory for a job.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/
    """
    return RUNS_ROOT / job_id


def get_runs_root() -> Path:
    """Get the root runs directory.

    Returns:
        Path to runs/
    """
    return RUNS_ROOT


def ensure_runs_root() -> Path:
    """Ensure the runs root directory exists.

    Returns:
        Path to runs/
    """
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return RUNS_ROOT


def create_job(request: JobCreateRequest, job_id: Optional[str] = None) -> str:
    """Create a new job workspace and persist initial state.

    Creates the directory structure under runs/{job_id}/ including:
    - uploads/, artifacts/, logs/ subdirectories
    - request.json with the job creation request
    - state.json with initial queued state

    Args:
        request: Job creation request
        job_id: Optional job ID (generated if not provided)

    Returns:
        The job_id for the created job
    """
    if job_id is None:
        job_id = str(uuid4())

    job_dir = _get_job_dir(job_id)
    if job_dir.exists():
        raise FileExistsError(f"Job workspace already exists: {job_dir}")

    # Create directory structure
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "uploads").mkdir(exist_ok=True)
    (job_dir / "artifacts").mkdir(exist_ok=True)
    (job_dir / "logs").mkdir(exist_ok=True)

    # Create initial timestamp
    created_at = datetime.now(timezone.utc)

    # Persist request.json
    request_data = {
        "objective": request.objective,
        "input_files": request.input_files,
        "stream_mode": request.stream_mode.value,
        "thread_id": request.thread_id,
        "created_at": created_at.isoformat(),
    }
    _write_json(job_dir / "request.json", request_data)

    # Create and persist initial state.json (queued state)
    state = JobState(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        current_node=None,
        created_at=created_at,
        started_at=None,
        ended_at=None,
        objective=request.objective,
        input_files=request.input_files,
        stream_mode=request.stream_mode,
        error=None,
    )
    write_job_state(state)

    return job_id


def read_job_state(job_id: str) -> JobState:
    """Read the current job state from state.json.

    Args:
        job_id: Job identifier

    Returns:
        JobState snapshot

    Raises:
        FileNotFoundError: If job workspace or state.json does not exist
    """
    state_file = _get_job_dir(job_id) / "state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"Job state not found: {state_file}")

    data = _read_json(state_file)

    # Parse datetime strings
    created_at = _parse_datetime(data.get("created_at"))
    started_at = _parse_datetime(data.get("started_at"))
    ended_at = _parse_datetime(data.get("ended_at"))

    # Reconstruct JobState
    return JobState(
        job_id=data["job_id"],
        status=JobStatus(data["status"]),
        progress=data["progress"],
        current_node=data.get("current_node"),
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        objective=data["objective"],
        input_files=data.get("input_files", []),
        stream_mode=StreamMode(data.get("stream_mode", StreamMode.UPDATES)),
        error=data.get("error"),
    )


def write_job_state(state: JobState) -> None:
    """Persist job state to state.json.

    Args:
        state: JobState to persist
    """
    state_file = _get_job_dir(state.job_id) / "state.json"

    state_data = {
        "job_id": state.job_id,
        "status": state.status.value,
        "progress": state.progress,
        "current_node": state.current_node,
        "created_at": _format_datetime(state.created_at),
        "started_at": _format_datetime(state.started_at),
        "ended_at": _format_datetime(state.ended_at),
        "objective": state.objective,
        "input_files": state.input_files,
        "stream_mode": state.stream_mode.value,
        "error": state.error,
    }

    _write_json(state_file, state_data)


def update_job_status(
    job_id: str,
    status: JobStatus,
    progress: Optional[int] = None,
    current_node: Optional[str] = None,
    error: Optional[dict] = None,
) -> JobState:
    """Update job status and persist.

    Args:
        job_id: Job identifier
        status: New job status
        progress: Optional progress value (0-100)
        current_node: Optional current node name
        error: Optional error details

    Returns:
        Updated JobState
    """
    state = read_job_state(job_id)

    # Update status
    state.status = status

    # Update optional fields
    if progress is not None:
        state.progress = progress
    if current_node is not None:
        state.current_node = current_node
    if error is not None:
        state.error = error

    # Update timestamps based on status transitions
    now = datetime.now(timezone.utc)
    if status == JobStatus.RUNNING and state.started_at is None:
        state.started_at = now
    elif status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        state.ended_at = now

    write_job_state(state)
    return state


def get_job_dir(job_id: str) -> Path:
    """Get the job workspace directory path.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/
    """
    return _get_job_dir(job_id)


def get_uploads_dir(job_id: str) -> Path:
    """Get the uploads directory for a job.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/uploads/
    """
    return _get_job_dir(job_id) / "uploads"


def get_artifacts_dir(job_id: str) -> Path:
    """Get the artifacts directory for a job.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/artifacts/
    """
    return _get_job_dir(job_id) / "artifacts"


def get_logs_dir(job_id: str) -> Path:
    """Get the logs directory for a job.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/logs/
    """
    return _get_job_dir(job_id) / "logs"


def get_events_path(job_id: str) -> Path:
    """Get the events.ndjson path for a job.

    Args:
        job_id: Job identifier

    Returns:
        Path to runs/{job_id}/events.ndjson
    """
    return _get_job_dir(job_id) / "events.ndjson"


def job_exists(job_id: str) -> bool:
    """Check if a job workspace exists.

    Args:
        job_id: Job identifier

    Returns:
        True if job workspace exists
    """
    return _get_job_dir(job_id).exists()


def delete_job(job_id: str) -> None:
    """Delete a job workspace.

    Args:
        job_id: Job identifier

    Raises:
        FileNotFoundError: If job workspace does not exist
    """
    job_dir = _get_job_dir(job_id)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job workspace not found: {job_dir}")

    shutil.rmtree(job_dir)


def copy_file_to_uploads(job_id: str, source_path: Path) -> Path:
    """Copy a file to the job's uploads directory.

    Args:
        job_id: Job identifier
        source_path: Path to the source file

    Returns:
        Path to the copied file in uploads/
    """
    uploads_dir = get_uploads_dir(job_id)
    target_path = uploads_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path


def add_uploaded_file(job_id: str, filename: str) -> str:
    """Add an uploaded file to the job's input_files list.

    This appends the relative path (uploads/{filename}) to the
    input_files array in request.json.

    Args:
        job_id: Job identifier
        filename: Name of the uploaded file

    Returns:
        The relative path that was added (e.g., "uploads/test.h5ad")
    """
    relative_path = f"uploads/{filename}"
    job_dir = _get_job_dir(job_id)
    request_file = job_dir / "request.json"

    # Read current request data
    request_data = _read_json(request_file)

    # Append to input_files if not already present
    if "input_files" not in request_data:
        request_data["input_files"] = []
    if relative_path not in request_data["input_files"]:
        request_data["input_files"].append(relative_path)

    # Write back
    _write_json(request_file, request_data)

    return relative_path


def _read_json(path: Path) -> dict:
    """Read JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    """Write data to JSON file.

    Args:
        path: Path to JSON file
        data: Data to write
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 datetime string.

    Args:
        value: ISO 8601 datetime string or None

    Returns:
        datetime object or None
    """
    if value is None:
        return None
    # Handle both with and without timezone
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    """Format datetime as ISO 8601 string.

    Args:
        value: datetime object or None

    Returns:
        ISO 8601 string or None
    """
    if value is None:
        return None
    return value.isoformat()
