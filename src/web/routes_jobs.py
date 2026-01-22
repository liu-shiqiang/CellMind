from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile, Form, Query
from fastapi.responses import StreamingResponse, FileResponse

from src.utils.langgraph_stream import run_agent_stream
from src.services.file_service import get_file_service
from src.web.schemas import (
    JobCreateResponse,
    JobStatusResponse,
    SSEvent,
)

router = APIRouter()

RUNS_ROOT = Path("runs").resolve()
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_ndjson(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


class _JobStreamState:
    def __init__(
        self,
        job_id: str,
        run_id: str,
        thread_id: str,
        objective: str,
        input_files: List[str],
        stream_mode: str,
        job_dir: Path,
    ) -> None:
        self.job_id = job_id
        self.run_id = run_id
        self.thread_id = thread_id
        self.objective = objective
        self.input_files = input_files
        self.stream_mode = stream_mode
        self.job_dir = job_dir
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.done = asyncio.Event()
        self.task: Optional[asyncio.Task[None]] = None
        self.status: str = "queued"
        self.progress: int = 0
        self.current_node: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at: datetime = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None

    @property
    def job_json_path(self) -> Path:
        return self.job_dir / "job.json"

    @property
    def events_path(self) -> Path:
        return self.job_dir / "events.ndjson"

    @property
    def uploads_dir(self) -> Path:
        return self.job_dir / "uploads"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "thread_id": self.thread_id,
            "objective": self.objective,
            "input_files": self.input_files,
            "stream_mode": self.stream_mode,
            "status": self.status,
            "progress": self.progress,
            "current_node": self.current_node,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "events_path": "events.ndjson",
            "artifacts_path": "artifacts/",
            "uploads_path": "uploads/",
        }

    def persist_job_json(self) -> None:
        _write_json(self.job_json_path, self.snapshot())


_job_states: Dict[str, _JobStreamState] = {}
_job_states_lock = asyncio.Lock()


def _format_sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _run_job(state: _JobStreamState) -> None:
    state.status = "running"
    state.started_at = datetime.now(timezone.utc)
    state.persist_job_json()

    async def handler(event: Dict[str, Any]) -> None:
        # Enrich event with job_id and normalized fields
        event.setdefault("job_id", state.job_id)
        if "step_id" not in event:
            event["step_id"] = event.get("node")
        # Persist event
        _append_ndjson(state.events_path, event)
        # Update lightweight status
        event_type = event.get("type")
        payload = event.get("payload", {}) or {}
        if event_type == "progress":
            state.progress = int(payload.get("progress", state.progress))
            state.current_node = event.get("node") or state.current_node
        elif event_type == "node_enter":
            state.current_node = event.get("node")
        elif event_type == "error":
            state.error = str(payload.get("detail", "unknown error"))
        elif event_type == "end":
            exec_status = payload.get("execution_status")
            state.status = exec_status or "succeeded"
            state.ended_at = datetime.now(timezone.utc)
            if payload.get("error"):
                state.error = str(payload.get("error"))
        # Enqueue for streaming
        await state.queue.put(event)

    try:
        await run_agent_stream(
            objective=state.objective,
            input_files=state.input_files,
            thread_id=state.thread_id,
            run_id=state.run_id,
            event_handler=handler,
            stream_mode=state.stream_mode,
        )
        if state.status not in {"failed", "cancelled"}:
            state.status = "succeeded"
    except Exception as exc:  # pragma: no cover - defensive
        state.status = "failed"
        state.error = str(exc)
        error_event: Dict[str, Any] = {
            "type": "error",
            "job_id": state.job_id,
            "node": "runtime",
            "step_id": None,
            "payload": {"detail": str(exc)},
            "ts": _iso_now(),
        }
        _append_ndjson(state.events_path, error_event)
        await state.queue.put(error_event)
    finally:
        state.ended_at = state.ended_at or datetime.now(timezone.utc)
        state.persist_job_json()
        state.done.set()


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    objective: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    file_id: Optional[str] = Form(default=None),
    thread_id: Optional[str] = Form(default=None),
    stream_mode: str = Form(default="updates"),
) -> JobCreateResponse:
    if not objective.strip():
        raise HTTPException(status_code=422, detail="objective is required")

    job_id = str(uuid4())
    run_id = job_id
    resolved_thread_id = thread_id or job_id
    job_dir = RUNS_ROOT / job_id
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    if file is not None:
        file_path = uploads_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        saved_files.append(str(file_path))
    elif file_id:
        file_service = get_file_service()
        metadata = await file_service.get_file(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="file_id not found")
        source_path = Path(metadata.filepath)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="file content not found")
        target_path = uploads_dir / metadata.original_name
        await asyncio.to_thread(shutil.copyfile, source_path, target_path)
        saved_files.append(str(target_path))

    state = _JobStreamState(
        job_id=job_id,
        run_id=run_id,
        thread_id=resolved_thread_id,
        objective=objective.strip(),
        input_files=saved_files,
        stream_mode=stream_mode,
        job_dir=job_dir,
    )
    state.persist_job_json()

    async with _job_states_lock:
        _job_states[job_id] = state

    state.task = asyncio.create_task(_run_job(state))

    return JobCreateResponse(job_id=job_id, thread_id=resolved_thread_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    async with _job_states_lock:
        state = _job_states.get(job_id)
    if state is None:
        job_json = RUNS_ROOT / job_id / "job.json"
        if not job_json.exists():
            raise HTTPException(status_code=404, detail="job not found")
        data = json.loads(job_json.read_text(encoding="utf-8"))
        return JobStatusResponse(
            job_id=data["job_id"],
            status=data["status"],
            progress=int(data.get("progress", 0)),
            current_node=data.get("current_node"),
            objective=data.get("objective", ""),
            input_files=data.get("input_files", []),
            stream_mode=data.get("stream_mode", "updates"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
        )

    return JobStatusResponse(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        current_node=state.current_node,
        objective=state.objective,
        input_files=state.input_files,
        stream_mode=state.stream_mode,
        error=state.error,
        created_at=state.created_at,
        started_at=state.started_at,
        ended_at=state.ended_at,
    )


@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    async with _job_states_lock:
        state = _job_states.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found or completed")

    async def event_generator():
        try:
            while True:
                if state.done.is_set() and state.queue.empty():
                    break
                event = await state.queue.get()
                # Validate/normalize event to SSE format
                try:
                    SSEvent.model_validate(event)
                except Exception:
                    pass
                yield _format_sse(event)
        finally:
            async with _job_states_lock:
                _job_states.pop(job_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/artifacts")
async def list_job_artifacts(
    job_id: str,
    artifact_type: Optional[str] = Query(default=None, alias="type"),
) -> Dict[str, Any]:
    job_dir = RUNS_ROOT / job_id / "artifacts"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job artifacts not found")

    artifacts: List[Dict[str, Any]] = []
    for path in job_dir.rglob("*"):
        if not path.is_file():
            continue
        if artifact_type and artifact_type not in path.parts:
            continue
        stat = path.stat()
        artifacts.append({
            "path": str(path.relative_to(job_dir)),
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })

    artifacts.sort(key=lambda item: item["modified_at"], reverse=True)

    return {"job_id": job_id, "artifacts": artifacts}


@router.get("/jobs/{job_id}/artifacts/download")
async def download_job_artifact(
    job_id: str,
    path: str = Query(..., min_length=1),
) -> FileResponse:
    job_dir = RUNS_ROOT / job_id / "artifacts"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job artifacts not found")

    resolved = (job_dir / path).resolve()
    if not resolved.is_relative_to(job_dir):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(path=str(resolved), filename=resolved.name)
