"""Job workspace management package.

Provides functions to create and manage job workspaces under runs/.
"""
from __future__ import annotations

from src.jobs.job_service import (
    add_uploaded_file,
    copy_file_to_uploads,
    create_job,
    delete_job,
    ensure_runs_root,
    get_artifacts_dir,
    get_events_path,
    get_job_dir,
    get_logs_dir,
    get_uploads_dir,
    job_exists,
    read_job_state,
    update_job_status,
    write_job_state,
    get_runs_root,
)

__all__ = [
    "create_job",
    "read_job_state",
    "write_job_state",
    "update_job_status",
    "job_exists",
    "delete_job",
    "get_job_dir",
    "get_uploads_dir",
    "get_artifacts_dir",
    "get_logs_dir",
    "get_events_path",
    "get_runs_root",
    "ensure_runs_root",
    "copy_file_to_uploads",
    "add_uploaded_file",
]
