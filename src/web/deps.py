"""Shared request models and validators for FastAPI routes."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.path_manager import validate_h5ad_file


class AgentRunRequest(BaseModel):
    objective: str = Field(..., min_length=1, description="Task description for the agent")
    input_files: Optional[List[str]] = Field(
        default=None,
        description="Optional list of .h5ad input file paths",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional identifier used to resume a conversation thread",
    )

    @field_validator("objective")
    @classmethod
    def _validate_objective(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("objective must not be empty")
        return cleaned

    @field_validator("thread_id")
    @classmethod
    def _validate_thread_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("input_files")
    @classmethod
    def _validate_files(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return []
        validated: List[str] = []
        for file_path in value:
            cleaned = file_path.strip()
            if not cleaned:
                continue
            info = validate_h5ad_file(cleaned)
            validated.append(str(Path(info.resolved_path)))
        return validated

    def normalized_files(self) -> List[str]:
        return list(self.input_files or [])

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
    }
