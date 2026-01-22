"""Web layer package.

Contains FastAPI routes, request/response schemas, and dependencies.
"""
from __future__ import annotations

from src.web.schemas import (
    ChatRequest,
    ChatResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
    SSEvent,
)

__all__ = [
    "JobCreateRequest",
    "JobCreateResponse",
    "JobStatusResponse",
    "SSEvent",
    "ChatRequest",
    "ChatResponse",
]
