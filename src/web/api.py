# src/web/api.py

"""API router that exposes both legacy (v1) and new (v2) endpoints."""
from fastapi import APIRouter

from . import routes_chat, routes_jobs, routes_v2

router = APIRouter()

# Job-based endpoints (new)
router.include_router(routes_jobs.router, tags=["jobs"])

# Chat endpoint (non-agent)
router.include_router(routes_chat.router, tags=["chat"])

# v2 endpoints (current)
router.include_router(routes_v2.router, prefix="/v2/agent", tags=["agent_v2"])

# Note: v1 endpoints have been removed. Use v2 endpoints instead.
