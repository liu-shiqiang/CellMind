# src/web/api.py

"""API router that exposes both legacy (v1) and new (v2) endpoints."""
from fastapi import APIRouter

from . import routes_v1, routes_v2

router = APIRouter()

router.include_router(routes_v1.router, prefix="/v1/agent", tags=["agent_v1"])
router.include_router(routes_v2.router, prefix="/v2/agent", tags=["agent_v2"])

# Backwards compatibility for legacy clients expecting /api/run
router.add_api_route(
    "/run",
    routes_v1.run_agent,
    methods=["POST"],
    tags=["agent_v1"],
)
