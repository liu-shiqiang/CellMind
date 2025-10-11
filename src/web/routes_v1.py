"""Legacy API routes powered by the original agent implementation."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from src.agent.agent1 import build_graph

router = APIRouter()


class ObjectiveInput(BaseModel):
    objective: str


@router.post("/run")
async def run_agent(input: ObjectiveInput) -> Dict[str, Any]:
    """Run the legacy LangGraph agent with the provided objective."""
    result = []

    graph = build_graph()
    async for event in graph.astream(
        {"input": input.objective},
        config={
            "recursion_limit": 50,
            "configurable": {"thread_id": "1"},
        },
    ):
        for key, value in event.items():
            if key != "__end__":
                result.append({key: value})

    return {"events": result}
