# src/web/api.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from src.agent.agent1 import build_graph

router = APIRouter()

class ObjectiveInput(BaseModel):
    objective: str

@router.post("/run")
async def run_agent(input: ObjectiveInput) -> Dict[str, Any]:
    """Run the LangGraph agent with the provided objective."""
    result = []

    graph = build_graph()
    async for event in graph.astream(
        {"input": input.objective},
        config={
            "recursion_limit": 50,
            "configurable": {"thread_id": "1"},
        },
    ):
        for k, v in event.items():
            if k != "__end__":
                result.append({k: v})

    return {"events": result}
