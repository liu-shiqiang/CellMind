# src/tools/planning_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field

class PlanningArgs(BaseModel):
    command: str = Field(..., description="Planning instruction: create/update/list/get/mark_step/delete")
    plan_id: str = Field(..., description="Unique ID for the plan")
    title: str = Field(None, description="Plan Title")
    steps: list[str] = Field(None, description="The steps of the plan")
    step_idx: int = Field(None, description="Current step number")
    status: str = Field(None, description="Current plan status")

_PLANS = {}

@tool(name="planning", args_schema=PlanningArgs)
def planning_tool(command, plan_id, title=None, steps=None, step_idx=None, status=None):
    """A tool for creating and managing multi-step task plans."""
    if command == "create":
        _PLANS[plan_id] = {"title": title, "steps": steps, "active_step": 0, "status": "created"}
        return {"msg": "created", "plan": _PLANS[plan_id]}
    if command == "update":
        if plan_id in _PLANS:
            if steps is not None:
                _PLANS[plan_id]["steps"] = steps
            if title:
                _PLANS[plan_id]["title"] = title
            return {"msg": "updated", "plan": _PLANS[plan_id]}
        return {"error": "plan not found"}
    if command == "mark_step":
        if plan_id in _PLANS and step_idx is not None:
            _PLANS[plan_id]["active_step"] = step_idx
            return {"msg": f"Step {step_idx} marked"}
        return {"error": "plan not found"}
    if command == "list":
        return list(_PLANS.keys())
    if command == "get":
        return _PLANS.get(plan_id, {})
    if command == "delete":
        _PLANS.pop(plan_id, None)
        return {"msg": "deleted"}
    return {"error": "unknown command"}


