"""Helpers that execute the Genomix multi-agent system for experiment runs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import BaseMessage

from src.agent.agent_new import AgentRuntimeConfig, run_objective
from .task_library import TaskSpec


@dataclass
class AgentRunResult:
    task: TaskSpec
    config_name: str
    run_index: int
    thread_id: str
    success: bool
    duration_sec: float
    final_message: str
    execution_status: str
    metrics: Dict[str, Any]
    tool_calls: int
    tool_errors: int
    planner_invocations: int
    plan_regenerations: int
    replanner_invocations: int
    keyword_hits: int
    keyword_total: int

    @property
    def keyword_recall(self) -> float:
        if self.keyword_total == 0:
            return 0.0
        return self.keyword_hits / float(self.keyword_total)


def _normalise_message(message: Any) -> str:
    if isinstance(message, BaseMessage):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(part) for part in content)
        return str(content)
    if message is None:
        return ""
    return str(message)


def run_agent_task(
    task: TaskSpec,
    dataset_path: Optional[Path],
    runtime_config: AgentRuntimeConfig,
    config_name: str,
    run_index: int,
    base_thread_id: str,
) -> AgentRunResult:
    """Execute a single objective and collect diagnostics for analysis."""

    if task.requires_dataset and not dataset_path:
        raise ValueError(f"Task {task.task_id} requires a dataset path")

    input_files = [str(dataset_path)] if task.requires_dataset and dataset_path else []
    thread_id = f"{base_thread_id}-{task.task_id}-{config_name}-{run_index}"

    start = time.perf_counter()
    final_message, final_state = asyncio.run(
        run_objective(
            objective=task.objective,
            input_files=input_files,
            thread_id=thread_id,
            runtime_config=runtime_config,
            return_diagnostics=True,
        )
    )
    duration = time.perf_counter() - start

    final_state = final_state or {}
    final_message = _normalise_message(final_message)
    metrics: Dict[str, Any] = dict(final_state.get("metrics", {}) or {})
    tool_history = final_state.get("tool_history", []) or []
    execution_status = str(final_state.get("execution_status", "unknown"))

    tool_calls = int(metrics.get("tool_calls", len(tool_history)))
    tool_errors = int(metrics.get("tool_errors", 0))
    planner_invocations = int(metrics.get("planner_invocations", 0))
    plan_regenerations = int(metrics.get("plan_regenerations", 0))
    replanner_invocations = int(metrics.get("replanner_invocations", 0))

    success = execution_status == "completed"
    if not success and not task.requires_dataset:
        success = bool(final_message.strip())

    keyword_hits = 0
    lowered = final_message.lower()
    for keyword in task.expected_keywords:
        if keyword.lower() in lowered:
            keyword_hits += 1

    return AgentRunResult(
        task=task,
        config_name=config_name,
        run_index=run_index,
        thread_id=thread_id,
        success=success,
        duration_sec=duration,
        final_message=final_message,
        execution_status=execution_status,
        metrics=metrics,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        planner_invocations=planner_invocations,
        plan_regenerations=plan_regenerations,
        replanner_invocations=replanner_invocations,
        keyword_hits=keyword_hits,
        keyword_total=len(task.expected_keywords),
    )


__all__ = ["AgentRunResult", "run_agent_task"]
