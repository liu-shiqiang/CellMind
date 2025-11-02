"""Utilities for aggregating metrics across experiment runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List

from src.experiments.workflows import WorkflowRun


@dataclass
class AggregateMetrics:
    label: str
    success_rate: float
    avg_runtime: float
    avg_tool_calls: float
    avg_clarity: float
    plan_edit_distance: float = 0.0
    plan_modifications: float = 0.0
    recovery_rate: float = 0.0
    memory_precision: float = 0.0
    memory_recall: float = 0.0
    knowledge_accuracy: float = 0.0


def _safe_mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(mean(items)) if items else 0.0


def summarise_runs(label: str, runs: Iterable[WorkflowRun]) -> AggregateMetrics:
    runs = list(runs)
    if not runs:
        return AggregateMetrics(label=label, success_rate=0.0, avg_runtime=0.0, avg_tool_calls=0.0, avg_clarity=0.0)

    success_rate = sum(1 for run in runs if run.success) / len(runs)
    avg_runtime = _safe_mean(run.runtime for run in runs)
    avg_tool_calls = _safe_mean(run.tool_call_count for run in runs)
    avg_clarity = _safe_mean(run.clarity_score for run in runs)
    plan_edit = _safe_mean(run.plan_edit_distance for run in runs if run.plan_edit_distance is not None)
    plan_mods = _safe_mean(run.plan_modifications for run in runs)
    recovery_rate = sum(1 for run in runs if run.recovered_from_failure) / len(runs)
    memory_precision = _safe_mean(run.memory_precision for run in runs if run.memory_precision is not None)
    memory_recall = _safe_mean(run.memory_recall for run in runs if run.memory_recall is not None)
    knowledge_accuracy = _safe_mean(run.knowledge_accuracy for run in runs if run.knowledge_accuracy is not None)

    return AggregateMetrics(
        label=label,
        success_rate=round(success_rate, 3),
        avg_runtime=round(avg_runtime, 3),
        avg_tool_calls=round(avg_tool_calls, 2),
        avg_clarity=round(avg_clarity, 2),
        plan_edit_distance=round(plan_edit, 3),
        plan_modifications=round(plan_mods, 2),
        recovery_rate=round(recovery_rate, 3),
        memory_precision=round(memory_precision, 3),
        memory_recall=round(memory_recall, 3),
        knowledge_accuracy=round(knowledge_accuracy, 3),
    )


def group_by_label(runs: Iterable[WorkflowRun], *, key) -> Dict[str, List[WorkflowRun]]:
    grouped: Dict[str, List[WorkflowRun]] = {}
    for run in runs:
        grouped.setdefault(key(run), []).append(run)
    return grouped


def summarise_by_dataset(runs: Iterable[WorkflowRun]) -> Dict[str, AggregateMetrics]:
    grouped = group_by_label(runs, key=lambda run: run.dataset.name)
    return {dataset: summarise_runs(dataset, subset) for dataset, subset in grouped.items()}


def summarise_by_task(runs: Iterable[WorkflowRun]) -> Dict[str, AggregateMetrics]:
    grouped = group_by_label(runs, key=lambda run: run.task.task_id)
    return {task: summarise_runs(task, subset) for task, subset in grouped.items()}


def build_table(rows: Iterable[AggregateMetrics]) -> List[Dict[str, float]]:
    table: List[Dict[str, float]] = []
    for row in rows:
        table.append(
            {
                "label": row.label,
                "success_rate": row.success_rate,
                "avg_runtime": row.avg_runtime,
                "avg_tool_calls": row.avg_tool_calls,
                "avg_clarity": row.avg_clarity,
                "plan_edit_distance": row.plan_edit_distance,
                "plan_modifications": row.plan_modifications,
                "recovery_rate": row.recovery_rate,
                "memory_precision": row.memory_precision,
                "memory_recall": row.memory_recall,
                "knowledge_accuracy": row.knowledge_accuracy,
            }
        )
    return table


__all__ = [
    "AggregateMetrics",
    "summarise_runs",
    "summarise_by_dataset",
    "summarise_by_task",
    "build_table",
]
