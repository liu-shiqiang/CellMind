"""Metric utilities shared across experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .agent_runner import AgentRunResult


@dataclass
class ConfusionMatrix:
    labels: List[str]
    matrix: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)


def results_dataframe(results: Iterable[AgentRunResult]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for item in results:
        records.append(
            {
                "task_id": item.task.task_id,
                "category": item.task.category,
                "difficulty": item.task.difficulty,
                "config": item.config_name,
                "run_index": item.run_index,
                "success": item.success,
                "duration_sec": item.duration_sec,
                "tool_calls": item.tool_calls,
                "tool_errors": item.tool_errors,
                "planner_invocations": item.planner_invocations,
                "plan_regenerations": item.plan_regenerations,
                "replanner_invocations": item.replanner_invocations,
                "keyword_recall": item.keyword_recall,
                "keyword_hits": item.keyword_hits,
                "keyword_total": item.keyword_total,
                "execution_status": item.execution_status,
                "final_message": item.final_message,
            }
        )
    return pd.DataFrame.from_records(records)


def summarise_by_config(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("config")
        .agg(
            success_rate=("success", "mean"),
            mean_duration=("duration_sec", "mean"),
            median_duration=("duration_sec", "median"),
            mean_tool_calls=("tool_calls", "mean"),
            planner_invocations=("planner_invocations", "mean"),
            replanner_invocations=("replanner_invocations", "mean"),
            keyword_recall=("keyword_recall", "mean"),
        )
        .reset_index()
    )
    return agg


def summarise_by_task(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["task_id", "config"])
        .agg(
            success_rate=("success", "mean"),
            mean_duration=("duration_sec", "mean"),
            mean_tool_calls=("tool_calls", "mean"),
            keyword_recall=("keyword_recall", "mean"),
        )
        .reset_index()
    )
    return agg


def failure_recovery_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["replanner_invocations"].notna()].copy()
    subset["replanner_invocations"] = subset["replanner_invocations"].fillna(0).astype(int)
    subset["tool_errors"] = subset["tool_errors"].fillna(0).astype(int)
    subset = subset[subset["tool_errors"] > 0]
    if subset.empty:
        return subset
    return (
        subset.groupby(["config", "replanner_invocations"])
        .agg(success_rate=("success", "mean"), count=("success", "size"))
        .reset_index()
    )


def memory_scores(df: pd.DataFrame) -> pd.DataFrame:
    memory_df = df[df["category"] == "memory"].copy()
    if memory_df.empty:
        return memory_df
    return (
        memory_df.groupby("config")
        .agg(
            memory_success_rate=("success", "mean"),
            memory_keyword_recall=("keyword_recall", "mean"),
        )
        .reset_index()
    )


def knowledge_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    knowledge_df = df[df["category"].isin(["knowledge_retrieval", "cell_communication", "pathway_analysis"])]
    if knowledge_df.empty:
        return knowledge_df
    return (
        knowledge_df.groupby("config")
        .agg(
            accuracy=("keyword_recall", "mean"),
            duration=("duration_sec", "mean"),
        )
        .reset_index()
    )


def build_confusion_matrix(
    ground_truth: Sequence[str], predictions: Sequence[str]
) -> ConfusionMatrix:
    labels = sorted(set(ground_truth) | set(predictions))
    label_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for gt, pred in zip(ground_truth, predictions):
        matrix[label_index[gt], label_index[pred]] += 1

    return ConfusionMatrix(labels=labels, matrix=matrix)


def classification_report(confusion: ConfusionMatrix) -> pd.DataFrame:
    df = confusion.as_dataframe()
    totals = df.sum(axis=1)
    precision = []
    recall = []
    f1 = []
    for idx, label in enumerate(confusion.labels):
        tp = df.iloc[idx, idx]
        fp = df.iloc[:, idx].sum() - tp
        fn = df.iloc[idx, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)
        if prec + rec > 0:
            f1.append(2 * prec * rec / (prec + rec))
        else:
            f1.append(0.0)
    return pd.DataFrame(
        {
            "label": confusion.labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": totals.tolist(),
        }
    )


__all__ = [
    "AgentRunResult",
    "ConfusionMatrix",
    "results_dataframe",
    "summarise_by_config",
    "summarise_by_task",
    "failure_recovery_table",
    "memory_scores",
    "knowledge_accuracy",
    "build_confusion_matrix",
    "classification_report",
]
