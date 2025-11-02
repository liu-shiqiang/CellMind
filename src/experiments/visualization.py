"""Plotting utilities for the experimental evaluation suite."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.experiments.metrics import AggregateMetrics


sns.set_theme(style="whitegrid", context="talk")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_grouped_bar(
    metrics_a: Sequence[AggregateMetrics],
    metrics_b: Sequence[AggregateMetrics],
    labels: Sequence[str],
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    values_a = [getattr(metrics_a[idx], metric) for idx in range(len(labels))]
    values_b = [getattr(metrics_b[idx], metric) for idx in range(len(labels))]
    ax.bar(x - width / 2, values_a, width, label="多智能体")
    ax.bar(x + width / 2, values_b, width, label="基线")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_violin_steps(
    plan_distances: Sequence[float],
    labels: Sequence[str],
    output_path: Path,
    title: str = "步骤修改分布",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(y=plan_distances, x=labels, ax=ax, inner="quart", cut=0)
    ax.set_ylabel("计划编辑距离")
    ax.set_xlabel("配置")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_survival_curve(attempts: Sequence[int], recoveries: Sequence[int], output_path: Path) -> None:
    attempts = np.asarray(attempts)
    recoveries = np.asarray(recoveries)
    survival = 1.0 - np.cumsum(recoveries) / np.maximum(1, attempts)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(range(1, len(survival) + 1), survival, where="post")
    ax.set_xlabel("重新计划尝试次数")
    ax.set_ylabel("剩余失败比例")
    ax.set_title("重规划恢复生存曲线")
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_radar_chart(
    labels: Sequence[str],
    values_a: Sequence[float],
    values_b: Sequence[float],
    output_path: Path,
    title: str,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_a = list(values_a)
    values_b = list(values_b)
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values_a, "-o", label="启用记忆")
    ax.fill(angles, values_a, alpha=0.25)
    ax.plot(angles, values_b, "-o", label="禁用记忆")
    ax.fill(angles, values_b, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_accuracy_latency_scatter(
    accuracy: Sequence[float],
    latency: Sequence[float],
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(latency, accuracy, s=100, c=np.linspace(0, 1, len(accuracy)), cmap="viridis")
    for x, y, label in zip(latency, accuracy, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("响应延迟 (秒)")
    ax.set_ylabel("事实准确率")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_table(data: Iterable[Dict[str, float]], output_path: Path) -> None:
    df = pd.DataFrame(list(data))
    _ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


__all__ = [
    "plot_grouped_bar",
    "plot_violin_steps",
    "plot_survival_curve",
    "plot_radar_chart",
    "plot_accuracy_latency_scatter",
    "plot_confusion_matrix",
    "export_table",
]
