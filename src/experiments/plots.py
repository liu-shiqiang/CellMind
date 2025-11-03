"""Plotting helpers for experiment visualisations."""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import ConfusionMatrix


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_grouped_bars(summary: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    metrics = ["success_rate", "mean_duration", "keyword_recall"]
    titles = ["成功率", "平均耗时 (s)", "关键词召回率"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), constrained_layout=True)
    configs = summary["config"].tolist()
    x = np.arange(len(configs))
    width = 0.35

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = summary[metric].to_numpy()
        ax.bar(x, values, color="#4F6BED")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric)
    fig.suptitle("实验 1：多智能体 vs. 基线")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_violin_tool_calls(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    grouped = [df[df["config"] == config]["tool_calls"].to_numpy() for config in df["config"].unique()]
    labels = df["config"].unique().tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ax.violinplot(grouped, showmeans=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4F6BED')
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("工具调用次数")
    ax.set_title("实验 2：规划器消融工具调用分布")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_survival_curve(survival_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for config, group in survival_df.groupby("config"):
        group = group.sort_values("replanner_invocations")
        attempts = group["replanner_invocations"].to_numpy()
        success = group["success_rate"].to_numpy()
        ax.step(attempts, success, where="post", label=config)
    ax.set_xlabel("重新规划尝试次数")
    ax.set_ylabel("恢复成功率")
    ax.set_title("实验 3：错误恢复生存曲线")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_radar_memory(memory_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    if memory_df.empty:
        return
    metrics = ["memory_success_rate", "memory_keyword_recall"]
    labels = ["成功率", "关键词召回"]
    angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)

    for _, row in memory_df.iterrows():
        values = row[metrics].to_numpy()
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, label=row["config"])
        ax.fill(angles, values, alpha=0.2)

    ax.set_thetagrids(angles[:-1] * 180 / math.pi, labels)
    ax.set_title("实验 4：记忆能力雷达图")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_latency(scatter_df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for config, group in scatter_df.groupby("config"):
        ax.scatter(group["duration"], group["accuracy"], label=config, s=60)
    ax.set_xlabel("平均耗时 (s)")
    ax.set_ylabel("事实准确率")
    ax.set_title("实验 5：知识检索准确率-延迟关系")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(confusion: ConfusionMatrix, output_path: Path) -> None:
    _ensure_parent(output_path)
    df = confusion.as_dataframe()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(df.values, cmap="Blues")
    ax.set_xticks(np.arange(len(confusion.labels)))
    ax.set_yticks(np.arange(len(confusion.labels)))
    ax.set_xticklabels(confusion.labels, rotation=45, ha="right")
    ax.set_yticklabels(confusion.labels)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("实验 6：意图分类混淆矩阵")

    for (i, j), value in np.ndenumerate(df.values):
        ax.text(j, i, int(value), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_grouped_bars",
    "plot_violin_tool_calls",
    "plot_survival_curve",
    "plot_radar_memory",
    "plot_accuracy_latency",
    "plot_confusion_matrix",
]
