"""Task definitions and synthetic prompt generation for experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

Difficulty = Literal["simple", "composite", "ambiguous"]


@dataclass(frozen=True)
class TaskDefinition:
    """Description of an evaluation task."""

    task_id: str
    description: str
    intents: List[str]
    difficulty: Difficulty
    requires_memory: bool = False
    requires_retrieval: bool = False
    expected_outputs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset specific metadata for experiment generation."""

    dataset_path: Path
    name: str
    default_question_bank: List[str]


def build_task_suite(dataset: DatasetConfig) -> List[TaskDefinition]:
    """Create a deterministic set of tasks spanning all difficulty levels."""

    base_intents = [
        "cell_annotation",
        "differential_expression",
        "pathway_analysis",
    ]

    tasks: List[TaskDefinition] = []

    tasks.append(
        TaskDefinition(
            task_id=f"{dataset.name}_cell_annotation",
            description=(
                "请基于提供的单细胞数据完成细胞类型注释，并返回排名靠前的候选细胞类型。"
            ),
            intents=["cell_annotation"],
            difficulty="simple",
            expected_outputs=["cluster_celltype_rank1.csv"],
        )
    )

    tasks.append(
        TaskDefinition(
            task_id=f"{dataset.name}_composite_analysis",
            description=(
                "先完成细胞聚类和差异表达，再输出关键通路的富集结果。"
            ),
            intents=["clustering_analysis", "pathway_analysis"],
            difficulty="composite",
            expected_outputs=[
                "*_clustered.h5ad",
                "*_diff_gene.csv",
                "*_pathway_enrichment.csv",
            ],
        )
    )

    tasks.append(
        TaskDefinition(
            task_id=f"{dataset.name}_ambiguous_followup",
            description=(
                "上次我们讨论的免疫细胞簇进展如何？请结合记忆给出更新，并补充一个新的知识检索问题。"
            ),
            intents=["status_check", "memory_query", "dataset_bio_qa"],
            difficulty="ambiguous",
            requires_memory=True,
            requires_retrieval=True,
            expected_outputs=[
                "memory_status_report.json",
                "qa_context.txt",
            ],
        )
    )

    # Provide additional synthetic prompts to reach desired variety
    for idx, question in enumerate(dataset.default_question_bank, start=1):
        tasks.append(
            TaskDefinition(
                task_id=f"{dataset.name}_knowledge_{idx}",
                description=question,
                intents=["dataset_bio_qa"],
                difficulty="simple",
                requires_retrieval=True,
                expected_outputs=[f"knowledge_answer_{idx}.json"],
            )
        )

    return tasks


def iter_datasets(dataset_paths: Iterable[str]) -> List[DatasetConfig]:
    configs: List[DatasetConfig] = []
    for path_str in dataset_paths:
        path = Path(path_str).expanduser().resolve()
        configs.append(
            DatasetConfig(
                dataset_path=path,
                name=path.stem,
                default_question_bank=[
                    "该数据集中最显著的免疫相关通路是什么？",
                    "识别与T细胞活化相关的高表达基因。",
                    "比较肿瘤相关巨噬细胞与树突状细胞的差异表达。",
                ],
            )
        )
    return configs


__all__ = ["TaskDefinition", "DatasetConfig", "build_task_suite", "iter_datasets"]
