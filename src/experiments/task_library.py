"""Task and intent benchmark definitions for Genomix experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from typing import Literal


@dataclass(frozen=True)
class TaskSpec:
    """Description of a single user objective used in the experiments."""

    task_id: str
    objective: str
    category: str
    difficulty: Literal["simple", "composite", "ambiguous"]
    requires_dataset: bool = True
    expected_keywords: Tuple[str, ...] = ()
    follow_up_of: Optional[str] = None


@dataclass(frozen=True)
class IntentBenchmarkSample:
    prompt: str
    label: str


def build_task_suite() -> List[TaskSpec]:
    """Return the canonical list of evaluation tasks (>10 covering key behaviours)."""

    tasks: List[TaskSpec] = [
        TaskSpec(
            task_id="cell_annotation_basic",
            objective="请基于上传的 scRNA-seq 数据完成细胞类型注释，并列出每个簇的代表性标记基因",
            category="cell_annotation",
            difficulty="simple",
            expected_keywords=("注释", "标记"),
        ),
        TaskSpec(
            task_id="quality_control_overview",
            objective="对该数据集执行质量控制并总结过滤标准，指出低质量细胞比例",
            category="quality_control",
            difficulty="simple",
            expected_keywords=("质量", "过滤"),
        ),
        TaskSpec(
            task_id="clustering_comparison",
            objective="比较不同聚类分辨率下的簇稳定性，并解释差异表达的核心基因",
            category="clustering_analysis",
            difficulty="composite",
            expected_keywords=("聚类", "差异"),
        ),
        TaskSpec(
            task_id="marker_gene_dive",
            objective="找出最显著的免疫相关簇，并列出上调的经典标记基因",
            category="marker_gene_analysis",
            difficulty="composite",
            expected_keywords=("免疫", "上调"),
        ),
        TaskSpec(
            task_id="pathway_enrichment_full",
            objective="执行 ssGSEA 富集分析，说明 T 细胞簇显著富集的通路",
            category="pathway_analysis",
            difficulty="composite",
            expected_keywords=("ssGSEA", "通路"),
        ),
        TaskSpec(
            task_id="regulatory_network_focus",
            objective="推断肿瘤相关簇的调控网络，指出潜在关键转录因子",
            category="regulatory_network",
            difficulty="ambiguous",
            expected_keywords=("调控", "转录因子"),
        ),
        TaskSpec(
            task_id="differential_expression_subset",
            objective="比较肿瘤相关簇与所有其他细胞的差异表达，并总结功能影响",
            category="differential_expression",
            difficulty="composite",
            expected_keywords=("差异表达", "功能"),
        ),
        TaskSpec(
            task_id="trajectory_analysis_time",
            objective="对拟胚系细胞构建发育轨迹并分析伪时间上的基因变化",
            category="trajectory_analysis",
            difficulty="composite",
            expected_keywords=("伪时间", "轨迹"),
        ),
        TaskSpec(
            task_id="data_visualization_umap",
            objective="生成 UMAP 可视化并标注主要细胞类型，突出显示稀有细胞簇",
            category="data_visualization",
            difficulty="simple",
            expected_keywords=("UMAP", "稀有"),
        ),
        TaskSpec(
            task_id="cell_communication_summary",
            objective="汇总推断的细胞间通信网络，强调免疫相关配体-受体对",
            category="cell_communication",
            difficulty="composite",
            expected_keywords=("通信", "配体"),
        ),
        TaskSpec(
            task_id="knowledge_validation_followup",
            objective="结合知识库回答：该数据集中与 T 细胞衰竭相关的生物学发现有哪些?",
            category="knowledge_retrieval",
            difficulty="ambiguous",
            expected_keywords=("知识", "衰竭"),
        ),
        TaskSpec(
            task_id="ambiguous_request_clarify",
            objective="也许做一点差异分析或者别的？你觉得下一步该怎么做",
            category="clarification",
            difficulty="ambiguous",
            requires_dataset=False,
        ),
        TaskSpec(
            task_id="memory_seed_project_state",
            objective="请记住我们当前研究针对的患者是 2024-AML-03，并建立后续分析都基于该患者",
            category="memory",
            difficulty="simple",
            requires_dataset=False,
            expected_keywords=("2024-AML-03",),
        ),
        TaskSpec(
            task_id="memory_followup_project_state",
            objective="提醒我之前提到的患者编号是什么，并更新最新的分析状态",
            category="memory",
            difficulty="ambiguous",
            requires_dataset=False,
            expected_keywords=("2024-AML-03", "状态"),
            follow_up_of="memory_seed_project_state",
        ),
        TaskSpec(
            task_id="status_request_active",
            objective="当前分析流程进展到哪一步了？下一步计划是什么",
            category="status_check",
            difficulty="simple",
            requires_dataset=False,
            expected_keywords=("进展", "下一步"),
        ),
    ]

    return tasks


def task_index() -> Dict[str, TaskSpec]:
    return {task.task_id: task for task in build_task_suite()}


def dataset_task_ids() -> List[str]:
    return [task.task_id for task in build_task_suite() if task.requires_dataset]


def composite_task_ids() -> List[str]:
    return [task.task_id for task in build_task_suite() if task.difficulty == "composite"]


def memory_task_sequences() -> List[Tuple[TaskSpec, TaskSpec]]:
    tasks = task_index()
    pairs: List[Tuple[TaskSpec, TaskSpec]] = []
    for task in tasks.values():
        if task.follow_up_of:
            seed = tasks[task.follow_up_of]
            pairs.append((seed, task))
    return pairs


def knowledge_task_ids() -> List[str]:
    return [
        task.task_id
        for task in build_task_suite()
        if task.category in {"knowledge_retrieval", "cell_communication", "pathway_analysis"}
    ]


def build_intent_benchmark(n_samples: int = 200) -> List[IntentBenchmarkSample]:
    """Synthetically expand labelled prompts covering planner intents."""

    base_samples: Dict[str, Sequence[str]] = {
        "cell_annotation": [
            "请完成 PBMC 数据的细胞类型注释",
            "Identify major immune cell types in the uploaded dataset",
            "给出每个聚类的细胞类型预测",
        ],
        "clustering_analysis": [
            "比较 0.4 与 1.0 分辨率下的聚类差异",
            "Can you rerun clustering with a higher resolution?",
        ],
        "marker_gene_analysis": [
            "列出 NK 细胞特异的上调基因",
            "Highlight canonical marker genes for each annotated cluster",
        ],
        "pathway_analysis": [
            "请对 CD8+ T 细胞做通路富集分析",
            "Run enrichment to find pathways active in macrophages",
        ],
        "regulatory_network": [
            "推断肿瘤细胞的调控网络",
            "Infer transcription factor activity changes",
        ],
        "differential_expression": [
            "比较肿瘤与免疫簇的差异表达",
            "Run differential expression between cluster 2 and others",
        ],
        "trajectory_analysis": [
            "构建发育轨迹并标注伪时间",
            "Perform pseudotime analysis for progenitor cells",
        ],
        "quality_control": [
            "请先做质量控制并说明过滤标准",
            "Report how many cells fail QC",
        ],
        "data_visualization": [
            "生成 UMAP 并按照注释着色",
            "Plot a heatmap of top markers",
        ],
        "knowledge_retrieval": [
            "结合文献解释 T 细胞衰竭机制",
            "Using the knowledge base, summarise NK cell biology",
        ],
        "memory_query": [
            "你还记得我之前的分析目标吗",
            "What patient ID did we mention last time?",
        ],
        "status_check": [
            "现在任务进度如何",
            "Have we finished the enrichment analysis yet?",
        ],
        "direct_response": [
            "谢谢你的帮助",
            "Just say hello to me",
        ],
        "clarification": [
            "你需要我提供什么参数才能继续",
            "Do you need the file path again?",
        ],
        "greeting": [
            "Hi assistant",
            "早上好",
        ],
        "chitchat": [
            "讲个笑话吧",
            "What music do you like?",
        ],
    }

    ordered_labels = list(base_samples.keys())
    samples: List[IntentBenchmarkSample] = []
    cycling_prompts = {label: cycle(prompts) for label, prompts in base_samples.items()}

    for idx in range(n_samples):
        label = ordered_labels[idx % len(ordered_labels)]
        prompt = next(cycling_prompts[label])
        samples.append(IntentBenchmarkSample(prompt=prompt, label=label))

    return samples


__all__ = [
    "TaskSpec",
    "IntentBenchmarkSample",
    "build_task_suite",
    "task_index",
    "dataset_task_ids",
    "composite_task_ids",
    "memory_task_sequences",
    "knowledge_task_ids",
    "build_intent_benchmark",
]
