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
            task_id="dataset_overview_basic",
            objective="快速浏览上传的 scRNA-seq 数据，报告细胞和基因数量，并指出是否包含已知批次信息",
            category="quality_control",
            difficulty="simple",
            expected_keywords=("细胞", "基因"),
        ),
        TaskSpec(
            task_id="qc_threshold_review",
            objective="检查质量控制过滤标准（如线粒体比例、最小基因数）是否合理，并提出如需调整的建议",
            category="quality_control",
            difficulty="simple",
            expected_keywords=("质量", "过滤"),
        ),
        TaskSpec(
            task_id="cell_annotation_basic",
            objective="完成基础的细胞类型注释，为每个聚类给出 2-3 个代表性标记基因",
            category="cell_annotation",
            difficulty="simple",
            expected_keywords=("注释", "标记"),
        ),
        TaskSpec(
            task_id="marker_gene_panel",
            objective="列出免疫相关簇的上调标记基因，帮助快速识别主要免疫亚群",
            category="marker_gene_analysis",
            difficulty="simple",
            expected_keywords=("免疫", "标记"),
        ),
        TaskSpec(
            task_id="differential_expression_pair",
            objective="对最常见的两个聚类执行差异表达，并总结前 5 个上调基因的功能",
            category="differential_expression",
            difficulty="composite",
            expected_keywords=("差异", "上调"),
        ),
        TaskSpec(
            task_id="ssgsea_enrichment_focus",
            objective="在已注释的主要免疫簇上运行 ssGSEA，列出显著富集的免疫相关通路",
            category="pathway_analysis",
            difficulty="composite",
            expected_keywords=("ssGSEA", "通路"),
        ),
        TaskSpec(
            task_id="dataset_consistency_check",
            objective="检查是否存在明显的批次或样本分层迹象，并给出可视化建议",
            category="clustering_analysis",
            difficulty="simple",
            expected_keywords=("批次", "可视化"),
        ),
        TaskSpec(
            task_id="plan_next_steps",
            objective="如果已完成聚类和标注，建议下一个高价值的分析步骤并说明原因",
            category="clarification",
            difficulty="ambiguous",
            requires_dataset=False,
        ),
        TaskSpec(
            task_id="memory_seed_analysis_goal",
            objective="请记住本次分析目标：快速完成细胞类型注释并输出差异基因摘要",
            category="memory",
            difficulty="simple",
            requires_dataset=False,
            expected_keywords=("目标", "注释"),
        ),
        TaskSpec(
            task_id="memory_followup_project_state",
            objective="之前记录的分析目标是什么？请复述并说明当前分析状态",
            category="memory",
            difficulty="ambiguous",
            requires_dataset=False,
            expected_keywords=("目标", "状态"),
            follow_up_of="memory_seed_analysis_goal",
        ),
        TaskSpec(
            task_id="status_request_active",
            objective="当前分析流程进展到哪一步了？还缺少哪些结果",
            category="status_check",
            difficulty="simple",
            requires_dataset=False,
            expected_keywords=("进展", "结果"),
        ),
        TaskSpec(
            task_id="greeting_chitchat",
            objective="简单问候并确认已准备好接收下一步指令",
            category="greeting",
            difficulty="simple",
            requires_dataset=False,
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
