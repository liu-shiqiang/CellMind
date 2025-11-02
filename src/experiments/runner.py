"""Entry point orchestrating all experiments and generating outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import typer

from src.experiments.analysis_pipeline import SingleCellAnalysisPipeline, ToolFailureInjector
from src.experiments.metrics import (
    AggregateMetrics,
    build_table,
    summarise_by_dataset,
    summarise_runs,
)
from src.experiments.tasks import DatasetConfig, TaskDefinition, build_task_suite, iter_datasets
from src.experiments.visualization import (
    export_table,
    plot_accuracy_latency_scatter,
    plot_confusion_matrix,
    plot_grouped_bar,
    plot_radar_chart,
    plot_survival_curve,
    plot_violin_steps,
)
from src.experiments.workflows import MultiAgentWorkflow, SingleAgentWorkflow, WorkflowRun
from src.memory.conversation_memory import ConversationMemoryStore


@dataclass
class ExperimentOutputs:
    experiment_1: Dict[str, List[WorkflowRun]]
    experiment_2: Dict[str, List[WorkflowRun]]
    experiment_3: Dict[str, List[WorkflowRun]]
    experiment_4: Dict[str, List[WorkflowRun]]
    experiment_5: Dict[str, List[WorkflowRun]]
    experiment_6: Dict[str, np.ndarray]


class RuleBasedIntentClassifier:
    """Lightweight classifier used for Experiment 6 benchmarking."""

    def __init__(self) -> None:
        self.intent_keywords: Dict[str, Sequence[str]] = {
            "cell_annotation": ("注释", "标注", "cell type", "annotate"),
            "clustering_analysis": ("聚类", "cluster"),
            "pathway_analysis": ("通路", "pathway", "富集"),
            "memory_query": ("记得", "记忆", "上次", "之前"),
            "status_check": ("进度", "状态", "完成了吗", "update"),
            "dataset_bio_qa": ("文献", "知识", "解释", "explain"),
        }

    def predict(self, text: str) -> str:
        lowered = text.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                return intent
        return "generic"


class ExperimentSuite:
    def __init__(
        self,
        datasets: Sequence[DatasetConfig],
        *,
        output_dir: Path,
        runs_per_task: int = 20,
        random_seed: Optional[int] = None,
    ) -> None:
        self.datasets = list(datasets)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_per_task = max(1, runs_per_task)
        self.random_state = np.random.default_rng(random_seed)
        self.memory_store = ConversationMemoryStore(self.output_dir / "conversation_memory.json")
        self.task_suites: Dict[str, List[TaskDefinition]] = {
            dataset.name: build_task_suite(dataset) for dataset in self.datasets
        }

    # ------------------------------------------------------------------
    def run(self) -> ExperimentOutputs:
        exp1 = self._run_experiment_1()
        exp2 = self._run_experiment_2()
        exp3 = self._run_experiment_3()
        exp4 = self._run_experiment_4()
        exp5 = self._run_experiment_5()
        exp6 = self._run_experiment_6()

        self._generate_outputs(exp1, exp2, exp3, exp4, exp5, exp6)
        return ExperimentOutputs(exp1, exp2, exp3, exp4, exp5, exp6)

    # ------------------------------------------------------------------
    def _create_pipeline(self, *, failure_rate: float = 0.0) -> SingleCellAnalysisPipeline:
        return SingleCellAnalysisPipeline(
            cache_intermediate=True,
            memory_store=self.memory_store,
            failure_injector=ToolFailureInjector(failure_rate=failure_rate, seed=int(self.random_state.integers(0, 1_000_000))),
        )

    def _question_for_task(self, task: TaskDefinition, run_idx: int) -> str:
        suffix = f" (运行{run_idx+1})"
        return f"{task.description}{suffix}"

    # Experiment 1 -----------------------------------------------------
    def _run_experiment_1(self) -> Dict[str, List[WorkflowRun]]:
        pipeline_multi = self._create_pipeline()
        pipeline_baseline = self._create_pipeline()
        multi_agent = MultiAgentWorkflow(pipeline_multi, enable_memory=True, enable_replanner=True)
        baseline = SingleAgentWorkflow(pipeline_baseline)

        results_multi: List[WorkflowRun] = []
        results_baseline: List[WorkflowRun] = []

        for dataset in self.datasets:
            tasks = self.task_suites[dataset.name]
            for run_idx in range(self.runs_per_task):
                for task in tasks:
                    question = self._question_for_task(task, run_idx)
                    results_multi.append(multi_agent.run_task(dataset, task, question=question, run_idx=run_idx))
                    results_baseline.append(baseline.run_task(dataset, task, question=question, run_idx=run_idx))

        return {"multi_agent": results_multi, "baseline": results_baseline}

    # Experiment 2 -----------------------------------------------------
    def _run_experiment_2(self) -> Dict[str, List[WorkflowRun]]:
        class NoPlannerWorkflow(MultiAgentWorkflow):
            def build_plan(self, task: TaskDefinition) -> List[str]:
                return ["直接执行"]

            def _intents_for_step(self, step: str, task: TaskDefinition) -> List[str]:
                return list(task.intents)

        pipeline_with_planner = self._create_pipeline()
        pipeline_without_planner = self._create_pipeline()
        planner_workflow = MultiAgentWorkflow(pipeline_with_planner, enable_memory=True, enable_replanner=True)
        no_planner_workflow = NoPlannerWorkflow(pipeline_without_planner, enable_memory=True, enable_replanner=False)

        planned_runs: List[WorkflowRun] = []
        flat_runs: List[WorkflowRun] = []

        for dataset in self.datasets:
            tasks = [task for task in self.task_suites[dataset.name] if task.difficulty != "simple"]
            for run_idx in range(self.runs_per_task):
                for task in tasks:
                    question = self._question_for_task(task, run_idx)
                    planned_runs.append(planner_workflow.run_task(dataset, task, question=question, run_idx=run_idx))
                    flat_runs.append(no_planner_workflow.run_task(dataset, task, question=question, run_idx=run_idx))

        return {"planner": planned_runs, "no_planner": flat_runs}

    # Experiment 3 -----------------------------------------------------
    def _run_experiment_3(self) -> Dict[str, List[WorkflowRun]]:
        pipeline_with_replanner = self._create_pipeline(failure_rate=0.3)
        pipeline_without_replanner = self._create_pipeline(failure_rate=0.3)
        replanner_workflow = MultiAgentWorkflow(pipeline_with_replanner, enable_memory=True, enable_replanner=True)
        no_replanner_workflow = MultiAgentWorkflow(pipeline_without_replanner, enable_memory=True, enable_replanner=False)

        runs_with = []
        runs_without = []

        for dataset in self.datasets:
            tasks = [task for task in self.task_suites[dataset.name] if task.difficulty != "simple"]
            for run_idx in range(self.runs_per_task):
                for task in tasks:
                    question = self._question_for_task(task, run_idx)
                    runs_with.append(replanner_workflow.run_task(dataset, task, question=question, run_idx=run_idx))
                    runs_without.append(no_replanner_workflow.run_task(dataset, task, question=question, run_idx=run_idx))

        return {"with_replanner": runs_with, "without_replanner": runs_without}

    # Experiment 4 -----------------------------------------------------
    def _run_experiment_4(self) -> Dict[str, List[WorkflowRun]]:
        pipeline_memory = self._create_pipeline()
        pipeline_no_memory = self._create_pipeline()
        workflow_memory = MultiAgentWorkflow(pipeline_memory, enable_memory=True, enable_replanner=True)
        workflow_no_memory = MultiAgentWorkflow(pipeline_no_memory, enable_memory=False, enable_replanner=True)

        memory_runs: List[WorkflowRun] = []
        no_memory_runs: List[WorkflowRun] = []

        for dataset in self.datasets:
            tasks = [task for task in self.task_suites[dataset.name] if task.requires_memory]
            for run_idx in range(self.runs_per_task):
                for task in tasks:
                    question = self._question_for_task(task, run_idx)
                    memory_runs.append(workflow_memory.run_task(dataset, task, question=question, run_idx=run_idx))
                    no_memory_runs.append(workflow_no_memory.run_task(dataset, task, question=question, run_idx=run_idx))

        return {"with_memory": memory_runs, "without_memory": no_memory_runs}

    # Experiment 5 -----------------------------------------------------
    def _run_experiment_5(self) -> Dict[str, List[WorkflowRun]]:
        pipeline_with_rag = self._create_pipeline()
        pipeline_without_rag = self._create_pipeline()

        class NoRetrievalWorkflow(MultiAgentWorkflow):
            def _intents_for_step(self, step: str, task: TaskDefinition) -> List[str]:
                if step == "知识检索":
                    return []
                return super()._intents_for_step(step, task)

        workflow_with_rag = MultiAgentWorkflow(pipeline_with_rag, enable_memory=True, enable_replanner=True)
        workflow_without_rag = NoRetrievalWorkflow(pipeline_without_rag, enable_memory=True, enable_replanner=True)

        rag_runs: List[WorkflowRun] = []
        no_rag_runs: List[WorkflowRun] = []

        for dataset in self.datasets:
            tasks = [task for task in self.task_suites[dataset.name] if task.requires_retrieval or "dataset_bio_qa" in task.intents]
            for run_idx in range(self.runs_per_task):
                for task in tasks:
                    question = self._question_for_task(task, run_idx)
                    rag_runs.append(workflow_with_rag.run_task(dataset, task, question=question, run_idx=run_idx))
                    no_rag_runs.append(workflow_without_rag.run_task(dataset, task, question=question, run_idx=run_idx))

        return {"with_rag": rag_runs, "without_rag": no_rag_runs}

    # Experiment 6 -----------------------------------------------------
    def _run_experiment_6(self) -> Dict[str, np.ndarray]:
        classifier = RuleBasedIntentClassifier()
        classes = ["cell_annotation", "clustering_analysis", "pathway_analysis", "memory_query", "status_check", "dataset_bio_qa", "generic"]
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        matrix = np.zeros((len(classes), len(classes)), dtype=int)

        prompts = self._generate_intent_dataset(classes, total=200)
        for prompt, label in prompts:
            prediction = classifier.predict(prompt)
            matrix[label_to_idx[label], label_to_idx.get(prediction, label_to_idx["generic"])] += 1

        return {"confusion_matrix": matrix, "class_labels": np.array(classes)}

    def _generate_intent_dataset(self, classes: Sequence[str], total: int = 200) -> List[Tuple[str, str]]:
        templates: Dict[str, List[str]] = {
            "cell_annotation": [
                "请帮我注释这个簇的细胞类型",
                "使用marker基因标注细胞",
            ],
            "clustering_analysis": [
                "重新聚类并展示主要的簇",
                "运行scRNA聚类分析",
            ],
            "pathway_analysis": [
                "对免疫相关通路做富集",
                "看看KEGG通路的显著性",
            ],
            "memory_query": [
                "你还记得上次讨论的结果吗",
                "回顾之前的总结",
            ],
            "status_check": [
                "当前分析进展如何",
                "任务完成了吗",
            ],
            "dataset_bio_qa": [
                "根据文献解释这个细胞簇的功能",
                "检索相关研究给出答案",
            ],
            "generic": [
                "你好，最近怎么样",
                "谢谢你的帮助",
            ],
        }
        samples: List[Tuple[str, str]] = []
        per_class = max(1, total // len(classes))
        for label in classes:
            template_list = templates.get(label, templates["generic"])
            for idx in range(per_class):
                template = template_list[idx % len(template_list)]
                samples.append((template, label))
        return samples

    # Output generation ------------------------------------------------
    def _generate_outputs(
        self,
        exp1: Dict[str, List[WorkflowRun]],
        exp2: Dict[str, List[WorkflowRun]],
        exp3: Dict[str, List[WorkflowRun]],
        exp4: Dict[str, List[WorkflowRun]],
        exp5: Dict[str, List[WorkflowRun]],
        exp6: Dict[str, np.ndarray],
    ) -> None:
        fig_dir = self.output_dir / "figures"

        # Experiment 1 figures (grouped bar)
        grouped_multi = summarise_by_dataset(exp1["multi_agent"])
        grouped_baseline = summarise_by_dataset(exp1["baseline"])
        labels = list(grouped_multi.keys())
        metrics_multi = [grouped_multi[label] for label in labels]
        metrics_baseline = [grouped_baseline.get(label, AggregateMetrics(label,0,0,0,0)) for label in labels]

        plot_grouped_bar(metrics_multi, metrics_baseline, labels, fig_dir / "figure_3A_success.png", "success_rate", "任务成功率", "图3A: 成功率对比")
        plot_grouped_bar(metrics_multi, metrics_baseline, labels, fig_dir / "figure_3A_runtime.png", "avg_runtime", "平均运行时间 (秒)", "图3A: 运行时间对比")
        plot_grouped_bar(metrics_multi, metrics_baseline, labels, fig_dir / "figure_3A_clarity.png", "avg_clarity", "平均清晰度评分", "图3A: 可理解性对比")

        # Experiment 2 violin plot
        plan_distances = [run.plan_edit_distance for run in exp2["planner"] if run.plan_edit_distance is not None]
        control_distances = [run.plan_edit_distance for run in exp2["no_planner"] if run.plan_edit_distance is not None]
        plot_violin_steps(plan_distances + control_distances, ["启用计划"] * len(plan_distances) + ["禁用计划"] * len(control_distances), fig_dir / "figure_3B_violin.png")

        # Experiment 3 survival curve
        attempts = [max(1, run.plan_modifications + 1) for run in exp3["with_replanner"]]
        recoveries = [1 if run.recovered_from_failure else 0 for run in exp3["with_replanner"]]
        plot_survival_curve(attempts, recoveries, fig_dir / "figure_3C_survival.png")

        # Experiment 4 radar
        memory_metrics = summarise_runs("memory", exp4["with_memory"])
        no_memory_metrics = summarise_runs("no_memory", exp4["without_memory"])
        radar_labels = ["记忆精确率", "记忆召回率", "满意度(清晰度)"]
        radar_with = [memory_metrics.memory_precision, memory_metrics.memory_recall, memory_metrics.avg_clarity]
        radar_without = [no_memory_metrics.memory_precision, no_memory_metrics.memory_recall, no_memory_metrics.avg_clarity]
        plot_radar_chart(radar_labels, radar_with, radar_without, fig_dir / "figure_3D_radar.png", "图3D: 记忆贡献")

        # Experiment 5 scatter
        rag_metrics = summarise_runs("with_rag", exp5["with_rag"])
        no_rag_metrics = summarise_runs("without_rag", exp5["without_rag"])
        plot_accuracy_latency_scatter(
            [rag_metrics.knowledge_accuracy, no_rag_metrics.knowledge_accuracy],
            [rag_metrics.avg_runtime, no_rag_metrics.avg_runtime],
            ["启用检索", "禁用检索"],
            fig_dir / "figure_3E_scatter.png",
            "图3E: 知识检索准确率与延迟",
        )

        # Experiment 6 confusion matrix
        matrix = exp6["confusion_matrix"]
        labels_cm = exp6["class_labels"]
        plot_confusion_matrix(matrix, labels_cm, fig_dir / "figure_3F_confusion.png", "图3F: 意图识别混淆矩阵")

        # Table 2 metrics
        table_rows = []
        table_rows.extend(build_table([summarise_runs("多智能体", exp1["multi_agent"]), summarise_runs("基线", exp1["baseline"])]))
        table_rows.extend(build_table([summarise_runs("规划", exp2["planner"]), summarise_runs("无规划", exp2["no_planner"])]))
        table_rows.extend(build_table([summarise_runs("重规划", exp3["with_replanner"]), summarise_runs("无重规划", exp3["without_replanner"])]))
        table_rows.extend(build_table([summarise_runs("记忆", exp4["with_memory"]), summarise_runs("无记忆", exp4["without_memory"])]))
        table_rows.extend(build_table([summarise_runs("检索", exp5["with_rag"]), summarise_runs("无检索", exp5["without_rag"])]))
        export_table(table_rows, self.output_dir / "table_2_metrics.csv")

    # CLI --------------------------------------------------------------


def run_all_experiments(
    dataset_paths: List[str],
    output_dir: str = "results",
    runs_per_task: int = 20,
    seed: Optional[int] = None,
) -> ExperimentOutputs:
    datasets = iter_datasets(dataset_paths)
    suite = ExperimentSuite(datasets, output_dir=Path(output_dir), runs_per_task=runs_per_task, random_seed=seed)
    return suite.run()


def main(
    dataset: List[str] = typer.Option(..., help="路径到.h5ad文件，可重复传入多次。"),
    output_dir: str = typer.Option("results", help="实验结果输出目录"),
    runs_per_task: int = typer.Option(20, help="每个任务的运行次数"),
    seed: Optional[int] = typer.Option(None, help="随机种子"),
) -> None:
    run_all_experiments(dataset, output_dir=output_dir, runs_per_task=runs_per_task, seed=seed)


if __name__ == "__main__":
    typer.run(main)
