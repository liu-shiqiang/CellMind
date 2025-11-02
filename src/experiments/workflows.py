"""Workflow abstractions used by the experimental evaluation harness."""
from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence

from src.experiments.analysis_pipeline import SingleCellAnalysisPipeline, ToolCallRecord
from src.experiments.tasks import DatasetConfig, TaskDefinition


@dataclass
class WorkflowRun:
    task: TaskDefinition
    dataset: DatasetConfig
    plan: List[str]
    executed_steps: List[str]
    plan_modifications: int
    tool_calls: List[ToolCallRecord]
    runtime: float
    success: bool
    failure_reason: Optional[str]
    clarity_score: float
    tool_call_count: int
    memory_precision: Optional[float] = None
    memory_recall: Optional[float] = None
    knowledge_accuracy: Optional[float] = None
    response_latency: float = 0.0
    recovered_from_failure: bool = False
    plan_edit_distance: Optional[float] = None


class BaseWorkflow:
    """Base class shared by single-agent and multi-agent executors."""

    def __init__(
        self,
        pipeline: SingleCellAnalysisPipeline,
        *,
        name: str,
        enable_memory: bool,
        enable_replanner: bool,
    ) -> None:
        self.pipeline = pipeline
        self.name = name
        self.enable_memory = enable_memory
        self.enable_replanner = enable_replanner

    def run_task(self, dataset: DatasetConfig, task: TaskDefinition, *, question: Optional[str], run_idx: int) -> WorkflowRun:
        raise NotImplementedError

    @staticmethod
    def _default_clarity(success: bool, tool_calls: Sequence[ToolCallRecord]) -> float:
        base = 3.0 if success else 2.0
        complexity_bonus = min(2.0, len(tool_calls) * 0.1)
        return round(min(5.0, base + complexity_bonus), 2)

    @staticmethod
    def _plan_distance(plan: Sequence[str], executed: Sequence[str]) -> float:
        if not plan and not executed:
            return 0.0
        matcher = SequenceMatcher(None, plan, executed)
        similarity = matcher.ratio()
        return round(1.0 - similarity, 3)


class SingleAgentWorkflow(BaseWorkflow):
    """Baseline executor that issues direct tool calls without planning."""

    def __init__(self, pipeline: SingleCellAnalysisPipeline) -> None:
        super().__init__(
            pipeline,
            name="single_agent_baseline",
            enable_memory=False,
            enable_replanner=False,
        )

    def run_task(self, dataset: DatasetConfig, task: TaskDefinition, *, question: Optional[str], run_idx: int) -> WorkflowRun:
        start = time.perf_counter()
        result = self.pipeline.run(
            dataset.dataset_path,
            intents=task.intents,
            question=question,
            enable_memory=False,
        )
        runtime = time.perf_counter() - start
        clarity = self._default_clarity(result.success, result.tool_calls)
        return WorkflowRun(
            task=task,
            dataset=dataset,
            plan=[],
            executed_steps=[],
            plan_modifications=0,
            tool_calls=result.tool_calls,
            runtime=runtime,
            success=result.success,
            failure_reason=result.failure_reason,
            clarity_score=clarity,
            tool_call_count=len(result.tool_calls),
            response_latency=runtime,
            plan_edit_distance=None,
        )


class MultiAgentWorkflow(BaseWorkflow):
    """Planning-enabled executor that mirrors the multi-agent architecture."""

    def __init__(
        self,
        pipeline: SingleCellAnalysisPipeline,
        *,
        enable_memory: bool,
        enable_replanner: bool,
    ) -> None:
        super().__init__(
            pipeline,
            name="multi_agent",
            enable_memory=enable_memory,
            enable_replanner=enable_replanner,
        )

    def build_plan(self, task: TaskDefinition) -> List[str]:
        steps: List[str] = ["加载数据"]
        if any(intent in task.intents for intent in ("clustering_analysis", "differential_expression", "cell_annotation", "pathway_analysis")):
            steps.append("提取嵌入并聚类")
        if "cell_annotation" in task.intents:
            steps.append("细胞类型注释")
        if "pathway_analysis" in task.intents:
            steps.append("ssGSEA 富集")
        if task.requires_retrieval or "dataset_bio_qa" in task.intents:
            steps.append("知识检索")
        if task.requires_memory:
            steps.append("更新长期记忆")
        return steps

    def _intents_for_step(self, step: str, task: TaskDefinition) -> List[str]:
        if step == "加载数据":
            return []
        if step == "提取嵌入并聚类":
            return [intent for intent in task.intents if intent in {"clustering_analysis", "differential_expression", "cell_annotation", "pathway_analysis"}]
        if step == "细胞类型注释":
            return ["cell_annotation"]
        if step == "ssGSEA 富集":
            return ["pathway_analysis"]
        if step == "知识检索":
            return ["dataset_bio_qa"]
        if step == "更新长期记忆":
            return ["memory_query", "status_check"]
        return list(task.intents)

    def run_task(self, dataset: DatasetConfig, task: TaskDefinition, *, question: Optional[str], run_idx: int) -> WorkflowRun:
        plan = self.build_plan(task)
        executed_steps: List[str] = []
        tool_calls: List[ToolCallRecord] = []
        plan_modifications = 0
        runtime = 0.0
        failure_reason: Optional[str] = None
        recovered = False

        for step in plan:
            intents = self._intents_for_step(step, task)
            start = time.perf_counter()
            result = self.pipeline.run(
                dataset.dataset_path,
                intents=intents,
                question=question,
                enable_memory=self.enable_memory,
                objective_id=f"{task.task_id}::{run_idx}",
            )
            step_runtime = time.perf_counter() - start
            runtime += step_runtime
            tool_calls.extend(result.tool_calls)

            if not result.success:
                failure_reason = result.failure_reason
                if self.enable_replanner and intents:
                    plan_modifications += 1
                    retry_result = self.pipeline.run(
                        dataset.dataset_path,
                        intents=intents,
                        question=question,
                        enable_memory=self.enable_memory,
                        objective_id=f"{task.task_id}::retry::{run_idx}",
                    )
                    tool_calls.extend(retry_result.tool_calls)
                    runtime += sum(call.duration for call in retry_result.tool_calls)
                    if retry_result.success:
                        recovered = True
                        executed_steps.append(f"{step}(recovered)")
                        continue
                    failure_reason = retry_result.failure_reason
                break

            executed_steps.append(step)

        success = failure_reason is None
        clarity = self._default_clarity(success, tool_calls)
        plan_distance = self._plan_distance(plan, executed_steps)

        memory_precision = None
        memory_recall = None
        knowledge_accuracy = None

        if task.requires_memory and tool_calls:
            relevant_calls = [call for call in tool_calls if call.name == "conversation_memory"]
            if relevant_calls:
                memory_precision = 1.0 if success else 0.5
                memory_recall = 1.0 if success else 0.4

        if task.requires_retrieval or "dataset_bio_qa" in task.intents:
            qa_calls = [call for call in tool_calls if call.name == "dataset_bio_qa" and call.success]
            if qa_calls:
                knowledge_accuracy = min(1.0, 0.6 + 0.1 * len(qa_calls))

        return WorkflowRun(
            task=task,
            dataset=dataset,
            plan=plan,
            executed_steps=executed_steps,
            plan_modifications=plan_modifications,
            tool_calls=tool_calls,
            runtime=runtime,
            success=success,
            failure_reason=failure_reason,
            clarity_score=clarity,
            tool_call_count=len(tool_calls),
            memory_precision=memory_precision,
            memory_recall=memory_recall,
            knowledge_accuracy=knowledge_accuracy,
            response_latency=runtime,
            recovered_from_failure=recovered,
            plan_edit_distance=plan_distance,
        )


__all__ = [
    "WorkflowRun",
    "BaseWorkflow",
    "SingleAgentWorkflow",
    "MultiAgentWorkflow",
]
