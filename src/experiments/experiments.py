"""Experiment orchestration for evaluating the Genomix multi-agent system."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.agent.agent_new import (
    AgentRuntimeConfig,
    FailureInjectionConfig,
    _normalize_intent_label,
    run_objective,
)
from .agent_runner import AgentRunResult, run_agent_task
from .metrics import (
    build_confusion_matrix,
    classification_report,
    failure_recovery_table,
    knowledge_accuracy,
    memory_scores,
    results_dataframe,
    summarise_by_config,
    summarise_by_task,
)
from .plots import (
    plot_accuracy_latency,
    plot_confusion_matrix,
    plot_grouped_bars,
    plot_radar_memory,
    plot_survival_curve,
    plot_violin_tool_calls,
)
from .task_library import (
    IntentBenchmarkSample,
    TaskSpec,
    build_intent_benchmark,
    composite_task_ids,
    dataset_task_ids,
    knowledge_task_ids,
    memory_task_sequences,
    task_index,
)


@dataclass
class ExperimentResult:
    all_runs: List[AgentRunResult]
    intent_predictions: pd.DataFrame


class ExperimentSuite:
    def __init__(
        self,
        dataset_path: Optional[Path],
        output_dir: Path,
        runs_per_task: int = 1,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.runs_per_task = max(1, runs_per_task)
        self.seed = seed
        self.rng = random.Random(seed)
        self.tasks: Dict[str, TaskSpec] = task_index()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment runners
    # ------------------------------------------------------------------
    def _experiment_baseline_vs_multi(self) -> List[AgentRunResult]:
        dataset_tasks = [self.tasks[task_id] for task_id in dataset_task_ids()]
        default_config = AgentRuntimeConfig()
        baseline_config = AgentRuntimeConfig(
            planner_mode="linear",
            enable_replanner=False,
            enable_memory=False,
            enable_rag=False,
            enable_dataset_qa=False,
        )

        runs: List[AgentRunResult] = []
        for run_idx in range(self.runs_per_task):
            base_thread = f"exp1-{self.seed}-{run_idx}"
            for task in dataset_tasks:
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        default_config,
                        "multi_agent",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        baseline_config,
                        "linear_baseline",
                        run_idx,
                        base_thread,
                    )
                )
        return runs

    def _experiment_planner_ablation(self) -> List[AgentRunResult]:
        composite_tasks = [self.tasks[task_id] for task_id in composite_task_ids()]
        default_config = AgentRuntimeConfig()
        ablated_config = AgentRuntimeConfig(
            planner_mode="disabled",
            enable_replanner=False,
            enable_memory=False,
            enable_rag=False,
        )
        runs: List[AgentRunResult] = []
        for run_idx in range(self.runs_per_task):
            base_thread = f"exp2-{self.seed}-{run_idx}"
            for task in composite_tasks:
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        default_config,
                        "multi_agent",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        ablated_config,
                        "no_planner",
                        run_idx,
                        base_thread,
                    )
                )
        return runs

    def _experiment_replanner(self) -> List[AgentRunResult]:
        dataset_tasks = [self.tasks[task_id] for task_id in dataset_task_ids()]
        runs: List[AgentRunResult] = []
        for run_idx in range(self.runs_per_task):
            failure_cfg = FailureInjectionConfig(
                tool_names=[
                    "load_h5ad_data",
                    "cluster_and_diff",
                    "annotate_with_markers",
                    "run_ssgsea_enrichment",
                ],
                rate=0.3,
                seed=self.seed + run_idx,
            )
            base_thread = f"exp3-{self.seed}-{run_idx}"
            with_replanner = AgentRuntimeConfig(
                failure_injection=failure_cfg,
                max_replan_attempts=4,
            )
            without_replanner = AgentRuntimeConfig(
                failure_injection=failure_cfg,
                enable_replanner=False,
                max_replan_attempts=1,
            )
            for task in dataset_tasks:
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        with_replanner,
                        "replanner",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        without_replanner,
                        "no_replanner",
                        run_idx,
                        base_thread,
                    )
                )
        return runs

    def _experiment_memory(self) -> List[AgentRunResult]:
        memory_pairs = memory_task_sequences()
        runs: List[AgentRunResult] = []
        memory_config = AgentRuntimeConfig(enable_memory=True)
        no_memory_config = AgentRuntimeConfig(enable_memory=False)

        for run_idx in range(self.runs_per_task):
            base_thread = f"exp4-{self.seed}-{run_idx}"
            for seed_task, follow_up in memory_pairs:
                thread_id = f"{base_thread}-{seed_task.task_id}"
                runs.append(
                    run_agent_task(
                        seed_task,
                        self.dataset_path,
                        memory_config,
                        "memory_enabled",
                        run_idx,
                        thread_id,
                    )
                )
                runs.append(
                    run_agent_task(
                        follow_up,
                        self.dataset_path,
                        memory_config,
                        "memory_enabled",
                        run_idx,
                        thread_id,
                    )
                )

                cold_thread = f"{base_thread}-cold-{seed_task.task_id}"
                runs.append(
                    run_agent_task(
                        seed_task,
                        self.dataset_path,
                        no_memory_config,
                        "memory_disabled",
                        run_idx,
                        cold_thread,
                    )
                )
                runs.append(
                    run_agent_task(
                        follow_up,
                        self.dataset_path,
                        no_memory_config,
                        "memory_disabled",
                        run_idx,
                        cold_thread,
                    )
                )
        return runs

    def _experiment_knowledge(self) -> List[AgentRunResult]:
        knowledge_tasks = [self.tasks[task_id] for task_id in knowledge_task_ids()]
        with_rag = AgentRuntimeConfig()
        without_rag = AgentRuntimeConfig(enable_dataset_qa=False, enable_rag=False)

        runs: List[AgentRunResult] = []
        for run_idx in range(self.runs_per_task):
            base_thread = f"exp5-{self.seed}-{run_idx}"
            for task in knowledge_tasks:
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        with_rag,
                        "rag_enabled",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    run_agent_task(
                        task,
                        self.dataset_path,
                        without_rag,
                        "rag_disabled",
                        run_idx,
                        base_thread,
                    )
                )
        return runs

    def _experiment_intent(self) -> pd.DataFrame:
        samples: List[IntentBenchmarkSample] = build_intent_benchmark(200)
        config = AgentRuntimeConfig(
            planner_mode="disabled",
            enable_memory=False,
            enable_rag=False,
            enable_dataset_qa=False,
            allow_tool_execution=False,
            track_metrics=False,
        )
        rows: List[Dict[str, str]] = []
        for idx, sample in enumerate(samples):
            thread_id = f"intent-{self.seed}-{idx}"
            final_text, final_state = asyncio.run(
                run_objective(
                    objective=sample.prompt,
                    input_files=None,
                    thread_id=thread_id,
                    runtime_config=config,
                    return_diagnostics=True,
                )
            )
            state_dict = final_state or {}
            recognized = state_dict.get("recognized_intents", []) or []
            if recognized:
                item = recognized[0]
                if isinstance(item, dict):
                    predicted = _normalize_intent_label(item.get("intent"))
                else:
                    predicted = _normalize_intent_label(getattr(item, "intent", str(item)))
            else:
                predicted = "unknown"
            rows.append(
                {
                    "prompt": sample.prompt,
                    "ground_truth": sample.label,
                    "predicted": predicted,
                    "response": final_text,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def run(self) -> ExperimentResult:
        all_runs: List[AgentRunResult] = []
        all_runs.extend(self._experiment_baseline_vs_multi())
        all_runs.extend(self._experiment_planner_ablation())
        all_runs.extend(self._experiment_replanner())
        all_runs.extend(self._experiment_memory())
        all_runs.extend(self._experiment_knowledge())

        intent_predictions = self._experiment_intent()

        self._export_results(all_runs, intent_predictions)
        return ExperimentResult(all_runs=all_runs, intent_predictions=intent_predictions)

    # ------------------------------------------------------------------
    def _export_results(
        self,
        results: List[AgentRunResult],
        intent_predictions: pd.DataFrame,
    ) -> None:
        df = results_dataframe(results)
        df.to_csv(self.output_dir / "aggregated_results.csv", index=False)

        summary_config = summarise_by_config(df)
        summary_config.to_csv(self.output_dir / "summary_by_config.csv", index=False)

        summary_task = summarise_by_task(df)
        summary_task.to_csv(self.output_dir / "summary_by_task.csv", index=False)

        plot_grouped_bars(summary_config, self.output_dir / "fig_3a_grouped_bars.png")
        planner_subset = df[df["config"].isin(["multi_agent", "no_planner"])]
        if not planner_subset.empty:
            plot_violin_tool_calls(
                planner_subset,
                self.output_dir / "fig_3b_planner_violin.png",
            )

        survival = failure_recovery_table(df)
        if not survival.empty:
            plot_survival_curve(survival, self.output_dir / "fig_3c_survival.png")

        memory_df = memory_scores(df)
        if not memory_df.empty:
            plot_radar_memory(memory_df, self.output_dir / "fig_3d_memory_radar.png")

        knowledge_df = knowledge_accuracy(df)
        if not knowledge_df.empty:
            plot_accuracy_latency(knowledge_df, self.output_dir / "fig_3e_accuracy_latency.png")

        if not intent_predictions.empty:
            confusion = build_confusion_matrix(
                intent_predictions["ground_truth"], intent_predictions["predicted"]
            )
            plot_confusion_matrix(confusion, self.output_dir / "fig_3f_confusion.png")
            intent_predictions.to_csv(
                self.output_dir / "intent_predictions.csv", index=False
            )
            classification_report(confusion).to_csv(
                self.output_dir / "intent_classification_report.csv",
                index=False,
            )

        metadata = {
            "runs_per_task": self.runs_per_task,
            "seed": self.seed,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "total_runs": len(results),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )


__all__ = ["ExperimentSuite", "ExperimentResult"]
