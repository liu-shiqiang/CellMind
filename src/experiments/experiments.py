"""Experiment orchestration for evaluating the Genomix multi-agent system."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self.run_log_dir = self.output_dir / "run_logs"
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment runners
    # ------------------------------------------------------------------
    def _log_json_line(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, default=self._json_default)
                + "\n"
            )

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return str(obj)

    def _record_run_result(self, result: AgentRunResult, experiment: str) -> None:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "task_id": result.task.task_id,
            "task_category": result.task.category,
            "task_difficulty": result.task.difficulty,
            "requires_dataset": result.task.requires_dataset,
            "task_objective": result.task.objective,
            "config": result.config_name,
            "run_index": result.run_index,
            "thread_id": result.thread_id,
            "success": result.success,
            "execution_status": result.execution_status,
            "duration_sec": result.duration_sec,
            "metrics": result.metrics,
            "tool_calls": result.tool_calls,
            "tool_errors": result.tool_errors,
            "planner_invocations": result.planner_invocations,
            "plan_regenerations": result.plan_regenerations,
            "replanner_invocations": result.replanner_invocations,
            "keyword_hits": result.keyword_hits,
            "keyword_total": result.keyword_total,
            "final_message": result.final_message,
        }
        path = self.run_log_dir / f"{experiment}.jsonl"
        self._log_json_line(path, log_entry)

    def _record_task_exception(
        self,
        experiment: str,
        task: TaskSpec,
        config_name: str,
        run_index: int,
        base_thread_id: str,
        error: Exception,
    ) -> None:
        thread_id = f"{base_thread_id}-{task.task_id}-{config_name}-{run_index}"
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "task_id": task.task_id,
            "task_category": task.category,
            "task_difficulty": task.difficulty,
            "requires_dataset": task.requires_dataset,
            "task_objective": task.objective,
            "config": config_name,
            "run_index": run_index,
            "thread_id": thread_id,
            "success": False,
            "execution_status": "exception",
            "duration_sec": None,
            "metrics": {},
            "tool_calls": 0,
            "tool_errors": 0,
            "planner_invocations": 0,
            "plan_regenerations": 0,
            "replanner_invocations": 0,
            "keyword_hits": 0,
            "keyword_total": len(task.expected_keywords),
            "final_message": "",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        path = self.run_log_dir / f"{experiment}.jsonl"
        self._log_json_line(path, log_entry)

    def _execute_task(
        self,
        experiment: str,
        task: TaskSpec,
        runtime_config: AgentRuntimeConfig,
        config_name: str,
        run_index: int,
        base_thread_id: str,
    ) -> AgentRunResult:
        try:
            result = run_agent_task(
                task,
                self.dataset_path,
                runtime_config,
                config_name,
                run_index,
                base_thread_id,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._record_task_exception(
                experiment,
                task,
                config_name,
                run_index,
                base_thread_id,
                exc,
            )
            raise
        self._record_run_result(result, experiment)
        return result

    def _record_intent_prediction(
        self,
        sample: IntentBenchmarkSample,
        predicted: str,
        response: str,
        index: int,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": "experiment6_intent",
            "sample_index": index,
            "prompt": sample.prompt,
            "ground_truth": sample.label,
            "predicted": predicted,
            "is_correct": predicted == sample.label,
            "response": response,
        }
        self._log_json_line(self.run_log_dir / "experiment6_intent.jsonl", payload)

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
                    self._execute_task(
                        "experiment1_baseline_vs_multi",
                        task,
                        default_config,
                        "multi_agent",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment1_baseline_vs_multi",
                        task,
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
                    self._execute_task(
                        "experiment2_planner_ablation",
                        task,
                        default_config,
                        "multi_agent",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment2_planner_ablation",
                        task,
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
                    self._execute_task(
                        "experiment3_replanner",
                        task,
                        with_replanner,
                        "replanner",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment3_replanner",
                        task,
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
                    self._execute_task(
                        "experiment4_memory",
                        seed_task,
                        memory_config,
                        "memory_enabled",
                        run_idx,
                        thread_id,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment4_memory",
                        follow_up,
                        memory_config,
                        "memory_enabled",
                        run_idx,
                        thread_id,
                    )
                )

                cold_thread = f"{base_thread}-cold-{seed_task.task_id}"
                runs.append(
                    self._execute_task(
                        "experiment4_memory",
                        seed_task,
                        no_memory_config,
                        "memory_disabled",
                        run_idx,
                        cold_thread,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment4_memory",
                        follow_up,
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
                    self._execute_task(
                        "experiment5_knowledge",
                        task,
                        with_rag,
                        "rag_enabled",
                        run_idx,
                        base_thread,
                    )
                )
                runs.append(
                    self._execute_task(
                        "experiment5_knowledge",
                        task,
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
            self._record_intent_prediction(sample, predicted, final_text, idx)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Public runners used by the dedicated experiment scripts
    # ------------------------------------------------------------------
    def run_experiment1(self) -> List[AgentRunResult]:
        runs = self._experiment_baseline_vs_multi()
        self._export_results(
            runs,
            pd.DataFrame(),
            experiment_name="experiment1_baseline_vs_multi",
        )
        return runs

    def run_experiment2(self) -> List[AgentRunResult]:
        runs = self._experiment_planner_ablation()
        self._export_results(
            runs,
            pd.DataFrame(),
            experiment_name="experiment2_planner_ablation",
        )
        return runs

    def run_experiment3(self) -> List[AgentRunResult]:
        runs = self._experiment_replanner()
        self._export_results(
            runs,
            pd.DataFrame(),
            experiment_name="experiment3_replanner",
        )
        return runs

    def run_experiment4(self) -> List[AgentRunResult]:
        runs = self._experiment_memory()
        self._export_results(
            runs,
            pd.DataFrame(),
            experiment_name="experiment4_memory",
        )
        return runs

    def run_experiment5(self) -> List[AgentRunResult]:
        runs = self._experiment_knowledge()
        self._export_results(
            runs,
            pd.DataFrame(),
            experiment_name="experiment5_knowledge",
        )
        return runs

    def run_experiment6(self) -> pd.DataFrame:
        intent_predictions = self._experiment_intent()
        self._export_results(
            [],
            intent_predictions,
            experiment_name="experiment6_intent",
        )
        return intent_predictions

    def run(self) -> ExperimentResult:
        all_runs: List[AgentRunResult] = []
        all_runs.extend(self.run_experiment1())
        all_runs.extend(self.run_experiment2())
        all_runs.extend(self.run_experiment3())
        all_runs.extend(self.run_experiment4())
        all_runs.extend(self.run_experiment5())

        intent_predictions = self.run_experiment6()

        self._export_results(all_runs, intent_predictions, experiment_name="all")
        return ExperimentResult(all_runs=all_runs, intent_predictions=intent_predictions)

    # ------------------------------------------------------------------
    def _export_results(
        self,
        results: List[AgentRunResult],
        intent_predictions: pd.DataFrame,
        experiment_name: str,
    ) -> None:
        output_dir = self.output_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        df = results_dataframe(results) if results else pd.DataFrame()
        if not df.empty:
            df.to_csv(output_dir / "aggregated_results.csv", index=False)

            summary_config = summarise_by_config(df)
            summary_config.to_csv(output_dir / "summary_by_config.csv", index=False)

            summary_task = summarise_by_task(df)
            summary_task.to_csv(output_dir / "summary_by_task.csv", index=False)

            if experiment_name.startswith("experiment1") or experiment_name == "all":
                plot_grouped_bars(summary_config, output_dir / "fig_3a_grouped_bars.png")

            if experiment_name.startswith("experiment2") or experiment_name == "all":
                planner_subset = df[df["config"].isin(["multi_agent", "no_planner"])]
                if not planner_subset.empty:
                    plot_violin_tool_calls(
                        planner_subset,
                        output_dir / "fig_3b_planner_violin.png",
                    )

            if experiment_name.startswith("experiment3") or experiment_name == "all":
                survival = failure_recovery_table(df)
                if not survival.empty:
                    plot_survival_curve(survival, output_dir / "fig_3c_survival.png")

            if experiment_name.startswith("experiment4") or experiment_name == "all":
                memory_df = memory_scores(df)
                if not memory_df.empty:
                    plot_radar_memory(memory_df, output_dir / "fig_3d_memory_radar.png")

            if experiment_name.startswith("experiment5") or experiment_name == "all":
                knowledge_df = knowledge_accuracy(df)
                if not knowledge_df.empty:
                    plot_accuracy_latency(
                        knowledge_df, output_dir / "fig_3e_accuracy_latency.png"
                    )

        if (experiment_name.startswith("experiment6") or experiment_name == "all") and not intent_predictions.empty:
            confusion = build_confusion_matrix(
                intent_predictions["ground_truth"], intent_predictions["predicted"]
            )
            plot_confusion_matrix(confusion, output_dir / "fig_3f_confusion.png")
            intent_predictions.to_csv(output_dir / "intent_predictions.csv", index=False)
            classification_report(confusion).to_csv(
                output_dir / "intent_classification_report.csv",
                index=False,
            )

        metadata = {
            "runs_per_task": self.runs_per_task,
            "seed": self.seed,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "total_runs": len(results),
            "experiment": experiment_name,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )


__all__ = ["ExperimentSuite", "ExperimentResult"]
