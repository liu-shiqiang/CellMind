"""Execution primitives for running Genomix analysis workflows in experiments."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.memory.conversation_memory import ConversationMemoryStore
from src.tools.annotate_with_markers import annotate_with_markers
from src.tools.cluster_diff import cluster_and_diff
from src.tools.dataset_qa import retrieve_bio_context
from src.tools.enrichment_analysis.ssgsea import run_ssgsea_enrichment
from src.tools.extract_embeddings import extract_embeddings_with_scgpt
from src.tools.load_h5ad import load_h5ad_data


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""

    name: str
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisArtifacts:
    """Collection of artefacts generated during analysis."""

    dataset_name: str
    raw_path: Path
    work_dir: Optional[Path] = None
    preprocessed_path: Optional[Path] = None
    embedding_path: Optional[Path] = None
    clustered_path: Optional[Path] = None
    diff_gene_path: Optional[Path] = None
    annotated_path: Optional[Path] = None
    annotation_summary: Optional[Path] = None
    enrichment_outputs: Dict[str, Path] = field(default_factory=dict)
    qa_outputs: List[Path] = field(default_factory=list)
    memory_reports: List[Path] = field(default_factory=list)

    def clone(self) -> "AnalysisArtifacts":
        return AnalysisArtifacts(
            dataset_name=self.dataset_name,
            raw_path=self.raw_path,
            work_dir=self.work_dir,
            preprocessed_path=self.preprocessed_path,
            embedding_path=self.embedding_path,
            clustered_path=self.clustered_path,
            diff_gene_path=self.diff_gene_path,
            annotated_path=self.annotated_path,
            annotation_summary=self.annotation_summary,
            enrichment_outputs=dict(self.enrichment_outputs),
            qa_outputs=list(self.qa_outputs),
            memory_reports=list(self.memory_reports),
        )


@dataclass
class PipelineResult:
    """Result object returned by :class:`SingleCellAnalysisPipeline`."""

    artifacts: AnalysisArtifacts
    tool_calls: List[ToolCallRecord]
    success: bool
    failure_reason: Optional[str] = None


class ToolFailureInjector:
    """Optional helper to simulate tool failures for robustness experiments."""

    def __init__(self, failure_rate: float = 0.0, seed: Optional[int] = None) -> None:
        self.failure_rate = float(max(0.0, min(1.0, failure_rate)))
        self.random_state = np.random.default_rng(seed)

    def should_fail(self, tool_name: str) -> bool:
        if self.failure_rate <= 0.0:
            return False
        return bool(self.random_state.random() < self.failure_rate)


class SingleCellAnalysisPipeline:
    """Utility class that orchestrates canonical tool execution for experiments."""

    def __init__(
        self,
        *,
        cache_intermediate: bool = True,
        memory_store: Optional[ConversationMemoryStore] = None,
        failure_injector: Optional[ToolFailureInjector] = None,
    ) -> None:
        self._cache_enabled = cache_intermediate
        self._cache: Dict[Path, AnalysisArtifacts] = {}
        self._memory = memory_store
        self._injector = failure_injector or ToolFailureInjector(0.0)

    # ------------------------------------------------------------------
    def run(
        self,
        dataset_path: Path,
        *,
        intents: Iterable[str],
        work_dir_override: Optional[Path] = None,
        question: Optional[str] = None,
        enable_memory: bool = True,
        objective_id: Optional[str] = None,
    ) -> PipelineResult:
        """Execute the minimum set of tools required for the given intents."""

        dataset_path = dataset_path.expanduser().resolve()
        base_artifacts = self._cache.get(dataset_path)
        artifacts = base_artifacts.clone() if base_artifacts else AnalysisArtifacts(
            dataset_name=dataset_path.stem,
            raw_path=dataset_path,
        )

        tool_calls: List[ToolCallRecord] = []
        failure_reason: Optional[str] = None

        try:
            self._ensure_loaded(artifacts, tool_calls, work_dir_override)
            if any(
                intent in ("clustering_analysis", "differential_expression", "cell_annotation", "pathway_analysis")
                for intent in intents
            ):
                self._ensure_embeddings(artifacts, tool_calls)
                self._ensure_clustering(artifacts, tool_calls)

            if "cell_annotation" in intents:
                self._ensure_annotation(artifacts, tool_calls)

            if "pathway_analysis" in intents:
                self._ensure_annotation(artifacts, tool_calls)
                self._ensure_enrichment(artifacts, tool_calls)

            if "dataset_bio_qa" in intents or "knowledge_retrieval" in intents:
                self._run_knowledge_retrieval(
                    artifacts,
                    tool_calls,
                    question or "请结合数据集给出关键的生物学发现。",
                )

            if "memory_query" in intents or "status_check" in intents:
                if enable_memory and self._memory is not None:
                    self._write_memory_report(
                        artifacts,
                        tool_calls,
                        objective_id or f"memory::{artifacts.dataset_name}",
                        question or "记录最新的分析进展。",
                    )
        except Exception as exc:  # pragma: no cover - defensive guard
            failure_reason = str(exc)

        success = failure_reason is None

        if success and self._cache_enabled:
            self._cache.setdefault(dataset_path, artifacts.clone())

        return PipelineResult(
            artifacts=artifacts,
            tool_calls=tool_calls,
            success=success,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    def _ensure_loaded(
        self,
        artifacts: AnalysisArtifacts,
        tool_calls: List[ToolCallRecord],
        work_dir_override: Optional[Path],
    ) -> None:
        if artifacts.work_dir and artifacts.preprocessed_path:
            return

        start = time.perf_counter()
        work_dir_param = work_dir_override or Path("data/experiments") / artifacts.dataset_name
        work_dir_param.mkdir(parents=True, exist_ok=True)

        if self._injector.should_fail("load_h5ad_data"):
            raise RuntimeError("Injected failure in load_h5ad_data")

        result = load_h5ad_data(file_path=str(artifacts.raw_path), output_dir=str(work_dir_param))
        duration = time.perf_counter() - start
        payload = json.loads(result)
        artifacts.work_dir = Path(payload["work_dir"]).expanduser().resolve()
        artifacts.preprocessed_path = Path(payload["preproc_path"]).expanduser().resolve()
        tool_calls.append(
            ToolCallRecord(
                name="load_h5ad_data",
                duration=duration,
                success=True,
                metadata={"work_dir": str(artifacts.work_dir)},
            )
        )

    def _ensure_embeddings(self, artifacts: AnalysisArtifacts, tool_calls: List[ToolCallRecord]) -> None:
        if artifacts.embedding_path and artifacts.embedding_path.exists():
            return

        if artifacts.preprocessed_path is None or artifacts.work_dir is None:
            raise RuntimeError("Attempted to compute embeddings before preprocessing.")

        start = time.perf_counter()
        if self._injector.should_fail("extract_embeddings_with_scgpt"):
            raise RuntimeError("Injected failure in extract_embeddings_with_scgpt")

        result = extract_embeddings_with_scgpt(
            preproc_path=str(artifacts.preprocessed_path),
            work_dir=str(artifacts.work_dir),
        )
        duration = time.perf_counter() - start
        payload = json.loads(result)
        artifacts.embedding_path = Path(payload["embeddings_path"]).expanduser().resolve()
        tool_calls.append(
            ToolCallRecord(
                name="extract_embeddings_with_scgpt",
                duration=duration,
                success=True,
                metadata={"embeddings_path": str(artifacts.embedding_path)},
            )
        )

    def _ensure_clustering(self, artifacts: AnalysisArtifacts, tool_calls: List[ToolCallRecord]) -> None:
        if artifacts.clustered_path and artifacts.diff_gene_path:
            return
        if artifacts.embedding_path is None or artifacts.work_dir is None:
            raise RuntimeError("Attempted clustering before embeddings were created.")

        start = time.perf_counter()
        if self._injector.should_fail("cluster_and_diff"):
            raise RuntimeError("Injected failure in cluster_and_diff")

        result = cluster_and_diff(
            embedding_path=str(artifacts.embedding_path),
            work_dir=str(artifacts.work_dir),
        )
        duration = time.perf_counter() - start
        payload = json.loads(result)
        artifacts.clustered_path = Path(payload["clustered_path"]).expanduser().resolve()
        artifacts.diff_gene_path = Path(payload["diff_gene_path"]).expanduser().resolve()
        tool_calls.append(
            ToolCallRecord(
                name="cluster_and_diff",
                duration=duration,
                success=True,
                metadata={
                    "clustered_path": str(artifacts.clustered_path),
                    "diff_gene_path": str(artifacts.diff_gene_path),
                },
            )
        )

    def _ensure_annotation(self, artifacts: AnalysisArtifacts, tool_calls: List[ToolCallRecord]) -> None:
        if artifacts.annotated_path and artifacts.annotation_summary:
            return
        if artifacts.clustered_path is None or artifacts.diff_gene_path is None:
            raise RuntimeError("Attempted annotation before clustering.")

        start = time.perf_counter()
        if self._injector.should_fail("annotate_with_markers"):
            raise RuntimeError("Injected failure in annotate_with_markers")

        result = annotate_with_markers(
            clustered_path=str(artifacts.clustered_path),
            diff_gene_path=str(artifacts.diff_gene_path),
            work_dir=str(artifacts.work_dir),
        )
        duration = time.perf_counter() - start
        payload = json.loads(result)
        artifacts.annotated_path = Path(payload["annoted_Path"]).expanduser().resolve()
        artifacts.annotation_summary = Path(payload["anno_result"]).expanduser().resolve()
        tool_calls.append(
            ToolCallRecord(
                name="annotate_with_markers",
                duration=duration,
                success=True,
                metadata={
                    "annotated_path": str(artifacts.annotated_path),
                    "summary_path": str(artifacts.annotation_summary),
                },
            )
        )

    def _ensure_enrichment(self, artifacts: AnalysisArtifacts, tool_calls: List[ToolCallRecord]) -> None:
        if "ssgsea" in artifacts.enrichment_outputs:
            return
        if artifacts.annotated_path is None or artifacts.work_dir is None:
            raise RuntimeError("Attempted enrichment without annotations.")

        start = time.perf_counter()
        if self._injector.should_fail("run_ssgsea_enrichment"):
            raise RuntimeError("Injected failure in run_ssgsea_enrichment")

        msg = run_ssgsea_enrichment(
            file_path=str(artifacts.annotated_path),
            work_dir=str(artifacts.work_dir),
            gene_set="KEGG",
        )
        duration = time.perf_counter() - start
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            payload = {}
        summary_path = None
        if isinstance(payload, dict):
            summary_value = payload.get("summary") or payload.get("summary_path")
            if summary_value:
                summary_path = Path(summary_value).expanduser().resolve()
        if summary_path is None:
            summary_path = artifacts.work_dir / "enrichment" / "ssgsea" / "ssgsea_summary.txt"
        artifacts.enrichment_outputs["ssgsea"] = summary_path
        tool_calls.append(
            ToolCallRecord(
                name="run_ssgsea_enrichment",
                duration=duration,
                success=True,
                metadata={"summary_path": str(summary_path)},
            )
        )

    def _run_knowledge_retrieval(
        self,
        artifacts: AnalysisArtifacts,
        tool_calls: List[ToolCallRecord],
        question: str,
    ) -> None:
        if artifacts.work_dir is None:
            raise RuntimeError("Knowledge retrieval requires a work directory.")

        start = time.perf_counter()
        if self._injector.should_fail("dataset_bio_qa"):
            raise RuntimeError("Injected failure in dataset_bio_qa")

        context_payload = retrieve_bio_context(
            question=question,
            work_dir=str(artifacts.work_dir),
            top_k_local=3,
            top_k_pubmed=3,
        )
        duration = time.perf_counter() - start
        output_path = artifacts.work_dir / f"qa_context_{len(artifacts.qa_outputs)+1}.json"
        output_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.qa_outputs.append(output_path)
        tool_calls.append(
            ToolCallRecord(
                name="dataset_bio_qa",
                duration=duration,
                success=True,
                metadata={"qa_context": str(output_path)},
            )
        )

    def _write_memory_report(
        self,
        artifacts: AnalysisArtifacts,
        tool_calls: List[ToolCallRecord],
        objective_id: str,
        question: str,
    ) -> None:
        if self._memory is None or artifacts.work_dir is None:
            return

        start = time.perf_counter()
        context = self._memory.load_context(thread_id=artifacts.dataset_name, objective=question)
        summary_path = artifacts.work_dir / f"memory_status_{len(artifacts.memory_reports)+1}.json"
        payload = {
            "objective": question,
            "thread_id": artifacts.dataset_name,
            "records": [record.__dict__ for record in context.records],
            "summary": context.summary,
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.memory_reports.append(summary_path)
        tool_calls.append(
            ToolCallRecord(
                name="conversation_memory",
                duration=time.perf_counter() - start,
                success=True,
                metadata={"memory_report": str(summary_path)},
            )
        )

        self._memory.store_conversation(
            thread_id=artifacts.dataset_name,
            objective=question,
            messages=[],
            result_text=f"生成了工作目录 {artifacts.work_dir} 的状态报告。",
            metadata={
                "project_state": {
                    "work_dir": str(artifacts.work_dir),
                    "last_update": summary_path.name,
                }
            },
        )


__all__ = [
    "AnalysisArtifacts",
    "PipelineResult",
    "SingleCellAnalysisPipeline",
    "ToolCallRecord",
    "ToolFailureInjector",
]
