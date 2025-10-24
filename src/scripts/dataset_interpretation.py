"""Utilities for combining single-cell analysis artefacts into dataset-level narratives."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.scripts.interpretation_types import ClusterSummary, InterpretationOutput

logger = logging.getLogger(__name__)


@dataclass
class DatasetReportArtifacts:
    """Paths and raw content returned by :func:`generate_dataset_report`."""

    report_path: Optional[Path]
    context_json_path: Optional[Path]
    report_content: str


def _to_serialisable(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects to JSON-serialisable types."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serialisable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback for exotic objects
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback for exotic objects
            return str(value)
    return str(value)


def _simplify_marker_candidates(candidates: Sequence) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for candidate in candidates[:5]:
        entry = {
            "cell_type": getattr(candidate, "cell_type", None),
            "score": _to_serialisable(getattr(candidate, "score", None)),
            "markers": list(getattr(candidate, "markers", [])[:10]),
            "metadata": _to_serialisable(getattr(candidate, "metadata", {})),
        }
        simplified.append(entry)
    return simplified


def _simplify_enrichment_terms(terms: Sequence) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for term in terms[:10]:
        entry = {
            "term": getattr(term, "term", None),
            "score": _to_serialisable(getattr(term, "score", None)),
            "p_value": _to_serialisable(getattr(term, "p_value", None)),
            "metadata": _to_serialisable(getattr(term, "metadata", {})),
        }
        simplified.append(entry)
    return simplified


def build_dataset_context(
    clusters: Sequence[ClusterSummary],
    interpretations: Sequence[InterpretationOutput],
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Combine multiple tool outputs into a dataset-level JSON summary."""

    interpretation_by_cluster = {
        item.cluster_id: item
        for item in interpretations
        if getattr(item, "cluster_id", None) is not None
    }

    cluster_entries: List[Dict[str, Any]] = []
    total_cells = 0
    for cluster in clusters:
        cluster_id = cluster.cluster_id
        interpretation = interpretation_by_cluster.get(cluster_id)
        metadata = _to_serialisable(cluster.metadata)
        n_cells = 0
        if isinstance(cluster.metadata, Mapping):
            n_cells = int(cluster.metadata.get("n_cells") or 0)
        total_cells += n_cells

        entry = {
            "cluster_id": cluster_id,
            "n_cells": n_cells,
            "top_genes": list(cluster.top_genes[:10]),
            "marker_candidates": _simplify_marker_candidates(cluster.marker_candidates),
            "enrichment_terms": _simplify_enrichment_terms(cluster.enrichment_terms),
            "metadata": metadata,
        }

        if interpretation is not None:
            entry["interpretation"] = _to_serialisable(interpretation.result)
        cluster_entries.append(entry)

    global_enrichment: Dict[str, Any] = {}
    cell_communication: Optional[Dict[str, Any]] = None
    pseudotime: Optional[Dict[str, Any]] = None

    for cluster in clusters:
        meta = cluster.metadata or {}
        if not isinstance(meta, Mapping):
            continue
        if cell_communication is None and meta.get("cell_communication"):
            cell_communication = _to_serialisable(meta.get("cell_communication"))
        if pseudotime is None and meta.get("pseudotime"):
            pseudotime = _to_serialisable(meta.get("pseudotime"))
        enrichment = meta.get("global_enrichment")
        if isinstance(enrichment, Mapping):
            for key, value in enrichment.items():
                if key not in global_enrichment:
                    global_enrichment[key] = _to_serialisable(value)

    dataset_context: Dict[str, Any] = {
        "dataset": dataset_name,
        "statistics": {
            "total_clusters": len(cluster_entries),
            "total_cells": total_cells,
        },
        "clusters": cluster_entries,
        "global_signals": {
            "enrichment": global_enrichment or None,
            "cell_communication": cell_communication,
            "pseudotime": pseudotime,
        },
    }

    return dataset_context


def generate_dataset_report(
    llm: BaseLanguageModel,
    clusters: Sequence[ClusterSummary],
    interpretations: Sequence[InterpretationOutput],
    output_dir: Path,
    dataset_name: Optional[str] = None,
) -> DatasetReportArtifacts:
    """Generate a narrative report that summarises the single-cell dataset."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_context = build_dataset_context(
        clusters,
        interpretations,
        dataset_name=dataset_name,
    )

    context_json_path = output_dir / "dataset_context.json"
    with context_json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_context, handle, ensure_ascii=False, indent=2)

    system_message = SystemMessage(
        content=(
            "You are an experienced single-cell transcriptomics analyst. "
            "Synthesize comprehensive biological insights using the provided analysis outputs."
        )
    )
    human_message = HumanMessage(
        content=(
            "以下是单细胞分析各项工具的整合结果(JSON格式)：\n"
            f"{json.dumps(dataset_context, ensure_ascii=False, indent=2)}\n\n"
            "请基于这些信息撰写一份中文解读报告，至少涵盖以下部分：\n"
            "1. 数据集概述与主要细胞类型构成；\n"
            "2. 各细胞群的功能状态或富集通路亮点；\n"
            "3. 细胞间通讯或调控网络的关键发现；\n"
            "4. 可能的发育轨迹/拟时序变化与生物学意义；\n"
            "5. 可供后续实验验证或临床转化的建议。\n"
            "请使用 Markdown 小节和条列式表达，突出关键信息并给出推理依据。"
        )
    )

    try:
        response = llm.invoke([system_message, human_message])  # type: ignore[arg-type]
        if isinstance(response, AIMessage):
            report_text = response.content
        else:
            report_text = getattr(response, "content", str(response))
        if isinstance(report_text, list):
            report_text = "\n".join(str(part) for part in report_text)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to generate dataset report: %s", exc)
        report_text = (
            "# 数据集解读报告\n\n"
            "自动生成报告失败，以下提供整合数据的概览以供人工分析：\n\n"
            f"```json\n{json.dumps(dataset_context, ensure_ascii=False, indent=2)}\n```"
        )

    report_path = output_dir / "dataset_interpretation_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(str(report_text))

    return DatasetReportArtifacts(
        report_path=report_path,
        context_json_path=context_json_path,
        report_content=str(report_text),
    )


def load_interpretation_outputs_from_disk(output_dir: Path) -> List[InterpretationOutput]:
    """Reload saved cluster interpretations for downstream consumers."""

    output_dir = output_dir.expanduser().resolve()
    if not output_dir.exists():
        return []

    outputs: List[InterpretationOutput] = []
    for path in sorted(output_dir.glob("cluster_*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - I/O guard
            logger.warning("Failed to parse interpretation file %s: %s", path, exc)
            continue

        cluster_id = str(payload.get("cluster_id") or path.stem.replace("cluster_", ""))
        result = payload.get("model_output") or {}

        outputs.append(
            InterpretationOutput(
                cluster_id=cluster_id,
                result=_to_serialisable(result),
                context=[],
                output_path=str(path),
            )
        )

    return outputs
