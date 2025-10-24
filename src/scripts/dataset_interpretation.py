"""Utilities for combining single-cell analysis artefacts into dataset-level narratives."""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


@dataclass
class CellTypeReportArtifacts:
    """Paths and raw content returned by :func:`generate_celltype_report`."""

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


def _resolve_primary_celltype(
    cluster: ClusterSummary,
    interpretation: Optional[InterpretationOutput],
) -> str:
    """Infer the primary cell-type label for a cluster."""

    label_candidates: List[Tuple[str, int]] = []
    meta_counts = cluster.metadata.get("meta_counts") if cluster.metadata else None
    if isinstance(meta_counts, Mapping):
        for key in ("celltype_l4", "celltype_l3", "celltype_l2", "celltype_l1", "pred_celltype"):
            counts = meta_counts.get(key)
            if isinstance(counts, Mapping):
                for celltype, count in counts.items():
                    label = str(celltype).strip()
                    if not label or label.lower() in {"unknown", "nan"}:
                        continue
                    label_candidates.append((label, int(count)))
            if label_candidates:
                break

    if label_candidates:
        label_candidates.sort(key=lambda item: (-item[1], item[0]))
        return label_candidates[0][0]

    if interpretation and isinstance(interpretation.result, Mapping):
        identity = interpretation.result.get("putative_identity")
        if isinstance(identity, str) and identity.strip():
            return identity.strip()

    return "未注释细胞群"


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

        entry["assigned_celltype"] = _resolve_primary_celltype(cluster, interpretation)
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
    dataset_context: Optional[Dict[str, Any]] = None,
) -> DatasetReportArtifacts:
    """Generate a narrative report that summarises the single-cell dataset."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_context is None:
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


def build_celltype_context(
    clusters: Sequence[ClusterSummary],
    interpretations: Sequence[InterpretationOutput],
    dataset_name: Optional[str] = None,
    dataset_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate cluster interpretations into cell type-level summaries."""

    interpretation_by_cluster = {
        item.cluster_id: item
        for item in interpretations
        if getattr(item, "cluster_id", None) is not None
    }

    aggregated: Dict[str, Dict[str, Any]] = {}
    total_cells = 0

    for cluster in clusters:
        interpretation = interpretation_by_cluster.get(cluster.cluster_id)
        celltype = _resolve_primary_celltype(cluster, interpretation)
        entry = aggregated.setdefault(
            celltype,
            {
                "cell_type": celltype,
                "total_cells": 0,
                "clusters": [],
                "functional_notes": [],
                "_marker_counter": Counter(),
                "_pathway_counter": Counter(),
                "_identity_counter": Counter(),
            },
        )

        metadata = cluster.metadata if isinstance(cluster.metadata, Mapping) else {}
        n_cells = 0
        if isinstance(metadata, Mapping):
            n_cells = int(metadata.get("n_cells") or 0)
        entry["total_cells"] += n_cells
        total_cells += n_cells

        interpretation_payload = _to_serialisable(interpretation.result) if interpretation else None
        if isinstance(interpretation_payload, Mapping):
            identity = interpretation_payload.get("putative_identity")
            if isinstance(identity, str) and identity.strip():
                entry["_identity_counter"][identity.strip()] += 1
            pathways = interpretation_payload.get("pathways")
            if isinstance(pathways, Sequence):
                entry["_pathway_counter"].update(
                    str(pathway).strip() for pathway in pathways if str(pathway).strip()
                )
            caveats = interpretation_payload.get("caveats")
        else:
            pathways = []
            caveats = None

        cluster_entry = {
            "cluster_id": cluster.cluster_id,
            "n_cells": n_cells,
            "top_genes": list(cluster.top_genes[:10]),
            "marker_candidates": _simplify_marker_candidates(cluster.marker_candidates),
            "enrichment_terms": _simplify_enrichment_terms(cluster.enrichment_terms),
            "interpretation": interpretation_payload,
        }
        entry["clusters"].append(cluster_entry)

        markers: List[str] = []
        markers.extend(cluster.top_genes[:10])
        for candidate in cluster.marker_candidates:
            markers.extend(candidate.markers[:10])
        entry["_marker_counter"].update(
            str(marker).strip() for marker in markers if str(marker).strip()
        )

        note = {
            "cluster_id": cluster.cluster_id,
            "putative_identity": interpretation_payload.get("putative_identity")
            if isinstance(interpretation_payload, Mapping)
            else None,
            "pathways": pathways,
            "caveats": caveats,
        }
        entry["functional_notes"].append(_to_serialisable(note))

    celltype_entries: List[Dict[str, Any]] = []
    for celltype, entry in aggregated.items():
        marker_counter: Counter = entry.pop("_marker_counter")  # type: ignore[assignment]
        pathway_counter: Counter = entry.pop("_pathway_counter")  # type: ignore[assignment]
        identity_counter: Counter = entry.pop("_identity_counter")  # type: ignore[assignment]

        entry["representative_markers"] = [gene for gene, _ in marker_counter.most_common(20)] or None
        entry["pathway_highlights"] = [term for term, _ in pathway_counter.most_common(15)] or None
        consensus = identity_counter.most_common(1)
        entry["consensus_identity"] = consensus[0][0] if consensus else celltype
        entry["n_clusters"] = len(entry["clusters"])
        entry["cluster_ids"] = [item["cluster_id"] for item in entry["clusters"]]
        celltype_entries.append(entry)

    celltype_entries.sort(key=lambda item: item.get("total_cells", 0), reverse=True)

    context: Dict[str, Any] = {
        "dataset": dataset_name,
        "statistics": {
            "total_cell_types": len(celltype_entries),
            "total_clusters": sum(entry.get("n_clusters", 0) for entry in celltype_entries),
            "total_cells": total_cells,
        },
        "cell_types": celltype_entries,
    }

    if dataset_context and isinstance(dataset_context.get("global_signals"), Mapping):
        context["global_signals"] = dataset_context["global_signals"]

    return context


def generate_celltype_report(
    llm: BaseLanguageModel,
    clusters: Sequence[ClusterSummary],
    interpretations: Sequence[InterpretationOutput],
    output_dir: Path,
    dataset_name: Optional[str] = None,
    dataset_context: Optional[Dict[str, Any]] = None,
    celltype_context: Optional[Dict[str, Any]] = None,
) -> CellTypeReportArtifacts:
    """Generate high-level biological interpretation for annotated cell types."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if celltype_context is None:
        celltype_context = build_celltype_context(
            clusters,
            interpretations,
            dataset_name=dataset_name,
            dataset_context=dataset_context,
        )

    context_json_path = output_dir / "celltype_context.json"
    with context_json_path.open("w", encoding="utf-8") as handle:
        json.dump(celltype_context, handle, ensure_ascii=False, indent=2)

    system_message = SystemMessage(
        content=(
            "You are a systems immunologist specialising in single-cell atlases. "
            "Explain the biological roles of annotated cell types using the provided summaries."
        )
    )
    human_message = HumanMessage(
        content=(
            "以下是按细胞类型整合的聚类解读信息(JSON格式)：\n"
            f"{json.dumps(celltype_context, ensure_ascii=False, indent=2)}\n\n"
            "请基于这些信息撰写中文总结，包含：\n"
            "1. 各主要细胞类型的核心功能或状态；\n"
            "2. 关键信号通路/富集主题；\n"
            "3. 细胞间通讯或互作要点；\n"
            "4. 对疾病或实验的启示与后续建议。\n"
            "请按照细胞类型组织内容，使用Markdown小节并注明依据。"
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
        logger.exception("Failed to generate cell-type report: %s", exc)
        report_text = (
            "# 细胞类型解读报告\n\n"
            "自动生成报告失败，以下提供细胞类型整合数据以供人工分析：\n\n"
            f"```json\n{json.dumps(celltype_context, ensure_ascii=False, indent=2)}\n```"
        )

    report_path = output_dir / "celltype_interpretation_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(str(report_text))

    return CellTypeReportArtifacts(
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
