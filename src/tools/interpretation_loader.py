"""Utilities for consolidating clustering analysis outputs for interpretation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from src.scripts.interpretation_types import (
    ClusterSummary,
    EnrichmentTerm,
    MarkerCandidate,
)

logger = logging.getLogger(__name__)


@dataclass
class ResultProvider:
    """Abstract base class for loading analysis artefacts."""

    name: str

    def load(
        self,
        work_dir: Path,
        sample_name: str,
        summaries: Dict[str, ClusterSummary],
    ) -> None:
        raise NotImplementedError


class DiffGeneProvider(ResultProvider):
    def __init__(self) -> None:
        super().__init__(name="diff_genes")

    def load(
        self,
        work_dir: Path,
        sample_name: str,
        summaries: Dict[str, ClusterSummary],
    ) -> None:
        diff_path = _resolve_file(work_dir, f"{sample_name}_diff_gene.csv")
        if diff_path is None:
            logger.warning("Differential gene file not found in %s", work_dir)
            return

        df = pd.read_csv(diff_path)
        for _, row in df.iterrows():
            cluster_id = str(row.get("cluster", ""))
            if not cluster_id:
                continue
            genes_raw = str(row.get("top_20_diff_genes", ""))
            genes = [gene.strip() for gene in genes_raw.split(",") if gene.strip()]
            summary = summaries.setdefault(cluster_id, ClusterSummary(cluster_id=cluster_id))
            if genes:
                summary.top_genes = genes
            summary.diff_gene_stats.update({
                "top_genes_raw": genes_raw,
                "source_path": str(diff_path),
            })


class MarkerProvider(ResultProvider):
    def __init__(self) -> None:
        super().__init__(name="marker_scores")

    def load(
        self,
        work_dir: Path,
        sample_name: str,
        summaries: Dict[str, ClusterSummary],
    ) -> None:
        marker_path = _resolve_file(work_dir, f"{sample_name}_markers.json")
        if marker_path is None:
            # Fallback to previous text output when JSON is not available.
            logger.info("Marker JSON not found for %s, skipping marker candidates", sample_name)
            return

        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to parse marker JSON %s: %s", marker_path, exc)
            return

        clusters = payload.get("clusters") if isinstance(payload, dict) else None
        if clusters is None and isinstance(payload, dict):
            clusters = payload

        if not isinstance(clusters, dict):
            logger.warning("Marker payload has unexpected format: %s", type(payload))
            return

        for cluster_id, cluster_data in clusters.items():
            summary = summaries.setdefault(str(cluster_id), ClusterSummary(cluster_id=str(cluster_id)))
            candidates = _normalise_marker_candidates(cluster_data)
            if candidates:
                summary.marker_candidates = candidates


class EnrichmentProvider(ResultProvider):
    def __init__(self) -> None:
        super().__init__(name="enrichment")

    def load(
        self,
        work_dir: Path,
        sample_name: str,
        summaries: Dict[str, ClusterSummary],
    ) -> None:
        enrichment_path = _resolve_file(work_dir, f"{sample_name}_enrichment.csv")
        if enrichment_path is None:
            return

        df = pd.read_csv(enrichment_path)
        for _, row in df.iterrows():
            cluster_id = str(row.get("cluster", ""))
            term = row.get("term") or row.get("pathway") or row.get("name")
            if not cluster_id or not term:
                continue
            enrichment = EnrichmentTerm(
                term=str(term),
                score=_safe_float(row.get("score")),
                p_value=_safe_float(row.get("p_value") or row.get("pvalue") or row.get("padj")),
                metadata={
                    "source_path": str(enrichment_path),
                    "raw": row.to_dict(),
                },
            )
            summary = summaries.setdefault(cluster_id, ClusterSummary(cluster_id=cluster_id))
            summary.enrichment_terms.append(enrichment)


PROVIDERS: List[ResultProvider] = [
    DiffGeneProvider(),
    MarkerProvider(),
    EnrichmentProvider(),
]


def load_cluster_results(work_dir: Path | str) -> List[ClusterSummary]:
    """Load cluster summaries by merging outputs from multiple analysis steps."""

    work_path = Path(work_dir).expanduser().resolve()
    if not work_path.exists():
        raise FileNotFoundError(f"Work directory not found: {work_path}")

    sample_name = work_path.name
    summaries: Dict[str, ClusterSummary] = {}

    _populate_cluster_metadata(work_path, sample_name, summaries)

    for provider in PROVIDERS:
        try:
            provider.load(work_path, sample_name, summaries)
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - provider level guard
            logger.exception("Provider %s failed: %s", provider.name, exc)

    ordered = sorted(summaries.values(), key=lambda item: item.cluster_id)
    return ordered


def _resolve_file(work_dir: Path, filename: str) -> Optional[Path]:
    if not filename:
        return None
    explicit = work_dir / filename
    if explicit.exists():
        return explicit

    matches = list(work_dir.glob(f"*{Path(filename).suffix}"))
    if not matches:
        return None

    for path in matches:
        if path.name == filename:
            return path
        if path.name.startswith(filename.split("_")[0]):
            return path

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _populate_cluster_metadata(
    work_dir: Path,
    sample_name: str,
    summaries: Dict[str, ClusterSummary],
) -> None:
    clustered_path = _resolve_file(work_dir, f"{sample_name}_clustered.h5ad")
    if clustered_path is None:
        logger.warning("Clustered AnnData file not found in %s", work_dir)
        return

    adata = sc.read_h5ad(clustered_path)
    cluster_key = "scGPT_clusters" if "scGPT_clusters" in adata.obs else None
    if cluster_key is None:
        logger.warning("Cluster key 'scGPT_clusters' missing from AnnData.obs")
        return

    clusters = adata.obs[cluster_key].astype(str)
    available_meta = [col for col in ("sample", "batch", "celltype_l1", "celltype_l2", "celltype_l3", "celltype_l4") if col in adata.obs.columns]

    for cluster_id in sorted(clusters.unique()):
        mask = clusters == cluster_id
        summary = summaries.setdefault(cluster_id, ClusterSummary(cluster_id=cluster_id))
        subset = adata[mask]
        summary.metadata.update(
            {
                "n_cells": int(subset.n_obs),
                "available_metadata": available_meta,
                "meta_counts": {
                    meta: _collect_unique(subset.obs[meta]) for meta in available_meta
                },
            }
        )
        if "X_scgpt" in subset.obsm:
            centroid = np.asarray(subset.obsm["X_scgpt"]).mean(axis=0)
            summary.embedding = centroid.astype(float).tolist()


def _collect_unique(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("Unknown").astype(str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _safe_float(value: Optional[object]) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _normalise_marker_candidates(cluster_data: object) -> List[MarkerCandidate]:
    candidates: List[MarkerCandidate] = []

    if isinstance(cluster_data, dict):
        raw_candidates = []
        if isinstance(cluster_data.get("candidates"), list):
            raw_candidates = cluster_data["candidates"]
        elif isinstance(cluster_data.get("marker_scores"), list):
            raw_candidates = cluster_data["marker_scores"]
        elif isinstance(cluster_data.get("predictions"), list):
            raw_candidates = cluster_data["predictions"]
        elif isinstance(cluster_data.get("cell_types"), list):
            raw_candidates = cluster_data["cell_types"]

        for item in raw_candidates:
            candidate = _marker_candidate_from_payload(item)
            if candidate:
                candidates.append(candidate)

    elif isinstance(cluster_data, list):
        for item in cluster_data:
            candidate = _marker_candidate_from_payload(item)
            if candidate:
                candidates.append(candidate)

    return candidates


def _marker_candidate_from_payload(payload: object) -> Optional[MarkerCandidate]:
    if not isinstance(payload, dict):
        return None

    cell_type = payload.get("cell_type") or payload.get("name") or payload.get("label")
    if not cell_type:
        return None

    markers: Iterable[str]
    raw_markers = payload.get("markers") or payload.get("marker_genes") or payload.get("genes")
    if isinstance(raw_markers, str):
        markers = [gene.strip() for gene in raw_markers.split(",") if gene.strip()]
    elif isinstance(raw_markers, Iterable):
        markers = [str(gene).strip() for gene in raw_markers if str(gene).strip()]
    else:
        markers = []

    score = payload.get("score") or payload.get("confidence") or payload.get("probability")
    metadata = {key: value for key, value in payload.items() if key not in {"cell_type", "name", "label", "markers", "marker_genes", "genes", "score", "confidence", "probability"}}

    return MarkerCandidate(
        cell_type=str(cell_type),
        score=_safe_float(score),
        markers=list(markers),
        metadata=metadata,
    )
