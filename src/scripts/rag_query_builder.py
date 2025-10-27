"""Utilities to translate cluster summaries into retrieval topics."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from src.scripts.interpretation_types import ClusterSummary, RagTopic

GENE_CHUNK_SIZE = 12
MAX_MARKER_TOPICS = 3
MAX_ENRICHMENT_TOPICS = 5


def build_topics_for_cluster(cluster: ClusterSummary) -> List[RagTopic]:
    """Generate a diverse list of retrieval topics for a given cluster."""

    topics: List[RagTopic] = []
    topics.extend(_gene_topics(cluster))
    topics.extend(_marker_topics(cluster))
    topics.extend(_enrichment_topics(cluster))
    topics.extend(_similarity_topics(cluster))
    return topics


def _gene_topics(cluster: ClusterSummary) -> List[RagTopic]:
    genes = [gene for gene in cluster.top_genes if gene]
    if not genes:
        return []

    topics: List[RagTopic] = []
    for chunk_index, chunk in enumerate(_chunk_sequence(genes, GENE_CHUNK_SIZE), 1):
        query = (
            "Cluster {cluster_id} exhibits strong expression of genes {genes}. "
            "Summarise the shared biological functions, cell types, or disease "
            "associations commonly linked to these genes in single-cell RNA-seq studies."
        ).format(cluster_id=cluster.cluster_id, genes=", ".join(chunk))
        topics.append(
            RagTopic(
                cluster_id=cluster.cluster_id,
                query_text=query,
                topic_type="differential_genes",
                supporting_genes=chunk,
                metadata={"chunk_index": chunk_index},
            )
        )
    return topics


def _marker_topics(cluster: ClusterSummary) -> List[RagTopic]:
    if not cluster.marker_candidates:
        return []

    topics: List[RagTopic] = []
    for candidate in cluster.marker_candidates[:MAX_MARKER_TOPICS]:
        if not candidate.cell_type:
            continue
        markers = ", ".join(candidate.markers[:8]) if candidate.markers else ""
        query = (
            "Cluster {cluster_id} shows marker evidence for {cell_type}. "
            "Discuss literature supporting this assignment and highlight the "
            "roles of markers {markers}."
        ).format(
            cluster_id=cluster.cluster_id,
            cell_type=candidate.cell_type,
            markers=markers or "(no explicit markers provided)",
        )
        topics.append(
            RagTopic(
                cluster_id=cluster.cluster_id,
                query_text=query,
                topic_type="marker_support",
                supporting_genes=candidate.markers,
                metadata={
                    "candidate_cell_type": candidate.cell_type,
                    "candidate_score": candidate.score,
                },
            )
        )
    return topics


def _enrichment_topics(cluster: ClusterSummary) -> List[RagTopic]:
    if not cluster.enrichment_terms:
        return []

    topics: List[RagTopic] = []
    for enrichment in cluster.enrichment_terms[:MAX_ENRICHMENT_TOPICS]:
        query = (
            "Cluster {cluster_id} is enriched for the pathway "
            "'{pathway}'. Explain the biological processes and potential "
            "phenotypic implications of this enrichment in immune or tissue contexts."
        ).format(cluster_id=cluster.cluster_id, pathway=enrichment.term)
        topics.append(
            RagTopic(
                cluster_id=cluster.cluster_id,
                query_text=query,
                topic_type="enrichment",
                supporting_genes=cluster.top_genes,
                metadata={
                    "pathway": enrichment.term,
                    "score": enrichment.score,
                    "p_value": enrichment.p_value,
                },
            )
        )
    return topics


def _similarity_topics(cluster: ClusterSummary) -> List[RagTopic]:
    similar = cluster.metadata.get("similar_clusters") if cluster.metadata else None
    if not similar:
        return []

    topics: List[RagTopic] = []
    for item in similar:
        reference = item.get("reference") or item.get("cell_type")
        if not reference:
            continue
        query = (
            "Compare cluster {cluster_id} with reference cell population "
            "'{reference}'. Highlight shared and divergent pathways, "
            "marker genes, and functional roles."
        ).format(cluster_id=cluster.cluster_id, reference=reference)
        topics.append(
            RagTopic(
                cluster_id=cluster.cluster_id,
                query_text=query,
                topic_type="similarity",
                supporting_genes=cluster.top_genes,
                metadata=item,
            )
        )
    return topics


def _chunk_sequence(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])
