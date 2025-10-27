"""Shared dataclasses for single-cell interpretation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document


@dataclass
class MarkerCandidate:
    """Candidate cell-type assignment derived from marker scoring."""

    cell_type: str
    score: Optional[float] = None
    markers: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentTerm:
    """Representation of a pathway or functional enrichment hit."""

    term: str
    score: Optional[float] = None
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterSummary:
    """Aggregated view of analysis artefacts for a single cluster."""

    cluster_id: str
    top_genes: List[str] = field(default_factory=list)
    diff_gene_stats: Dict[str, Any] = field(default_factory=dict)
    marker_candidates: List[MarkerCandidate] = field(default_factory=list)
    enrichment_terms: List[EnrichmentTerm] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class RagTopic:
    """A single retrieval topic constructed for RAG queries."""

    cluster_id: str
    query_text: str
    topic_type: str
    supporting_genes: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagTopicContext:
    """Container for retrieved documents associated with a topic."""

    topic: RagTopic
    documents: List[Document] = field(default_factory=list)
    combined_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretationOutput:
    """Final structured interpretation returned to downstream consumers."""

    cluster_id: str
    result: Dict[str, Any]
    context: List[RagTopicContext] = field(default_factory=list)
    output_path: Optional[str] = None
