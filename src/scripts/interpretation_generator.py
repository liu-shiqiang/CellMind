"""High-level orchestration for generating narrative interpretations."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.scripts.interpretation_types import (
    ClusterSummary,
    InterpretationOutput,
    RagTopicContext,
)

logger = logging.getLogger(__name__)


DEFAULT_JSON_SCHEMA = {
    "cluster_id": "string",
    "putative_identity": "string",
    "supporting_genes": ["string"],
    "pathways": ["string"],
    "literature_context": [
        {
            "source": "string",
            "snippet": "string",
        }
    ],
    "confidence": "float",
    "caveats": "string",
}


def _build_context_section(contexts: Sequence[RagTopicContext]) -> str:
    lines: List[str] = []
    for ctx in contexts:
        lines.append(f"Topic ({ctx.topic.topic_type}): {ctx.topic.query_text}")
        if ctx.documents:
            for doc in ctx.documents[:3]:
                source = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
                prefix = f"- Source: {source} | " if source else "- "
                snippet = doc.page_content.strip().replace("\n", " ")
                lines.append(f"  {prefix}{snippet[:400]}")
        else:
            lines.append("- No supporting documents retrieved.")
    return "\n".join(lines)


def _build_cluster_header(cluster: ClusterSummary) -> str:
    meta_lines = [f"Cluster ID: {cluster.cluster_id}"]
    if cluster.metadata.get("n_cells"):
        meta_lines.append(f"Cell count: {cluster.metadata['n_cells']}")
    meta_counts = cluster.metadata.get("meta_counts", {}) if cluster.metadata else {}
    if meta_counts:
        formatted = []
        for key, values in meta_counts.items():
            formatted.append(
                f"{key}: " + ", ".join(f"{k} ({v})" for k, v in values.items())
            )
        meta_lines.append("Annotations: " + "; ".join(formatted))
    return "\n".join(meta_lines)


def build_interpretation_prompt(
    cluster: ClusterSummary,
    contexts: Sequence[RagTopicContext],
    schema: Optional[Dict[str, object]] = None,
) -> List[SystemMessage | HumanMessage]:
    """Construct a chat prompt guiding the LLM to produce JSON output."""

    schema = schema or DEFAULT_JSON_SCHEMA
    header = _build_cluster_header(cluster)
    diff_genes = ", ".join(cluster.top_genes[:20]) if cluster.top_genes else "N/A"
    markers = "; ".join(
        f"{cand.cell_type} (score={cand.score if cand.score is not None else 'NA'})"
        for cand in cluster.marker_candidates
    ) or "None"
    enrichments = ", ".join(term.term for term in cluster.enrichment_terms[:5]) or "None"

    context_section = _build_context_section(contexts)
    schema_lines = json.dumps(schema, ensure_ascii=False, indent=2)

    system_message = SystemMessage(
        content=(
            "You are a senior single-cell RNA-seq analyst. "
            "Generate concise but information-rich biological interpretations."
        )
    )
    human_message = HumanMessage(
        content=(
            f"Cluster summary:\n{header}\n\n"
            f"Top differential genes: {diff_genes}\n"
            f"Marker candidates: {markers}\n"
            f"Enrichment highlights: {enrichments}\n\n"
            f"Retrieved knowledge base context:\n{context_section}\n\n"
            "Please analyse the evidence, propose the most likely cell identity, key pathways, "
            "and potential disease associations. Return a valid JSON object following this schema: "
            f"{schema_lines}."
        )
    )
    return [system_message, human_message]


def _parse_llm_output(raw_output: object) -> Dict[str, object]:
    if isinstance(raw_output, AIMessage):
        raw_text = raw_output.content
    else:
        raw_text = getattr(raw_output, "content", str(raw_output))

    if isinstance(raw_text, list):
        raw_text = "\n".join(str(part) for part in raw_text)

    text = str(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            logger.error("Failed to parse LLM output as JSON: %s", text)
            return {
                "putative_identity": "Unknown",
                "supporting_genes": [],
                "pathways": [],
                "literature_context": [],
                "confidence": 0.0,
                "caveats": "Failed to parse model output.",
            }


def generate_cluster_interpretation(
    llm: BaseLanguageModel,
    cluster: ClusterSummary,
    contexts: Sequence[RagTopicContext],
    output_dir: Path,
    schema: Optional[Dict[str, object]] = None,
) -> InterpretationOutput:
    """Generate interpretation JSON for a single cluster and persist the result."""

    messages = build_interpretation_prompt(cluster, contexts, schema=schema)
    try:
        response = llm.invoke(messages)  # type: ignore[call-arg]
    except Exception as exc:
        logger.error("LLM invocation failed for cluster %s: %s", cluster.cluster_id, exc)
        parsed = {
            "cluster_id": cluster.cluster_id,
            "putative_identity": "Unknown",
            "supporting_genes": cluster.top_genes[:5],
            "pathways": [term.term for term in cluster.enrichment_terms[:3]],
            "literature_context": [],
            "confidence": 0.0,
            "caveats": f"LLM invocation failed: {exc}",
        }
    else:
        parsed = _parse_llm_output(response)
        parsed.setdefault("cluster_id", cluster.cluster_id)

    output_payload = {
        "cluster_id": cluster.cluster_id,
        "model_output": parsed,
        "topics": [
            {
                "query": ctx.topic.query_text,
                "topic_type": ctx.topic.topic_type,
                "supporting_genes": list(ctx.topic.supporting_genes),
                "retrieved_documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in ctx.documents
                ],
                "metadata": ctx.metadata,
            }
            for ctx in contexts
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cluster_{cluster.cluster_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    return InterpretationOutput(
        cluster_id=cluster.cluster_id,
        result=parsed,
        context=list(contexts),
        output_path=str(output_path),
    )
