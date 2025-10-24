"""LangChain tool that orchestrates cluster interpretation via RAG."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.setting import settings
from src.memory.conversation_memory import ConversationMemoryStore
from src.scripts.dataset_interpretation import (
    DatasetReportArtifacts,
    generate_dataset_report,
)
from src.scripts.interpretation_generator import generate_cluster_interpretation
from src.scripts.interpretation_types import ClusterSummary, InterpretationOutput
from src.scripts.rag import BioKnowledgeRag, CellRag, find_similar_clusters
from src.scripts.rag_query_builder import build_topics_for_cluster
from src.tools.interpretation_loader import load_cluster_results
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)


class InterpretClustersArgs(BaseModel):
    work_dir: str = Field(..., description="Work directory containing clustering outputs.")
    collection_name: Optional[str] = Field(
        default=None,
        description="Override the default knowledge base collection name.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="LLM model identifier used for generation.",
    )
    max_clusters: Optional[int] = Field(
        default=None,
        description="If provided, limit the number of clusters processed.",
    )
    persist_memory: bool = Field(
        default=False,
        description="Persist interpretations into long-term memory store.",
    )
    memory_thread_id: Optional[str] = Field(
        default=None,
        description="Identifier for memory thread when persisting results.",
    )
    objective: Optional[str] = Field(
        default=None,
        description="Objective string stored with memory records.",
    )
    enable_cell_rag: bool = Field(
        default=False,
        description="Query the cell-level embedding index for similar populations.",
    )
    cell_rag_results: int = Field(
        default=5,
        description="Number of similar populations to retrieve when CellRag is enabled.",
    )


def _append_similarity_metadata(
    clusters: List[ClusterSummary],
    cell_rag: Optional[CellRag],
    n_results: int,
) -> None:
    if not cell_rag:
        return

    for cluster in clusters:
        if cluster.embedding is None:
            continue
        similar = find_similar_clusters(cell_rag, cluster.embedding, n_results=n_results)
        if similar:
            cluster.metadata.setdefault("similar_clusters", similar)


@tool("interpret_cluster_results", args_schema=InterpretClustersArgs)
def interpret_cluster_results(
    work_dir: str,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    max_clusters: Optional[int] = None,
    persist_memory: bool = False,
    memory_thread_id: Optional[str] = None,
    objective: Optional[str] = None,
    enable_cell_rag: bool = False,
    cell_rag_results: int = 5,
) -> str:
    """Generate biological interpretations for clusters within a work directory."""

    work_path = Path(work_dir).expanduser().resolve()
    clusters = load_cluster_results(work_path)
    if max_clusters is not None:
        clusters = clusters[: max_clusters]

    if not clusters:
        raise ValueError(f"No clusters found in work directory: {work_path}")

    rag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
    collection = collection_name or settings.CHROMADB_lit_collection_name
    rag.init_vector_store(collection)

    if enable_cell_rag:
        try:
            cell_rag = CellRag(settings.CHROMADB_PERSIST_DIR, settings.CHROMADB_cell_collection_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to initialise CellRag: %s", exc)
            cell_rag = None
    else:
        cell_rag = None

    _append_similarity_metadata(clusters, cell_rag, n_results=cell_rag_results)

    llm = get_llm(model_name)
    contexts_by_cluster: Dict[str, List] = {}

    for cluster in clusters:
        topics = build_topics_for_cluster(cluster)
        if topics:
            contexts = rag.interpret_topics_sync(topics, collection_name=collection)
        else:
            contexts = []
        contexts_by_cluster[cluster.cluster_id] = contexts

    output_dir = work_path / "interpretation"
    interpretations: List[InterpretationOutput] = []
    for cluster in clusters:
        contexts = contexts_by_cluster.get(cluster.cluster_id, [])
        interpretation = generate_cluster_interpretation(
            llm,
            cluster,
            contexts,
            output_dir=output_dir,
        )
        interpretations.append(interpretation)

    try:
        dataset_report: DatasetReportArtifacts = generate_dataset_report(
            llm,
            clusters,
            interpretations,
            output_dir=output_dir,
            dataset_name=work_path.name,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to build dataset-level report: %s", exc)
        dataset_report = DatasetReportArtifacts(
            report_path=None,
            context_json_path=None,
            report_content="",
        )

    result_payload = {
        "work_dir": str(work_path),
        "collection": collection,
        "clusters": [
            {
                "cluster_id": item.cluster_id,
                "output_path": item.output_path,
                "putative_identity": item.result.get("putative_identity"),
                "confidence": item.result.get("confidence"),
            }
            for item in interpretations
        ],
        "dataset_report": {
            "report_path": str(dataset_report.report_path)
            if dataset_report.report_path
            else None,
            "context_path": str(dataset_report.context_json_path)
            if dataset_report.context_json_path
            else None,
        },
    }

    if persist_memory and memory_thread_id:
        store = ConversationMemoryStore()
        messages = [
            AIMessage(content=json.dumps(result_payload, ensure_ascii=False)),
        ]
        store.store_conversation(
            thread_id=memory_thread_id,
            objective=objective or f"Cluster interpretation for {work_path.name}",
            messages=messages,
            result_text=json.dumps(result_payload, ensure_ascii=False),
            metadata={"work_dir": str(work_path)},
        )

    return json.dumps(result_payload, ensure_ascii=False)
