"""Tooling that enables dataset-level bioinformatics question answering."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from metapub import PubMedFetcher
from pydantic import BaseModel, Field

from config.setting import settings
from src.scripts.dataset_interpretation import (
    build_dataset_context,
    load_interpretation_outputs_from_disk,
)
from src.scripts.pubmed_rag.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
from src.scripts.pubmed_rag.data_repository.local_data_store import LocalJSONStore
from src.scripts.pubmed_rag.rag_pipeline.chromadb_rag import ChromaDbRag
from src.scripts.pubmed_rag.rag_pipeline.embeddings import embeddings as pubmed_embeddings
from src.scripts.rag import BioKnowledgeRag
from src.tools.interpretation_loader import load_cluster_results
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)


class DatasetQAArgs(BaseModel):
    """Arguments for dataset-level bioinformatics Q&A."""

    work_dir: str = Field(..., description="工作目录，包含单细胞分析结果。")
    question: str = Field(..., description="用户提出的生物信息学问题。")
    model_name: Optional[str] = Field(
        default=None,
        description="用于回答的语言模型名称，可选。",
    )
    top_k_local: int = Field(
        default=3,
        description="本地文献知识库返回的参考条目数量。",
    )
    top_k_pubmed: int = Field(
        default=3,
        description="PubMed RAG 返回的摘要数量。",
    )


def _read_text(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - I/O guard
        logger.warning("Failed to read %s: %s", path, exc)
        return ""
    if limit and len(content) > limit:
        return content[:limit] + "\n\n…(内容已截断)"
    return content


def _format_documents(docs: List, prefix: str) -> str:
    formatted: List[str] = []
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        source = metadata.get("title") or metadata.get("source") or metadata.get("doi")
        header = f"[{prefix}#{idx}] {source or '未提供标题'}"
        snippet = doc.page_content.replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        formatted.append(f"{header}\n{snippet}")
    return "\n\n".join(formatted)


def _collect_local_rag(question: str, top_k: int) -> List:
    try:
        rag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
        collection = settings.CHROMADB_lit_collection_name
        rag.init_vector_store(collection)
        return rag.query(question, collection_name=collection, top_k=top_k)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Local knowledge retrieval failed: %s", exc)
        return []


def _collect_pubmed_rag(
    work_path: Path,
    question: str,
    top_k: int,
) -> List:
    cache_root = work_path / "interpretation" / "qa_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    store_path = cache_root / "pubmed_store"
    store_path.mkdir(parents=True, exist_ok=True)
    persist_dir = cache_root / "pubmed_chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    try:
        retriever = PubMedAbstractRetriever(PubMedFetcher())
        abstracts = retriever.get_abstract_data(question)
        if not abstracts:
            return []

        store = LocalJSONStore(str(store_path))
        query_id = store.save_dataset(abstracts, question)
        documents = store.read_documents(query_id)

        rag = ChromaDbRag(str(persist_dir), pubmed_embeddings)
        vector_index = rag.create_vector_index_for_user_query(documents, query_id)
        results = vector_index.similarity_search(question, k=top_k)

        try:
            rag.client.delete_collection(query_id)
        except Exception:  # pragma: no cover - cleanup best effort
            pass

        store.delete_dataset(query_id)
        return results
    except Exception as exc:  # pragma: no cover - external dependency guard
        logger.warning("PubMed retrieval failed: %s", exc)
        return []


def _prepare_dataset_context(
    work_path: Path,
) -> Dict[str, Any]:
    clusters = load_cluster_results(work_path)
    interpretation_dir = work_path / "interpretation"
    interpretations = load_interpretation_outputs_from_disk(interpretation_dir)
    context = build_dataset_context(
        clusters,
        interpretations,
        dataset_name=work_path.name,
    )
    clusters_sorted = sorted(
        context.get("clusters", []),
        key=lambda item: item.get("n_cells", 0),
        reverse=True,
    )
    context_for_qa = dict(context)
    context_for_qa["clusters"] = clusters_sorted[: min(len(clusters_sorted), 8)]
    return context_for_qa


@tool("dataset_bio_qa", args_schema=DatasetQAArgs)
def dataset_bio_qa(
    work_dir: str,
    question: str,
    model_name: Optional[str] = None,
    top_k_local: int = 3,
    top_k_pubmed: int = 3,
) -> str:
    """回答基于单细胞分析结果的生物信息学问题。"""

    work_path = Path(work_dir).expanduser().resolve()
    if not work_path.exists():
        raise FileNotFoundError(f"Work directory not found: {work_path}")

    dataset_context = _prepare_dataset_context(work_path)
    interpretation_dir = work_path / "interpretation"

    dataset_report_path = interpretation_dir / "dataset_interpretation_report.md"
    if not dataset_report_path.exists():
        dataset_report_path = interpretation_dir / "overall_interpretation_report.md"
    dataset_report_text = _read_text(dataset_report_path)

    local_docs = _collect_local_rag(question, top_k=top_k_local)
    pubmed_docs = _collect_pubmed_rag(work_path, question, top_k=top_k_pubmed)

    context_sections = [
        "【整合分析结果】\n" + json.dumps(dataset_context, ensure_ascii=False, indent=2)
    ]
    if dataset_report_text:
        context_sections.append("【数据集解读报告】\n" + dataset_report_text)
    if local_docs:
        context_sections.append("【本地知识库检索】\n" + _format_documents(local_docs, "Local"))
    if pubmed_docs:
        context_sections.append("【PubMed 检索摘要】\n" + _format_documents(pubmed_docs, "PubMed"))

    combined_context = "\n\n".join(context_sections)

    llm = get_llm(model_name)
    system_message = SystemMessage(
        content=(
            "You are a biomedical scientist who explains single-cell RNA-seq results. "
            "Synthesize dataset analytics, local literature embeddings, and PubMed abstracts to answer in Chinese."
        )
    )
    human_message = HumanMessage(
        content=(
            "请综合以下资料回答用户问题，明确区分信息来源（单细胞分析、本地知识库、PubMed）：\n\n"
            f"{combined_context}\n\n"
            f"问题：{question}\n\n"
            "请按照以下结构作答：\n"
            "## 解答\n- 直接回答问题\n"
            "## 关键依据\n- 单细胞分析：...\n- 本地知识库：...\n- PubMed：...\n"
            "## 建议\n- 如有需要，给出后续实验或验证建议\n"
        )
    )

    response = llm.invoke([system_message, human_message])  # type: ignore[arg-type]
    answer_text = getattr(response, "content", str(response))
    if isinstance(answer_text, list):
        answer_text = "\n".join(str(part) for part in answer_text)

    qa_history_path = interpretation_dir / "qa_history.jsonl"
    record = {
        "question": question,
        "answer": answer_text,
        "dataset_report": str(dataset_report_path) if dataset_report_path.exists() else None,
        "local_hits": [doc.metadata for doc in local_docs],
        "pubmed_hits": [doc.metadata for doc in pubmed_docs],
    }
    try:
        with qa_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Failed to write QA history: %s", exc)

    return json.dumps(
        {
            "question": question,
            "answer": answer_text,
            "dataset_report_path": str(dataset_report_path) if dataset_report_path.exists() else None,
            "qa_history_path": str(qa_history_path),
            "local_reference_count": len(local_docs),
            "pubmed_reference_count": len(pubmed_docs),
        },
        ensure_ascii=False,
    )
