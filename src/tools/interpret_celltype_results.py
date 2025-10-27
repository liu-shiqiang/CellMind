"""LangChain tool that produces cell type-level interpretations."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.scripts.dataset_interpretation import (
    CellTypeReportArtifacts,
    build_celltype_context,
    build_dataset_context,
    generate_celltype_report,
    load_interpretation_outputs_from_disk,
)
from src.tools.interpretation_loader import load_cluster_results
from src.utils.llm_manager import get_llm

logger = logging.getLogger(__name__)


class InterpretCellTypesArgs(BaseModel):
    """Arguments accepted by :func:`interpret_celltype_results`."""

    work_dir: str = Field(..., description="Work directory containing clustering outputs.")
    model_name: Optional[str] = Field(
        default=None,
        description="Optional override for the language model used to draft the report.",
    )


@tool("interpret_celltype_results", args_schema=InterpretCellTypesArgs)
def interpret_celltype_results(work_dir: str, model_name: Optional[str] = None) -> str:
    """Summarise clusters into annotated cell type narratives."""

    work_path = Path(work_dir).expanduser().resolve()
    clusters = load_cluster_results(work_path)
    if not clusters:
        raise ValueError(f"No clusters found in work directory: {work_path}")

    interpretation_dir = work_path / "interpretation"
    interpretations = load_interpretation_outputs_from_disk(interpretation_dir)

    dataset_context = build_dataset_context(
        clusters,
        interpretations,
        dataset_name=work_path.name,
    )

    celltype_context = build_celltype_context(
        clusters,
        interpretations,
        dataset_name=work_path.name,
        dataset_context=dataset_context,
    )

    llm = get_llm(model_name)
    try:
        report: CellTypeReportArtifacts = generate_celltype_report(
            llm,
            clusters,
            interpretations,
            output_dir=interpretation_dir,
            dataset_name=work_path.name,
            dataset_context=dataset_context,
            celltype_context=celltype_context,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to build cell-type report: %s", exc)
        report = CellTypeReportArtifacts(
            report_path=None,
            context_json_path=None,
            report_content="",
        )

    payload = {
        "work_dir": str(work_path),
        "celltype_context_path": str((interpretation_dir / "celltype_context.json")),
        "report_path": str(report.report_path) if report.report_path else None,
        "context_path": str(report.context_json_path) if report.context_json_path else None,
    }

    return json.dumps(payload, ensure_ascii=False)

