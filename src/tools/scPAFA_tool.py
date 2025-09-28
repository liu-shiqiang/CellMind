import json
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from scPAFA_analysis import run_pas_mofa_pipeline  


class PASMOFAToolInput(BaseModel):
    adata_path: str = Field(..., description="Path to the AnnData file in .h5ad format.")
    sample_column: str = Field(..., description="Column in adata.obs that identifies biological samples (e.g., patient_id).")
    view_column: str = Field(..., description="Column in adata.obs that defines views (e.g., cell_type).")
    label_column: str = Field(..., description="Clinical label column for downstream analysis (e.g., response, disease_status).")
    sample_metadata_path: Optional[str] = Field(None, description="Optional path to external sample metadata CSV/TSV.")
    batch_column: Optional[str] = Field(None, description="Optional batch column for covariate adjustment.")
    pas_method: str = Field("UCell", description="Pathway scoring method: 'UCell' or 'score_genes'.")
    factor_number: int = Field(10, description="Number of MOFA+ latent factors to infer.")
    min_cells_per_sample_view: int = Field(3, description="Minimum number of cells per (sample, view) group.")
    output_dir: str = Field("./pas_mofa_results", description="Directory to save results.")
    random_seed: int = Field(42, description="Random seed for reproducibility.")


class PASMOFATool(BaseTool):
    name: str = "run_scpafa_analysis"
    description: str = (
        "Runs an end-to-end single-cell pathway activit y scoring (PAS) and MOFA+ multi-factor analysis pipeline. "
        "Input: paths to AnnData and pathway JSON, plus metadata column names. "
        "Output: paths to MOFA+ model, factor scores, pseudobulk matrix, and visualizations."
    )
    args_schema: Type[BaseModel] = PASMOFAToolInput

    def _run(self, **kwargs: Any) -> str:
        try:
            result = run_pas_mofa_pipeline(**kwargs)
            # 返回简洁、结构化的自然语言摘要 + 关键路径
            summary = (
                f"✅ PAS-MOFA+ analysis completed successfully.\n"
                f"• MOFA+ model saved to: {result['mofa_model_path']}\n"
                f"• Factor scores: {result['factors_path']}\n"
                f"• Pseudobulk matrix: {result['pseudobulk_path']}\n"
                f"• Visualizations: {result['figures_dir'] or 'None'}\n"
                f"• Analyzed {result['n_samples']} sample-view groups with {result['n_factors']} factors."
            )
            return summary
        except Exception as e:
            return f"❌ PAS-MOFA+ pipeline failed with error: {str(e)}"

    async def _arun(self, **kwargs: Any) -> str:
        # 如果你不需要异步，可以直接调用 _run
        return self._run(**kwargs)
    
run_scpafa_analysis = PASMOFATool()