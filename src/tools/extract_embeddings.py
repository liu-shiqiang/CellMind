import os
import json
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper
import scanpy as sc
from config.setting import settings

class ExtractEmbeddingsArgs(BaseModel):
    preproc_path: str = Field(description="Path to the  *_preprocessed.h5ad file to extract embeddings from.")
    work_dir: str = Field(description="Pre-sample folder created by load_h5ad_data")

@tool(
    "extract_embeddings_with_scgpt",
    args_schema=ExtractEmbeddingsArgs
)
def extract_embeddings_with_scgpt(
    preproc_path: str,
    work_dir: str,
) -> str:
    """
    Run scGPT to generate 512-dim cell embeddings, merge them into the preprocessed AnnData, and save <sample>_emb.h5ad in *work_dir*.
    """

    work = Path(work_dir).expanduser().resolve()
    preproc = Path(preproc_path).expanduser().resolve()

    if not preproc.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {preproc}")
    if not work.exists():
        raise FileNotFoundError(f"Work directory not found: {work}")

    sample_name = work.name
    emb_path = work / f"{sample_name}_emb.h5ad"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = ScGPTModelWrapper.from_pretrained(pretrained_model_name_or_path=settings.SCGPT_MODEL_DIR, device=device)
        adata_emb = model.extract_sample_embedding(
            adata_or_file=preproc_path,
            gene_col="gene_name",
            max_length=1200,
            cell_embedding_mode="cls",
            batch_size=64,
            obs_to_save=None,
            return_new_adata=True,
        )

        adata_preproc = sc.read_h5ad(preproc_path)
        adata_preproc.obsm["X_scGPT"] = adata_emb.X.copy()
        adata_preproc.write(emb_path)
    except Exception as exc:
        raise RuntimeError(f"scGPT embedding extraction failed: {exc}")from exc

    return json.dumps(
        {
            "embeddings_path": str(emb_path)
        }
    )
