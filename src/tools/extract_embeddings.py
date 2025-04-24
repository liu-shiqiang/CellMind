import os
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.bio_pretrained_model.data_prep._scgpt_data_processor import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper
import scanpy as sc
from config.setting import settings

class ExtractEmbeddingsArgs(BaseModel):
    preproc_path: str = Field(description="Path to the  *_preprocessed.h5ad file to extract embeddings from.")
    work_dir: str = Field(description="Pre-sample folder created by load_h5ad_data")
    model_path: str = Field(default=settings.SCGPT_MODEL_DIR,scription="Directory containing scGPT weight and vocab.json.")
    device: str = Field(default="cuda", description="Device to run inference on: 'cuda' or 'cpu'.")

@tool(
    name="extract_embeddings_with_scgpt",
    description=(
        "Compute 512-dim cell embeddings with scGPT for the preprocessed AnnData. "
        "If only work_dir is given the function auto-detects <work_dir>/<sample>_preprocessed.h5ad. "
        "Writes <sample>_emb.h5ad in the same folder. "
        "Returns JSON {work_dir, embedding_path, n_cells, n_genes}."
    ),
    args_schema=ExtractEmbeddingsArgs
)
def extract_embeddings_with_scgpt(
    file_path: str,
    work_dir: str,
    model_path: str = settings.SCGPT_MODEL_DIR,
    device: str = "cuda"
) -> str:
    """
    Extract cell embeddings using scGPT model and save the result as a new .h5ad file.
    """

    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    preprocessed_path = os.path.join(output_dir, f"{base_file_name}_preprocessed.h5ad")
    embedding_output_path = os.path.join(output_dir, f"{base_file_name}_with_embedding.h5ad")

    model = ScGPTModelWrapper.from_pretrained(pretrained_model_name_or_path=model_path, device=device)
    adata_emb = model.extract_sample_embedding(
        adata_or_file=file_path,
        gene_col="gene_name",
        max_length=1200,
        cell_embedding_mode="cls",
        batch_size=64,
        obs_to_save=None,
        return_new_adata=True,
    )

    adata_emb.write_h5ad(embedding_output_path)

    return f"Embedding extraction complete. Saved to: {embedding_output_path}"
