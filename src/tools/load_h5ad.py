import os
import json
from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.tools import tool

from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
from config.setting import settings

class LoadH5ADArgs(BaseModel):
    file_path: str = Field(description="Path to the input .h5ad file.")
    output_dir: str = Field(default=settings.OUTPUT_DIR, description="Directory to save the processed files.")
    model_path: str = Field(default=settings.SCGPT_MODEL_DIR, description="Path to the pre-trained scGPT model directory.")
    n_hvg: PositiveInt = Field(default=1200, description="Number of highly variable genes to keep.")
    use_raw:bool = Field(default=True, description="Whether to use raw counts for preprocessing.")

@tool(
    name="load_h5ad_data",
    description=(
        "Pre-process a single0cell .h5ad file with the scGPT pipeline."
        "Creates a sub-folder <output_root>/<sample_name>/ and writes "
        "<sample_name>_preprocessed.h5ad there. "
        "Returns JSON {validated: bool, work_dir: str, preproc_path: str}"
        ),
    args_schema=LoadH5ADArgs,
    return_direct=False,
)
def load_h5ad_data(
    file_path: str,
    output_dir: str = settings.OUTPUT_DIR,
    model_path: str = settings.SCGPT_MODEL_DIR,
    n_hvg: int = 1200,
    use_raw: bool = True
) -> str:
    """
    Preprocess an h5ad file with scGPT and save results in a dedicated sub-folder.

    JSON return payload
    -------------------
    {
      "validated": true,
      "work_dir": "<absolute path to per-file folder>",
      "preproc_path": "<absolute path to *_preprocessed.h5ad>"
    }
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".h5ad":
        raise FileNotFoundError(f"File not found or not .h5ad: {path}")
    
    base_name = path.stem
    work_dir = Path(output_dir).expanduser().resolve()/base_name
    work_dir.mkdir(parents=True, exist_ok=True)

    preproc_path = work_dir / f"{base_name}_preprocessed.h5ad"

    try:
        processor = ScGPTDataProcessor(
            raw_adata_file_name=file_path,
            is_count_raw_data=True
        )

        adata_preprocessed = processor.preprocess_data(
            gene_vocab=os.path.join(model_path, "vocab.json"),
            output_dir=str(work_dir),
            use_raw=use_raw,
            n_hvg=n_hvg,
            gene_col="gene_name",
        )
        adata_preprocessed.write_h5ad(preproc_path)
        if not preproc_path.exists():
            raise FileNotFoundError(f"Preprocessing finished but file not found: {preproc_path}")
    except Exception as exc:
        raise RuntimeError(f"scGPT preprocessing failed: {exc}")

    return json.dumps(
        {
            "validated": True,
            "work_dir": str(work_dir),
            "preproc_path": str(preproc_path),
        }
    )