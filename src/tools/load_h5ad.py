import os
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.tools import tool

from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
from config.setting import settings
from src.tools.artifact_paths import resolve_artifact_dir

class LoadH5ADArgs(BaseModel):
    file_path: str = Field(description="Path to the input .h5ad file.")

@tool(
    "load_h5ad_data_scgpt",
    args_schema=LoadH5ADArgs,
    return_direct=False,
)
def load_h5ad_data_scgpt(
    file_path: str,
    output_dir: Optional[str] = None,
    model_path: str = settings.SCGPT_MODEL_DIR,
) -> str:
    """
    Load and preprocess an h5ad file using the scGPT pipeline.

    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".h5ad":
        raise FileNotFoundError(f"File not found or not .h5ad: {path}")
    
    base_name = path.stem
    work_dir = resolve_artifact_dir(
        input_path=path,
        work_dir=output_dir,
        subdir=f"scgpt/{base_name}",
    )
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
            use_raw=True,
            n_hvg=1200,
            gene_col="gene_name",
        )
        adata_preprocessed.write_h5ad(preproc_path)
        if not preproc_path.exists():
            raise FileNotFoundError(f"Preprocessing finished but file not found: {preproc_path}")
    except Exception as exc:
        raise RuntimeError(f"scGPT preprocessing failed: {exc}")
    
    result={
            "work_dir": str(work_dir),
            "preproc_path": str(preproc_path),
        }
        

    return json.dumps(result)
    
