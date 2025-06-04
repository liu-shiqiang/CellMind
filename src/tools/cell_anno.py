import os
import json
import torch
import scanpy as sc
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.tools import tool

from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper
from config.setting import settings


model_path = settings.SCGPT_MODEL_DIR
output_dir = settings.OUTPUT_DIR

def cell_anno(
        adata_path: str,
        ):
    
    path = Path(adata_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".h5ad":
        raise FileNotFoundError(f"File not found or not .h5ad: {path}")
    base_name = path.stem
    work_dir = Path(output_dir).expanduser().resolve()/base_name
    work_dir.mkdir(parents=True, exist_ok=True)

    processor = ScGPTDataProcessor(
        raw_adata_file_name=adata_path,
        is_count_raw_data=True
    )

    adata_preprocessed = processor.preprocess_data(
        gene_vocab=os.path.join(model_path, "vocab.json"),
        output_dir=work_dir,
        use_raw=True,
        n_hvg=1200,
        gene_col="gene_name",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ScGPTModelWrapper.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device=device,
    )

    adata_emb = model.extract_sample_embedding(
        adata_or_file=adata_preprocessed,
        gene_col="gene_name",  # 如果报错可以换成 'index'
        max_length=1200,
        cell_embedding_mode="cls",
        batch_size=64,
        obs_to_save=None,
        return_new_adata=True,
    )

    adata_preprocessed.obsm["X_scgpt"] = adata_emb.X.copy()

    sc.pp.neighbors(adata_preprocessed, use_rep="X_scgpt", n_neighbors=15)

    sc.tl.leiden(
        adata_preprocessed,
        key_added="scGPT_clusters",
        resolution=0.5,
        flavor="igraph",
        n_iterations=2,
        directed=False
    )

    sc.tl.umap(adata_preprocessed)
    sc.pl.umap(adata_preprocessed, color="scGPT_clusters", save="_umap_scgpt_clustered.png", show=False)

    adata_preprocessed.var_names = adata_preprocessed.var["gene_name"].astype(str)
    adata_preprocessed.var_names_make_unique()

    sc.tl.rank_genes_groups(
        adata_preprocessed,
        groupby="scGPT_clusters",
        method="wilcoxon",
    )

    result = adata_preprocessed.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    diff_expr_df = pd.DataFrame({
        group: result["name"][group][:100]
        for group in groups
    })
    diff_expr_df = diff_expr_df.melt(var_name="group", value_name="names")

    
    



       
    