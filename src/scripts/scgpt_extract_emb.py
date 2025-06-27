import os
import torch
import json

import scanpy as sc
import pandas as pd

from src.bio_pretrained_model.data_prep._scgpt_data_processor import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper

from config.setting import settings

scgpt_modeldir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/save/scgpt/scgpt_cima_1200"

def run_scgpt_inference(adata_path):

    processor = ScGPTDataProcessor(
        raw_adata_file_name=adata_path,
        is_count_raw_data=True
    )

    adata_preprocessed = processor.preprocess_data(
        gene_vocab=os.path.join(scgpt_modeldir, "vocab.json"),
        output_dir="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_cima_1200",
        use_raw=True,
        n_hvg=1200,
        gene_col="gene_name",
    )

    base_file_name = os.path.splitext(os.path.basename(adata_path))[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ScGPTModelWrapper.from_pretrained(
        pretrained_model_name_or_path=scgpt_modeldir,
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

    embedding_output_path = os.path.join("/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_cima_1200", f"{base_file_name}_with_embedding.h5ad")
    adata_preprocessed.write_h5ad(embedding_output_path)

    return adata_preprocessed

if __name__ == "__main__":

    adata_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/test_l3_stratified_5pct.h5ad"
    run_scgpt_inference(adata_path)
    print(f"Processed data with embeddings saved")