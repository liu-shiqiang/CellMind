import os
import torch
import json
from pathlib import Path
import scanpy as sc
import pandas as pd

from src.bio_pretrained_model.data_prep._scgpt_data_processor import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper

from config.setting import settings

scgpt_modeldir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/save/scgpt/scgpt_human"

def run_scgpt_inference(adata_path):

    processor = ScGPTDataProcessor(
        raw_adata_file_name=adata_path,
        is_count_raw_data=True
    )

    adata_preprocessed = processor.preprocess_data(
        gene_vocab=os.path.join(scgpt_modeldir, "vocab.json"),
        output_dir="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_human/b2af3247",
        use_raw=True,
        n_hvg=1200,
        gene_col="gene_name",
    )

    sc.pp.filter_cells(adata_preprocessed, min_genes=1)

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

    embedding_output_path = os.path.join("/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_human/b2af3247", f"{base_file_name}_with_embedding.h5ad")
    adata_preprocessed.write_h5ad(embedding_output_path)

    return adata_preprocessed
 

def cluster_ledien(adata_path:str):

    adata = sc.read_h5ad(adata_path)
    
    sc.pp.neighbors(adata,use_rep="X_scgpt", n_neighbors=15)
    for res in [0.2,0.5,0.8,1.2,1.5,1.8,2.1,2.4,2.7]:
        sc.tl.leiden(
            adata,
            key_added=f"leiden_res_{res:4.2f}",
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            directed=False
        )

    sc.tl.umap(adata)

    clustered_path = os.path.join(os.path.dirname(adata_path), "test_row_clustered.h5ad")

    adata.write_h5ad(clustered_path)

    return None


def run_scgpt_and_cluster(adata_path):
    processor = ScGPTDataProcessor(
        raw_adata_file_name=adata_path,
        is_count_raw_data=True
    )

        # 提取不带扩展名的文件名
    file_stem = Path(adata_path).stem  # 得到 "9fd987e8"

    # 目标目录
    base_output_dir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_human"

    # 构造新文件夹路径
    new_dir_path = os.path.join(base_output_dir, file_stem)

    # 创建新目录（如果不存在）
    os.makedirs(new_dir_path, exist_ok=True)

    print(f"New directory created at: {new_dir_path}")

    adata_preprocessed = processor.preprocess_data(
        gene_vocab=os.path.join(scgpt_modeldir, "vocab.json"),
        output_dir=new_dir_path,
        use_raw=True,
        n_hvg=1200,
        gene_col="gene_name",
    )

    sc.pp.filter_cells(adata_preprocessed, min_genes=1)

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

    sc.pp.neighbors(adata_preprocessed,use_rep="X_scgpt", n_neighbors=15)
    for res in [0.2,0.5,0.8,1.2,1.5,1.8,2.1,2.4,2.7]:
        sc.tl.leiden(
            adata_preprocessed,
            key_added=f"leiden_res_{res:4.2f}",
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            directed=False
        )

    sc.tl.umap(adata_preprocessed)

    output_h5ad_path = os.path.join(new_dir_path, f"{file_stem}_filtered.h5ad")

    adata_preprocessed.write_h5ad(output_h5ad_path)

    print(f"Processed AnnData saved to: {output_h5ad_path}")

    return None




if __name__ == "__main__":

    adata_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/scgpt_human/b2af3247/b2af3247_with_embedding.h5ad"
    file_list = [
        "/home/share/huadjyin/home/zhangxinyu4/project/cell_type/analysize/method/9fd987e8.h5ad",
        "/home/share/huadjyin/home/zhangxinyu4/project/cell_type/analysize/method/e9a2cccd.h5ad",
        "/home/share/huadjyin/home/zhangxinyu4/project/cell_type/analysize/method/bf28d870.h5ad",
        "/home/share/huadjyin/home/zhangxinyu4/project/cell_type/analysize/method/fd9aef61.h5ad",
        "/home/share/huadjyin/home/zhangxinyu4/project/cell_type/analysize/method/c4a7e716.h5ad",
    ]
    for file_path in file_list:
        print(f"Processing file: {file_path}")
        run_scgpt_and_cluster(file_path)
    print(f"Processed data with embeddings saved")