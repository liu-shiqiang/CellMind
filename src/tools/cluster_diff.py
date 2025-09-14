# tools/clustering_marker.py
import json
from pathlib import Path
import scanpy as sc
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from config.setting import settings



class ClusterMarkerArgs(BaseModel):
    embedding_path: str = Field(description="Path to the *_emb.h5ad file with cell embeddings.")
    work_dir: str = Field(description="Per-sample folder created by load_h5ad_data.")
    resolution: float = Field(default= 1.8 , description="Resolution for Leiden clustering.")  
    
@tool(
        "cluster_and_diff",
        args_schema=ClusterMarkerArgs
        )
def cluster_and_diff(
    embedding_path: str, 
    work_dir: str,
    resolution: float = 1.8,
    ) -> str: 
    """
    Perform clustering and differential expression analysis on the embedded AnnData.
    Saves UMAP plots and differentially expressed genes tables into the work directory.
    """
    work = Path(work_dir).expanduser().resolve()
    emb = Path(embedding_path).expanduser().resolve()
    if not emb.exists():
        raise FileNotFoundError(f"Embedding file {embedding_path} does not exist.")
    if not work.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    
    sample = work.name
    clustered_path = work / f"{sample}_clustered.h5ad"
    diff_gene_path = work / f"{sample}_diff_gene.csv"
    sc.settings.figdir = str(work)
    
    adata = sc.read_h5ad(emb)

    sc.pp.neighbors(adata,use_rep="X_scgpt", n_neighbors=15)

    sc.tl.leiden(
        adata,
        key_added="scGPT_clusters",
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False
    )

    sc.tl.umap(adata)
    sc.pl.umap(adata,color="scGPT_clusters",save="_umap_scgpt_clustered.png",show=False)

    adata.var_names = adata.var["gene_name"].astype(str)
    adata.var_names_make_unique()

    sc.tl.rank_genes_groups(
        adata,
        groupby="scGPT_clusters",
        method="wilcoxon",
        layer='X_log1p',
        use_raw=False,
    )

    # Compute gene difference 
    diff_genes_df = sc.get.rank_genes_groups_df(adata, group=None)
    result_df = pd.DataFrame({
    'cluster': diff_genes_df['group'].unique(),
    'top_20_diff_genes': [
        ', '.join(diff_genes_df[diff_genes_df['group'] == cluster]
                 .sort_values('scores', ascending=False)
                 .head(20)['names'].tolist())
        for cluster in diff_genes_df['group'].unique()
    ]
    })
    result_df.to_csv(diff_gene_path, index=False)

    adata.var.index.name = None
    adata.write_h5ad(clustered_path)

    return json.dumps({
         "work_dir": str(work),
         "clustered_path": str(clustered_path),
         "diff_gene_path": str(diff_gene_path),
    })
