# tools/clustering_marker.py
import json
from pathlib import Path
import scanpy as sc
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config.setting import settings

class ClusterMarkerArgs(BaseModel):
    embedding_path: str = Field(..., description="Path to the *_emb.h5ad file with cell embeddings.")
    work_dir: str = Field(..., description="Per-sample folder created by load_h5ad_data.")

@tool(
        "cluster_and_rank_markers",
        args_schema=ClusterMarkerArgs
        )
def cluster_and_rank_markers(
    embedding_path: str, 
    work_dir: str,
    ) -> str: 
    """
    Perform clustering and marker gene ranking on the embedded AnnData.
    Saves UMAP plots and marker gene tables into the work directory.
    """
    work = Path(work_dir).expanduser().resolve()
    emb = Path(embedding_path).expanduser().resolve()
    if not emb.exists():
        raise FileNotFoundError(f"Embedding file {embedding_path} does not exist.")
    if not work.exists():
        raise FileNotFoundError(f"Work directory {work_dir} does not exist.")
    
    sample = work.name
    clustered_path = work / f"{sample}_clustered.h5ad"
    matched_path = work / f"{sample}_matched.csv"
    
    adata = sc.read_h5ad(emb)

    sc.pp.neighbors(adata,use_rep="X_scGPT", n_neighbors=15)

    sc.tl.leiden(
        adata,
        key_added="scGPT_clusters",
        resolution=0.5,
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
    )

    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    marker_df = pd.DataFrame({
        group: result["names"][group][:100]
        for group in groups
    })
    marker_df = marker_df.melt(var_name="group", value_name="names")

    # Using Celltype Markergene reference
    if settings.MARKER_REFERENCE_PATH:
            reference_df = pd.read_csv(settings.MARKER_REFERENCE_PATH)
            reference_grouped = reference_df.groupby("celltype")["markergene"].apply(set).to_dict()
            # Add an overlap rating column
            def compute_overlap_score(g):
                candidate = set(marker_df[marker_df["group"] == g]["names"].tolist())
                overlaps = {ct: len(candidate & ref_genes) for ct, ref_genes in reference_grouped.items()}
                return sorted(overlaps.items(), key=lambda x: x[1], reverse=True)

            marker_df["matched_celltype"] = marker_df["group"].apply(lambda g: compute_overlap_score(g)[:3])

    
    marker_df.to_csv(matched_path, index=False)

    adata.var.index.name = None
    adata.write_h5ad(clustered_path)

    return json.dumps({
         "work_dir": str(work),
         "clustered_path": str(clustered_path),
         "matched_path": str(matched_path),
    })
