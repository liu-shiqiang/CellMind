# tools/clustering_marker.py
import json
from pathlib import Path
import scanpy as sc
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from config.setting import settings


NEO4J_URI = "neo4j+s://c0651eec.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "eWRgS3xons7xBhxaoZM0fr1SJZeANZoS6d_334ykH1k"
NEO4J_DATABASE = "neo4j"


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
    sc.settings.figdir = str(work)
    
    adata = sc.read_h5ad(emb)

    sc.pp.neighbors(adata,use_rep="X_scgpt", n_neighbors=15)

    sc.tl.leiden(
        adata,
        key_added="scGPT_clusters",
        resolution=2.0,
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
        layer='X_log1p'
    )

    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    deg_df = pd.DataFrame({
        group: result["names"][group][:100]
        for group in groups
    })
    deg_df = deg_df.melt(var_name="group", value_name="names")

    def query_neo4j(gene_set: set):
        query = """
        UNWIND $genes AS gene_name
        MATCH (c:CellType)-[:MARKERED_BY]->(g:MarkerGene {name: gene_name})
        RETURN c.name AS celltype, collect(g.name) AS matched_genes, count(*) AS score
        ORDER BY score DESC

        """
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(query, genes=list(gene_set))
            top3 = []
            for record in records:
                top3.append({
                    "celltype": record["celltype"],
                    "score": record["score"],
                    "matched_genes": record["matched_genes"]
                })
                if len(top3) >= 3:
                    break
        driver.close()
        return top3

    matched_results = []
    for group in deg_df["group"].unique():
        gene_list = deg_df[deg_df["group"] == group]["names"].dropna().tolist()
        gene_set = set(gene_list)
        match_info = query_neo4j(gene_set)
        for m in match_info:
            matched_results.append({
                "group": group,
                "matched_celltype": m["celltype"],
                "score": m["score"],
                "matched_genes": ",".join(m["matched_genes"])
            })
    
    matched_df = pd.DataFrame(matched_results)

    # # Using Celltype Markergene reference
    # if settings.MARKER_REFERENCE_PATH:
    #         reference_df = pd.read_csv(settings.MARKER_REFERENCE_PATH)
    #         reference_grouped = reference_df.groupby("celltype")["markergene"].apply(set).to_dict()
    #         # Add an overlap rating column
    #         def compute_overlap_score(g):
    #             candidate = set(marker_df[marker_df["group"] == g]["names"].tolist())
    #             overlaps = {ct: len(candidate & ref_genes) for ct, ref_genes in reference_grouped.items()}
    #             return sorted(overlaps.items(), key=lambda x: x[1], reverse=True)

    #         marker_df["matched_celltype"] = marker_df["group"].apply(lambda g: compute_overlap_score(g)[:3])

    
    matched_df.to_csv(matched_path, index=False)

    adata.var.index.name = None
    adata.write_h5ad(clustered_path)

    return json.dumps({
         "work_dir": str(work),
         "clustered_path": str(clustered_path),
         "matched_path": str(matched_path),
    })
