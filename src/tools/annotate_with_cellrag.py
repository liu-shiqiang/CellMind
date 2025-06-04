# tools/annotate_with_cellrag.py
import os
import json
import scanpy as sc
import pandas as pd
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.scripts.utils import extract_json_from_response
from src.scripts.rag import CellRag
from config.setting import settings

llm = ChatOllama(model="deepseek-r1:32b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )

class CellRAGAnnoArgs(BaseModel):
    clustered_path: str = Field(description="Path to the clustered AnnData h5ad file.")
    matched_path: str = Field(description="File that have undergone cell type matching")
    work_dir: str = Field(..., description="Per-sample folder created by load_h5ad_data.")

def build_anno_prompt(clusters_id, retrieved_dos, matched_df):

    prompt = f"""
You are responsible for annotating cell types on single-cell RNA-seq data.
The single-cell data has been preprocessed, using scgpt to extract embeddings, clustering embeddings.
Now you need to combine external knowledge to determine the cell type of each cluster.



"""

def anno_prompt(cluster_id, retrieved_cell_context, matched_df):

    marker_genes = matched_df[matched_df["group"] == cluster_id]["names"].tolist()
    neighbors = [doc for doc in retrieved_cell_context["documents"][0]]

    match_info = matched_df[matched_df["group"] == cluster_id]["matched_celltype"].tolist()
    match_text = ""
    if match_info and isinstance(match_info[0], list):
        match_text += "\n\nCell types that match known reference markers (in descending order of matching degree):\n"
        for celltype, score in match_info[0]:
            match_text += f"- {celltype} (overlap={score})\n"

    prompt = f"""
You are responsible for annotating cell types on single-cell RNA-seq data.
The single-cell data has been preprocessed, using scgpt to extract embeddings, clustering embeddings.
Now you need to combine external knowledge to determine the cell type of each cluster

Objective: Please assign a unique cell type label (English name) to cluster {cluster_id} based on the following information, and explain the criteria for judgment.

Retrieve neighboring cells:
{chr(10).join(neighbors)}

The candidate marker genes for this cluster are:
{', '.join(marker_genes)}
{match_text}

Please output in the following JSON format:
{{
"cluster_id": {cluster_id},
celltype ":"<English name of cell type>",
reason ":"<Explain the reason>"
}}

"""
    return prompt
        

@tool(
    "annotate_with_cellrag",
    args_schema=CellRAGAnnoArgs,
)
def annotate_with_cellrag(
    clustered_path: str,
    matched_path: str,
    work_dir: str,
) -> str:
    """
    Annotate clusters using Cell-RAG vector database. Return annotations result path.
    """
    work = Path(work_dir).expanduser().resolve()
    clustered_path = Path(clustered_path).expanduser().resolve()
    matched_path = Path(matched_path).expanduser().resolve()

    if not clustered_path.exists():
        raise FileNotFoundError(f"Clustered file not found: {clustered_path}")
    if not matched_path.exists():
        raise FileNotFoundError(f"Marker gene file not found: {matched_path}")

    adata = sc.read_h5ad(clustered_path)
    matched_df = pd.read_csv(matched_path)

    db = CellRag(chromadb_path="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/chroma_data", collection_name="my_collection")

    cluster_ids = sorted(set(adata.obs["scGPT_clusters"]))
    cluster2celltype = {}

    for cluster_id in cluster_ids:
        mean_emb = adata[adata.obs["scGPT_clusters"] == cluster_id].obsm["X_scGPT"].mean(axis=0).tolist()
        retrieved_cell_context = db.query(mean_emb, n_results=settings.RETRIVE_TOP_K)

        prompt = anno_prompt(cluster_id, retrieved_cell_context, matched_df)

        response = llm.invoke(prompt)
        sample = work.name
        output_subdir = work / f"{sample}clusters_llm_explanation"
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir,f"cluster_{cluster_id}_explanation.md"), "w") as f:
            f.write(prompt + "\n\n---\n\n" + response.content)

        parsed = extract_json_from_response(response.content)
        if parsed and "celltype" in parsed:
            cluster2celltype[str(cluster_id)] = parsed["celltype"]
        else:
            print(f"cluster {cluster_id} failed to parse successfully, marked as Unknown")
            cluster2celltype[str(cluster_id)] = "Unknown"  

    cluster2celltype_path = work / "cluster_celltype_map.json"
    with open(cluster2celltype_path, "w") as f:
        json.dump(cluster2celltype, f, indent=2)

    print("\n All cluster annotations have been completed and saved to cluster_celltype_map.json")
    adata.obs["scGPT_celltype"] = adata.obs["scGPT_clusters"].astype(str).map(cluster2celltype)
    anno_umap_path = work / f"{sample}_llm_celltypes.png"
    sc.pl.umap(adata, color="scGPT_celltype", save=anno_umap_path, show=False)
    print(" UMAP visualization image of cell types with LLM annotations generated ")

    return json.dumps({
        "work_dir": str(work),
        "anno_umap_path": str(anno_umap_path),
        "cluster_celltype_map": str(cluster2celltype_path),
    })   
    # initial_annotations = {}
    # low_confidence_clusters = []

    # for cluster_id in cluster_ids:
    #     mean_emb = adata[adata.obs["scGPT_clusters"] == cluster_id].obsm["X_scGPT"].mean(axis=0).tolist()
    #     retrieved_docs = db.query(mean_emb, n_results=top_k)

    #     if not retrieved_docs:
    #         low_confidence_clusters.append(str(cluster_id))
    #         continue
        
    #     best_match = retrieved_docs[0]
    #     if best_match["score"] >= similarity_threshold:
    #         initial_annotations[str(cluster_id)] = best_match["metadata"]["cell_type"]
    #     else:
    #         low_confidence_clusters.append(str(cluster_id))

    # return json.dumps({
    #     "initial_annotations": initial_annotations,
    #     "low_confidence_clusters": low_confidence_clusters
    # })
