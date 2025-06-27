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
from src.scripts.rag import CellRag, BioKnowledgeRag
from config.setting import settings

llm = ChatOllama(model="deepseek-r1:32b", 
                 temperature=0.6,
                 base_url="http://localhost:11434"
                 )


def lit_rag(match_text: str, neighbor_text: str) -> str:
    bioknowledgerag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
    bioknowledgerag.init_vector_store(settings.CHROMADB_lit_collection_name)
    
    rag_query = f"""
You are identifying the most likely cell type for a cluster of single-cell RNA-seq data.

The cluster expresses the following marker genes:{match_text}
Its neareast neighbors in the embeddingspaces include cell type : {neighbor_text}

Retrieve scientific literature that discusses any of these genes in relation to spectific immune or tissue cell types.
"""
    result = bioknowledgerag.query(rag_query,settings.CHROMADB_lit_collection_name, top_k=5)

    context = "\n".join([doc.page_content for doc in result])

    return context
    
def extract_predined_celltypes(db,celltype_level: str = "celltype_l3") -> list[str]:
    
    all_metadata = db.get_all_metadata()  # Must return list of dicts

    # Extract non-null, unique celltype names
    allowed_celltypes = sorted({
        m[celltype_level].strip()
        for m in all_metadata
        if celltype_level in m and m[celltype_level] and isinstance(m[celltype_level], str)
    })

    return allowed_celltypes

def anno_prompt(cluster_id, match_text, neighbor_text, lit_context, predined_celltype) -> str:

    prompt = f"""You are a domain expert responsible for annotation cell clusters in a single_cell RNA-seq dataset.
The data has been clustered using scGPT-derived embeddings.
Now you are given supporting information to help assign a **biologically meaningful cell type** label to cluster {cluster_id}.

You will conside the following three sources of evidence:
1. Nearest neighbor cells from embedding space:
{neighbor_text}
2. Marker gene matching to known cell types:
{match_text}
3、Knowledge from external literature
{lit_context}
4、Your own biologicalprior knowledge

Please now detetmine the most likeing **Englist cell type name** for cluster{cluster_id},strictly at the L3 hierarchy level,and explain your reasoning briedly based on the above evidence
You MUST choose the cell type from the following predefined list (L3 level cell types):
{predined_celltype}
If none are appropriate, choose the **closest** match from this list and explain why.

Output in the following JSON format:
{{
    "cluster_id":{cluster_id},
    "celltype":"<English name of cell type>",
    "reason":"<Explain reasoning with reference to markers or neighbors>"
}}

"""


    return prompt
        
class CellRAGAnnoArgs(BaseModel):
    clustered_path: str = Field(description="Path to the clustered AnnData h5ad file.")
    matched_path: str = Field(description="File that have undergone cell type matching")
    work_dir: str = Field(..., description="Per-sample folder created by load_h5ad_data.")


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

    db = CellRag(chromadb_path=settings.CHROMADB_PERSIST_DIR, collection_name=settings.CHROMADB_cell_collection_name)

    predined_celltype = extract_predined_celltypes(db, celltype_level = "celltype_l3")

    cluster_ids = sorted(set(adata.obs["scGPT_clusters"]))
    cluster2celltype = {}

    for cluster_id in cluster_ids:
        mean_emb = adata[adata.obs["scGPT_clusters"] == cluster_id].obsm["X_scgpt"].mean(axis=0).tolist()
        cellcontext = db.query(mean_emb, n_results=5)
        print(cellcontext)

        cellcontext_meta  = cellcontext.get("metadatas",[[]])[0]
        print(cellcontext_meta)

        neighbor_text = ""
        for i, cell in enumerate(cellcontext_meta):
            sample = cell.get("sample", "Unknown")
            celltype_l1 = cell.get("celltype_l1","Unknown")
            celltype_l2 = cell.get("celltype_l2", "Unknown")
            celltype_l3 = cell.get("celltype_l3","Unknown")
            celltype_l4 = cell.get("final_annotation", "Unknown")
            n_genes = cell.get("n_genes_by_counts","?")
            total_counts = cell.get("total_counts","?")
            neighbor_text += f"-Neighbor{i+1}:(L1 = {celltype_l1}, l2 = {celltype_l2}, l3 = {celltype_l3}, L4 = {celltype_l4}, Sample = {sample}, Gene detected = {n_genes} Total counts = {total_counts})\n"
        
        print(neighbor_text)

        candidates = matched_df[matched_df["group"] == int(cluster_id)].to_dict("records")

        print(candidates)

        match_text = ""
        if candidates:
            match_text += "Based on the marker gene overlap with known references: \n"
            for cand in candidates:
                celltype = cand.get("matched_celltype","Unknown")
                score = cand.get("score", 0)
                genes = cand.get("matched_genes","")
                match_text += f"- {celltype} (Score: {score})\n marker gene mathced:{genes}\n"

        print(match_text)
        
        lit_context = lit_rag(match_text=match_text, neighbor_text=neighbor_text)

        print(match_text, neighbor_text, lit_context,sep="\n")

        prompt = anno_prompt(cluster_id, match_text, neighbor_text, lit_context, predined_celltype)

        print(prompt)

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
    adata.write_h5ad(work / f"{sample}_annotated.h5ad")
    sc.settings.figdir = str(work)
    sc.pl.umap(adata, color="scGPT_celltype", save="_llm_celltypes.png", show=False)
    print(" UMAP visualization image of cell types with LLM annotations generated ")

    return json.dumps({
        "work_dir": str(work),
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

if __name__ == "__main__":
    clustered_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/c_data/c_data_clustered.h5ad"
    matched_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/c_data/c_data_matched.csv"
    work_dir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/c_data"
    result = annotate_with_cellrag(clustered_path=clustered_path, matched_path=matched_path, work_dir=work_dir)
