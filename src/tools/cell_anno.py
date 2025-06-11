import os
import json
import torch
import scanpy as sc
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt

from neo4j import GraphDatabase
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from src.scripts.rag import CellRag, BioKnowledgeRag

from src.scripts.utils import extract_json_from_response
from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper
from config.setting import settings


model_path = settings.SCGPT_MODEL_DIR
output_dir = settings.OUTPUT_DIR
NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
NEO4J_DATABASE = settings.NEO4J_DATABASE

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

def anno_prompt(cluster_id, match_text, neighbor_text, lit_context) -> str:

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

Output in the following JSON format:
{{
    "cluster_id":{cluster_id},
    "celltype":"<English name of cell type>",
    "reason":"<Explain reasoning with reference to markers or neighbors>"
}}

"""


    return prompt

class CellAnnoArgs(BaseModel):
    adata_path: str = Field(description="Path to the input .h5ad file.")


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
    sc.pl.umap(adata_preprocessed, color="scGPT_clusters", save= work_dir + "_umap_scgpt_clustered.png", show=False)

    adata_preprocessed.var_names = adata_preprocessed.var["gene_name"].astype(str)
    adata_preprocessed.var_names_make_unique()

    sc.tl.rank_genes_groups(
        adata_preprocessed,
        groupby="scGPT_clusters",
        method="wilcoxon",
    )

    result = adata_preprocessed.uns["rank_genes_groups"]
    groups = result["names"].dtype.names

    deg_df = pd.DataFrame({
        group: result["name"][group][:100]
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

    sample = work_dir.name
    clustered_path = work_dir / f"{sample}_clustered.h5ad"
    matched_path = work_dir / f"{sample}_matched.csv"

    matched_df.to_csv(matched_path, index=False)

    adata_preprocessed.var.index.name = None
    adata_preprocessed.write_h5ad(clustered_path)

    db = CellRag(chromadb_path=settings.CHROMADB_PERSIST_DIR, collection_name=settings.CHROMADB_cell_collection_name)


    cluster_ids = sorted(set(adata_preprocessed.obs["scGPT_clusters"]))
    cluster2celltype = {}

    for cluster_id in cluster_ids:
        mean_emb = adata_preprocessed[adata_preprocessed.obs["scGPT_clusters"] == cluster_id].obsm["X_scgpt"].mean(axis=0).tolist()
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

        prompt = anno_prompt(cluster_id, match_text, neighbor_text, lit_context)

        print(prompt)

        response = llm.invoke(prompt)
    
        output_subdir = work_dir / f"{sample}clusters_llm_explanation"
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir,f"cluster_{cluster_id}_explanation.md"), "w") as f:
            f.write(prompt + "\n\n---\n\n" + response.content)

        parsed = extract_json_from_response(response.content)
        if parsed and "celltype" in parsed:
            cluster2celltype[str(cluster_id)] = parsed["celltype"]
        else:
            print(f"cluster {cluster_id} failed to parse successfully, marked as Unknown")
            cluster2celltype[str(cluster_id)] = "Unknown"  

    cluster2celltype_path = work_dir / "cluster_celltype_map.json"
    with open(cluster2celltype_path, "w") as f:
        json.dump(cluster2celltype, f, indent=2)

    print("\n All cluster annotations have been completed and saved to cluster_celltype_map.json")
    adata_preprocessed.obs["scGPT_celltype"] = adata_preprocessed.obs["scGPT_clusters"].astype(str).map(cluster2celltype)
    anno_umap_path = work_dir / f"{sample}_llm_celltypes.png"
    sc.pl.umap(adata_preprocessed, color="scGPT_celltype", save=anno_umap_path, show=False)
    print(" UMAP visualization image of cell types with LLM annotations generated ")
    
    return None

if __name__ == "__main__":

    data_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/cima_test_set_final.h5ad"
    cell_anno(adata_path=data_path)
       
    