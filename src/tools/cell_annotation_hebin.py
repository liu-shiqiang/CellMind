import os
import json
import torch
import scanpy as sc
import pandas as pd
import numpy as np
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
    sc.pl.umap(adata_preprocessed, color="scGPT_clusters", save= "_umap_scgpt_clustered.png", show=False)

    adata_preprocessed.var_names = adata_preprocessed.var["gene_name"].astype(str)
    adata_preprocessed.var_names_make_unique()

    sc.tl.rank_genes_groups(
        adata_preprocessed,
        groupby="scGPT_clusters",
        method="wilcoxon",
        use_raw=False,
    )

    # Compute gene difference 
    diff_genes_df = sc.get.rank_genes_groups_df(adata_preprocessed, group=None)
    result_df = pd.DataFrame({
    'cluster': diff_genes_df['group'].unique(),
    'top_20_diff_genes': [
        ', '.join(diff_genes_df[diff_genes_df['group'] == cluster]
                 .sort_values('scores', ascending=False)
                 .head(20)['names'].tolist())
        for cluster in diff_genes_df['group'].unique()
    ]
    })
    path_dif_genes = os.path.join(work_dir, 'diff_genes.csv')
    # 4. 保存为CSV（保留引号）
    result_df.to_csv(path_dif_genes, index=False, quoting=1)
    
    # result = adata_preprocessed.uns["rank_genes_groups"]
    # groups = result["names"].dtype.names

    # deg_df = pd.DataFrame({
    #     group: result["name"][group][:100]
    #     for group in groups
    # })
    # deg_df = deg_df.melt(var_name="group", value_name="names")
    marker_df = pd.read_csv(settings.Blood)
    diff_gene_df = result_df
    # cluster中计算每个celltype的Marker基因的表达量之和×overlap / N，选取排名前K的celltype作为推荐注释的celltype
    manual_cluster_annotation(adata_preprocessed, marker_df, diff_gene_df, str(work_dir))  
    generate_rank1_in_cluster(work_dir)


def generate_rank1_in_cluster(work_dir):
    with open(os.path.join(work_dir, 'cluster_celltype_annotation.txt'), 'r') as f:
        lines = f.readlines()   

    results = []
    current_cluster = None
    for line in lines:
        if line.startswith('Cluster '):
            current_cluster = line.split()[1].strip(':')
        if 'Suggested cell type for Cluster' in line:
            celltype = line.split(':')[-1].strip()
            results.append({'Cluster': f'{current_cluster}', 'CellType': celltype})
    out_dir = os.path.join(work_dir, 'cluster_celltype_rank1.csv')
    pd.DataFrame(results).to_csv(out_dir, index=False)

def calculate_expression_weight(adata, marker_genes):
    # 筛选出存在于adata中的marker基因
    valid_genes = [gene for gene in marker_genes if gene in adata.var_names]
    
    if not valid_genes:
        return 0.0  # 无有效基因时返回0权重
    
    # 计算这些基因的平均表达量（稀疏矩阵安全处理）
    expr_matrix = adata[:, valid_genes].X.toarray()      
    mean_expression = np.mean(expr_matrix, axis=0)  # 按基因求均值
    return np.sum(mean_expression)  # 返回所有基因的平均值

def manual_cluster_annotation(adata, marker_df, diff_gene_df, work_dir, cluster_key="scGPT_clusters"):
    # 读取差异基因文件
    cluster_celltype = diff_gene_df
    diff_genes_df = diff_gene_df

    # 将top_20_diff_genes列从字符串转换为列表
    diff_genes_df['top_20_diff_genes'] = diff_genes_df['top_20_diff_genes'].str.split(', ')
    
    output_file = os.path.join(work_dir, 'cluster_celltype_annotation.txt')
    with open(output_file, 'w') as f:
        # 遍历所有cluster
        for cluster in diff_genes_df['cluster'].unique():
            cluster_str = str(cluster)
            key = adata.obs[cluster_key] == cluster_str
            cluster_adata = adata[key].copy()
            
            # 获取该cluster的差异基因
            cluster_diff_genes = diff_genes_df[diff_genes_df['cluster'] == cluster]['top_20_diff_genes'].iloc[0]
            print(f"\nCluster {cluster} differential genes:", cluster_diff_genes, file=f)
            # CellType_Composition = cluster_celltype[cluster_celltype['Cluster'] == cluster]['CellType_Composition'].iloc[0]
            # print(f"Cluster {cluster} cellType composition: {CellType_Composition}", file=f)
            
            # 计算与所有细胞类型的匹配分数
            celltype_scores = []
            expr_scores = []
            for idx, row in marker_df.iterrows():
                celltype = row['Low-hierarchy cell types']
                if pd.notna(row['markergene']) and row['markergene']:
                    genes = [g.strip() for g in str(row['markergene']).split(',') if g.strip()]
                    marker_genes = set(genes)
                else:
                    comp_genes = str(row['computational_gene']).split(',')
                    clean_genes = [g.strip() for g in comp_genes if g.strip()]
                    marker_genes = set(clean_genes[:20])
                score = calculate_expression_weight(cluster_adata, marker_genes)
                expr_scores.append((celltype, score, marker_genes))
            
            expr_scores.sort(key=lambda x: x[1], reverse=True)
            top5_candidates = expr_scores

            all_overlaps_zero = True
            for celltype, expr_score, marker_genes in top5_candidates:
                overlap = len(set(cluster_diff_genes) & marker_genes)
                score = overlap / len(marker_genes) * expr_score
                celltype_scores.append((celltype, score, expr_score, overlap, marker_genes))    
                if overlap > 0:
                    all_overlaps_zero = False
            
            # 按分数排序，获取前n个候选
            # celltype_scores.sort(key=lambda x: x[1], reverse=True)
            # top_candidates = celltype_scores[:5]
            # 按分数排序，获取前n个候选
            #celltype_scores.sort(key=lambda x: x[1], reverse=True)
            if all_overlaps_zero:
                # 所有overlap为0时按expr_score降序
                celltype_scores.sort(key=lambda x: x[2], reverse=True)
            else:
                # 否则按score降序，然后expr_score降序
                celltype_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = celltype_scores[:10]
            
            # 打印候选信息
            print(f"Top candidate cell types for Cluster {cluster}:", file=f)
            for i, (celltype, score, expr_score, overlap, marker_genes) in enumerate(top_candidates, 1):
                #marker = marker_df[marker_df['Low-hierarchy cell types'] == celltype]['markergene'].tolist()
                #marker = parse_genes(marker)
                print(f"{i}. {celltype} (score: {score:.2f}, expr_score: {expr_score}, overlap: {overlap}, marker: {len(marker_genes)})", file=f)
                print(f"Marker genes for {celltype}: {marker_genes}", file=f)
            
            # 获取最佳匹配的marker genes
            best_match = top_candidates[0][0] if top_candidates else 'Unknown'
            # suggested_markers = marker_df[marker_df['cell_type_ontology_term_id'] == best_match]['markergene'].tolist()
            # suggested_markers = suggested_markers[0] if suggested_markers else []
            # suggested_markers_str = ", ".join(suggested_markers) if suggested_markers else "N/A"
            
            # 用户选择
            print(f"Suggested cell type for Cluster {cluster}: {best_match}", file=f)
            #print(f"Marker genes for {best_match}: {suggested_markers_str}", file=f)


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

    data_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/test_l3_stratified_5pct.h5ad"
    cell_anno(adata_path=data_path)
       
    