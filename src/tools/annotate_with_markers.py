# tools/annotate_with_cellrag.py
import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
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
    
    return output_file


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

    return out_dir

        
class CellAnnoArgs(BaseModel):
    clustered_path: str = Field(description="Path to the clustered AnnData h5ad file.")
    diff_gene_path: str = Field(description="File that have undergone cell type matching")
    work_dir: str = Field(..., description="Per-sample folder created by load_h5ad_data.")


@tool(
    "annotate_with_markers",
    args_schema=CellAnnoArgs,
)
def annotate_with_markers(
    clustered_path: str,
    diff_gene_path: str,
    work_dir: str,
) -> str:
    """
    Annotate clusters using marker genes. Return annotations result path.
    """
    work = Path(work_dir).expanduser().resolve()
    clustered_path = Path(clustered_path).expanduser().resolve()
    diff_gene_path = Path(diff_gene_path).expanduser().resolve()
    sc.settings.figdir = str(work_dir)

    if not clustered_path.exists():
        raise FileNotFoundError(f"Clustered file not found: {clustered_path}")
    if not diff_gene_path.exists():
        raise FileNotFoundError(f"Marker gene file not found: {diff_gene_path}")

    anno_h5ad = work / "annotated_with_celltype.h5ad"
    anno_candidate_path = work / "cluster_celltype_annotation.txt"
    anno_result_path = work / "cluster_celltype_rank1.csv"

    if anno_h5ad.exists() and anno_candidate_path.exists() and anno_result_path.exists():
        return json.dumps(
            {
                "work_dir": str(work),
                "annoted_Path": str(anno_h5ad),
                "anno_candidate": str(anno_candidate_path),
                "anno_result": str(anno_result_path),
            }
        )

    adata = sc.read_h5ad(clustered_path)
    diff_gene_df = pd.read_csv(diff_gene_path)

    marker_df = pd.read_csv(settings.MARKER_GENE_FILE)

    anno_candidate_file = manual_cluster_annotation(adata, marker_df, diff_gene_df, str(work_dir))
    anno_result_file = generate_rank1_in_cluster(work_dir)

    df = pd.read_csv(anno_result_file, dtype={'Cluster': str, 'CellType': str})
    cluster_to_celltype = dict(zip(df.Cluster, df.CellType))
    adata.obs['pred_celltype'] = adata.obs['scGPT_clusters'].map(cluster_to_celltype)
    adata.write_h5ad(anno_h5ad)
    sc.pl.umap(adata,color="pred_celltype",save="_umap_annoted.png",show=False, title='Cell Type UMAP', frameon=False)

    sc.tl.rank_genes_groups(
        adata,
        groupby="pred_celltype",
        method="wilcoxon",
        layer='X_log1p',
        use_raw=False,
    )

    sc.pl.rank_genes_groups_dotplot(
    adata, groupby="pred_celltype", standard_scale="var", n_genes=5,save = "_dotplot_annoted.png", show=False
    )

    return json.dumps({
        "work_dir": str(work),
        "annoted_Path": str(anno_h5ad),
        "anno_candidate": str(anno_candidate_file),
        "anno_result": str(anno_result_file)
    })