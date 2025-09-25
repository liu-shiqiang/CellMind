import os
import json
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from src.scripts.rag import CellRag, BioKnowledgeRag

from src.scripts.utils import extract_json_from_response
from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper
from config.setting import settings

import matplotlib.pyplot as plt

model_path = settings.SCGPT_MODEL_DIR
output_dir = settings.OUTPUT_DIR

llm = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0.6,
    base_url="http://localhost:11434")

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

#只保留3w个细胞和基因测试
data_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/cima_test_set_final.h5ad"
adata = sc.read_h5ad(data_path)
adata = adata[:30000, :30000].copy()
temp_path = "/home/share/huadjyin/home/zhangzilin/test/debug_subset.h5ad"
adata.write_h5ad(temp_path)


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
        raw_adata_file_name=temp_path,   #adata_path,
        is_count_raw_data=True
    )

    adata_preprocessed = processor.preprocess_data(
        gene_vocab=os.path.join(model_path, "vocab.json"),
        output_dir=work_dir,
        use_raw=True,
        n_hvg=1200,
        gene_col="gene_name",
        min_gene_vocab_matched_frac=0.1, #测试加
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

    sc.pp.neighbors(adata_preprocessed, use_rep="X_scgpt", n_neighbors=10)#15)

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
    # 使用正确的列名
    celltype_col = 'Low-hierarchy cell types'
    marker_col = 'markergene'
    comp_gene_col = 'computational_gene'
    
    # === 新增：在保持逻辑不变的前提下进行限制 ===
    
    # 1. 过滤掉没有marker基因且计算基因也空的细胞类型
    valid_mask = (marker_df[marker_col].notna()) | (marker_df[comp_gene_col].notna())
    marker_df = marker_df[valid_mask].copy()
    print(f"有效细胞类型数量: {len(marker_df)}")
    
    # 2. 限制分析的细胞类型数量（保持算法逻辑，但减少计算量）
    if len(marker_df) > 40:
        print(f"细胞类型过多({len(marker_df)})，限制为前40个最常见类型")
        # 可以根据count列排序选择最常见的
        if 'count' in marker_df.columns:
            marker_df = marker_df.nlargest(40, 'count')
        else:
            marker_df = marker_df.head(40)
    
    # === 保持原有的核心逻辑不变 ===
    
    # 读取差异基因文件（使用传入的diff_gene_df）
    cluster_celltype = diff_gene_df
    diff_genes_df = diff_gene_df  # 这里赋值给局部变量

    # 将top_20_diff_genes列从字符串转换为列表
    diff_genes_df['top_20_diff_genes'] = diff_genes_df['top_20_diff_genes'].str.split(', ')
    
    # 3. 限制cluster数量（如果聚类太多）- 现在可以安全使用diff_genes_df了
    clusters_to_analyze = diff_genes_df['cluster'].unique()
    if len(clusters_to_analyze) > 30:
        print(f"聚类数量过多({len(clusters_to_analyze)})，限制为前30个cluster")
        clusters_to_analyze = clusters_to_analyze[:30]
    
    output_file = os.path.join(work_dir, 'cluster_celltype_annotation.txt')
    with open(output_file, 'w') as f:
        # 遍历所有cluster（使用限制后的cluster列表）
        for cluster in clusters_to_analyze:
            cluster_str = str(cluster)
            key = adata.obs[cluster_key] == cluster_str
            cluster_adata = adata[key].copy()
            
            # 获取该cluster的差异基因
            cluster_diff_genes = diff_genes_df[diff_genes_df['cluster'] == cluster]['top_20_diff_genes'].iloc[0]
            print(f"\nCluster {cluster} differential genes:", cluster_diff_genes, file=f)
            
            # 计算与所有细胞类型的匹配分数
            celltype_scores = []
            expr_scores = []
            
            # 遍历限制后的marker_df
            for idx, row in marker_df.iterrows():
                celltype = row[celltype_col]  # 使用正确的列名
                
                # 保持原有的基因选择逻辑完全不变
                if pd.notna(row[marker_col]) and row[marker_col]:
                    genes = [g.strip() for g in str(row[marker_col]).split(',') if g.strip()]
                    marker_genes = set(genes)
                else:
                    comp_genes = str(row[comp_gene_col]).split(',')
                    clean_genes = [g.strip() for g in comp_genes if g.strip()]
                    marker_genes = set(clean_genes[:20])  # 保持20个不变
                
                score = calculate_expression_weight(cluster_adata, marker_genes)
                expr_scores.append((celltype, score, marker_genes))
            
            # 保持原有的排序和选择逻辑完全不变
            expr_scores.sort(key=lambda x: x[1], reverse=True)
            top5_candidates = expr_scores

            all_overlaps_zero = True
            for celltype, expr_score, marker_genes in top5_candidates:
                overlap = len(set(cluster_diff_genes) & marker_genes)
                score = overlap / len(marker_genes) * expr_score
                celltype_scores.append((celltype, score, expr_score, overlap, marker_genes))    
                if overlap > 0:
                    all_overlaps_zero = False
            
            # 保持原有的排序逻辑完全不变
            if all_overlaps_zero:
                celltype_scores.sort(key=lambda x: x[2], reverse=True)
            else:
                celltype_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = celltype_scores[:10]  # 保持10个候选不变
            
            # 打印候选信息
            print(f"Top candidate cell types for Cluster {cluster}:", file=f)
            for i, (celltype, score, expr_score, overlap, marker_genes) in enumerate(top_candidates, 1):
                print(f"{i}. {celltype} (score: {score:.2f}, expr_score: {expr_score}, overlap: {overlap}, marker: {len(marker_genes)})", file=f)
                print(f"Marker genes for {celltype}: {marker_genes}", file=f)
            
            # 获取最佳匹配
            best_match = top_candidates[0][0] if top_candidates else 'Unknown'
            print(f"Suggested cell type for Cluster {cluster}: {best_match}", file=f)
            
            marker_dict = {}
            for idx, row in marker_df.iterrows():
                celltype = row[celltype_col]
                if pd.notna(row[marker_col]) and row[marker_col]:
                     genes = [g.strip() for g in str(row[marker_col]).split(',') if g.strip()]
                     marker_dict[celltype] = genes
                else:
                    comp_genes = str(row['computational_gene']).split(',')
                    clean_genes = [g.strip() for g in comp_genes if g.strip()][:10]
                    marker_dict[celltype] = clean_genes

    # 跳过Neo4j查询，直接使用本地marker list进行匹配
    matched_results = []
    for group in diff_gene_df["cluster"].unique():
        gene_raw = diff_gene_df[diff_gene_df["cluster"] == str(group)]["top_20_diff_genes"].iloc[0]
        if isinstance(gene_raw, list):
             gene_list = gene_raw
        else:
            gene_list = gene_raw.split(',')
        gene_set = set(g.strip() for g in gene_list)
        
        # 本地匹配逻辑
        for idx, row in marker_df.iterrows():
            celltype = row['Low-hierarchy cell types']
            if pd.notna(row['markergene']) and row['markergene']:
                marker_genes = set([g.strip() for g in str(row['markergene']).split(',') if g.strip()])
            else:
                comp_genes = str(row['computational_gene']).split(',')
                marker_genes = set([g.strip() for g in comp_genes if g.strip()][:20])
            
            matched_genes = gene_set & marker_genes
            score = len(matched_genes)
            
            if score > 0:
                matched_results.append({
                    "group": group,
                    "matched_celltype": celltype,
                    "score": score,
                    "matched_genes": ",".join(matched_genes)
                })
    
    matched_df = pd.DataFrame(matched_results)

    sample = Path(work_dir).name
    clustered_path = Path(work_dir) / f"{sample}_clustered.h5ad"
    matched_path = Path(work_dir) / f"{sample}_matched.csv"

    matched_df.to_csv(matched_path, index=False)

    adata.var.index.name = None
    adata.write_h5ad(clustered_path)

    db = CellRag(chromadb_path=settings.CHROMADB_PERSIST_DIR, collection_name=settings.CHROMADB_cell_collection_name)

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

        prompt = anno_prompt(cluster_id, match_text, neighbor_text, lit_context)

        print(prompt)

        response = llm.invoke(prompt)
    
        output_subdir = Path(work_dir) / f"{sample}clusters_llm_explanation"
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir,f"cluster_{cluster_id}_explanation.md"), "w") as f:
            f.write(prompt + "\n\n---\n\n" + response.content)

        parsed = extract_json_from_response(response.content)
        if parsed and "celltype" in parsed:
            cluster2celltype[str(cluster_id)] = parsed["celltype"]
        else:
            print(f"cluster {cluster_id} failed to parse successfully, marked as Unknown")
            cluster2celltype[str(cluster_id)] = "Unknown"  

    cluster2celltype_path = Path(work_dir) / "cluster_celltype_map.json"
    with open(cluster2celltype_path, "w") as f:
        json.dump(cluster2celltype, f, indent=2)

    print("\n All cluster annotations have been completed and saved to cluster_celltype_map.json")
    adata.obs["scGPT_celltype"] = adata.obs["scGPT_clusters"].astype(str).map(cluster2celltype)
    anno_umap_path = Path(work_dir) / f"{sample}_llm_celltypes.png"
    
    

    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color="scGPT_celltype", show=False)
    plt.savefig(anno_umap_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"UMAP visualization of cell types saved to {anno_umap_path}")
    print(" UMAP visualization image of cell types with LLM annotations generated ")
    
    # === 1. 用 LLM 精选 marker ===
    selected_markers, reasoning = select_discriminative_markers_with_llm(
        diff_genes_df=diff_genes_df,
        cluster_celltype_map=cluster2celltype,
        n_markers_per_cluster=8
    )

# 可选：保存 reasoning（调试用）
    (Path(work_dir) / "marker_selection_reasoning.md").write_text(reasoning)

# === 2. 生成优化 dotplot ===
    generate_optimized_dotplot(
        adata=adata,
        selected_markers=selected_markers,
        work_dir=work_dir,
        sample=sample
    )

# === 3. 基于精选 marker 做文献解读 ===
    generate_celltype_interpretation_with_selected_markers(
        adata=adata,
        selected_markers=selected_markers,
        work_dir=work_dir
    )

    return None


def select_discriminative_markers_with_llm(diff_genes_df, cluster_celltype_map, n_markers_per_cluster=8):
    """
    使用LLM选择能够区分不同细胞类型的marker基因
    """
    # 构建cluster到细胞类型的映射
    cluster_to_celltype = {}
    for cluster_id, celltype in cluster_celltype_map.items():
        cluster_to_celltype[f"Cluster_{cluster_id}"] = celltype
    
    # 准备LLM提示词
    prompt = f"""
您是一个单细胞数据分析专家。请从每个cluster的差异基因中选择最具区分度的marker基因。

聚类注释结果:
{json.dumps(cluster_to_celltype, indent=2)}

每个cluster的top 20差异基因:
{json.dumps(diff_genes_df.set_index('cluster')['top_20_diff_genes'].to_dict(), indent=2)}

请为每个cluster选择 {n_markers_per_cluster} 个最能代表其细胞类型且能与其他cluster区分的基因。
优先选择:
1. 该细胞类型的经典marker基因
2. 在其他cluster中表达量较低的基因
3. 具有明确生物学功能的基因

输出JSON格式:
{{
    "selected_markers": {{
        "Cluster_0": ["gene1", "gene2", ...],
        "Cluster_1": ["gene1", "gene2", ...],
        ...
    }},
    "reasoning": "简要说明选择理由"
}}
"""
    
    try:
        response = llm.invoke(prompt)
        result = extract_json_from_response(response.content)
        
        if result and "selected_markers" in result:
            return result["selected_markers"], result.get("reasoning", "")
        else:
            print("LLM返回格式错误，使用备用方案")
            return select_markers_fallback(diff_genes_df, n_markers_per_cluster)
            
    except Exception as e:
        print(f"LLM选择marker失败: {e}, 使用备用方案")
        return select_markers_fallback(diff_genes_df, n_markers_per_cluster)

def select_markers_fallback(diff_genes_df, n_markers_per_cluster):
    """备用方案：基于基因表达特异性选择marker"""
    selected_markers = {}
    
    for _, row in diff_genes_df.iterrows():
        cluster = f"Cluster_{row['cluster']}"
        genes = row['top_20_diff_genes']
        
        if isinstance(genes, str):
            genes = genes.split(', ')
        
        # 简单选择前n个基因
        selected_markers[cluster] = genes[:n_markers_per_cluster]
    
    return selected_markers, "使用前N个差异基因作为备用方案"

def generate_optimized_dotplot(adata, selected_markers, work_dir, sample):
    """
    生成优化布局的dotplot
    """
    try:
        # 计算合适的图形尺寸
        n_clusters = len(selected_markers)
        n_genes_total = sum(len(genes) for genes in selected_markers.values())
        
        # 动态调整图形大小
        fig_width = max(12, n_genes_total * 0.4)  # 每个基因0.4英寸
        fig_height = max(8, n_clusters * 1.2)     # 每个cluster 1.2英寸
        
        print(f"生成dotplot: {n_clusters}个cluster, {n_genes_total}个基因")
        print(f"图形尺寸: {fig_width:.1f}x{fig_height:.1f}英寸")
        
        # 创建dotplot
        dot_plot = sc.pl.dotplot(
            adata,
            var_names=selected_markers,
            groupby='scGPT_celltype',
            standard_scale='var',
            dendrogram=True,
            figsize=(fig_width, fig_height),
            show=False
        )
        
        # 调整布局
        dotplot_path = Path(work_dir) / f"{sample}_optimized_dotplot.pdf"
        plt.savefig(dotplot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ 优化dotplot保存到: {dotplot_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成dotplot失败: {e}")
        return generate_alternative_visualization(adata, selected_markers, work_dir, sample)

def generate_alternative_visualization(adata, selected_markers, work_dir, sample):
    """备用可视化方案"""
    try:
        # 使用热图作为备用
        import seaborn as sns
        
        # 准备数据
        expression_data = []
        for cluster_name, genes in selected_markers.items():
            cluster_num = cluster_name.replace('Cluster_', '')
            celltype = adata.obs[adata.obs['scGPT_clusters'] == cluster_num]['scGPT_celltype'].iloc[0]
            
            for gene in genes:
                if gene in adata.var_names:
                    cluster_mask = adata.obs['scGPT_clusters'] == cluster_num
                    mean_expr = np.mean(adata[cluster_mask, gene].X.toarray())
                    
                    expression_data.append({
                        'CellType': celltype,
                        'Gene': gene,
                        'MeanExpression': mean_expr,
                        'Cluster': cluster_name
                    })
        
        df = pd.DataFrame(expression_data)
        
        # 创建热图
        pivot_df = df.pivot_table(index='Gene', columns='CellType', 
                                 values='MeanExpression', aggfunc='mean')
        
        plt.figure(figsize=(max(12, len(pivot_df.columns) * 1.5), 
                           max(8, len(pivot_df.index) * 0.6)))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f',
                   cbar_kws={'label': 'Mean Expression'})
        plt.title('Marker Gene Expression by Cell Type')
        plt.tight_layout()
        
        heatmap_path = Path(work_dir) / f"{sample}_marker_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 热图备用方案保存到: {heatmap_path}")
        return True
        
    except Exception as e:
        print(f"❌ 热图生成也失败: {e}")
        return False

def generate_celltype_interpretation_with_selected_markers(adata, selected_markers, work_dir):
    """
    使用选择的marker基因进行文献解读
    """
    interpretations = {}
    
    try:
        bioknowledgerag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
        bioknowledgerag.init_vector_store(settings.CHROMADB_lit_collection_name)
    except Exception as e:
        print(f"❌ RAG初始化失败: {e}")
        return generate_generic_interpretations(selected_markers)
    
    # 为每个细胞类型解读其选择的marker
    celltype_to_markers = {}
    for cluster_name, markers in selected_markers.items():
        cluster_num = cluster_name.replace('Cluster_', '')
        celltype = adata.obs[adata.obs['scGPT_clusters'] == cluster_num]['scGPT_celltype'].iloc[0]
        
        if celltype not in celltype_to_markers:
            celltype_to_markers[celltype] = []
        celltype_to_markers[celltype].extend(markers)
    
    # 去重
    for celltype in celltype_to_markers:
        celltype_to_markers[celltype] = list(set(celltype_to_markers[celltype]))
    
    # 进行RAG查询
    for celltype, markers in celltype_to_markers.items():
        print(f"解读 {celltype} 的marker基因: {markers}")
        
        marker_interpretations = []
        for marker in markers[:5]:  # 每个细胞类型最多解读5个marker
            try:
                query = f"""Biological function and clinical significance of {marker} gene in {celltype} cells.
                Role as cell marker, expression pattern, and functional importance."""
                
                docs = bioknowledgerag.query(query, settings.CHROMADB_lit_collection_name, top_k=3)
                
                if docs:
                    summary = summarize_marker_function(docs, marker, celltype)
                    marker_interpretations.append(f"{marker}: {summary}")
                else:
                    marker_interpretations.append(f"{marker}: Known marker for {celltype}")
                    
            except Exception as e:
                print(f"解读 {marker} 时出错: {e}")
                marker_interpretations.append(f"{marker}: Cell type marker")
        
        interpretations[celltype] = "; ".join(marker_interpretations)
    
    # 保存结果
    interp_path = Path(work_dir) / "celltype_marker_interpretation.json"
    with open(interp_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    
    print(f"✅ Marker基因解读保存到: {interp_path}")
    return interpretations

def summarize_marker_function(docs, marker, celltype):
    """从文献中提取marker功能信息"""
    content = " ".join([doc.page_content for doc in docs[:2]])
    
    # 提取关键句子
    key_sentences = []
    for doc in docs[:2]:
        sentences = doc.page_content.split('.')
        for sentence in sentences:
            if (marker.lower() in sentence.lower() and 
                any(keyword in sentence.lower() for keyword in ['function', 'role', 'express', 'marker'])):
                key_sentences.append(sentence.strip() + '.')
    
    if key_sentences:
        return " ".join(key_sentences[:2])
    else:
        return f"Functions as a marker for {celltype} cells"

def generate_generic_interpretations(selected_markers):
    """生成通用解读"""
    interpretations = {}
    known_markers = {
        'CD4': 'Helper T cell marker',
        'CD8': 'Cytotoxic T cell marker', 
        'CD19': 'B cell marker',
        'CD14': 'Monocyte marker',
        'CD56': 'NK cell marker',
        'CD3': 'T cell marker',
        'MS4A1': 'B cell marker (CD20)',
        'IL7R': 'T cell marker (CD127)',
        'GZMB': 'Cytotoxic granule protein',
        'GNLY': 'NK and cytotoxic T cell marker',
        # 添加更多已知marker...
    }
    
    for cluster_name, markers in selected_markers.items():
        celltype = f"Cluster_{cluster_name.replace('Cluster_', '')}"
        interpretations[celltype] = []
        
        for marker in markers:
            if marker in known_markers:
                interpretations[celltype].append(f"{marker}: {known_markers[marker]}")
            else:
                interpretations[celltype].append(f"{marker}: Cell type marker")
    
    return interpretations


if __name__ == "__main__":
    data_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/cima_test_set_final.h5ad"
    cell_anno(adata_path=data_path)