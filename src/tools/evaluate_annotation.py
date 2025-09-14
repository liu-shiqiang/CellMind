import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.scripts.rag import BioKnowledgeRag, CellRag
from config.setting import settings

# 初始化LLM
llm = ChatOllama(model="qwen3:30b", 
                 temperature=0.6,
                 base_url="http://localhost:11434")

def read_cluster_annotation(annotation_file: str) -> dict:
    """
    读取cluster注释文件并解析内容
    """
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    clusters = {}
    current_cluster = None
    
    for line in lines:
        if line.startswith('Cluster '):
            current_cluster = line.split()[1].strip(':')
            clusters[current_cluster] = {
                'differential_genes': [],
                'candidates': [],
                'suggested_celltype': None
            }
        elif 'differential genes:' in line:
            genes_str = line.split(':')[1].strip().strip('[]')
            if genes_str:
                clusters[current_cluster]['differential_genes'] = [
                    gene.strip().strip("'") for gene in genes_str.split(',') if gene.strip()
                ]
        elif 'Suggested cell type for Cluster' in line:
            celltype = line.split(':')[-1].strip()
            clusters[current_cluster]['suggested_celltype'] = celltype
        elif '. ' in line and 'marker genes for' in line.lower():
            # 解析候选细胞类型信息
            pass
            
    return clusters


def query_gene_literature(gene_list: List[str], celltype: str) -> str:
    """
    使用BioKnowledgeRag查询基因相关文献信息
    """
    try:
        bioknowledgerag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
        bioknowledgerag.init_vector_store(settings.CHROMADB_lit_collection_name)
        
        genes_str = ", ".join(gene_list[:10])  # 限制基因数量避免过长查询
        
        rag_query = f"""
        Discuss the biological functions and roles of these genes: {genes_str}
        in relation to {celltype} cell type. Include information about:
        1. Known biological pathways these genes are involved in
        2. Functional implications of their expression levels
        3. Relationship to cell activation, differentiation or other states
        4. Any disease or immunological relevance
        """
        
        result = bioknowledgerag.query(rag_query, settings.CHROMADB_lit_collection_name, top_k=5)
        context = "\n".join([doc.page_content for doc in result])
        
        return context
    except Exception as e:
        print(f"Error querying literature for genes {gene_list}: {e}")
        return ""

def interpret_celltype_biology(cluster_id: str, celltype: str, marker_genes: List[str], 
                              diff_genes: List[str], adata, cluster_key: str) -> dict:
    """
    为特定细胞类型生成生物学解读报告
    """
    # 生成dotplot

    
    # 查询文献信息
    literature_info = query_gene_literature(marker_genes, celltype)
    
    # 构建解读报告
    report = {
        "cluster_id": cluster_id,
        "celltype": celltype,
        "marker_genes": marker_genes,
        "differential_genes": diff_genes,
        "dotplot_path": plot_path,
        "literature_insights": literature_info,
        "biological_interpretation": f"""
基于marker基因表达模式，对{celltype} (cluster {cluster_id})的生物学解读：

1. 关键marker基因:
   - {', '.join(marker_genes[:5])} 等基因的表达指示了该细胞类型的特征

2. 生物学功能推断:
   - 根据marker基因表达模式，该细胞群体显示出典型的{celltype}特征
   - 基因表达模式暗示细胞可能处于特定的激活或分化状态
00000000000                                                                                                      0
3. 文献支持的生物学意义:
   - 相关文献表明这些基因在{celltype}功能中发挥重要作用
   - 表达水平的变化可能与细胞的激活、分化或效应功能相关
        """
    }
    
    return report

def parse_marker_genes_from_annotation(annotation_file: str) -> dict:
    """
    从注释文件中解析每个cluster的marker基因
    """
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    clusters = {}
    current_cluster = None
    
    for line in lines:
        if line.startswith('Cluster '):
            current_cluster = line.split()[1].strip(':')
            clusters[current_cluster] = {
                'marker_genes': {},
                'suggested_celltype': None
            }
        elif 'Suggested cell type for Cluster' in line:
            celltype = line.split(':')[-1].strip()
            clusters[current_cluster]['suggested_celltype'] = celltype
        elif 'Marker genes for' in line and ':' in line:
            # 提取marker基因信息
            try:
                celltype_name = line.split('Marker genes for ')[1].split(':')[0]
                genes_str = line.split(':')[-1].strip()
                genes = eval(genes_str) if genes_str.startswith('{') else set()
                clusters[current_cluster]['marker_genes'][celltype_name] = list(genes)
            except:
                pass
    
    return clusters



def integrate_celltype_annotations(adata, anno_result: str) -> sc.AnnData:
    """
    Integrate cell annotation results into adata

    """
    cluster_map = pd.read_csv(anno_result)
    cluster2celltype = dict(zip(cluster_map['Cluster'], cluster_map['CellType']))
    adata.obs['scGPT_clusters'] = adata.obs['scGPT_clusters'].astype(int)
    adata.obs['pred_celltype'] = adata.obs['scGPT_clusters'].map(cluster2celltype)

    return adata

def get_top_marker_dict(adata, top_n):
    # 提取 rank_genes_groups 的结果
    result = adata.uns["rank_genes_groups"]
    
    # 转换为 DataFrame 便于操作
    groups = result["names"].dtype.names  # 每一列对应一个 group
    marker_dict = {}

    for group in groups:
        # 取出该 group 的前 top_n 个基因
        top_genes = result["names"][group][:top_n].tolist()
        marker_dict[group] = top_genes
    return marker_dict

def generate_plot(adata, work_dir: str):
    """
    Generate basic visualization charts
    """

    # 存储生成的图表路径
    plot_paths = {}
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True)

    # UMAP可视化
    sc.pl.umap(adata, 
               color='pred_celltype', 
               save='anno_result_umap.png', 
               show=False, 
               title = '_Cell_Annotate'
               )
    
    umap_path = work_path / 'umap_anno_result_umap.png'
    if umap_path.exists():
        plot_paths['umap'] = str(umap_path)
    
    sc.tl.rank_genes_groups(
        adata,
        groupby="pred_celltype",
        method="wilcoxon",
        layer='X_log1p',
        use_raw=False,
    )
    sc.pl.rank_genes_groups_dotplot(
        adata, 
        groupby="pred_celltype", 
        standard_scale="var", 
        n_genes=5,
        save='_anno_result_dotplot.png',
        )
    
    dotplot_path = work_path / 'dotplot_anno_result_dotplot.png'
    if dotplot_path.exists():
        plot_paths['dotplot'] = str(dotplot_path)

    
    marker_genes_dict = get_top_marker_dict(adata, top_n=5)
    def prepare_heatmap_inputs(marker_genes_dict):
        var_names = []
        var_group_positions = []
        var_group_labels = []

        start = 0
        for label, genes in marker_genes_dict.items():
            end = start + len(genes) - 1
            var_names.extend(genes)
            var_group_positions.append((start, end))
            var_group_labels.append(label)
            start = end + 1

        return var_names, var_group_positions, var_group_labels
    
    var_names, var_group_positions, var_group_labels = prepare_heatmap_inputs(marker_genes_dict)

    # 绘制热图
    sc.pl.heatmap(
        adata,
        var_names=var_names,
        groupby="pred_celltype",   # 这里替换成你分组的 obs 列名
        layer="X_log1p",           # 或者 raw
        cmap="viridis",
        standard_scale="var",
        swap_axes=True,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        show_gene_labels=True,
        dendrogram=True,
        save = '_anno_result_heatmap.png',
    )

    heatmap_path = work_path / 'heatmap_anno_result_heatmap.png'
    if heatmap_path.exists():
        plot_paths['heatmap'] = str(heatmap_path)

    print(f"Visualization plots saved in {work_dir}")

    return json.dumps(plot_paths)


def query_celltype_literature(celltype: str, marker_genes: List[str]) -> str:
    """
    查询细胞类型相关文献信息
    """
    try:
        bioknowledgerag = BioKnowledgeRag(settings.CHROMADB_PERSIST_DIR)
        bioknowledgerag.init_vector_store(settings.CHROMADB_lit_collection_name)
        
        genes_str = ", ".join(marker_genes[:5])  # 限制基因数量
        
        rag_query = f"""
        Provide detailed information about {celltype} cell type with focus on:
        1. Biological functions and roles in immune system
        2. Key marker genes: {genes_str}
        3. Differentiation pathways and developmental stages
        4. Activation states and functional implications
        5. Disease associations and clinical relevance
        """
        
        result = bioknowledgerag.query(rag_query, settings.CHROMADB_lit_collection_name, top_k=5)
        context = "\n".join([doc.page_content for doc in result])
        
        return context
    except Exception as e:
        print(f"查询细胞类型 {celltype} 文献时出错: {e}")
        return "无法获取文献信息"

def comprehensive_cell_interpretation(adata, marker_gene_dict ,work_dir: str):
    """
    Perform full cell annotation results interpretation
    """
    interpretations = {}
    summary_data = []

    for celltype, markers in marker_gene_dict.items():
        literature_info = query_celltype_literature(celltype, markers)

        prompt = f"""
You are a bioinformatics expert. Please analyze the cell annotation results for me based on the cell annotation results and related literature.
1. Key Marker Genes:
- The expression of genes such as {', '.join(markers[:5])} indicates the characteristics of this cell type.

2. Inferred Biological Function:
- Based on the marker gene expression pattern, this cell population displays typical characteristics of {celltype}.
- Gene expression patterns suggest that cells may be in a specific activation or differentiation state.

3. Biological Significance Supported by Literature:
- Related literature indicates that these genes play an important role in the function of {celltype}.
- Changes in expression levels may be associated with cell activation, differentiation, or effector functions.

Literature content:
{literature_info[:2000]}
"""
        
        llm_response = llm.invoke(prompt)

        interpretation = {
            "celltype": celltype,
            "marker_genes": markers,
            "literature_insights": literature_info,
            "biological_interpretation": llm_response.content,
        }

        interpretations[celltype] = interpretation

        summary_data.append({
            "CellType": celltype,
            "NumberOfMarkers": len(markers),
            "KeyMarkerGenes": ", ".join(markers[:5])
        })

    # 保存每个细胞类型的解读到单独的文件
    output_path = Path(work_dir)
    output_path.mkdir(exist_ok=True)
    
    for celltype, interp in interpretations.items():
        # 保存单个细胞类型的解读为JSON文件
        celltype_filename = celltype.replace('/', '_').replace(' ', '_').replace('-', '_')
        with open(output_path / f"{celltype_filename}_interpretation.json", 'w', encoding='utf-8') as f:
            json.dump(interp, f, indent=2, ensure_ascii=False)
    
    # 生成综合报告
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "celltype_interpretation_summary.csv", index=False)
    
    # 生成综合解读报告
    with open(output_path / "comprehensive_celltype_interpretation.md", 'w', encoding='utf-8') as f:
        f.write("# 单细胞注释结果综合生物学解读报告（按细胞类型）\n\n")
        f.write("## 概述\n\n")
        f.write(f"本次分析共识别了 {len(interpretations)} 种不同的细胞类型。\n\n")
        
        # 为每种细胞类型生成详细解读
        for celltype, interp in interpretations.items():
            f.write(f"## {celltype}\n\n")
            f.write(f"### 关键marker基因\n")
            f.write(f"- {', '.join(interp['marker_genes'][:15])}")
            if len(interp['marker_genes']) > 15:
                f.write("...")
            f.write("\n\n")
            f.write(f"### 生物学解读\n")
            f.write(f"{interp['biological_interpretation']}\n\n")
            f.write(f"### 文献支持信息\n")
            f.write(f"<details>\n<summary>点击查看详细文献信息</summary>\n\n")
            f.write(f"{interp['literature_insights'][:2000]}...\n\n")
            f.write(f"</details>\n\n")
            f.write("---\n\n")
    
    # 生成细胞类型特异性marker基因总结报告
    with open(output_path / "celltype_marker_summary.md", 'w', encoding='utf-8') as f:
        f.write("# 细胞类型特异性Marker基因总结报告\n\n")
        
        # 为每种细胞类型生成总结
        for celltype, data in interpretations.items():
            marker_genes = data['marker_genes']
            
            f.write(f"## {celltype}\n\n")
            f.write(f"### 识别到的Marker基因 (共{len(marker_genes)}个)\n")
            f.write(f"- {', '.join(marker_genes[:20])}")
            if len(marker_genes) > 20:
                f.write("...")
            f.write("\n\n")
    
    # 整合所有细胞类型的生物学解读，生成总体解读报告
    overall_interpretation_prompt = "As a bioinformatics expert, please generate an overall analysis report based on the following biological interpretations of various cell types:：\n\n"
    
    for celltype, interp in interpretations.items():
        overall_interpretation_prompt += f"celltype: {celltype}\n"
        overall_interpretation_prompt += f"Biological interpretation: {interp['biological_interpretation']}\n\n"
    
    overall_interpretation_prompt += "Please combine the above information to generate an overall interpretation report containing the following content：\n"
    overall_interpretation_prompt += "1. The composition and proportion of cell types in the sample\n"
    overall_interpretation_prompt += "2. Functional relationships between different cell types\n"
    overall_interpretation_prompt += "3. Possible biological status or characteristics of the sample\n"
    overall_interpretation_prompt += "4. Discoveries or inferences of biological significance\n"
    
    # 调用LLM生成总体解读
    overall_interpretation_response = llm.invoke(overall_interpretation_prompt)
    
    # 保存总体解读报告
    with open(output_path / "overall_interpretation_report.md", 'w', encoding='utf-8') as f:
        f.write("# 总体细胞类型生物学解读报告\n\n")
        f.write(overall_interpretation_response.content)
    
    print(f"综合生物学解读报告已生成:")
    print(f"- 综合解读报告: {output_path / 'comprehensive_celltype_interpretation.md'}")
    print(f"- 细胞类型marker总结: {output_path / 'celltype_marker_summary.md'}")
    print(f"- 解读摘要: {output_path / 'celltype_interpretation_summary.csv'}")
    print(f"- 总体解读报告: {output_path / 'overall_interpretation_report.md'}")
    print(f"- 各细胞类型单独的解读文件已保存到 {work_dir} 目录")

    return interpretations


class CellAnnoInterpreArgs(BaseModel):
    clustered_path: str = Field(description="Path to the clustered AnnData h5ad file.")
    annotation_file: str = Field(description = "Path to the cluster annotation text file.")
    anno_result: str = Field(description="Path to the cluster annotation result file.")
    work_dir: str = Field(description="Per-sample folder created by load_h5ad_data.")

# @tool(
#         "cell_annotation_interpretation",
#         args_schema=CellAnnoInterpreArgs,
# )
def run_cell_interpretation(
    clustered_path: str, 
    anno_result:str,
    work_dir: str):
    """
    Run the complete interpretation workflow for cell annotation results
    """
    print("Start interpreting cell annotation results...")
    # Set scanpy save path
    sc.settings.figdir = str(work_dir)

    adata = sc.read_h5ad(clustered_path)
    #Mapping the clustering results to cell types
    adata = integrate_celltype_annotations(adata, anno_result)

    plot_paths_json = generate_plot(adata, work_dir)
    plot_paths = json.loads(plot_paths_json) if plot_paths_json else {}

    marker_gene_dict = get_top_marker_dict(adata, top_n=5)

    report = comprehensive_cell_interpretation(adata, marker_gene_dict, work_dir)
    
    print(f"Interpretation completed! Results saved to{work_dir}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 请根据实际路径修改以下参数
    clustered_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_clustered.h5ad"
    annotation_file = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/cluster_celltype_annotation.txt"
    anno_result = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/cluster_celltype_rank1.csv"
    work_dir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/interpretation"
    
    # 运行解读
    results = run_cell_interpretation(clustered_path, anno_result, work_dir)