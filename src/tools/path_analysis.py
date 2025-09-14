from langchain_core.tools import tool
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from config.setting import settings
import argparse

import gseapy as gp
from gseapy import gsva
from gseapy import ssgsea

import scanpy as sc

class PathwayArgs(BaseModel):
    method: str = Field(description="分析方法: GSEA/ssGSEA/GSVA/GO/KEGG")
    input_file: str = Field(description="输入表达矩阵文件路径 (CSV, TSV, H5AD)")
    geneset: Optional[str] = Field(default="KEGG", description="基因集数据库: GO/KEGG/MSigDB, 或自定义GMT文件路径")


@tool("pathway_analysis", return_direct=False, args_schema=PathwayArgs)
def pathway_analysis(method: str, input_file: str, geneset: str = "KEGG") -> dict:
    """
    在输入表达数据上执行通路富集分析。
    支持方法:  GSEA, ssGSEA, GSVA, GO富集, KEGG富集。
    返回包含图表路径、显著性结果和JSON数据的字典。
    """
    # 1. 读取并预处理表达数据
    expr = load_expression(input_file)
    
    # 2. 选择并执行分析方法
    if method == "GSEA":
        res = run_gsea(expr, geneset=geneset)
    elif method == "ssGSEA":
        res = run_ssgsea(expr, geneset=geneset)
    elif method == "GSVA":
        res = run_gsva(expr, geneset=geneset)
    elif method == "GO":
        res = run_go(expr, geneset=geneset)
    elif method == "KEGG":
        res = run_kegg(expr, geneset=geneset)
    else:
        raise ValueError(f"不支持的分析方法: {method}")
    
    with open(os.path.join(settings.OUTPUT_DIR, "pathway_analysis.json"), "w") as f:
        json.dump({
            "method": method,
            "top_terms": res.top_terms.to_dict(orient="records") if hasattr(res.top_terms, 'to_dict') else res.top_terms,
            "pvalues": res.pvalues,
        }, f, indent=4)
        
    # # 3. 构建并返回结果字典
    # return json.dumps({
    #     "method": method,
    #     "top_terms": res.top_terms.to_dict(orient="records") if hasattr(res.top_terms, 'to_dict') else res.top_terms,
    #     "pvalues": res.pvalues,
    # })

# ===== 辅助函数实现 =====
def load_expression(file_path: str) -> pd.DataFrame:
    """加载表达矩阵，支持CSV、TSV和H5AD格式"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, index_col=0)
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        return pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)
        return adata.to_df()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

class AnalysisResult:
    def __init__(self, top_terms: pd.DataFrame, pvalues: Dict[str, float], 
                 scores: Optional[pd.DataFrame] = None, 
                 gene_sets: Optional[Dict[str, List[str]]] = None):
        self.top_terms = top_terms  # 富集结果表格
        self.pvalues = pvalues      # 通路p值字典
        self.scores = scores        # 通路得分矩阵(仅适用于某些方法)
        self.gene_sets = gene_sets  # 基因集定义(仅适用于某些方法)

# ===== 通路分析方法实现 =====
def run_gsea(expr: pd.DataFrame, geneset: str = "KEGG") -> AnalysisResult:
    """执行GSEA分析"""
    # 准备输入数据：需要排序后的基因列表
    # 计算基因表达均值作为排序指标
    gene_list = expr.columns.to_list()
    
    # 确定基因集数据库或文件
    if geneset.lower() in ["kegg", "go", "msigdb"]:
        gene_sets = load_pathway_genesets(geneset)
    
    # 执行GSEA分析
    gsea_results = gp.gsea(
        data=expr,
        cls=gene_list,
        gene_sets=gene_sets,
        permutation_num=1000,  # 实际分析可增加到1000
        processes=4,
        outdir=None  # 不保存到文件
    )
    
    # 提取结果
    if gsea_results is not None and hasattr(gsea_results, 'res2d'):
        top_terms = gsea_results.res2d.head(20)
        pvalues = dict(zip(top_terms['Term'], top_terms['p.value']))
        return AnalysisResult(top_terms=top_terms, pvalues=pvalues)
    else:
        # 处理分析失败的情况
        return AnalysisResult(
            top_terms=pd.DataFrame(),
            pvalues={}
        )

def run_ssgsea(expr: pd.DataFrame, geneset: str = "KEGG") -> AnalysisResult:
    """执行ssGSEA分析"""
    gene_sets = load_pathway_genesets(geneset)

    try:
        # 使用gseapy的ssgsea函数
        ssgsea_results = ssgsea(
            data=expr,
            gene_sets=gene_sets,
            permutation_num=1000, 
            outdir=None
        )
        
        # 提取结果
        if ssgsea_results is not None and hasattr(ssgsea_results, 'score'):
            scores = ssgsea_results.score
            # 计算每个通路的平均得分
            pathway_scores = scores.mean(axis=0).reset_index(name='mean_score')
            pathway_scores = pathway_scores.rename(columns={'index': 'pathway'})
            
            # 这里需要获取p值，实际分析中可能需要额外的统计检验
            # 简化处理：使用负对数转换的均值作为排序指标
            pathway_scores['pvalue'] = 1 - np.exp(-pathway_scores['mean_score'])
            pathway_scores = pathway_scores.sort_values('mean_score', ascending=False).head(20)
            
            pvalues = dict(zip(pathway_scores['pathway'], pathway_scores['pvalue']))
            return AnalysisResult(
                top_terms=pathway_scores,
                pvalues=pvalues,
                scores=scores
            )
        else:
            return AnalysisResult(
                top_terms=pd.DataFrame(),
                pvalues={}
            )
    except Exception as e:
        print(f"ssGSEA分析出错: {e}")
        return AnalysisResult(
            top_terms=pd.DataFrame(),
            pvalues={}
        )

def run_gsva(expr: pd.DataFrame, geneset: str = "KEGG") -> AnalysisResult:
    """执行GSVA分析"""   
    gene_sets = load_pathway_genesets(geneset)

    # 执行GSVA
    try:
        gsva_results = gsva(
            data=expr,
            gene_sets=gene_sets,
            method='gsva',
            outdir=None
        )
        
        if gsva_results is not None and hasattr(gsva_results, 'score'):
            scores = gsva_results.score
            # 计算每个通路的方差
            pathway_variance = scores.var(axis=0).reset_index(name='variance')
            pathway_variance = pathway_variance.rename(columns={'index': 'pathway'})
            pathway_variance = pathway_variance.sort_values('variance', ascending=False).head(20)
            
            # 这里需要获取p值，实际分析中可能需要额外检验
            # 简化处理：使用方差的倒数作为p值估计
            pathway_variance['pvalue'] = 1 / (1 + pathway_variance['variance'])
            
            pvalues = dict(zip(pathway_variance['pathway'], pathway_variance['pvalue']))
            return AnalysisResult(
                top_terms=pathway_variance,
                pvalues=pvalues,
                scores=scores
            )
        else:
            return AnalysisResult(
                top_terms=pd.DataFrame(),
                pvalues={}
            )
    except Exception as e:
        print(f"GSVA分析出错: {e}")
        return AnalysisResult(
            top_terms=pd.DataFrame(),
            pvalues={}
        )

def run_go(expr: pd.DataFrame, geneset: str = "GO") -> AnalysisResult:
    """执行GO富集分析"""
    # 提取差异表达基因(这里简化为所有基因)
    gene_list = expr.columns.to_list()
    gene_sets = load_pathway_genesets(geneset)
    # 执行GO富集
    try:
        # 使用gseapy的enrichgo函数
        go_results = gp.enrichr(
            gene_list=gene_list,
            organism='Human',  
            gene_sets=gene_sets,
            outdir=None,
            cutoff=0.05
        )
        
        if go_results is not None and hasattr(go_results, 'res2d'):
            top_terms = go_results.res2d.head(20)
            # 重命名列以保持一致性
            top_terms = top_terms.rename(columns={
                'ID': 'ID',
                'Description': 'Description',
                'GeneRatio': 'GeneRatio',
                'BgRatio': 'BgRatio',
                'pvalue': 'pvalue',
                'p.adjust': 'pAdjust',
                'qvalue': 'qvalue',
                'geneID': 'geneID',
                'Count': 'Count'
            })
            pvalues = dict(zip(top_terms['ID'], top_terms['pvalue']))
            return AnalysisResult(top_terms=top_terms, pvalues=pvalues)
        else:
            return AnalysisResult(
                top_terms=pd.DataFrame(),
                pvalues={}
            )
    except Exception as e:
        print(f"GO富集分析出错: {e}")
        return AnalysisResult(
            top_terms=pd.DataFrame(),
            pvalues={}
        )

def run_kegg(expr: pd.DataFrame, geneset: str = "KEGG") -> AnalysisResult:
    """执行KEGG富集分析"""
    # 提取差异表达基因(这里简化为所有基因)
    gene_list = expr.columns.tolist()
    gene_sets = load_pathway_genesets(geneset)

    # 执行KEGG富集
    kegg_results = gp.enrichr(
        gene_list=gene_list,
        organism='Human',  # 人类
        gene_sets=gene_sets,
        outdir=None,
        cutoff=0.05
    )
    
    if kegg_results is not None and hasattr(kegg_results, 'res2d'):
        top_terms = kegg_results.res2d
        return AnalysisResult(top_terms=top_terms, pvalues=kegg_results.res2d['P-value'].to_dict())
    else:
        return AnalysisResult(
            top_terms=pd.DataFrame(),
            pvalues={}
    )

# ===== 辅助函数 =====
def load_pathway_genesets(db_name: str) -> Dict[str, List[str]]:
    """加载通路基因集"""

    msig = gp.Msigdb()
    if db_name.lower() == "kegg":
        return msig.get_gmt(category='h.all',dbver=settings.MSIGDB_VERSION)
    elif db_name.lower() == "go":
        return msig.get_gmt(category='c5.go',dbver=settings.MSIGDB_VERSION)
    elif db_name.lower() == "msigdb":
        return msig.get_gmt(category='msigdb',dbver=settings.MSIGDB_VERSION)
    else:
        raise ValueError(f"不支持的基因集数据库: {db_name}")


# # 可视化函数
# def plot_enrichment_bubble(result: AnalysisResult) -> str:
#     """生成富集气泡图"""
#     if result.top_terms.empty:
#         return ""
    
#     # 创建图表目录
#     os.makedirs('results/plots', exist_ok=True)
#     output_path = 'results/plots/bubble_plot.png'
    
#     # 绘图
#     plt.figure(figsize=(12, 8))
#     ax = plt.gca()
    
#     # 确保包含所需列
#     required_cols = ['pathway', 'NES', 'pvalue', 'size']
#     if all(col in result.top_terms.columns for col in required_cols):
#         scatter = ax.scatter(
#             result.top_terms['pathway'], 
#             result.top_terms['NES'],
#             s=result.top_terms['size'] * 10,  # 气泡大小与通路大小相关
#             c=-np.log10(result.top_terms['pvalue']),  # 颜色与p值负对数相关
#             cmap='viridis',
#             alpha=0.7
#         )
#         plt.colorbar(scatter, label='-log10(pvalue)')
#         plt.title('GSEA Pathway Enrichment Bubble Plot')
#     else:
#         # 兼容不同分析方法的结果
#         if 'pathway' in result.top_terms.columns and 'pvalue' in result.top_terms.columns:
#             scatter = ax.scatter(
#                 result.top_terms['pathway'], 
#                 range(len(result.top_terms)),
#                 s=100/result.top_terms['pvalue'], 
#                 c=result.top_terms['pvalue'], 
#                 cmap='viridis'
#             )
#             plt.colorbar(scatter, label='p-value')
#             plt.title('Pathway Enrichment Bubble Plot')
    
#     plt.xticks(rotation=90, ha='right')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
    
#     return output_path

# def plot_enrichment_bar(result: AnalysisResult) -> str:
#     """生成富集柱状图"""
#     if result.top_terms.empty:
#         return ""
    
#     output_path = 'results/plots/bar_plot.png'
#     plt.figure(figsize=(12, 8))
    
#     # 确保包含p值列
#     if 'pvalue' in result.top_terms.columns:
#         plt.bar(
#             result.top_terms['pathway'], 
#             -np.log10(result.top_terms['pvalue']),
#             color='skyblue'
#         )
#         plt.ylabel('-log10(pvalue)')
#     elif 'mean_score' in result.top_terms.columns:
#         plt.bar(
#             result.top_terms['pathway'], 
#             result.top_terms['mean_score'],
#             color='lightgreen'
#         )
#         plt.ylabel('Mean Score')
#     else:
#         plt.bar(
#             result.top_terms['pathway'], 
#             range(len(result.top_terms)),
#             color='lightcoral'
#         )
    
#     plt.title('Pathway Enrichment Bar Plot')
#     plt.xticks(rotation=90, ha='right')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
    
#     return output_path

parser = argparse.ArgumentParser(description="Genomix-Agent: AI-Powered Multi-Omics Analysis Platform", add_help=False)
parser.add_argument('--file', type=str, help='Path to .h5ad file')
parser.add_argument('--method', type=str, choices=['GSEA', 'ssGSEA', 'GSVA', 'GO', 'KEGG'])

args = parser.parse_args()

# 在args定义后立即使用
results = pathway_analysis.invoke({
    "method": args.method,
    "input_file": args.file,
    "geneset": "KEGG"
})
print(results)