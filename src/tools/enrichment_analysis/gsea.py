import gseapy as gp
import pandas as pd

from typing import Dict, List, Optional

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import load_pathway_genesets, load_expression, AnalysisResult


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """在 df 中按候选名（忽略大小写）找第一列名，找不到返回 None"""
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None

class GSEAAnalyzer(EnrichmentAnalysiszer):

    @staticmethod
    def run(file_path: str, gene_set: str = "KEGG"):
        # Implement GSEA analysis logic here
        expr = load_expression(file_path)
        gene_list = expr.columns.to_list()
    
        # 确定基因集数据库或文件
        if gene_set.lower() in ["kegg", "go", "msigdb"]:
            gene_sets = load_pathway_genesets(gene_set)
        
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
        
class GSEAVisualizer(EnrichmentVisualizer):

    @staticmethod
    def plot(results: AnalysisResult):
        # Implement visualization logic here
        if results.top_terms.empty:
            return "No enriched terms found."
        
        # 创建图表
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.barh(results.top_terms['Term'], results.top_terms['NES'])

class GSEAEvaluator(EnrichmentEvaluator):

    @staticmethod
    def evaluate(results: AnalysisResult):
        # Implement evaluation logic here
        if not results.pvalues:
            return "No p-values available for evaluation."
        
        significant_terms = [term for term, pval in results.pvalues.items() if pval < 0.05]
        return f"Found {len(significant_terms)} significantly enriched terms."
    
    
class GSEAFactory(EnrichmentFactory):
    def create_analyzer(self): return GSEAAnalyzer()
    def create_visualizer(self): return GSEAVisualizer()
    def create_evaluator(self): return GSEAEvaluator()
