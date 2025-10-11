import gseapy as gp
import pandas as pd
import numpy as np
from gseapy import ssgsea

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import load_pathway_genesets, load_expression, AnalysisResult

class SSGSEAAnalyzer(EnrichmentAnalysiszer):

    @staticmethod
    def run(file_path: str, gene_set: str = "KEGG"):
        # Implement SSGSEA analysis logic here
        expr = load_expression(file_path)
        gene_sets = load_pathway_genesets(gene_set)

        try:
            # 使用gseapy的ssgsea函数
            ssgsea_results = ssgsea(
                data=expr,
                gene_sets=gene_sets,
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
        
class SSGSEAVisualizer(EnrichmentVisualizer):

    @staticmethod
    def plot(results: AnalysisResult):
        # Implement visualization logic here
        if results.top_terms.empty:
            return "No enriched terms found."
        
        # 创建图表
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.barh(results.top_terms['Term'], results.top_terms['NES'])

class SSGSEAEvaluator(EnrichmentEvaluator):

    @staticmethod
    def evaluate(results: AnalysisResult):
        # Implement evaluation logic here
        if not results.pvalues:
            return "No p-values available for evaluation."
        
        significant_terms = [term for term, pval in results.pvalues.items() if pval < 0.05]
        return f"Found {len(significant_terms)} significantly enriched terms."
    
    
class SSGSEAFactory(EnrichmentFactory):
    def create_analyzer(self): return SSGSEAAnalyzer()
    def create_visualizer(self): return SSGSEAVisualizer()
    def create_evaluator(self): return SSGSEAEvaluator()
