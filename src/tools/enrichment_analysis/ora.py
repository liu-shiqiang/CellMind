import gseapy as gp
import pandas as pd

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import load_pathway_genesets, load_expression, AnalysisResult


class ORAAnalyzer(EnrichmentAnalysiszer):

    @staticmethod
    def run(file_path: str, gene_set: str = "KEGG"):
        # Load marker gene list
        
        gene_list = load_pathway_genesets(file_path)

        if gene_set.lower() in ["kegg", "go", "msigdb"]:
            gene_sets = load_pathway_genesets(gene_set)

        # Run ORA
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="Human",
            outdir=None
        )

        if enr is not None and hasattr(enr, "res2d"):
            top_terms = enr.res2d.head(20)
            pvalues = dict(zip(top_terms['Term'], top_terms['Adjusted P-value']))
            return AnalysisResult(top_terms=top_terms, pvalues=pvalues)
        else:
            return AnalysisResult(pd.DataFrame(), {})


class ORAVisualizer(EnrichmentVisualizer):

    @staticmethod
    def plot(results: AnalysisResult):
        if results.top_terms.empty:
            return "No enriched terms found."

        # Example: barplot
        # gp.barplot(results.top_terms, title="ORA results")


class ORAEvaluator(EnrichmentEvaluator):

    @staticmethod
    def evaluate(results: AnalysisResult):
        if not results.pvalues:
            return "No p-values available."

        sig_terms = [t for t, p in results.pvalues.items() if p < 0.05]
        return f"Found {len(sig_terms)} significantly enriched terms."


class ORAFactory(EnrichmentFactory):
    def create_analyzer(self): return ORAAnalyzer()
    def create_visualizer(self): return ORAVisualizer()
    def create_evaluator(self): return ORAEvaluator()