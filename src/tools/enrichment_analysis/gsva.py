import gseapy as gp
import pandas as pd

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import load_pathway_genesets, load_expression, AnalysisResult


class GSVAAnalyzer(EnrichmentAnalysiszer):

    @staticmethod
    def run(file_path: str, gene_set: str = "KEGG"):
        expr = load_expression(file_path)

        if gene_set.lower() in ["kegg", "go", "msigdb"]:
            gene_sets = load_pathway_genesets(gene_set)

        # Run GSVA
        gsva_results = gp.ssgsea(
            data=expr,
            gene_sets=gene_sets,
            outdir=None
        )

        if gsva_results is not None and hasattr(gsva_results, "res2d"):
            top_terms = gsva_results.res2d.head(20)
            scores = dict(zip(top_terms.index, top_terms["ES"]))
            return AnalysisResult(top_terms=top_terms, pvalues=scores)
        else:
            return AnalysisResult(pd.DataFrame(), {})


class GSVAVisualizer(EnrichmentVisualizer):

    @staticmethod
    def plot(results: AnalysisResult):
        if results.top_terms.empty:
            return "No enriched pathways found."

        # Example: heatmap
        # gp.heatmap(results.top_terms, title="GSVA results")


class GSVAEvaluator(EnrichmentEvaluator):

    @staticmethod
    def evaluate(results: AnalysisResult):
        if not results.pvalues:
            return "No scores available."

        # In GSVA, evaluate based on enrichment score thresholds
        high_score = [t for t, s in results.pvalues.items() if abs(s) > 0.5]
        return f"Found {len(high_score)} pathways with strong enrichment scores."


class GSVAFactory(EnrichmentFactory):
    def create_analyzer(self): return GSVAAnalyzer()
    def create_visualizer(self): return GSVAVisualizer()
    def create_evaluator(self): return GSVAEvaluator()