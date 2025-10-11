import gseapy as gp
import pandas as pd

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import extract_gene_list_from_celltype, load_expression, AnalysisResult


class ORAAnalyzer(EnrichmentAnalysiszer):

    @staticmethod
    def run(file_path: str,
            gene_set: str = "KEGG",
            celltype_col: str = "pred_celltype",
            target_celltype: str = "required",):
        # Load marker gene list
        
        gene_list = extract_gene_list_from_celltype(file_path, celltype_col= celltype_col,
            target_celltype = target_celltype, )

        ENRICHR_LIB_MAP = {
            "kegg": "KEGG_2021_Human",
            "go": "GO_Biological_Process_2021",
            "hallmark": "Hallmark_2020"
        }
        enrichr_lib = ENRICHR_LIB_MAP.get(gene_set.lower())
        if not enrichr_lib:
            raise ValueError(f"不支持的 gene_set: {gene_set}. 支持: {list(ENRICHR_LIB_MAP.keys())}")

        # Run ORA
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=enrichr_lib,
            organism="Human",
            outdir=None
        )

        if enr is not None and hasattr(enr, "res2d") and not enr.res2d.empty:
            top_terms = enr.res2d.head(20)
            pvalues = dict(zip(top_terms['Term'], top_terms['Adjusted P-value']))
            return AnalysisResult(top_terms=top_terms, pvalues=pvalues)
        else:
            return AnalysisResult(pd.DataFrame(), {})


import matplotlib.pyplot as plt
import numpy as np
import textwrap
import ast

class ORAVisualizer(EnrichmentVisualizer):
    @staticmethod
    def plot(results: AnalysisResult):
        if results.top_terms.empty:
            return "No enriched terms found."
        
        df = results.top_terms.copy()
        df = df.rename(columns={"Adjusted P-value": "adjP"})
        
        # 1. 数据预处理
        df = df.sort_values(by="adjP", ascending=True)
        df["logP"] = -np.log10(df["adjP"])
        
        # 2. 通路名称自动换行
        df["Path"] = df["Term"].apply(lambda x: textwrap.fill(x, width=30))
        
        # 3. 专业渐变蓝配色
        cmap = plt.get_cmap("Blues_r")
        colors = cmap(np.linspace(0.2, 0.8, len(df)))
        
        # 4. 创建水平条形图
        plt.figure(figsize=(9, 6))
        bars = plt.barh(
            y=df["Path"],
            width=df["logP"],
            color=colors,
            height=0.7,
            edgecolor="white"
        )
        
        # 5. 添加基因计数标签
        df['Count'] = df['Overlapping Genes'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
        )
        
        for bar, count in zip(bars, df['Count']):
            plt.text(
                x=bar.get_width() - 0.2,
                y=bar.get_y() + bar.get_height()/2,
                s=str(count),
                ha="right", va="center",
                color="white", fontweight="bold", fontsize=10
            )
        
        # 6. 优化x轴
        max_logP = df["logP"].max()
        plt.xlim(0, max_logP + 0.5)
        plt.xticks(np.arange(0, max_logP + 0.5, 1), fontsize=10)
        plt.xlabel("-log10(Adjusted P-value)", fontsize=12, labelpad=10)
        plt.ylabel("Pathway", fontsize=12, labelpad=10)
        
        # 7. 美化细节
        plt.title("Top 10 Enriched Pathways (ORA)", fontsize=14, fontweight="bold", pad=15)
        plt.yticks(fontsize=9)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        # 8. 保存高分辨率图
        output_path = "ora_barplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
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