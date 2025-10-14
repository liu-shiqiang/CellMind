import ast
import os
import textwrap

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Union

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
        expr = load_expression(file_path)

        if expr.empty:
            return AnalysisResult(top_terms=pd.DataFrame(), pvalues={})

        # 通过均值创建排名，若仅一列则直接使用该列
        if expr.shape[1] > 1:
            ranking = expr.mean(axis=1)
        else:
            ranking = expr.iloc[:, 0]

        ranking = ranking.dropna()
        ranking = ranking[~ranking.index.duplicated(keep="first")]
        if ranking.empty:
            return AnalysisResult(top_terms=pd.DataFrame(), pvalues={})

        ranking = ranking.sort_values(ascending=False)

        # 确定基因集来源
        gene_sets: Union[Dict[str, List[str]], str]
        if gene_set.lower() in {"kegg", "go", "hallmark", "msigdb"}:
            gene_sets = load_pathway_genesets(gene_set)
        elif os.path.isfile(gene_set):
            gene_sets = gene_set  # gseapy 支持直接读取GMT文件
        else:
            raise ValueError(f"Unsupported gene_set: {gene_set}")

        try:
            gsea_results = gp.prerank(
                rnk=ranking,
                gene_sets=gene_sets,
                processes=4,
                permutation_num=200,
                outdir=None,
                seed=42,
            )
        except Exception:
            return AnalysisResult(top_terms=pd.DataFrame(), pvalues={})

        if gsea_results is None or not hasattr(gsea_results, "res2d"):
            return AnalysisResult(top_terms=pd.DataFrame(), pvalues={})

        res = gsea_results.res2d.copy()
        if res.empty:
            return AnalysisResult(top_terms=pd.DataFrame(), pvalues={})

        term_col = _get_col(res, ["Term", "term", "name", "Pathway", "Description"])
        if term_col is None:
            term_col = res.columns[0]

        p_col = _get_col(res, [
            "FDR q-val", "FDR", "FDR q-value", "FDR (q-value)",
            "p.adjust", "p value", "p.value", "P-value", "P_val", "P-val", "NOM p-val"
        ])

        if p_col is None:
            # 若找不到显著性列，则使用排名顺序构造伪p值
            res["pseudo_p"] = np.linspace(0.001, 0.05, num=len(res))
            p_col = "pseudo_p"

        res = res.sort_values(by=p_col, ascending=True)
        top_terms = res.head(20)
        pvalues = dict(zip(top_terms[term_col], top_terms[p_col]))

        stored_gene_sets = gene_sets if isinstance(gene_sets, dict) else None

        return AnalysisResult(
            top_terms=top_terms,
            pvalues=pvalues,
            gene_sets=stored_gene_sets,
        )

class GSEAVisualizer(EnrichmentVisualizer):

    @staticmethod
    def plot(results: AnalysisResult):
        if results.top_terms.empty:
            return "No enriched terms found."

        df = results.top_terms.copy()

        term_col = _get_col(df, ["Term", "term", "name", "Pathway", "Description"])
        if term_col is None:
            term_col = df.columns[0]

        nes_col = _get_col(df, ["NES", "nes", "Normalized Enrichment Score"])
        p_col = _get_col(df, [
            "FDR q-val", "FDR", "FDR q-value", "FDR (q-value)",
            "p.adjust", "p value", "p.value", "P-value", "P_val", "P-val", "NOM p-val"
        ])

        if p_col is None:
            if results.pvalues:
                df["plot_p"] = df[term_col].map(results.pvalues)
                p_col = "plot_p"
            else:
                df["plot_p"] = np.linspace(0.001, 0.05, num=len(df))
                p_col = "plot_p"

        df = df.dropna(subset=[term_col])
        df = df.sort_values(by=p_col, ascending=True).head(15)

        df["neg_log10"] = -np.log10(df[p_col].clip(lower=1e-300))
        df["display_term"] = df[term_col].apply(lambda x: textwrap.fill(str(x), width=35))

        cmap = plt.get_cmap("coolwarm")
        color_values = None
        if nes_col is not None:
            nes_vals = df[nes_col].astype(float)
            vmax = np.nanmax(np.abs(nes_vals))
            if vmax == 0:
                vmax = 1
            color_values = (nes_vals / (2 * vmax)) + 0.5
        else:
            color_values = df["neg_log10"] / (df["neg_log10"].max() or 1)

        plt.figure(figsize=(9, 6))
        bars = plt.barh(
            y=df["display_term"],
            width=df["neg_log10"],
            color=cmap(color_values),
            edgecolor="white",
            height=0.7,
        )

        for bar, pval in zip(bars, df[p_col]):
            plt.text(
                x=bar.get_width() + 0.1,
                y=bar.get_y() + bar.get_height() / 2,
                s=f"FDR={pval:.2g}",
                va="center",
                ha="left",
                fontsize=9,
            )

        plt.xlabel("-log10(FDR)", fontsize=12)
        plt.ylabel("Pathway", fontsize=12)
        plt.title("Top Enriched Pathways (GSEA)", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = "gsea_barplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

class GSEAEvaluator(EnrichmentEvaluator):

    @staticmethod
    def evaluate(results: AnalysisResult):
        if results.top_terms.empty:
            return "未检测到显著富集的通路。"

        df = results.top_terms.copy()
        term_col = _get_col(df, ["Term", "term", "name", "Pathway", "Description"])
        if term_col is None:
            term_col = df.columns[0]

        nes_col = _get_col(df, ["NES", "nes", "Normalized Enrichment Score"])
        p_col = _get_col(df, [
            "FDR q-val", "FDR", "FDR q-value", "FDR (q-value)",
            "p.adjust", "p value", "p.value", "P-value", "P_val", "P-val", "NOM p-val"
        ])

        if not results.pvalues:
            if p_col is not None:
                pvalues = dict(zip(df[term_col], df[p_col]))
            else:
                pvalues = {}
        else:
            pvalues = results.pvalues

        significant_terms = [term for term, pval in pvalues.items() if pval is not None and pval < 0.05]

        summary_lines: List[str] = []
        summary_lines.append("GSEA富集分析总结：")
        if significant_terms:
            summary_lines.append(f"共有 {len(significant_terms)} 条通路显著富集(FDR < 0.05)。")
        else:
            summary_lines.append("未发现FDR < 0.05的显著通路，但以下通路表现最突出：")

        df_for_summary = df.copy()
        if pvalues:
            df_for_summary["_pval"] = df_for_summary[term_col].map(pvalues)
            df_for_summary = df_for_summary.sort_values("_pval", ascending=True)
        elif p_col is not None:
            df_for_summary = df_for_summary.sort_values(p_col, ascending=True)

        top_entries = df_for_summary.head(5)

        leading_col = _get_col(df, ["leadingEdge", "LeadingEdge", "leading_edge", "core_enrichment"])

        for idx, row in enumerate(top_entries.itertuples(index=False), start=1):
            term_name = getattr(row, term_col)
            nes_value = getattr(row, nes_col) if nes_col is not None else None
            p_value = pvalues.get(term_name) if term_name in pvalues else (getattr(row, p_col) if p_col is not None else None)

            detail = f"{idx}. {term_name}"
            if nes_value is not None and not pd.isna(nes_value):
                detail += f" (NES={float(nes_value):.2f})"
            if p_value is not None and not pd.isna(p_value):
                detail += f", FDR={float(p_value):.2g}"

            if leading_col is not None:
                leading_data = getattr(row, leading_col)
                if isinstance(leading_data, str):
                    try:
                        leading_genes = ast.literal_eval(leading_data)
                    except (ValueError, SyntaxError):
                        leading_genes = [g.strip() for g in leading_data.split(",") if g.strip()]
                elif isinstance(leading_data, (list, tuple, set)):
                    leading_genes = list(leading_data)
                else:
                    leading_genes = []

                if leading_genes:
                    detail += f"。核心富集基因: {', '.join(leading_genes[:5])}"

            summary_lines.append(detail)

        summary_lines.append("提示：NES > 0 表示上调通路，NES < 0 表示下调通路。")

        return "\n".join(summary_lines)
    
    
class GSEAFactory(EnrichmentFactory):
    def create_analyzer(self): return GSEAAnalyzer()
    def create_visualizer(self): return GSEAVisualizer()
    def create_evaluator(self): return GSEAEvaluator()
