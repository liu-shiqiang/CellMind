# ora.py (slim, drop冗余fallback与日志)
from typing import Optional, Callable
import numpy as np
import pandas as pd
import anndata as ad
import gseapy as gp

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import extract_gene_list_from_celltype, AnalysisResult

_DEF_TOPN = {"kegg": 150, "go": 250}
_LIB = {"kegg": "KEGG_2021_Human", "go": "GO_Biological_Process_2021"}

class ORAAnalyzer(EnrichmentAnalysiszer):
    def run(
        self,
        input_file: str,
        *,
        celltype_col: str = "pred_celltype",       # 可选：你也可以传别的列
        target_celltype: Optional[str] = None,     # 可选：不传则自动用该列里数量最多的类型
        gene_set: str = "kegg",                    # "kegg" 或 "go"
        top_n: Optional[int] = None,               # 不传则按 gene_set 取默认
        enrichr_lib_path: Optional[str] = None,    # 可选：本地GMT，离线环境时使用
        quiet: bool = True,                        # 默认不打log
        logger: Optional[Callable[[str], None]] = None,
    ) -> AnalysisResult:

        def _log(msg: str):
            if not quiet and logger:
                logger(msg)

        gs = gene_set.lower()
        if gs not in _LIB:
            raise ValueError("gene_set 仅支持 'kegg' 或 'go'")
        lib_name = _LIB[gs]
        if top_n is None:
            top_n = _DEF_TOPN[gs]

        # 1) 若未指定 target_celltype，自动选该列里细胞数最多的类别
        if target_celltype is None:
            adata = ad.read_h5ad(input_file)
            if celltype_col not in adata.obs.columns:
                raise ValueError(f"adata.obs 缺少列 '{celltype_col}'")
            ser = adata.obs[celltype_col]
            if hasattr(ser, "cat"):
                counts = ser.value_counts().reindex(ser.cat.categories, fill_value=0).astype(int)
            else:
                counts = ser.value_counts().astype(int)
            target_celltype = counts.idxmax()

        # 2) 提取 marker（内部已做展平与清洗）
        markers = extract_gene_list_from_celltype(
            input_file,
            celltype_col=celltype_col,
            target_celltype=target_celltype,
            top_n=top_n,
            return_info=False,
            logger=(lambda s: _log(s))
        )

        # 3) ORA（不写磁盘，不预筛）
        if enrichr_lib_path:
            gene_sets = enrichr_lib_path
        else:
            gene_sets = lib_name

        enr = gp.enrichr(
            gene_list=markers,
            gene_sets=gene_sets,
            organism="Human",
            outdir=None,
            no_plot=True,
            cutoff=1.0,      # 不要预过滤，交给我们自己判定
            verbose=0
        )

        if enr is None or not hasattr(enr, "res2d") or enr.res2d is None or enr.res2d.empty:
            return AnalysisResult(pd.DataFrame(), {})

        df = enr.res2d.copy()
        pcol = "Adjusted P-value" if "Adjusted P-value" in df.columns else "P-value"
        df = df.sort_values(pcol, ascending=True)
        top_terms = df.head(20).copy()
        pvalues = dict(zip(top_terms["Term"], top_terms[pcol]))
        return AnalysisResult(top_terms=top_terms,
                              pvalues=pvalues,
                              meta={"gene_set": lib_name,
                                    "celltype_col": celltype_col,
                                    "target_celltype": target_celltype,
                                    "top_n": top_n,
                                    "organism": "Human",})


class ORAVisualizer(EnrichmentVisualizer):
    def plot(
        self,
        result: AnalysisResult,
        *,
        output_path: str = None,
        outdir: str = None,
        db_name: str = None,
        title_prefix: str = "Top Enriched Pathways",
        annotate_count_in_label: bool = False,
        top_k: int = 10
    ):
        import os, matplotlib.pyplot as plt, numpy as np, textwrap

        if result.top_terms is None or result.top_terms.empty:
            return None

        df = result.top_terms.copy()

        # --- 1) 优先从 meta 里取信息（可选字段，缺失则回退） ---
        meta = getattr(result, "meta", {}) or {}
        # db_name 优先 meta['gene_set']，否则从 DataFrame 猜
        if db_name is None:
            db_name = (meta.get("gene_set")
                       or (df["Gene_set"].iloc[0] if "Gene_set" in df.columns and df["Gene_set"].notna().any() else "Enrichr"))
        celltype = meta.get("target_celltype")
        cellcol  = meta.get("celltype_col")

        # --- 2) 列名对齐 + 计算 ---
        if "Adjusted P-value" in df.columns:
            df = df.rename(columns={"Adjusted P-value": "adjP"})
        if "Genes" in df.columns and "Overlapping Genes" not in df.columns:
            df["Overlapping Genes"] = df["Genes"]
        df = df.sort_values(by="adjP", ascending=True).head(top_k)
        df["logP"] = -np.log10(df["adjP"])

        def _count_genes(x):
            if isinstance(x, str):
                sep = ";" if ";" in x else ("," if "," in x else None)
                if sep: return len([i for i in (g.strip() for g in x.split(sep)) if i])
                return 1 if x.strip() else 0
            elif isinstance(x, (list, tuple)):
                return len(x)
            return 0
        df["Count"] = df["Overlapping Genes"].apply(_count_genes)

        def _mk_label(term, cnt):
            text = textwrap.fill(str(term), width=30)
            return f"{text}  (n={cnt})" if annotate_count_in_label else text
        df["Label"] = [_mk_label(t, c) for t, c in zip(df["Term"], df["Count"])]

        # --- 3) 标题：带上库名 &（可选）celltype 信息 ---
        title = f"{title_prefix} · {db_name}"
        if celltype:
            if cellcol:
                title += f"  ({cellcol}: {celltype})"
            else:
                title += f"  ({celltype})"

        # --- 4) 画图 ---
        plt.figure(figsize=(9, 6))
        bars = plt.barh(y=df["Label"], width=df["logP"], height=0.7, edgecolor="white")
        for bar, c in zip(bars, df["Count"]):
            plt.text(bar.get_width() - 0.15, bar.get_y() + bar.get_height()/2,
                     str(c), ha="right", va="center", color="white", fontweight="bold", fontsize=10)
        plt.xlabel("-log10(Adjusted P-value)"); plt.ylabel("Pathway")
        plt.title(title); plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.7); plt.tight_layout()

        # --- 5) 默认文件名：也用 meta（若传了目录或未给文件名）---
        if output_path is None:
            fname = f"ora_barplot_{db_name.replace(' ', '_')}"
            if celltype:
                fname += f"_{str(celltype).replace(' ', '_')}"
            fname += ".png"
            output_path = os.path.join(outdir, fname) if outdir else fname
        else:
            if os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
                fname = f"ora_barplot_{db_name.replace(' ', '_')}.png"
                output_path = os.path.join(output_path, fname)
            else:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path


class ORAEvaluator(EnrichmentEvaluator):
    def evaluate(self, result: AnalysisResult) -> str:
        if not result.pvalues: return "No enriched terms."
        sig = sum(1 for p in result.pvalues.values() if p < 0.05)
        return f"{sig} enriched terms (adjP<0.05)."


class ORAFactory(EnrichmentFactory):
    def create_analyzer(self): return ORAAnalyzer()
    def create_visualizer(self): return ORAVisualizer()
    def create_evaluator(self): return ORAEvaluator()



if __name__=="__main__":
    
    analyzer = ORAAnalyzer()
    results = analyzer.run(
        input_file="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_annotated.h5ad",
        gene_set="KEGG",
        celltype_col="pred_celltype",
        target_celltype="CD8-positive, alpha-beta cytotoxic T cell"
    )
    print(results.top_terms)

    visualizer = ORAVisualizer()
    plot_path = visualizer.plot(results)
    print(f"Plot saved to: {plot_path}")

    evaluator = ORAEvaluator()
    summary = evaluator.evaluate(results)
    print(summary)