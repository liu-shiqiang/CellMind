# ssgsea.py — unified minimal outputs (auto: ranksum_boot if enough groups else zscore)
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
import os, json
from datetime import datetime

import numpy as np
import pandas as pd
import gseapy as gp
from scipy.stats import norm, ranksums
from statsmodels.stats.multitest import multipletests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import load_pathway_genesets, load_expression, AnalysisResult

_OUTDIR_ROOT = "enrichment_results"
_SUBDIR = "ssgsea"


def _ensure_outdir(path: Optional[str] = None) -> str:
    if path:
        outdir = Path(path).expanduser().resolve()
    else:
        outdir = Path(_OUTDIR_ROOT) / _SUBDIR
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)

def _ensure_gene_set_axis(scores: pd.DataFrame, gene_set_names: Iterable[str]) -> pd.DataFrame:
    gene_set_names = set(map(str, gene_set_names))
    idx_hit = len(gene_set_names.intersection(map(str, scores.index)))
    col_hit = len(gene_set_names.intersection(map(str, scores.columns)))
    if idx_hit >= col_hit:
        return scores.rename(index=str, columns=str)
    return scores.T.rename(index=str, columns=str)

def _others_only_z_pvalue(row: pd.Series) -> float:
    ct = row.idxmax()
    ls = float(row[ct])
    others = row.drop(index=ct).astype(float).values
    if others.size == 0:
        return 1.0
    mu = float(np.mean(others)); sd = float(np.std(others, ddof=0)) or 1e-9
    z = (ls - mu) / sd
    return float(2 * norm.sf(abs(z)))

def _bootstrap(values: np.ndarray, reps=30, frac=0.8, seed=2025) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values); k = max(2, int(round(frac * n)))
    out = []
    for _ in range(reps):
        idx = rng.choice(n, size=min(k, n), replace=(k > n))
        out.append(values[idx].mean())
    return np.asarray(out, dtype=float)

def _auto_pick_mode(cell_counts: Dict[str, int], n_cts: int, min_cells_per_group=30, min_groups_for_A=4) -> str:
    ok = sum(int(cell_counts.get(ct, 0) >= min_cells_per_group) for ct in cell_counts.keys())
    return "ranksum_boot" if (ok >= min_groups_for_A and n_cts >= min_groups_for_A) else "zscore"

def _extract_scores(ssgsea_obj, gene_set_names) -> Optional[pd.DataFrame]:
    # 1) 尝试从对象属性提取
    for attr in ("scores", "resultsOnSamples", "samples_scores"):
        df = getattr(ssgsea_obj, attr, None)
        if isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0:
            return df.copy()
    # 2) 尝试 res2d 透视 NES/ES
    res2d = getattr(ssgsea_obj, "res2d", None)
    if isinstance(res2d, pd.DataFrame) and {"Name", "Term"} <= set(res2d.columns):
        val = "NES" if "NES" in res2d.columns else ("ES" if "ES" in res2d.columns else None)
        if val:
            return res2d.pivot(index="Term", columns="Name", values=val).copy()
    return None


class SSGSEAAnalyzer(EnrichmentAnalysiszer):

    def run(
        self,
        file_path: str,
        *,
        gene_set: str = "KEGG",
        celltype_col: str = "pred_celltype",
        target_celltypes: Optional[Iterable[str]] = None,
        aggregation: str = "mean",
        min_cells: int = 15,
        mode: str = "auto",              # "auto" | "zscore" | "ranksum_boot"
        min_cells_per_group: int = 30,   # auto 用
        min_groups_for_A: int = 4,       # auto 用
        threads: int = 1,
    ) -> AnalysisResult:

        # 1) 表达（聚合到 celltype）
        expr = load_expression(
            file_path=file_path,
            celltype_col=celltype_col,
            target_celltypes=target_celltypes,
            aggregation=aggregation,
            min_cells=min_cells,
            normalize=True,
        )  # genes × celltypes

        # 2) 基因集（并做与表达基因的重叠过滤）
        gene_sets = load_pathway_genesets(gene_set)
        expr_genes = set(map(str, expr.index))
        gene_sets = {k: [g for g in v if g in expr_genes] for k, v in gene_sets.items()}
        gene_sets = {k: v for k, v in gene_sets.items() if len(v) >= 10}
        if not gene_sets:
            return AnalysisResult(pd.DataFrame(), {})

        # 3) 跑 ssGSEA
        ssgsea_res = gp.ssgsea(
            data=expr,
            gene_sets=gene_sets,
            sample_norm_method="rank",
            outdir=None,
            min_size=10,
            max_size=1000,
            permutation_num=0,
            no_plot=True,
            threads=threads,
            seed=2024,
            verbose=0,
        )
        score_df = _extract_scores(ssgsea_res, gene_sets.keys())
        if score_df is None:
            return AnalysisResult(pd.DataFrame(), {})

        # 统一轴：gene set × celltype
        score_df = _ensure_gene_set_axis(score_df, gene_sets.keys())
        score_df = score_df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
        if score_df.empty:
            return AnalysisResult(pd.DataFrame(), {})

        # 4) auto 模式判断
        chosen = mode
        if mode == "auto":
            try:
                import scanpy as sc
                adata = sc.read_h5ad(file_path)
                if celltype_col in adata.obs.columns:
                    counts = adata.obs[celltype_col].value_counts().astype(int).to_dict()
                else:
                    counts = {}
            except Exception:
                counts = {}
            chosen = _auto_pick_mode(counts, score_df.shape[1], min_cells_per_group, min_groups_for_A)

        # 5) 一对多统计
        lead_ct, lead_score, pvals = [], [], []
        for _, row in score_df.iterrows():
            row = row.astype(float)
            ct = row.idxmax(); ls = float(row[ct])
            lead_ct.append(ct); lead_score.append(ls)
            others = row.drop(index=ct).values

            if chosen == "ranksum_boot":
                # 用“伪重复”增强稳健性
                grp1 = _bootstrap(np.repeat(ls, 30), reps=30, frac=0.8)
                grp2 = []
                for _, ov in row.drop(index=ct).items():
                    grp2.append(_bootstrap(np.repeat(float(ov), 30), reps=10, frac=0.8))
                grp2 = np.concatenate(grp2) if len(grp2) else np.array([ls], float)
                if grp2.size < 2:
                    p = _others_only_z_pvalue(row)
                else:
                    _, p = ranksums(grp1, grp2, alternative="greater")
            else:
                p = _others_only_z_pvalue(row)

            if not np.isfinite(p):
                p = 1.0
            pvals.append(float(p))

        summary = pd.DataFrame({
            "Pathway": score_df.index.astype(str),
            "LeadingCelltype": lead_ct,
            "LeadingScore": lead_score,
            "PValue": pvals,
        })
        summary["FDR"] = multipletests(summary["PValue"], method="fdr_bh")[1]
        summary = summary.sort_values(["FDR", "LeadingScore"], ascending=[True, False]).reset_index(drop=True)

        top_terms = summary.head(20).copy()
        pvalue_map = dict(zip(summary["Pathway"], summary["FDR"]))

        return AnalysisResult(
            top_terms=top_terms,
            pvalues=pvalue_map,
            scores=score_df,
            gene_sets=gene_sets,
        )


class SSGSEAVisualizer(EnrichmentVisualizer):
    def plot(
        self,
        results: AnalysisResult,
        *,
        top_n: int = 12,
        outdir: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os

        if results.top_terms is None or results.top_terms.empty or results.scores is None:
            return None

        outdir = _ensure_outdir(outdir)

        top_pathways = results.top_terms["Pathway"].head(top_n).tolist()
        avail = [p for p in top_pathways if p in results.scores.index]
        if not avail:
            return None

        # 取出并标准化 Z-score，以增强色彩对比
        df = results.scores.loc[avail].astype(float)
        df_z = (df - df.mean(axis=1).values[:, None]) / (df.std(axis=1).values[:, None] + 1e-9)

        # 美观配色（coolwarm 或 RdBu_r）
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        plt.figure(figsize=(max(6, df.shape[1]*0.6+2), max(4, len(df)*0.4+2)))

        sns.heatmap(
            df_z,
            cmap=cmap,
            center=0,
            annot=False,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "ssGSEA Z-score"},
            xticklabels=True,
            yticklabels=True,
        )

        plt.title("ssGSEA Enrichment Heatmap", fontsize=13, pad=12)
        plt.xlabel("Cell type")
        plt.ylabel("Pathway")

        # 旋转标签 & 调整间距
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(outdir, "ssgsea_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return plot_path


class SSGSEAEvaluator(EnrichmentEvaluator):
    def evaluate(
        self,
        results: AnalysisResult,
        *,
        top_k: int = 20,
        fdr_threshold: float = 0.05,
        return_paths: bool = True,
        outdir: Optional[str] = None,
    ):
        outdir = _ensure_outdir(outdir)

        # 空结果：也输出最小三件套
        if results is None or results.top_terms is None or results.top_terms.empty:
            json_path = os.path.join(outdir, "ssgsea_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "method": "ssGSEA",
                    "gene_set": None,
                    "n_celltypes": None,
                    "n_significant": 0,
                    "top_pathways": [],
                }, f, ensure_ascii=False, indent=2)

            summary_path = os.path.join(outdir, "ssgsea_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("[ssGSEA Summary] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("No enriched pathways.\n")

            tsv_path = os.path.join(outdir, "ssgsea_top_terms.tsv")
            pd.DataFrame().to_csv(tsv_path, sep="\t", index=False)

            msg = "No enriched pathways."
            return (msg, {"json": json_path, "summary": summary_path, "tsv": tsv_path}) if return_paths else msg

        df = results.top_terms.copy()
        if "FDR" not in df.columns and "PValue" in df.columns:
            df["FDR"] = df["PValue"]

        # TSV
        save_df = df.sort_values(["FDR", "LeadingScore"], ascending=[True, False]).head(top_k)
        tsv_path = os.path.join(outdir, "ssgsea_top_terms.tsv")
        save_df.to_csv(tsv_path, sep="\t", index=False)

        # JSON（统一、轻量）
        payload: Dict[str, Any] = {
            "method": "ssGSEA",
            "gene_set": None,               # 保持轻量：此处不写 meta
            "n_celltypes": int(results.scores.shape[1]) if (results.scores is not None and results.scores.shape[1] > 0) else None,
            "n_significant": int((df["FDR"] < fdr_threshold).sum()),
            "top_pathways": [
                {
                    "Pathway": str(r["Pathway"]),
                    "LeadingCelltype": str(r.get("LeadingCelltype", "")),
                    "Score": float(r.get("LeadingScore", np.nan)),
                    "FDR": float(r.get("FDR", np.nan)),
                }
                for _, r in save_df.iterrows()
            ],
            "meta": (getattr(results, "meta", {}) or {})
        }
        json_path = os.path.join(outdir, "ssgsea_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Summary
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[ssGSEA Summary] {ts}", f"n_celltypes  : {payload['n_celltypes']}", ""]
        lines.append(f"显著富集通路数量: {payload['n_significant']}")
        if len(save_df) > 0:
            lines.append("Top pathways:")
            for _, r in save_df.head(min(10, len(save_df))).iterrows():
                lines.append(
                    f"  - {r['Pathway']} 在 {r.get('LeadingCelltype','?')} 中活性最高 "
                    f"(score={float(r.get('LeadingScore', np.nan)):.3f}, FDR={float(r.get('FDR', np.nan)):.3g})"
                )
        summary_path = os.path.join(outdir, "ssgsea_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        msg = f"{payload['n_significant']} significant pathways (FDR<{fdr_threshold})."
        return (msg, {"json": json_path, "summary": summary_path, "tsv": tsv_path}) if return_paths else msg


class SSGSEAFactory(EnrichmentFactory):
    def create_analyzer(self): return SSGSEAAnalyzer()
    def create_visualizer(self): return SSGSEAVisualizer()
    def create_evaluator(self): return SSGSEAEvaluator()


class SSGSEAEnrichmentArgs(BaseModel):
    file_path: str = Field(..., description="Path to the annotated AnnData (.h5ad) file.")
    work_dir: str = Field(..., description="Work directory where ssGSEA outputs will be stored.")
    gene_set: str = Field(default="KEGG", description="Gene set library to use (e.g. KEGG, GO, Hallmark).")
    celltype_col: str = Field(default="pred_celltype", description="Column in AnnData.obs that defines cell groups.")
    target_celltypes: Optional[List[str]] = Field(default=None, description="Subset of cell types to include in the analysis.")
    aggregation: str = Field(default="mean", description="Aggregation strategy when collapsing to cell types.")
    min_cells: int = Field(default=15, description="Minimum cells required per cell type for aggregation.")
    mode: str = Field(default="auto", description="Statistic used for ranking (auto, zscore, ranksum_boot).")
    min_cells_per_group: int = Field(default=30, description="Minimum cells for a group to be considered in auto mode.")
    min_groups_for_A: int = Field(default=4, description="Minimum groups required to trigger ranksum bootstrap in auto mode.")
    threads: int = Field(default=1, description="Number of threads for gseapy.ssgsea.")
    top_n_heatmap: int = Field(default=12, description="Number of pathways to display in the heatmap.")
    top_k_terms: int = Field(default=20, description="Number of pathways saved to the TSV/JSON outputs.")
    fdr_threshold: float = Field(default=0.05, description="FDR threshold used when counting significant pathways.")
    cluster_id: Optional[str] = Field(default=None, description="Optional cluster identifier associated with this analysis.")


@tool("run_ssgsea_enrichment", args_schema=SSGSEAEnrichmentArgs)
def run_ssgsea_enrichment(
    file_path: str,
    work_dir: str,
    gene_set: str = "KEGG",
    celltype_col: str = "pred_celltype",
    target_celltypes: Optional[List[str]] = None,
    aggregation: str = "mean",
    min_cells: int = 15,
    mode: str = "auto",
    min_cells_per_group: int = 30,
    min_groups_for_A: int = 4,
    threads: int = 1,
    top_n_heatmap: int = 12,
    top_k_terms: int = 20,
    fdr_threshold: float = 0.05,
    cluster_id: Optional[str] = None,
) -> str:
    """Run ssGSEA enrichment on aggregated cell type expression profiles."""

    input_path = Path(file_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    outdir = Path(_ensure_outdir(work_path / "enrichment" / "ssgsea"))

    json_path = outdir / "ssgsea_result.json"
    summary_path = outdir / "ssgsea_summary.txt"
    tsv_path = outdir / "ssgsea_top_terms.tsv"
    scores_path = outdir / "ssgsea_scores.tsv"
    heatmap_path = outdir / "ssgsea_heatmap.png"

    if json_path.exists() and summary_path.exists() and tsv_path.exists():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        top_terms = cached.get("top_pathways") or []
        n_sig = cached.get("n_significant")
        if not top_terms:
            message = "No enriched pathways."
        else:
            message = f"{int(n_sig)} significant pathways (FDR<{fdr_threshold})." if n_sig is not None else "No enriched pathways."
        result_paths = {
            "json": str(json_path.resolve()),
            "summary": str(summary_path.resolve()),
            "tsv": str(tsv_path.resolve()),
        }
        if scores_path.exists():
            result_paths["scores"] = str(scores_path.resolve())
        if heatmap_path.exists():
            result_paths["heatmap"] = str(heatmap_path.resolve())
        payload = {
            "status": "success",
            "message": message,
            "result_paths": result_paths,
            "top_terms": top_terms,
            "meta": cached.get("meta", {}),
        }
        return json.dumps(payload, ensure_ascii=False)

    analyzer = SSGSEAAnalyzer()
    result = analyzer.run(
        file_path=str(input_path),
        gene_set=gene_set,
        celltype_col=celltype_col,
        target_celltypes=target_celltypes,
        aggregation=aggregation,
        min_cells=min_cells,
        mode=mode,
        min_cells_per_group=min_cells_per_group,
        min_groups_for_A=min_groups_for_A,
        threads=threads,
    )

    meta = dict(getattr(result, "meta", {}) or {})
    meta.setdefault("gene_set", gene_set)
    meta.setdefault("work_dir", str(work_path))
    if cluster_id is not None:
        meta["cluster_id"] = str(cluster_id)
    result.meta = meta

    visualizer = SSGSEAVisualizer()
    heatmap_path = visualizer.plot(result, top_n=top_n_heatmap, outdir=str(outdir))

    evaluator = SSGSEAEvaluator()
    eval_msg, eval_paths = evaluator.evaluate(
        result,
        top_k=top_k_terms,
        fdr_threshold=fdr_threshold,
        return_paths=True,
        outdir=str(outdir),
    )

    paths_dict: Dict[str, str] = {}
    for key, value in (eval_paths or {}).items():
        if value:
            paths_dict[key] = str(Path(value).expanduser().resolve())
    if heatmap_path:
        paths_dict["heatmap"] = str(Path(heatmap_path).expanduser().resolve())

    scores_path = None
    if result.scores is not None and not result.scores.empty:
        scores_path = outdir / "ssgsea_scores.tsv"
        result.scores.to_csv(scores_path, sep="\t")
        paths_dict["scores"] = str(scores_path.resolve())

    top_terms_records: List[Dict[str, Any]] = []
    if result.top_terms is not None and not result.top_terms.empty:
        for row in result.top_terms.head(top_k_terms).to_dict(orient="records"):
            cleaned: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned[key] = float(value)
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    cleaned[key] = int(value)
                else:
                    cleaned[key] = value
            top_terms_records.append(cleaned)

    payload = {
        "status": "success",
        "message": eval_msg,
        "result_paths": paths_dict,
        "top_terms": top_terms_records,
        "meta": result.meta,
    }

    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    analyzer = SSGSEAAnalyzer()
    res = analyzer.run(
        file_path="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_annotated.h5ad",
        gene_set="KEGG",
        celltype_col="pred_celltype",
        mode="auto",
    )
    viz = SSGSEAVisualizer()
    plot_path = viz.plot(res)
    print("Heatmap:", plot_path)
    ev = SSGSEAEvaluator()
    msg, paths = ev.evaluate(res, return_paths=True)
    print(msg, paths)

