# cellphoneDB.py — Standalone (CellPhoneDB + ktplotspy plotting)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os, re
from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc
from pydantic import BaseModel, Field, ValidationError

from langchain_core.tools import tool

# CPDB core
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

# settings（可选）
try:
    from config.setting import settings  # 可能包含 CELLPHONEDB_ZIP
except Exception:
    settings = None  # type: ignore

# 非交互后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ktplotspy
import ktplotspy as kpy


# ==== 修改为你的数据库 zip ====
CELLPHONEDB = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cellphonedb/cellphonedb.zip"


# -----------------------------
# Helpers
# -----------------------------
def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _to_dense_frame(adata: sc.AnnData) -> pd.DataFrame:
    """cells × genes（写盘前转置为 genes × cells）"""
    if adata.isbacked:
        adata = adata.to_memory(copy=True)
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    elif isinstance(x, np.matrix):
        x = np.asarray(x)
    return pd.DataFrame(x, index=adata.obs_names, columns=adata.var_names)

def _latest_file(output_dir: Path, pattern: str) -> Optional[Path]:
    paths = sorted((Path(p) for p in glob(str(output_dir / pattern))),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None

def _split_pair(pair: str) -> Tuple[str, str]:
    if "|" in pair:
        lhs, rhs = pair.split("|", 1)
    elif ":" in pair:
        lhs, rhs = pair.split(":", 1)
    else:
        lhs, rhs = pair, pair
    return lhs.strip(), rhs.strip()

def _summarise_interactions(df: pd.DataFrame, top_n: int = 10) -> List[str]:
    if df.empty:
        return ["未检测到显著的配体-受体相互作用。"]
    lines = [
        f"共识别出 {df['interaction'].nunique()} 个显著配体-受体对，"
        f"涉及 {df['source'].nunique()} 个来源细胞类型与 {df['target'].nunique()} 个受体细胞类型。"
    ]
    top_df = df.sort_values(["pvalue", "value"], ascending=[True, False]).head(top_n)
    for _, row in top_df.iterrows():
        lines.append(
            f"• {row['source']} → {row['target']}：{row['interaction']}，"
            f"平均互作强度 {row['value']:.3f}，调整后P值 {row['pvalue']:.3g}。"
        )
    return lines

def _melt_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    id_columns = [c for c in df.columns if "|" not in str(c)]
    value_columns = [c for c in df.columns if "|" in str(c)]
    if not value_columns: return pd.DataFrame()
    melted = df.melt(id_vars=id_columns, value_vars=value_columns,
                     var_name="celltype_pair", value_name="value")
    return melted.dropna(subset=["value"])

def _tokenize_gene_like(s: str) -> set:
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s))
    return {t.upper() for t in s.split() if t}

def _filter_cpdb_by_genes(df: pd.DataFrame, genes: Optional[List[str]]) -> pd.DataFrame:
    if not genes or "interacting_pair" not in df.columns:
        return df
    target = {g.upper() for g in genes}
    mask = df["interacting_pair"].astype(str).apply(
        lambda x: bool(_tokenize_gene_like(x) & target)
    )
    out = df[mask].copy()
    return out if not out.empty else df

def _coerce_numeric_cpdb_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    pair_cols = [c for c in out.columns if '|' in str(c)]
    for c in pair_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[pair_cols] = out[pair_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if pair_cols:
        row_mask = (out[pair_cols].to_numpy()!=0).any(axis=1)
        out = out.loc[row_mask]
        col_mask = (out[pair_cols].to_numpy()!=0).any(axis=0)
        keep_cols = list(out.columns.difference(pair_cols)) + [c for c, k in zip(pair_cols, col_mask) if k]
        out = out.loc[:, keep_cols]
    return out

def _winsorize_wide(df: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """轻度缩尾，抑制极端大值导致的超粗弦。"""
    if df is None or df.empty: return df
    out = df.copy()
    pair_cols = [c for c in out.columns if '|' in str(c)]
    if not pair_cols: return out
    caps = out[pair_cols].quantile(q=q, axis=0).astype(float).replace(0, np.nan)
    for c in pair_cols:
        cap = float(caps.get(c, np.nan))
        if np.isfinite(cap):
            out[c] = np.minimum(pd.to_numeric(out[c], errors="coerce"), cap)
    return out

def _safe_for_clustering(df: pd.DataFrame) -> bool:
    if df is None or df.empty: return False
    pair_cols = [c for c in df.columns if '|' in str(c)]
    if len(pair_cols) < 2: return False
    vals = df[pair_cols].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): return False
    return (vals!=0).any(axis=0).sum() >= 2

def auto_select_focus_genes(
    merged: pd.DataFrame, *, max_genes: int = 16, top_pairs: int = 200
) -> List[str]:
    df = merged.sort_values(["pvalue", "value"], ascending=[True, False]).head(int(top_pairs)).copy()
    if {"gene_a", "gene_b"}.issubset(df.columns):
        df["gA"] = df["gene_a"].astype(str)
        df["gB"] = df["gene_b"].astype(str)
    elif "interacting_pair" in df.columns:
        gs = df["interacting_pair"].astype(str).str.split("_", n=1, expand=True)
        if gs.shape[1] < 2: return []
        df["gA"], df["gB"] = gs[0], gs[1]
    else:
        return []
    df["gA"] = df["gA"].str.upper(); df["gB"] = df["gB"].str.upper()
    df["pair"] = df["source"].astype(str) + "→" + df["target"].astype(str)
    tall = pd.concat([
        df[["pair","value","gA"]].rename(columns={"gA":"gene"}),
        df[["pair","value","gB"]].rename(columns={"gB":"gene"}),
    ], ignore_index=True).dropna(subset=["gene"])
    gstat = (tall.groupby("gene")
                  .agg(n_pairs=("pair","nunique"), strength=("value","sum"))
                  .reset_index())
    if gstat.empty: return []
    strength_max = max(float(gstat["strength"].max()), 1.0)
    gstat["score"] = gstat["n_pairs"] + (gstat["strength"] / strength_max)
    pairs_by_gene = {g:set(tall.loc[tall["gene"]==g,"pair"].unique()) for g in gstat["gene"]}
    selected, covered = [], set()
    work = gstat.set_index("gene").copy()
    while len(selected) < int(max_genes) and not work.empty:
        gains = []
        for g in work.index:
            new_pairs = pairs_by_gene[g] - covered
            gains.append((g, len(new_pairs), work.loc[g,"score"]))
        gains.sort(key=lambda x:(x[1], x[2]), reverse=True)
        best_g, new_cov, _ = gains[0]
        if new_cov <= 0 and selected: break
        selected.append(best_g); covered |= pairs_by_gene[best_g]
        work = work.drop(index=best_g)
    if len(selected) < int(max_genes):
        rest = [g for g in gstat.sort_values("strength", ascending=False)["gene"] if g not in selected]
        selected.extend(rest[:(int(max_genes)-len(selected))])
    return selected


# -----------------------------
# Return dataclass
# -----------------------------
@dataclass
class CellphoneDBResult:
    status: str
    message: str
    summary_path: Optional[Path]
    figures: Dict[str, Path]
    tables: Dict[str, Path]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "figures": {k: str(v) for k, v in self.figures.items()},
            "tables": {k: str(v) for k, v in self.tables.items()},
        }


# -----------------------------
# Args schema
# -----------------------------
class CellphoneDBArgs(BaseModel):
    file_path: str = Field(description="输入 .h5ad 文件路径")
    work_dir: Optional[str] = Field(default=None, description="输出目录；默认 {输入同级}/cellphonedb_results")
    celltype_column: str = Field(default="pred_celltype", description="AnnData.obs 中的细胞类型列名")
    database_path: Optional[str] = Field(default=None, description="CellPhoneDB 数据库 zip 路径")
    counts_data: str = Field(default="hgnc_symbol", description="表达矩阵使用的基因 ID 类型")
    threads: int = Field(default=4, ge=1, description="CPU 线程数")
    pvalue_threshold: float = Field(default=0.05, gt=0, description="置换检验 p 值阈值")
    # ktplotspy chord 相关
    focus_genes: Optional[List[str]] = Field(default=None, description="只保留包含这些基因的互作用于弦图")
    chord_cell_type1: str = Field(default=".", description="ktplotspy chord: cell_type1（'.'表示不限）")
    chord_cell_type2: str = Field(default=".", description="ktplotspy chord: cell_type2（'.'表示不限）")


# -----------------------------
# Core
# -----------------------------
@tool(
    "run_cellphonedb_core",
    args_schema=CellphoneDBArgs,
    return_direct=False,
)
def run_cellphonedb_core(
    file_path: str,
    work_dir: Optional[str] = None,
    celltype_column: str = "pred_celltype",
    database_path: Optional[str] = CELLPHONEDB,
    counts_data: str = "hgnc_symbol",
    threads: int = 4,
    pvalue_threshold: float = 0.05,
    focus_genes: Optional[List[str]] = None,
    chord_cell_type1: str = ".",
    chord_cell_type2: str = ".",
) -> Dict[str, Any]:
    """
    Perform cell communication analysis on annotated data
    """

    input_path = Path(file_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    base_dir = Path(work_dir).expanduser().resolve() if work_dir else (input_path.parent / "cellphonedb_results")
    _ensure_directory(base_dir)
    input_dir = _ensure_directory(base_dir / "inputs")
    output_dir = _ensure_directory(base_dir / "outputs")

    # 1) 准备输入
    adata = sc.read_h5ad(str(input_path))
    if celltype_column not in adata.obs:
        raise ValueError(f"AnnData.obs 不包含列 '{celltype_column}'")

    counts_df = _to_dense_frame(adata)
    (input_dir / "counts.txt").write_text(
        counts_df.T.to_csv(sep="\t"), encoding="utf-8"
    )  # genes × cells
    counts_path = input_dir / "counts.txt"

    meta_df = adata.obs[[celltype_column]].copy()
    meta_df.insert(0, "Cell", meta_df.index.astype(str))
    meta_df = meta_df.rename(columns={celltype_column: "cell_type"})
    meta_path = input_dir / "meta.txt"
    meta_df.to_csv(meta_path, sep="\t", index=False)

    # 2) 数据库 zip
    if database_path:
        db_path = Path(database_path).expanduser().resolve()
    else:
        db_path = None
        if settings is not None and hasattr(settings, "CELLPHONEDB_ZIP"):
            db_path = Path(getattr(settings, "CELLPHONEDB_ZIP")).expanduser().resolve()
    if not db_path or not db_path.exists():
        raise FileNotFoundError("未找到 CellPhoneDB 数据库压缩包。请提供有效路径。")

    # 3) 运行 CPDB
    cpdb_statistical_analysis_method.call(
        cpdb_file_path=str(db_path),
        meta_file_path=str(meta_path),
        counts_file_path=str(counts_path),
        counts_data=counts_data,
        output_path=str(output_dir),
        threshold=0.1,
        threads=threads,
        pvalue=pvalue_threshold,
        debug_seed=-1,
        score_interactions=True,
    )

    # 4) 解析输出
    significant_path = (_latest_file(output_dir, "*significant_means*.txt") or output_dir / "significant_means.txt")
    means_path       = (_latest_file(output_dir, "*means*.txt")              or output_dir / "means.txt")
    pvalues_path     = (_latest_file(output_dir, "*pvalues*.txt")            or output_dir / "pvalues.txt")
    deconv_path      = (_latest_file(output_dir, "*deconvoluted*.txt")       or output_dir / "deconvoluted.txt")
    interaction_scores_path = _latest_file(output_dir, "*interaction_scores*.txt")
    deconv_percents_path    = _latest_file(output_dir, "*deconvoluted_percents*.txt")

    tables: Dict[str, Path] = {}
    for name, path in (
        ("means", means_path),
        ("pvalues", pvalues_path),
        ("significant_means", significant_path),
        ("deconvoluted", deconv_path),
        ("interaction_scores", interaction_scores_path),
        ("deconvoluted_percents", deconv_percents_path),
    ):
        if path and Path(path).exists():
            tables[name] = Path(path)

    if not Path(significant_path).exists():
        summary_text_path = base_dir / "summary.txt"
        summary_text = "CellPhoneDB 未返回 significant_means.txt，可能缺乏显著的细胞互作。"
        summary_text_path.write_text(summary_text, encoding="utf-8")
        return CellphoneDBResult("success", summary_text, summary_text_path, {}, tables).to_dict()

    means_df = pd.read_csv(means_path, sep="\t") if Path(means_path).exists() else pd.DataFrame()
    pvalues_df = pd.read_csv(pvalues_path, sep="\t") if Path(pvalues_path).exists() else pd.DataFrame()
    significant_df = pd.read_csv(significant_path, sep="\t")

    sig_long = _melt_interaction_table(significant_df).rename(columns={"value": "value"})
    if sig_long.empty:
        summary_text_path = base_dir / "summary.txt"
        msg = "CellPhoneDB 检测到的显著互作为空，未生成进一步的统计结果。"
        summary_text_path.write_text(msg, encoding="utf-8")
        return CellphoneDBResult("success", msg, summary_text_path, {}, tables).to_dict()

    means_long = _melt_interaction_table(means_df).rename(columns={"value": "mean"})
    pvalues_long = _melt_interaction_table(pvalues_df).rename(columns={"value": "pvalue"})

    merged = sig_long.copy()
    key_cols = [c for c in merged.columns if c != "value"]
    if not means_long.empty:
        keys = [c for c in key_cols if c in means_long.columns]
        merged = merged.merge(means_long[keys + ["mean"]], on=keys, how="left")
    else:
        merged["mean"] = merged["value"].astype(float)
    if not pvalues_long.empty:
        keys = [c for c in key_cols if c in pvalues_long.columns]
        merged = merged.merge(pvalues_long[keys + ["pvalue"]], on=keys, how="left")
    else:
        merged["pvalue"] = 1.0

    if "interacting_pair" in merged.columns:
        merged["interaction"] = merged["interacting_pair"].astype(str)
    else:
        merged["interaction"] = (
            merged.get("gene_a", pd.Series(dtype=str)).astype(str) + "-" +
            merged.get("gene_b", pd.Series(dtype=str)).astype(str)
        )
    merged[["source","target"]] = merged["celltype_pair"].apply(lambda s: pd.Series(_split_pair(str(s))))
    merged["value"]  = merged.get("value", merged.get("mean", pd.Series(dtype=float))).astype(float)
    merged["mean"]   = merged.get("mean", merged["value"]).astype(float)
    merged["pvalue"] = merged.get("pvalue", pd.Series(1.0, index=merged.index)).astype(float)
    merged = merged.dropna(subset=["value"]).reset_index(drop=True)

    # 5) Top 表
    top_interactions = merged.sort_values(["pvalue","value"], ascending=[True, False]).head(50)
    top_path = base_dir / "top_interactions.tsv"
    top_interactions.to_csv(top_path, sep="\t", index=False)
    tables["top_interactions"] = top_path

    # ---- 自动基因（若未传 focus_genes） ----
    auto_genes = None
    if not focus_genes:
        auto_genes = auto_select_focus_genes(merged, max_genes=16, top_pairs=200)
    genes_for_chord = focus_genes or auto_genes

    # ---- 作图输入清洗 ----
    means_df   = _coerce_numeric_cpdb_wide(means_df)
    pvalues_df = _coerce_numeric_cpdb_wide(pvalues_df)

    means_for_chord = _filter_cpdb_by_genes(means_df, genes_for_chord)
    pvals_for_chord = _filter_cpdb_by_genes(pvalues_df, genes_for_chord)

    means_for_chord = _coerce_numeric_cpdb_wide(means_for_chord)
    pvals_for_chord = _coerce_numeric_cpdb_wide(pvals_for_chord)

    # 轻度缩尾，避免超级粗弦
    means_for_chord = _winsorize_wide(means_for_chord, q=0.95)

    # 6) 可视化（ktplotspy）
    figures: Dict[str, Path] = {}

    # 6.1 Heatmap（显著次数）
    heatmap_path = base_dir / "cpdb_heatmap.png"
    plt.figure()
    kpy.plot_cpdb_heatmap(
        pvals=pvalues_df,
        symmetrical=True,
        row_cluster=True,
        col_cluster=True,
        title="CellPhoneDB interaction intensity",
    )
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    figures["interaction_heatmap"] = heatmap_path

    # 6.2 Chord（弦图）
    decon_df = pd.read_csv(deconv_path, sep="\t") if Path(deconv_path).exists() else pd.DataFrame()
    chord_title = ("CellPhoneDB ligand–receptor network (filtered by genes)"
                   if focus_genes else
                   "CellPhoneDB ligand–receptor network (top links)")
    chord_path = base_dir / "cpdb_chord.png"
    if _safe_for_clustering(means_for_chord):
        plt.figure()
        kpy.plot_cpdb_chord(
            adata=adata,
            means=means_for_chord,
            pvals=pvals_for_chord,
            deconvoluted=decon_df if not decon_df.empty else pd.DataFrame(),
            celltype_key=celltype_column,
            cell_type1=chord_cell_type1,   # <== 你要的参数
            cell_type2=chord_cell_type2,   # <== 你要的参数
            edge_cmap=plt.cm.coolwarm,
            remove_self=True,
            figsize=(10, 10),
            title=chord_title,
            scale_lw=7,
            legend_params={"loc":"center left","bbox_to_anchor":(1,1),
                           "frameon":False,"fontsize":9},
        )
        plt.savefig(chord_path, dpi=300, bbox_inches="tight")
        plt.close()
        figures["chord"] = chord_path
    else:
        (base_dir / "cpdb_chord.SKIPPED.txt").write_text(
            "Chord plot skipped: not enough finite/non-zero data after sanitization.\n",
            encoding="utf-8"
        )

    # 7) Summary
    summary_lines = [
        "[CellPhoneDB Summary]",
        f"输入文件: {input_path.name}",
        f"细胞类型列: {celltype_column}",
        "",
    ]
    summary_lines.extend(_summarise_interactions(top_interactions))
    summary_path = base_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return CellphoneDBResult(
        status="success",
        message="CellPhoneDB 分析完成。",
        summary_path=summary_path,
        figures=figures,
        tables=tables,
    ).to_dict()





# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    try:
        args = CellphoneDBArgs(
            file_path="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_annotated.h5ad",
            work_dir="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/cellphonedb",
            celltype_column="pred_celltype",
            database_path=CELLPHONEDB,
            counts_data="hgnc_symbol",
            threads=4,
            pvalue_threshold=0.05,
            # focus_genes=["PTPRC","CD40","CLEC2D"],  # 可选
            chord_cell_type1=".",   # 可填具体细胞类型过滤
            chord_cell_type2=".",   # 可填具体细胞类型过滤
        )
    except ValidationError as e:
        print("参数校验失败：", e)
        raise SystemExit(2)

    out = run_cellphonedb_core(**args.model_dump())
    print(out)