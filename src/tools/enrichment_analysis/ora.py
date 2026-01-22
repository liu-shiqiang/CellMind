from pathlib import Path
from typing import Optional, Callable, Dict, List
import os, json
from datetime import datetime

import numpy as np
import pandas as pd
import anndata as ad
import gseapy as gp
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.tools.artifact_paths import resolve_artifact_dir

from .interface import EnrichmentAnalysiszer, EnrichmentVisualizer, EnrichmentEvaluator, EnrichmentFactory
from .data_setting import extract_gene_list_from_celltype, AnalysisResult

_DEF_TOPN = {"kegg": 150, "go": 250}
_LIB = {"kegg": "KEGG_2021_Human", "go": "GO_Biological_Process_2021"}

_OUTDIR_ROOT = "enrichment_results"
_SUBDIR = "ora"

def _ensure_outdir() -> str:
    outdir = os.path.join(_OUTDIR_ROOT, _SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


class ORAAnalyzer(EnrichmentAnalysiszer):
    def run(
        self,
        input_file: str,
        *,
        celltype_col: str = "pred_celltype",
        target_celltype: Optional[str] = None,
        gene_set: str = "kegg",
        top_n: Optional[int] = None,
        enrichr_lib_path: Optional[str] = None,
        # 新增：直接传入基因列表或基因文件（任一即可）
        gene_list: Optional[list] = None,
        gene_list_file: Optional[str] = None,
        list_label: Optional[str] = None,   # 例如 "early" / "late"，用于 meta 标注
        quiet: bool = True,
        logger: Optional[Callable[[str], None]] = None,
    ) -> AnalysisResult:
        gs = gene_set.lower()
        if gs not in _LIB:
            raise ValueError("gene_set 仅支持 'kegg' 或 'go'")
        lib_name = _LIB[gs]
        if top_n is None:
            top_n = _DEF_TOPN[gs]

        # ---- 1) 解析 gene_list 优先级：gene_list > gene_list_file > 自动提取 ----
        markers: list[str]
        if gene_list is not None:
            markers = [str(g).strip() for g in gene_list if str(g).strip()]
        elif gene_list_file is not None:
            with open(gene_list_file, "r") as fh:
                markers = [ln.strip() for ln in fh if ln.strip()]
        else:
            # 沿用你原有的“按 celltype 提 marker”的逻辑
            if target_celltype is None:
                adata = ad.read_h5ad(input_file)
                if celltype_col not in adata.obs.columns:
                    raise ValueError(f"adata.obs 缺少列 '{celltype_col}'")
                ser = adata.obs[celltype_col]
                counts = (ser.value_counts().astype(int) if not hasattr(ser, "cat")
                          else ser.value_counts().reindex(ser.cat.categories, fill_value=0).astype(int))
                target_celltype = counts.idxmax()

            markers = extract_gene_list_from_celltype(
                input_file,
                celltype_col=celltype_col,
                target_celltype=target_celltype,
                top_n=top_n,
                return_info=False,
            )

        if not markers:
            return AnalysisResult(pd.DataFrame(), {}, meta={"gene_set": lib_name})

        # ---- 2) ORA ----
        gene_sets = enrichr_lib_path if enrichr_lib_path else lib_name
        enr = gp.enrichr(
            gene_list=markers,
            gene_sets=gene_sets,
            organism="Human",
            outdir=None,
            no_plot=True,
            cutoff=1.0,
            verbose=0
        )
        if enr is None or not hasattr(enr, "res2d") or enr.res2d is None or enr.res2d.empty:
            return AnalysisResult(pd.DataFrame(), {}, meta={"gene_set": lib_name})

        df = enr.res2d.copy()
        # 统一 adjP
        if "Adjusted P-value" in df.columns:
            df = df.rename(columns={"Adjusted P-value": "adjP"})
        elif "P-value" in df.columns and "adjP" not in df.columns:
            df["adjP"] = df["P-value"]

        df = df.sort_values("adjP", ascending=True)
        top_terms = df.head(20).copy()
        pvalues = dict(zip(top_terms["Term"], top_terms["adjP"]))

        # ---- 3) meta 补充：记录这是来自哪个 list（early/late） ----
        meta = {
            "gene_set": lib_name,
            "organism": "Human",
            "top_n": top_n,
            "source": ("gene_list" if gene_list is not None else
                       "gene_list_file" if gene_list_file is not None else
                       "markers_from_celltype"),
            "list_label": list_label,               # e.g., "early" / "late"
            "gene_list_file": gene_list_file,
            "celltype_col": celltype_col if gene_list is None and gene_list_file is None else None,
            "target_celltype": target_celltype if gene_list is None and gene_list_file is None else None,
            "n_input_genes": len(markers),
        }

        return AnalysisResult(top_terms=top_terms, pvalues=pvalues, meta=meta)


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

        outdir = outdir or os.path.join("enrichment_results", "ora")
        os.makedirs(outdir, exist_ok=True)

        df = result.top_terms.copy()

        # --- 1) 优先从 meta 里取信息（可选字段，缺失则回退） ---
        meta = getattr(result, "meta", {}) or {}
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

        # 4) 动态画布高度（top_k 越大越高；兼顾长标签）
        base_h = 1.0 + 0.5 * len(df)          # 每个条 0.5 高度 + 边距
        base_h = max(4.5, min(base_h, 16))    # 限制上下限，避免过大或过小
        plt.figure(figsize=(9, base_h))

        bars = plt.barh(y=df["Label"], width=df["logP"], height=0.7, edgecolor="white")
        # 若需要在条内标出 Count（仅在 annotate_count_in_label=False 时可考虑额外标注）
        if not annotate_count_in_label and "Count" in df.columns:
            for bar, c in zip(bars, df["Count"]):
                if c and np.isfinite(bar.get_width()):
                    plt.text(bar.get_width() - 0.15, bar.get_y() + bar.get_height()/2,
                             str(c), ha="right", va="center", color="white", fontweight="bold", fontsize=10)

        plt.xlabel("-log10(Adjusted P-value)"); plt.ylabel("Pathway")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()

      # 5) 输出文件路径（与统一命名一致）
        if output_path is None:
            output_path = os.path.join(outdir, "ora_plot.png")
        else:
            # 如果传入是目录，则在目录下写文件
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, "ora_plot.png")
            else:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path


class ORAEvaluator(EnrichmentEvaluator):
    def evaluate(
        self,
        result: AnalysisResult,
        *,
        outdir: Optional[str] = None,   # ← 新增：允许自定义输出目录
        basename: Optional[str] = None, # ← 新增：文件名前缀（避免覆盖）
        top_k: int = 20,
        return_paths: bool = True,
    ):
        # 目录
        if outdir is None:
            outdir = _ensure_outdir()
        os.makedirs(outdir, exist_ok=True)

        # 文件名前缀
        prefix = basename or "ora"

        # ===== 空结果也写最小三件套 =====
        if result is None or result.top_terms is None or result.top_terms.empty:
            payload = {
                "method": "ORA",
                "database": None,
                "celltype_col": None,
                "target_celltype": None,
                "organism": "Human",
                "top_n": None,
                "n_terms": 0,
                "n_significant": 0,
                "pvalues": {},
                "top_terms": [],
                "meta": getattr(result, "meta", {}) or {},
            }
            json_path = os.path.join(outdir, f"{prefix}_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            summary_path = os.path.join(outdir, f"{prefix}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("[ORA Summary] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("No enriched terms.\n")

            tsv_path = os.path.join(outdir, f"{prefix}_top_terms.tsv")
            pd.DataFrame().to_csv(tsv_path, sep="\t", index=False)

            msg = "No enriched terms."
            return (msg, {"json": json_path, "summary": summary_path, "tsv": tsv_path}) if return_paths else msg

        # ===== 非空结果：与你原逻辑一致，只是文件名用 prefix =====
        df = result.top_terms.copy()
        if "adjP" not in df.columns:
            if "Adjusted P-value" in df.columns:
                df = df.rename(columns={"Adjusted P-value": "adjP"})
            elif "P-value" in df.columns:
                df["adjP"] = df["P-value"]

        save_df = df.sort_values("adjP", ascending=True).head(top_k)

        def _count_overlap(val):
            if isinstance(val, str):
                sep = ";" if ";" in val else ("," if "," in val else None)
                if sep:
                    return len([g for g in val.split(sep) if g.strip()])
                else:
                    return 1 if val.strip() else 0
            elif isinstance(val, (list, tuple, set)):
                return len(val)
            return 0

        if "Overlap" in save_df.columns:
            save_df["Overlapping Genes"] = save_df["Overlap"].astype(str).apply(
                lambda s: int(s.split("/")[0]) if "/" in s else _count_overlap(s)
            )
        elif "Overlapping Genes" in save_df.columns:
            save_df["Overlapping Genes"] = save_df["Overlapping Genes"].apply(_count_overlap)
        else:
            save_df["Overlapping Genes"] = np.nan

        tsv_path = os.path.join(outdir, f"{prefix}_top_terms.tsv")
        save_df.to_csv(tsv_path, sep="\t", index=False)

        pvals = {str(r["Term"]): _safe_float(r["adjP"]) for _, r in save_df.iterrows()}
        payload = {
            "method": "ORA",
            "database": str(df.get("Gene_set", pd.Series(["Enrichr"])).iloc[0]) if "Gene_set" in df.columns else "Enrichr",
            "celltype_col": None,
            "target_celltype": None,
            "organism": "Human",
            "top_n": int(len(df)) if len(df) else None,
            "n_terms": int(len(df)),
            "n_significant": int((df["adjP"] < 0.05).sum()) if "adjP" in df.columns else None,
            "pvalues": pvals,
            "top_terms": [
                {
                    "Term": str(r["Term"]),
                    "adjP": _safe_float(r["adjP"]),
                    "Overlapping Genes": int(r["Overlapping Genes"]) if not pd.isna(r["Overlapping Genes"]) else None
                }
                for _, r in save_df.iterrows()
            ],
            "meta": (getattr(result, "meta", {}) or {})
        }

        json_path = os.path.join(outdir, f"{prefix}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"[ORA Summary] {ts}",
            f"Database   : {payload['database']}",
            f"Result rows: {payload['n_terms']}",
            f"Significant: {payload['n_significant']} (adjP<0.05)",
            "",
            "Top significant terms:",
        ]
        for _, r in save_df.head(min(10, len(save_df))).iterrows():
            gc = int(r["Overlapping Genes"]) if not pd.isna(r["Overlapping Genes"]) else "-"
            lines.append(f"  - {r['Term']} (adjP={_safe_float(r['adjP'])}, n_genes={gc})")

        summary_path = os.path.join(outdir, f"{prefix}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        msg = f"{payload['n_significant']} enriched terms (adjP<0.05)."
        return (msg, {"json": json_path, "summary": summary_path, "tsv": tsv_path}) if return_paths else msg


class ORAFactory(EnrichmentFactory):
    def create_analyzer(self): return ORAAnalyzer()
    def create_visualizer(self): return ORAVisualizer()
    def create_evaluator(self): return ORAEvaluator()



class ORAEnrichmentArgs(BaseModel):
    input_file: str = Field(..., description="Path to the annotated AnnData (.h5ad) file.")
    work_dir: Optional[str] = Field(
        default=None,
        description="Work directory where enrichment outputs will be stored.",
    )
    celltype_col: str = Field(
        default="pred_celltype",
        description="Column in AnnData.obs containing cell type annotations.",
    )
    target_celltype: Optional[str] = Field(default=None, description="Specific cell type to extract marker genes for enrichment.")
    gene_set: str = Field(default="kegg", description="Gene set library to use (e.g. 'kegg' or 'go').")
    top_n: Optional[int] = Field(default=None, description="Number of top marker genes to include from the selected population.")
    enrichr_lib_path: Optional[str] = Field(default=None, description="Optional path to a local Enrichr library file.")
    gene_list: Optional[List[str]] = Field(default=None, description="Explicit list of genes to analyse. Overrides automatic extraction.")
    gene_list_file: Optional[str] = Field(default=None, description="Text file containing genes (one per line). Overrides automatic extraction.")
    list_label: Optional[str] = Field(default=None, description="Label describing the gene list (e.g. 'early' or 'late').")
    cluster_id: Optional[str] = Field(default=None, description="Cluster identifier associated with this enrichment run.")


@tool("run_ora_enrichment", args_schema=ORAEnrichmentArgs)
def run_ora_enrichment(
    input_file: str,
    work_dir: Optional[str] = None,
    celltype_col: str = "pred_celltype",
    target_celltype: Optional[str] = None,
    gene_set: str = "kegg",
    top_n: Optional[int] = None,
    enrichr_lib_path: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    gene_list_file: Optional[str] = None,
    list_label: Optional[str] = None, 
    cluster_id: Optional[str] = None,
) -> str:
    """Run ORA enrichment using marker genes derived from a cluster or a custom gene list."""

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists() and not (gene_list or gene_list_file):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    work_path = resolve_artifact_dir(
        input_path=input_path,
        work_dir=work_dir,
        subdir="enrichment/ora",
    )
    outdir = work_path

    analyzer = ORAAnalyzer()
    if input_path.exists() and not (gene_list or gene_list_file):
        adata_preview = ad.read_h5ad(str(input_path))
        if celltype_col not in adata_preview.obs.columns:
            if "cell_type" in adata_preview.obs.columns:
                celltype_col = "cell_type"
            elif "pred_celltype" in adata_preview.obs.columns:
                celltype_col = "pred_celltype"
            else:
                raise ValueError(f"AnnData.obs 不包含列 '{celltype_col}'")
    result = analyzer.run(
        input_file=str(input_path),
        celltype_col=celltype_col,
        target_celltype=target_celltype,
        gene_set=gene_set,
        top_n=top_n,
        enrichr_lib_path=enrichr_lib_path,
        gene_list=gene_list,
        gene_list_file=gene_list_file,
        list_label=list_label,
    )

    meta = dict(getattr(result, "meta", {}) or {})
    meta.setdefault("gene_set", gene_set)
    meta.setdefault("work_dir", str(work_path))
    if cluster_id is not None:
        meta["cluster_id"] = str(cluster_id)
    result.meta = meta

    sample_name = work_path.name or (input_path.stem if input_path.name else "sample")
    prefix_parts = [sample_name]
    if cluster_id:
        prefix_parts.append(f"cluster{cluster_id}")
    elif target_celltype:
        prefix_parts.append(str(target_celltype).replace(" ", "_"))
    if list_label:
        prefix_parts.append(str(list_label))
    basename = "_".join([part for part in prefix_parts if part]) or "ora"

    visualizer = ORAVisualizer()
    plot_path = visualizer.plot(
        result,
        outdir=str(outdir),
        title_prefix=f"ORA · {gene_set.upper()}",
        output_path=str(outdir / f"{basename}_plot.png"),
    )

    evaluator = ORAEvaluator()
    eval_msg, eval_paths = evaluator.evaluate(
        result,
        outdir=str(outdir),
        basename=basename,
        return_paths=True,
    )

    paths_dict: Dict[str, str] = {}
    for key, value in (eval_paths or {}).items():
        if value:
            paths_dict[key] = str(Path(value).expanduser().resolve())
    if plot_path:
        paths_dict["plot"] = str(Path(plot_path).expanduser().resolve())

    top_terms_records: List[Dict[str, object]] = []
    if result.top_terms is not None and not result.top_terms.empty:
        for row in result.top_terms.head(20).to_dict(orient="records"):
            cleaned = {}
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
        "pvalues": {str(k): float(v) for k, v in (result.pvalues or {}).items()},
        "meta": result.meta,
    }

    return json.dumps(payload, ensure_ascii=False)



if __name__=="__main__":
    
    analyzer = ORAAnalyzer()
    res = analyzer.run(
        input_file="/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_l3_stratified_5pct/test_l3_stratified_5pct_annotated.h5ad",
        gene_set="KEGG",
        celltype_col="pred_celltype",
        target_celltype="CD8-positive, alpha-beta cytotoxic T cell"
    )
    viz = ORAVisualizer()
    plot_path = viz.plot(res)
    print("Plot:", plot_path)
    ev = ORAEvaluator()
    msg, paths = ev.evaluate(res, return_paths=True)
    print(msg, paths)
