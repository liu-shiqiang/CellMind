# pseudotime_analysis.py — Basic unified version (v1 ⨉ v2)
from __future__ import annotations
from .enrichment_analysis.ora import ORAAnalyzer, ORAVisualizer, ORAEvaluator

import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# -------- path layout (aligned with your enrichment tools) --------
_OUTDIR_ROOT = "analysis_results"
_SUBDIR = "pseudotime"


def _ensure_outdir() -> str:
    outdir = os.path.join(_OUTDIR_ROOT, _SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _log(logger: Optional[Callable[[str], None]], msg: str) -> None:
    if logger:
        logger(msg)


def _is_log_transformed(X) -> bool:
    """Heuristic: values in reasonable log1p range, mostly >0."""
    try:
        if sparse.issparse(X):
            X = X.data if hasattr(X, "data") else X.toarray()
        X = np.asarray(X)
        if X.size == 0:
            return False
        if np.min(X) < 0 or np.max(X) > 20:
            return False
        # >90% positive looks like log1p-normalized
        return float(np.mean(X <= 0)) <= 0.1
    except Exception:
        return False


@dataclass
class PseudotimeResult:
    """Minimal container returned by Analyzer."""
    pseudotime_table: pd.DataFrame                 # columns: cell, pseudotime, [celltype]
    celltype_summary: pd.DataFrame                 # per-celltype stats (median/mean/min/max/count)
    embedding: Optional[pd.DataFrame]              # dim1, dim2, pseudotime, [celltype]
    meta: Dict[str, object]                        # run metadata (thresholds, root, etc.)


class PseudotimeAnalyzer:
    """Run Scanpy DPT; export tables & helper files for downstream ORA/ssGSEA."""

    def run(
        self,
        file_path: str,
        *,
        celltype_col: str = "pred_celltype",
        root_cell: Optional[str] = None,
        root_celltype: Optional[str] = None,
        n_neighbors: int = 30,
        n_pcs: int = 30,
        max_top_genes: int = 2000,
        logger: Optional[Callable[[str], None]] = None,
    ) -> PseudotimeResult:

        _log(logger, f"[pseudotime] loading {file_path}")
        adata = sc.read_h5ad(file_path)
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError("AnnData 为空，无法进行拟时序分析。")

        # --- light preprocess (only if not log-transformed) ---
        meta: Dict[str, object] = {
            "file_path": file_path,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "preprocess": {"normalize_total": False, "log1p": False},
        }
        if not _is_log_transformed(adata.X):
            _log(logger, "[pseudotime] normalize_total + log1p")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            meta["preprocess"] = {"normalize_total": True, "log1p": True}

        # HVG (cap) + scale + PCA + neighbors
        if "highly_variable" not in adata.var.columns:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=min(max_top_genes, adata.n_vars), flavor="seurat", span=0.3
            )
        hv = adata.var.get("highly_variable", pd.Series(False, index=adata.var_names)).values
        if hv.sum() >= 300:  # keep sanity
            adata = adata[:, hv].copy()
        meta["n_highly_variable"] = int(adata.n_vars)

        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars))
        sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, max(5, adata.n_obs - 1)), n_pcs=min(n_pcs, adata.n_vars))

        # Embedding (best effort)
        try:
            sc.tl.umap(adata)
            emb_key = "X_umap"
        except Exception:
            emb_key = None

        # Diffusion map + DPT
        sc.tl.diffmap(adata)
            # ---- resolve root cell ----
        root_idx = self._resolve_root_index(adata, celltype_col, root_cell, root_celltype, logger)
        meta["root_cell"] = str(adata.obs_names[root_idx])
        meta["root_celltype"] = (
            str(adata.obs[celltype_col].iloc[root_idx]) if celltype_col in adata.obs.columns else None
        )

        # ---- DPT (version-compatible) ----
        # 老版本：通过 adata.uns['iroot'] 指定根；新版本也能识别
        adata.uns["iroot"] = int(root_idx)

        # 估个 n_dcs（如果有 diffmap）
        n_dcs = None
        if "X_diffmap" in adata.obsm and adata.obsm["X_diffmap"].shape[1] > 0:
            n_dcs = min(10, adata.obsm["X_diffmap"].shape[1])

        # 兼容不同签名：部分版本只接受 n_dcs，部分不接受任何额外参数
        try:
            sc.tl.dpt(adata, n_dcs=n_dcs)
        except TypeError:
            sc.tl.dpt(adata)

        if "dpt_pseudotime" not in adata.obs.columns:
            raise RuntimeError("scanpy.tl.dpt 未能生成 'dpt_pseudotime'。")

        # ---- tables ----
        pt = adata.obs["dpt_pseudotime"].astype(float)
        tab = {"cell": adata.obs_names.astype(str), "pseudotime": pt.values}
        if celltype_col in adata.obs.columns:
            tab[celltype_col] = adata.obs[celltype_col].astype(str).values
        pt_table = pd.DataFrame(tab)

        # embedding df (2D) + annotate
        embedding_df = None
        candidate = emb_key if emb_key in (adata.obsm.keys() if hasattr(adata.obsm, "keys") else []) else "X_diffmap"
        if candidate in adata.obsm and adata.obsm[candidate].shape[1] >= 2:
            arr = np.asarray(adata.obsm[candidate])[:, :2]
            embedding_df = pd.DataFrame(arr, columns=["dim1", "dim2"], index=adata.obs_names.astype(str))
            embedding_df["pseudotime"] = pt.values
            if celltype_col in adata.obs.columns:
                embedding_df[celltype_col] = adata.obs[celltype_col].astype(str).values
        meta["embedding_key"] = candidate if candidate in adata.obsm else None

        # ---- celltype summary ----
        if celltype_col in pt_table.columns:
            summary = (
                pt_table.groupby(celltype_col)["pseudotime"]
                .agg(["median", "mean", "min", "max", "count"])
                .sort_values("median")
                .reset_index()
            )
        else:
            summary = pd.DataFrame(
                {
                    "median": [float(pt.median())],
                    "mean": [float(pt.mean())],
                    "min": [float(pt.min())],
                    "max": [float(pt.max())],
                    "count": [int(adata.n_obs)],
                }
            )

        # ---- early/late bin (median cut) + export ----
        med = float(pt.median())
        labels = np.where(pt <= med, "early", "late")
        bin_table = pt_table.copy()
        bin_table["pt_bin"] = labels
        outdir = _ensure_outdir()
        bin_table.to_csv(os.path.join(outdir, "pseudotime_bins.tsv"), sep="\t", index=False)
        meta["pt_cut"] = {"scheme": "early/late@median", "median": med}
        meta["bin_counts"] = {
            "early": int((labels == "early").sum()),
            "late": int((labels == "late").sum()),
        }

        # ---- Optional: quick DE for early vs late (export top lists when adequate) ----
        try:
            early_cells = bin_table.loc[bin_table["pt_bin"] == "early", "cell"].values
            late_cells = bin_table.loc[bin_table["pt_bin"] == "late", "cell"].values
            if len(early_cells) >= 50 and len(late_cells) >= 50:
                adata.obs["pt_bin"] = pd.Categorical(labels, categories=["early", "late"])
                sc.tl.rank_genes_groups(adata, groupby="pt_bin", groups=["early"], reference="late", method="wilcoxon")
                early_df = sc.get.rank_genes_groups_df(adata, group="early").sort_values("scores", ascending=False)
                early_genes = early_df["names"].head(150).astype(str).tolist()

                sc.tl.rank_genes_groups(adata, groupby="pt_bin", groups=["late"], reference="early", method="wilcoxon")
                late_df = sc.get.rank_genes_groups_df(adata, group="late").sort_values("scores", ascending=False)
                late_genes = late_df["names"].head(150).astype(str).tolist()

                with open(os.path.join(outdir, "early_markers.txt"), "w") as f:
                    f.write("\n".join(early_genes))
                with open(os.path.join(outdir, "late_markers.txt"), "w") as f:
                    f.write("\n".join(late_genes))

                meta["pt_markers"] = {
                    "early_topN": 150,
                    "late_topN": 150,
                    "files": {"early": "early_markers.txt", "late": "late_markers.txt"},
                }
        except Exception:
            # 非关键路径，静默失败
            pass
        
        # ========= ORA on early/late markers（若存在）=========
        ora_payloads = {}
        artifacts = {
            "plots": {},
            "tables": {
                "pseudotime_bins": os.path.join(outdir, "pseudotime_bins.tsv"),
                "early_markers": os.path.join(outdir, "early_markers.txt") if os.path.exists(os.path.join(outdir, "early_markers.txt")) else None,
                "late_markers":  os.path.join(outdir, "late_markers.txt")  if os.path.exists(os.path.join(outdir, "late_markers.txt"))  else None,
            },
            "summaries": {},
        }

        try:
            ora = ORAAnalyzer()
            oviz = ORAVisualizer()
            oev  = ORAEvaluator()

            if os.path.exists(os.path.join(outdir, "early_markers.txt")):
                res_e = ora.run(
                    input_file=None,
                    gene_list_file=os.path.join(outdir, "early_markers.txt"),  # 需要在 ORAAnalyzer 中支持
                    list_label="early",
                    gene_set="KEGG",
                )
                ora_early_plot = oviz.plot(
                    res_e, outdir=outdir, title_prefix="ORA · Early", top_k=10,
                    output_path=os.path.join(outdir, "ora_early_plot.png")
                )
                artifacts["plots"]["ora_early"] = ora_early_plot
                _, paths_e = oev.evaluate(res_e, outdir=outdir, basename="ora_early", return_paths=True)
                artifacts["tables"]["ora_early_top_terms"] = paths_e.get("tsv")
                artifacts["summaries"]["ora_early"]        = paths_e.get("summary")
                with open(paths_e["json"], "r", encoding="utf-8") as fh:
                    ora_payloads["early"] = json.load(fh)

            if os.path.exists(os.path.join(outdir, "late_markers.txt")):
                res_l = ora.run(
                    input_file=None,
                    gene_list_file=os.path.join(outdir, "late_markers.txt"),
                    list_label="late",
                    gene_set="KEGG",
                )
                ora_late_plot = oviz.plot(
                    res_l, outdir=outdir, title_prefix="ORA · Late", top_k=10,
                    output_path=os.path.join(outdir, "ora_late_plot.png")
                )
                artifacts["plots"]["ora_late"] = ora_late_plot
                _, paths_l = oev.evaluate(res_l, outdir=outdir, basename="ora_late", return_paths=True)
                artifacts["tables"]["ora_late_top_terms"] = paths_l.get("tsv")
                artifacts["summaries"]["ora_late"]        = paths_l.get("summary")
                with open(paths_l["json"], "r", encoding="utf-8") as fh:
                    ora_payloads["late"] = json.load(fh)
        except Exception as e:
            _log(logger, f"[pseudotime] ORA pipeline failed: {e}")

        # ========= 固定 schema 的 master JSON / TXT =========
        schema_version = "1.0"

        # 若可视化已由 PseudotimeVisualizer 生成，则这里补齐路径（不存在就跳过）
        for k, fn in [("embedding", "pseudotime_embedding.png"),
                      ("hist", "pseudotime_hist.png"),
                      ("celltype_box", "pseudotime_celltype_boxplot.png")]:
            p = os.path.join(outdir, fn)
            if os.path.exists(p):
                artifacts["plots"][k] = p

        master_json = {
            "schema_version": schema_version,
            "pseudotime": {
                "meta": {
                    "file_path": file_path,
                    "n_cells": meta["n_cells"],
                    "n_genes": meta["n_genes"],
                    "embedding_key": meta.get("embedding_key"),
                    "root_cell": meta.get("root_cell"),
                    "root_celltype": meta.get("root_celltype"),
                    "pt_cut": meta["pt_cut"],
                    "bin_counts": meta["bin_counts"],
                    "timestamp": datetime.now().isoformat(),
                },
                "celltype_summary": summary.to_dict(orient="records"),
            },
            "enrichment": { "ORA": ora_payloads },
            "artifacts": artifacts,
        }
        with open(os.path.join(outdir, "pseudotime_master.json"), "w", encoding="utf-8") as fh:
            json.dump(master_json, fh, ensure_ascii=False, indent=2)

        # 人类可读汇总
        lines = []
        lines.append("[Pseudotime+Enrichment Summary] " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append(f"- Cells: {meta['n_cells']}, Genes: {meta['n_genes']}")
        lines.append(f"- Root: {meta.get('root_cell')} ({meta.get('root_celltype')})")
        lines.append(f"- Split: early/late@median = {meta['pt_cut']['median']:.3f}")
        lines.append(f"- Counts: {meta['bin_counts']}")
        lines.append("")
        if ora_payloads:
            for stage in ("early", "late"):
                if stage in ora_payloads:
                    pj = ora_payloads[stage]
                    lines.append(f"ORA-{stage}: {pj.get('n_significant', 0)} significant terms (adjP<0.05)")
                    for t in (pj.get("top_terms", [])[:5]):
                        term = t.get("Term"); ap = t.get("adjP"); og = t.get("Overlapping Genes")
                        lines.append(f"  - {term} (adjP={ap}, n_genes={og})")
                    lines.append("")
        with open(os.path.join(outdir, "pseudotime_master.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        return PseudotimeResult(
            pseudotime_table=pt_table,
            celltype_summary=summary,
            embedding=embedding_df,
            meta=meta,
        )

    @staticmethod
    def _resolve_root_index(
        adata: "sc.AnnData",
        celltype_col: str,
        root_cell: Optional[str],
        root_celltype: Optional[str],
        logger: Optional[Callable[[str], None]],
    ) -> int:
        if root_cell and root_cell in adata.obs_names:
            return int(np.where(adata.obs_names == root_cell)[0][0])
        if root_celltype and celltype_col in adata.obs.columns:
            mask = adata.obs[celltype_col].astype(str).str.lower() == root_celltype.lower()
            if mask.any():
                if "X_diffmap" in adata.obsm:
                    diff1 = np.asarray(adata.obsm["X_diffmap"])[:, 0]
                    idxs = np.where(mask.values)[0]
                    # pick the most "early" along diffusion component 1
                    best = idxs[np.argmin(diff1[idxs])]
                    return int(best)
                return int(np.where(mask.values)[0][0])
            _log(logger, f"[pseudotime] root_celltype='{root_celltype}' 未找到，自动推断")
        if "X_diffmap" in adata.obsm:
            diff1 = np.asarray(adata.obsm["X_diffmap"])[:, 0]
            return int(np.argmin(diff1))
        return 0


class PseudotimeVisualizer:
    """Two quick plots: embedding colored by pseudotime + histogram with median cut; optional boxplot by celltype."""

    def plot(
        self,
        result: PseudotimeResult,
        *,
        with_celltype_box: bool = True,
        celltype_col: str = "pred_celltype",
        cmap: str = "viridis",
    ) -> Dict[str, Optional[str]]:
        import matplotlib.pyplot as plt
        import numpy as np
        outdir = _ensure_outdir()
        paths: Dict[str, Optional[str]] = {"embedding": None, "hist": None, "celltype_box": None}

        # 1) embedding colored by pseudotime
        if result.embedding is not None and not result.embedding.empty:
            df = result.embedding.copy()
            plt.figure(figsize=(6.6, 5.4))
            scp = plt.scatter(df["dim1"], df["dim2"], c=df["pseudotime"], s=10, cmap=cmap, linewidths=0, alpha=0.9)
            plt.colorbar(scp, label="Pseudotime")
            plt.xlabel("dim1"); plt.ylabel("dim2"); plt.title("Pseudotime trajectory")
            emb_path = os.path.join(outdir, "pseudotime_embedding.png")
            plt.tight_layout(); plt.savefig(emb_path, dpi=220); plt.close()
            paths["embedding"] = emb_path

        # 2) histogram + median cut
        pt = result.pseudotime_table["pseudotime"].values
        med = float(np.median(pt))
        plt.figure(figsize=(6.6, 4.2))
        plt.hist(pt, bins=30, alpha=0.9)
        plt.axvline(med, color="red", linestyle="--", linewidth=1, label=f"median={med:.3f}")
        plt.legend(frameon=False)
        plt.xlabel("Pseudotime"); plt.ylabel("Cell count"); plt.title("Pseudotime distribution")
        hist_path = os.path.join(outdir, "pseudotime_hist.png")
        plt.tight_layout(); plt.savefig(hist_path, dpi=220); plt.close()
        paths["hist"] = hist_path

        # 3) optional: boxplot by celltype (order by median)
       # --- 替换 celltype_box 段 ---
        if celltype_col in result.pseudotime_table.columns:
            import textwrap
            df = result.pseudotime_table[[celltype_col, "pseudotime"]].copy()

            # 顺序：按中位数从早到晚
            order = (
                result.celltype_summary[celltype_col].tolist()
                if celltype_col in result.celltype_summary.columns
                else sorted(df[celltype_col].unique())
            )

            # 根据类别数动态调整：<=12 纵排；>12 自动横排（更易读）
            n_ct = len(order)
            horizontal = n_ct > 12

            # 文本换行，避免超长标签挤在一起
            def _wrap(lbl, width=28):
                return "\n".join(textwrap.wrap(str(lbl), width=width))

            labels_wrapped = [_wrap(ct, width=26 if horizontal else 22) for ct in order]
            values = [df.loc[df[celltype_col] == ct, "pseudotime"].values for ct in order]

            # 动态画布大小（越多类型越宽/高，但限制最大尺寸）
            if horizontal:
                # 横向箱线：高随类别数变化
                h = min(0.34 * n_ct + 2.5, 18)
                w = 8.5
                plt.figure(figsize=(w, h))
                box = plt.boxplot(
                    values, vert=False, tick_labels=labels_wrapped,
                    patch_artist=True, widths=0.6
                )
                plt.xlabel("Pseudotime"); plt.ylabel(celltype_col)
            else:
                # 纵向箱线：宽随类别数变化
                w = min(max(8, 0.55 * n_ct + 2), 22)
                h = 5.2
                plt.figure(figsize=(w, h))
                box = plt.boxplot(
                    values, vert=True, tick_labels=labels_wrapped,
                    patch_artist=True, widths=0.6
                )
                plt.ylabel("Pseudotime"); plt.xlabel(celltype_col)
                plt.xticks(rotation=0, ha="center")

            # 上色 + 叠加抖动点
            for patch in box["boxes"]:
                patch.set_facecolor("#90CAF9")
                patch.set_alpha(0.7)

            import numpy as np
            for i, pts in enumerate(values, start=1):
                if pts.size == 0: 
                    continue
                jitter = np.random.uniform(-0.08, 0.08, size=pts.size)
                if horizontal:
                    plt.scatter(pts, np.full_like(pts, i) + jitter, s=8, color="#1E88E5", alpha=0.35, linewidths=0)
                else:
                    plt.scatter(np.full_like(pts, i) + jitter, pts, s=8, color="#1E88E5", alpha=0.35, linewidths=0)

            plt.title("Pseudotime per cell type")
            plt.tight_layout()
            box_path = os.path.join(outdir, "pseudotime_celltype_boxplot.png")
            plt.savefig(box_path, dpi=220)
            plt.close()
            paths["celltype_box"] = box_path


        return paths


class PseudotimeEvaluator:
    """Generate a textual summary for pseudotime analysis results."""

    def evaluate(
        self,
        result: PseudotimeResult,
        top_k: int = 5,
        *,
        outdir: Optional[str] = None,
        celltype_col: str = "pred_celltype",
        save_text: bool = True,
        filename: str = "pseudotime_summary.txt",
        include_ora: bool = True,                 # 新增：是否合并 ORA 段
        master_json_name: str = "pseudotime_master.json",  # 新增：master 文件名
    ) -> str:
        import json, os
        from datetime import datetime
        import numpy as np

        # ---- 基础统计（和你之前一致）----
        if result is None or result.pseudotime_table is None or result.pseudotime_table.empty:
            return "拟时序结果为空。"
        pt_series = result.pseudotime_table["pseudotime"].astype(float)
        meta = result.meta or {}

        stats = {
            "min": float(np.min(pt_series)),
            "max": float(np.max(pt_series)),
            "median": float(np.median(pt_series)),
            "q1": float(np.quantile(pt_series, 0.25)),
            "q3": float(np.quantile(pt_series, 0.75)),
        }

        lines = ["拟时序分析摘要："]
        lines.append(f"  • 细胞数：{len(pt_series):,}")
        lines.append(f"  • 基因数：{meta.get('n_genes', 'NA')}")
        root_ct = meta.get("root_celltype")
        root_cell = meta.get("root_cell") or (meta.get("root_cells", [None])[0] if isinstance(meta.get("root_cells"), list) else None)
        root_part = f"{root_cell}" if root_cell is not None else "NA"
        if root_ct:
            root_part += f" ({root_ct})"
        lines.append(f"  • 根细胞：{root_part}")

        lines.append(
            "  • 拟时序范围：{min:.3f}–{max:.3f}（中位数 {median:.3f}；Q1={q1:.3f}, Q3={q3:.3f})"
            .format(**stats)
        )

        if celltype_col in result.pseudotime_table.columns and result.celltype_summary is not None and not result.celltype_summary.empty:
            lead = result.celltype_summary.nsmallest(top_k, "median")
            tail = result.celltype_summary.nlargest(top_k, "median")

            def _blk(df, title):
                arr = [
                    f"     - {r[celltype_col]}: median={r['median']:.3f}, n={int(r['count'])}"
                    for _, r in df.iterrows()
                ]
                return [title] + arr if arr else []

            lines += _blk(lead, "  • 最早进入拟时序的细胞类型：")
            lines += _blk(tail, "  • 最晚进入拟时序的细胞类型：")

        if "pt_cut" in meta:
            cut = meta["pt_cut"] or {}
            lines.append(f"  • 切分：{cut.get('scheme','NA')}，median={cut.get('median','NA')}")
        if "bin_counts" in meta:
            lines.append(f"  • 计数：{meta['bin_counts']}")
        if "pt_markers" in meta:
            m = meta["pt_markers"]
            files = m.get("files", {})
            lines.append(
                f"  • 已导出阶段 marker：early({files.get('early','-')}), "
                f"late({files.get('late','-')})  [topN={m.get('early_topN','NA')}]"
            )

        # ---- 追加 ORA 段（从 master JSON 读取，避免文件命名差异）----
        if include_ora:
            dflt_dir = os.path.join("analysis_results", "pseudotime")
            use_dir = outdir or dflt_dir
            master_path = os.path.join(use_dir, master_json_name)
            if os.path.exists(master_path):
                try:
                    with open(master_path, "r", encoding="utf-8") as fh:
                        mj = json.load(fh)
                    ora_payloads = (mj.get("enrichment", {}) or {}).get("ORA", {}) or {}
                    if ora_payloads:
                        lines.append("")
                        lines.append("ORA 富集摘要：")
                        for stage in ("early", "late"):
                            if stage in ora_payloads:
                                pj = ora_payloads[stage] or {}
                                n_sig = pj.get("n_significant")
                                lines.append(f"  • {stage}: 显著通路 {n_sig if n_sig is not None else 0} 个 (adjP<0.05)")
                                top_terms = pj.get("top_terms", [])[:min(5, len(pj.get('top_terms', [])))]
                                for t in top_terms:
                                    term = t.get("Term")
                                    ap = t.get("adjP")
                                    og = t.get("Overlapping Genes")
                                    lines.append(f"     - {term} (adjP={ap}, n_genes={og})")
                except Exception:
                    # 读取失败不影响主体
                    pass

        summary_text = "\n".join(lines)

        # ---- 落盘：summary.txt + result.json（保留你原来的输出）----
        if save_text:
            use_dir = outdir or os.path.join("analysis_results", "pseudotime")
            os.makedirs(use_dir, exist_ok=True)
            with open(os.path.join(use_dir, filename), "w", encoding="utf-8") as f:
                f.write(summary_text + "\n")

            payload = {
                "meta": meta,
                "pseudotime_stats": stats,
                "timestamp": datetime.now().isoformat(),
            }
            with open(os.path.join(use_dir, "pseudotime_result.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        return summary_text


# Factory (与其它工具保持一致接口)
class PseudotimeFactory:
    def create_analyzer(self) -> PseudotimeAnalyzer: return PseudotimeAnalyzer()
    def create_visualizer(self) -> PseudotimeVisualizer: return PseudotimeVisualizer()
    def create_evaluator(self) -> PseudotimeEvaluator: return PseudotimeEvaluator()


if __name__ == "__main__":
    # Minimal smoke test (修改为你的 .h5ad 路径)
    in_h5ad = "/home/share/huadjyin/home/zhangzilin/genomix-agent/data/test_file/test_l3_stratified_5pct_annotated.h5ad"
    celltype_col = "pred_celltype"

    if os.path.exists(in_h5ad):
        analyzer = PseudotimeAnalyzer()
        res = analyzer.run(in_h5ad, celltype_col=celltype_col, root_celltype=None)

        viz = PseudotimeVisualizer()
        artifacts = viz.plot(res)
        print("[plots]", artifacts)

        eva = PseudotimeEvaluator()
        print(eva.evaluate(res, celltype_col=celltype_col))
    else:
        print(f"Please update __main__ input path: {in_h5ad}")