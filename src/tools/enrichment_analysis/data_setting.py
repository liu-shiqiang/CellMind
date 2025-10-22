from typing import List, Tuple, Dict, Union, Optional, Callable, Any, Iterable

import difflib,ast
from collections import OrderedDict
import pandas as pd
import scanpy as sc
import gseapy as gp
from config.setting import settings

import anndata as ad
import numpy as np
import scipy.sparse as sp

def extract_gene_list_from_celltype(
    file_path: str,
    celltype_col: str = "pred_celltype",
    target_celltype: Optional[str] = None,    # ← 允许 None：不传则取该列里最多的类别
    top_n: int = 200,
    method: str = "wilcoxon",
    return_info: bool = False,
    logger: Optional[Callable[[str], None]] = None
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:

    def _log(msg: str):
        if logger: logger(msg)

    adata = ad.read_h5ad(file_path)
    obs_cols = list(map(str, adata.obs.columns))
    if celltype_col not in adata.obs.columns:
        close_cols = difflib.get_close_matches(celltype_col, obs_cols, n=5, cutoff=0.3)
        raise ValueError(f"adata.obs 缺少列 '{celltype_col}'；相似：{close_cols or '（无）'}")

    ser = adata.obs[celltype_col]
    categories = list(ser.cat.categories) if hasattr(ser, "cat") else sorted(list(map(str, ser.astype(str).unique())))

    # 默认 celltype：该列里数量最多的类别
    if target_celltype is None:
        counts = ser.value_counts()
        if hasattr(ser, "cat"): counts = counts.reindex(ser.cat.categories, fill_value=0)
        target_celltype = counts.idxmax()

    # 规范化匹配
    t_raw = str(target_celltype); t_lc = t_raw.lower()
    cats_lc = [str(c).lower() for c in categories]
    if t_lc not in cats_lc:
        close_cats = difflib.get_close_matches(t_raw, list(map(str, categories)), n=10, cutoff=0.3)
        raise ValueError(f"目标细胞类型 '{t_raw}' 不在 '{celltype_col}'。相近：{close_cats or '（无）'}")
    target_resolved = str(categories[cats_lc.index(t_lc)])
    n_cells = int(np.sum(ser.astype(str) == target_resolved))

    # 预处理（略，保持你的原逻辑）
    def _is_log_transformed(X):
        if sp.issparse(X): X = X.toarray()
        X = np.asarray(X)
        if X.size == 0: return False
        try:
            if np.min(X) < 0 or np.max(X) > 20: return False
            return np.mean(X <= 0) <= 0.1
        except Exception:
            return False

    did_norm = did_log1p = False
    if not _is_log_transformed(adata.X):
        sc.pp.normalize_total(adata, target_sum=1e4); did_norm = True
        sc.pp.log1p(adata); did_log1p = True

    # 差异分析
    key_added = "_ora_temp_rank_genes"
    sc.tl.rank_genes_groups(adata, groupby=celltype_col, method=method, key_added=key_added)

    # 提取 top_n
    try:
        try:
            df = sc.get.rank_genes_groups_df(adata, key=key_added)
            if df is None or df.empty: raise RuntimeError("rank_genes_groups_df 为空")
            sort_key = "scores" if "scores" in df.columns else "logfoldchanges"
            sub = df[df["group"] == target_resolved].sort_values(sort_key, ascending=False)
            markers = sub["names"].head(top_n).astype(str).tolist()
        except Exception:
            rg = adata.uns.get(key_added, {})
            names_obj = rg["names"]
            if isinstance(names_obj, dict):
                arr = names_obj[target_resolved]; markers_all = list(map(str, (arr.tolist() if hasattr(arr,"tolist") else list(arr))))
                markers = markers_all[:top_n]
            else:
                groups = list(ser.cat.categories) if hasattr(ser, "cat") else []
                gi = groups.index(target_resolved)
                arr_np = np.array(names_obj)
                col = arr_np[:, gi] if (arr_np.ndim == 2 and arr_np.shape[1] > gi) else (arr_np[gi, :] if arr_np.ndim == 2 else np.array(list(names_obj)).astype(object))
                markers_all = list(map(str, (col.tolist() if hasattr(col, "tolist") else list(col))))
                markers = markers_all[:top_n]

        # 展平 + 清洗
        def _flatten(seq):
            out = []
            for g in seq:
                if isinstance(g, (list, tuple)): out.extend(map(str, g)); continue
                s = str(g).strip()
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple)): out.extend(map(str, val)); continue
                except Exception:
                    pass
                if s.startswith("(") and s.endswith(")"): s = s[1:-1]
                if "," in s:
                    out.extend([p.strip().strip("'\"") for p in s.split(",") if p.strip()])
                else:
                    out.append(s)
            return out

        def _clean(s):
            s = str(s).strip()
            if not s: return None
            s = s.split(".")[0].replace(" ", "")
            return s.upper()

        markers = [_clean(m) for m in _flatten(markers)]
        markers = [m for m in markers if m]
        markers = list(OrderedDict.fromkeys(markers))
        if top_n and len(markers) > top_n:
            markers = markers[:top_n]
        if not markers:
            raise ValueError("展平清洗后 marker 为空")
    finally:
        if key_added in adata.uns:
            try: del adata.uns[key_added]
            except Exception: pass

    info = {
        "file_path": file_path,
        "celltype_col": celltype_col,
        "target_celltype_input": t_raw,
        "target_celltype_resolved": target_resolved,
        "n_cells_in_group": n_cells,
        "method": method,
        "preprocess": {"normalize_total": did_norm, "log1p": did_log1p},
        "top_n_requested": top_n,
        "top_n_returned": len(markers),
    }
    _log(f"[extract_gene_list] col='{celltype_col}', group='{target_resolved}', n_cells={n_cells}, top_n={top_n} -> {len(markers)}")
    return (markers, info) if return_info else markers


def load_pathway_genesets(db_name: str) -> Dict[str, List[str]]:
    import os
    from glob import glob
    import gseapy as gp

    name = db_name.lower()
    ver = getattr(settings, "MSIGDB_VERSION", "7.4")
    base = getattr(settings, "MSIGDB_PATH", ".")

    preferred = {
        "kegg":     f"c2.cp.kegg.v{ver}.symbols.gmt",
        "go":       f"c5.go.bp.v{ver}.symbols.gmt",
        "hallmark": f"h.all.v{ver}.symbols.gmt",
    }
    fallbacks = {
        "kegg":     ["c2.cp.kegg.symbols.gmt"],
        "go":       ["c5.go.bp.symbols.gmt"],
        "hallmark": ["h.all.symbols.gmt"],
    }
    fullfile = f"msigdb.v{ver}.symbols.gmt"

    def _read_gmt(path: str) -> Dict[str, List[str]]:
        return gp.parser.read_gmt(path)

    # 子集本地优先
    cand = os.path.join(base, preferred.get(name, ""))
    if os.path.isfile(cand):
        gs = _read_gmt(cand)
        print(f"加载 {db_name}（local subset）: {cand}  集合数={len(gs)}")
        return gs

    for patt in fallbacks.get(name, []):
        hits = sorted(glob(os.path.join(base, patt)))
        if hits:
            gs = _read_gmt(hits[-1])
            print(f"加载 {db_name}（local fallback）: {hits[-1]}  集合数={len(gs)}")
            return gs

    # 全集 + 筛选前缀
    full_path = os.path.join(base, fullfile)
    if os.path.isfile(full_path):
        all_sets = _read_gmt(full_path)
        if name == "kegg":
            gs = {k: v for k, v in all_sets.items() if str(k).startswith("KEGG_")}
            if gs:
                print(f"加载 {db_name}（full -> KEGG 前缀）: {full_path} 数={len(gs)}")
                return gs
        elif name == "hallmark":
            gs = {k: v for k, v in all_sets.items() if str(k).startswith("HALLMARK_")}
            if gs:
                print(f"加载 {db_name}（full -> HALLMARK 前缀）: {full_path} 数={len(gs)}")
                return gs
        elif name == "go":
            print("警告：从全集无法可靠区分 GO-BP，建议使用 c5.go.bp 子 GMT")

    # 在线 fallback（不带 outdir 参数）
    msig = gp.Msigdb()
    category = {"kegg": "c2.cp.kegg", "go": "c5.go.bp", "hallmark": "h.all"}[name]
    genesets = msig.get_gmt(category=category, dbver=ver)
    genesets = {str(k): [_norm_symbol(g) for g in v] for k, v in genesets.items()}
    print(f"加载 {db_name}（online）集合数={len(genesets)}")
    return genesets

    
def _is_log_transformed(X: Any) -> bool:
    """简单判断矩阵是否已经 log 转换"""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.size == 0:
        return False
    try:
        if np.min(X) < 0 or np.max(X) > 20:
            return False
        zero_ratio = np.mean(X <= 0)
        return zero_ratio <= 0.1
    except Exception:
        return False


def _to_iterable(value: Optional[Iterable[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)

def _norm_symbol(s: str) -> str:
    s = str(s).strip()
    if not s: return s
    s = s.split('.')[0]
    return s.upper()

def load_expression(
    file_path: str,
    celltype_col: Optional[str] = None,
    target_celltypes: Optional[Iterable[str]] = None,
    aggregation: str = "mean",
    min_cells: int = 10,
    normalize: bool = True,
) -> pd.DataFrame:
    """加载用于 ssGSEA 的表达矩阵

    参数:
        file_path: 输入的表达矩阵文件路径, 支持 csv/tsv/txt/h5ad
        celltype_col: (可选) 指定 .h5ad 文件中用于分组的细胞类型列
        target_celltypes: (可选) 仅保留的细胞类型集合
        aggregation: 在聚合到细胞类型水平时的聚合方法, 支持 "mean" 或 "median"
        min_cells: 聚合前细胞数量的最小阈值, 低于该阈值的细胞类型将被丢弃
        normalize: 是否自动对单细胞表达矩阵进行标准化 + log1p 处理

    返回:
        pd.DataFrame: 基因 × 样本的表达矩阵, 行为基因、列为细胞或细胞类型
    """

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)

        if normalize and not _is_log_transformed(adata.X):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        if celltype_col is not None:
            if celltype_col not in adata.obs.columns:
                raise ValueError(f"adata.obs 中缺少列 '{celltype_col}'")

            target_celltypes_list = _to_iterable(target_celltypes)
            obs_series = adata.obs[celltype_col].astype(str)

            if target_celltypes_list is not None:
                missing = set(target_celltypes_list) - set(obs_series.unique())
                if missing:
                    raise ValueError(f"目标细胞类型 {missing} 不存在于列 '{celltype_col}' 中")
                mask = obs_series.isin(target_celltypes_list)
                adata = adata[mask].copy()
                obs_series = adata.obs[celltype_col].astype(str)

            group_counts = obs_series.value_counts()
            valid_groups = group_counts[group_counts >= max(min_cells, 1)].index
            if len(valid_groups) == 0:
                raise ValueError("筛选后没有满足 min_cells 条件的细胞类型")

            mask = obs_series.isin(valid_groups)
            adata = adata[mask].copy()
            obs_series = adata.obs[celltype_col].astype(str)

            expr_df = adata.to_df()  # cells × genes

            if aggregation not in {"mean", "median"}:
                raise ValueError("aggregation 仅支持 'mean' 或 'median'")

            if aggregation == "mean":
                aggregated = expr_df.groupby(obs_series).mean()
            else:
                aggregated = expr_df.groupby(obs_series).median()

            df = aggregated.T  # genes × celltypes
        else:
            df = adata.to_df().T  # genes × cells
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("表达矩阵为空或格式不正确")

    # 如果行数少于列数, 可能是 cells × genes, 自动转置
    if df.shape[0] < df.shape[1]:
        df = df.T

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 表达矩阵：load_expression() 末尾
    df.index = df.index.map(_norm_symbol)
    df = df[~df.index.duplicated(keep="first")]

    return df.astype(float)
    
class AnalysisResult:
    def __init__(
        self,
        top_terms: pd.DataFrame,
        pvalues: Dict[str, float],
        scores: Optional[pd.DataFrame] = None,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        meta: Optional[Dict[str, Any]] = None,   
    ):
        self.top_terms = top_terms
        self.pvalues = pvalues
        self.scores = scores
        self.gene_sets = gene_sets
        self.meta = meta 