from typing import Optional, Dict, Any, List

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
    target_celltype: str = "required",
    top_n: int = 50,
    method: str = "wilcoxon"
) -> List[str]:
    """
    从单细胞 .h5ad 文件中提取指定细胞类型的 Top N Marker 基因，用于 ORA 分析。
    
    参数:
        file_path (str): 输入的 .h5ad 文件路径
        celltype_col (str): adata.obs 中的细胞类型注释列名（默认 "pred_celltype"）
        target_celltype (str): 要提取 marker 的目标细胞类型（必须存在）
        top_n (int): 返回前 N 个差异最显著的基因（默认 50）
        method (str): 差异分析方法（如 "wilcoxon", "t-test"；默认 "wilcoxon"）
    
    返回:
        List[str]: 提取的 marker 基因列表，如 ["CD3D", "CD8A", "GZMB", ...]
    
    异常:
        ValueError: 当输入无效（如列不存在、细胞类型不匹配等）时抛出
    """
    if target_celltype == "required":
        raise ValueError("必须指定 target_celltype 参数")

    # 1. 读取 adata
    adata = ad.read_h5ad(file_path)

    # 2. 验证注释列和细胞类型
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"adata.obs 中缺少列 '{celltype_col}'")
    
    categories = adata.obs[celltype_col].cat.categories if hasattr(adata.obs[celltype_col], 'cat') else adata.obs[celltype_col].unique()
    if target_celltype not in categories:
        raise ValueError(f"目标细胞类型 '{target_celltype}' 不在 '{celltype_col}' 列中。可用类型: {list(categories)}")

    # 3. 自动判断是否需要预处理
    def _is_log_transformed(X):
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

    if not _is_log_transformed(adata.X):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 4. 执行差异分析
    key_added = "_ora_temp_rank_genes"
    sc.tl.rank_genes_groups(
        adata,
        groupby=celltype_col,
        method=method,
        key_added=key_added
    )

    # 5. 提取结果
    try:
        names = adata.uns[key_added]["names"]
        markers = names[target_celltype][:top_n].tolist()
    except Exception as e:
        raise ValueError(f"提取 marker 基因失败: {e}")
    finally:
        # 清理临时结果，避免污染 adata
        if key_added in adata.uns:
            del adata.uns[key_added]

    return markers


def load_pathway_genesets(db_name: str) -> Dict[str, List[str]]:
    """加载通路基因集"""

    mapping = {'kegg': 'c2.cp.kegg', 'go': 'c5.all', 'hallmark': 'h.all'}
    
    if db_name.lower() not in mapping:
        raise ValueError(f"只支持: {list(mapping.keys())}")
    
    msig = gp.Msigdb()
    genesets = msig.get_gmt(category=mapping[db_name.lower()], dbver=settings.MSIGDB_VERSION)
    print(f"加载 {db_name}: {len(genesets)} 个基因集")
    return genesets
    
def load_expression(file_path: str) -> pd.DataFrame:
    """加载表达矩阵,支持CSV、TSV和H5AD格式
        ssGSEA 要求: genes × cells
        增加检查是否转置
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)
        df = adata.to_df()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    if df.shape[0] < df.shape[1]:
            # 通常细胞数远大于基因数，如果行数少于列数，可能是 cells × genes
        df = df.T

    return df
    
class AnalysisResult:
    def __init__(self, top_terms: pd.DataFrame, pvalues: Dict[str, float], 
                 scores: Optional[pd.DataFrame] = None, 
                 gene_sets: Optional[Dict[str, List[str]]] = None):
        self.top_terms = top_terms  # 富集结果表格
        self.pvalues = pvalues      # 通路p值字典
        self.scores = scores        # 通路得分矩阵(仅适用于某些方法)
        self.gene_sets = gene_sets  # 基因集定义(仅适用于某些方法)