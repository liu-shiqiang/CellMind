from typing import Optional, Dict, Any, List

import pandas as pd
import scanpy as sc
import gseapy as gp
from config.setting import settings


def load_pathway_genesets(db_name: str) -> Dict[str, List[str]]:
    """加载通路基因集"""

    msig = gp.Msigdb()
    if db_name.lower() == "kegg":
        return msig.get_gmt(category='h.all',dbver=settings.MSIGDB_VERSION)
    elif db_name.lower() == "go":
        return msig.get_gmt(category='c5.go',dbver=settings.MSIGDB_VERSION)
    elif db_name.lower() == "msigdb":
        return msig.get_gmt(category='msigdb',dbver=settings.MSIGDB_VERSION)
    else:
        raise ValueError(f"不支持的基因集数据库: {db_name}")
    
def load_expression(file_path: str) -> pd.DataFrame:
    """加载表达矩阵,支持CSV、TSV和H5AD格式"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, index_col=0)
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        return pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)
        return adata.to_df()
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
class AnalysisResult:
    def __init__(self, top_terms: pd.DataFrame, pvalues: Dict[str, float], 
                 scores: Optional[pd.DataFrame] = None, 
                 gene_sets: Optional[Dict[str, List[str]]] = None):
        self.top_terms = top_terms  # 富集结果表格
        self.pvalues = pvalues      # 通路p值字典
        self.scores = scores        # 通路得分矩阵(仅适用于某些方法)
        self.gene_sets = gene_sets  # 基因集定义(仅适用于某些方法)