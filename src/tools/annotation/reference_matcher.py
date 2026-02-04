"""细胞类型参考数据库匹配工具

提供基于参考数据库的细胞类型注释功能：
- 计算基因集合与标记基因的重叠分数
- 查找最佳匹配的细胞类型
- 生成注释报告
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc

from src.tools.base import ToolResult, create_tool_result, fix_adata_var_index_name
from src.tools.utils.marker_loader import MarkerLoader
from src.tools.utils.validation import ToolValidator
from src.tools.utils.path_resolver import PathResolver

logger = logging.getLogger(__name__)

# 初始化工具
marker_loader = MarkerLoader()
validator = ToolValidator()
path_resolver = PathResolver()


def calculate_overlap_score(
    cluster_markers: Set[str],
    reference_markers: Set[str]
) -> float:
    """计算基因集合的重叠分数

    Args:
        cluster_markers: 聚类的标记基因集合
        reference_markers: 参考标记基因集合

    Returns:
        重叠分数 (0-1)
    """
    if not reference_markers:
        return 0.0

    overlap = cluster_markers & reference_markers
    return len(overlap) / len(reference_markers)


def calculate_expression_weight(
    adata: sc.AnnData,
    marker_genes: List[str]
) -> float:
    """计算标记基因在细胞群体中的平均表达量

    Args:
        adata: AnnData 对象（仅包含目标cluster）
        marker_genes: 标记基因列表

    Returns:
        平均表达量
    """
    # 筛选出存在于adata中的marker基因
    valid_genes = [gene for gene in marker_genes if gene in adata.var_names]

    if not valid_genes:
        return 0.0

    # 计算这些基因的平均表达量（稀疏矩阵安全处理）
    expr_matrix = adata[:, valid_genes].X
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()

    mean_expression = np.mean(expr_matrix)
    return float(mean_expression)


def find_best_match(
    cluster_markers: Set[str],
    marker_dict: Dict[str, List[str]],
    top_n: int = 5
) -> List[Dict[str, Union[str, float, List[str]]]]:
    """查找与标记基因集合最佳匹配的细胞类型

    Args:
        cluster_markers: 聚类的标记基因集合
        marker_dict: 细胞类型到标记基因的映射
        top_n: 返回前 N 个最佳匹配

    Returns:
        最佳匹配列表，格式: [{"cell_type": str, "score": float, "markers": List[str]}, ...]
    """
    results = []
    for cell_type, markers in marker_dict.items():
        marker_set = set(markers)
        score = calculate_overlap_score(cluster_markers, marker_set)
        if score > 0:
            results.append({
                "cell_type": cell_type,
                "score": score,
                "markers": markers,
                "overlap": list(cluster_markers & marker_set)
            })

    # 按分数降序排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def annotate_cluster_with_reference(
    adata: sc.AnnData,
    cluster_id: str,
    cluster_key: str,
    marker_dict: Dict[str, List[str]],
    top_n_markers: int = 10
) -> Dict[str, any]:
    """为单个聚类进行细胞类型注释

    Args:
        adata: AnnData 对象
        cluster_id: 聚类ID
        cluster_key: 聚类键名
        marker_dict: 细胞类型到标记基因的映射
        top_n_markers: 使用前 N 个标记基因进行匹配

    Returns:
        注释结果字典
    """
    # 获取该聚类的top标记基因
    try:
        # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
        genes_df = sc.get.rank_genes_groups_df(
            adata,
            group=str(cluster_id)
        )
        cluster_markers = set(genes_df['names'].head(top_n_markers).tolist())
    except Exception as e:
        logger.warning(f"获取cluster {cluster_id} 的标记基因失败: {e}")
        return {
            "cluster": cluster_id,
            "cell_type": f"Cluster_{cluster_id}",
            "score": 0.0,
            "error": str(e)
        }

    # 查找最佳匹配
    matches = find_best_match(cluster_markers, marker_dict, top_n=3)

    if matches and matches[0]["score"] > 0:
        best_match = matches[0]
        return {
            "cluster": cluster_id,
            "cell_type": best_match["cell_type"],
            "score": best_match["score"],
            "overlap_genes": best_match["overlap"],
            "top_candidates": [
                {"cell_type": m["cell_type"], "score": m["score"]}
                for m in matches
            ]
        }
    else:
        return {
            "cluster": cluster_id,
            "cell_type": f"Cluster_{cluster_id}",
            "score": 0.0,
            "top_candidates": []
        }


def generate_annotation_report(
    annotations: List[Dict[str, any]],
    output_path: Optional[str] = None
) -> str:
    """生成注释报告

    Args:
        annotations: 注释结果列表
        output_path: 输出文件路径

    Returns:
        报告内容
    """
    lines = [
        "# 细胞类型注释报告",
        "",
        f"**聚类数量**: {len(annotations)}",
        "",
        "## 注释结果",
        "",
        "| Cluster | Cell Type | Score | Overlap Genes |",
        "|---------|-----------|-------|---------------|",
    ]

    for anno in annotations:
        cluster = anno["cluster"]
        cell_type = anno["cell_type"]
        score = anno["score"]
        overlap = anno.get("overlap_genes", [])
        overlap_str = ", ".join(overlap[:5])
        if len(overlap) > 5:
            overlap_str += f" ... ({len(overlap)} total)"
        lines.append(f"| {cluster} | {cell_type} | {score:.3f} | {overlap_str} |")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


# 导出
__all__ = [
    "calculate_overlap_score",
    "calculate_expression_weight",
    "find_best_match",
    "annotate_cluster_with_reference",
    "generate_annotation_report",
]
