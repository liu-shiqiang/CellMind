"""基于标记基因的细胞类型注释工具

提供多种注释方法：
1. 简单标记基因匹配 - 使用内置的常见标记基因
2. CIMA 参考文件匹配 - 使用 CIMA 数据库
3. 血液细胞标记匹配 - 使用血液细胞参考数据
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import scanpy as sc
from langchain_core.tools import tool

from src.tools.base import ToolResult, create_tool_result, detect_cluster_key
from src.tools.utils.marker_loader import get_marker_loader, load_common_markers, load_cima_markers
from src.tools.utils.path_resolver import PathResolver
from src.tools.utils.validation import ToolValidator

logger = logging.getLogger(__name__)

# 初始化工具
marker_loader = get_marker_loader()
path_resolver = PathResolver()
validator = ToolValidator()


def _resolve_input_path(file_path: str) -> Path:
    """解析输入文件路径"""
    return path_resolver.resolve_input_path(file_path, None)


def _resolve_artifact_dirs(input_path: Path):
    """解析输出目录"""
    return path_resolver.resolve_all_output_dirs(input_path)


@tool("annotate_with_simple_markers", return_direct=False)
def annotate_with_simple_markers(
    file_path: str,
    cluster_key: Optional[str] = None,
    species: str = "human",
    save_result: bool = True,
) -> str:
    """使用简单标记基因进行细胞类型注释

    使用内置的常见细胞类型标记基因进行快速注释。
    适用于初步分析和常见细胞类型（T细胞、B细胞、NK细胞等）。

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        species: 物种类型 ("human" 或 "mouse")
        save_result: 是否保存结果

    Returns:
        注释结果摘要

    Example:
        >>> annotate_with_simple_markers("data.h5ad", species="human")
        '{"annotations": {"0": "T cells", ...}, "n_annotated": 15}'
    """
    try:
        logger.info(f"开始简单标记基因注释: {file_path}")

        # 验证参数
        species = validator.validate_choices(species, ['human', 'mouse'], 'species')

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 加载常见标记基因
        marker_dict = load_common_markers(species)

        if not marker_dict:
            return create_tool_result(
                status="error",
                message=f"未找到 {species} 物种的标记基因数据",
                error="标记基因数据为空"
            )

        # 获取聚类的标记基因
        if "rank_genes_groups" not in adata.uns:
            # 如果没有标记基因，先计算
            sc.tl.rank_genes_groups(
                adata,
                groupby=cluster_key,
                method='wilcoxon',
                n_genes=25
            )

        # 为每个聚类进行注释
        clusters = adata.obs[cluster_key].cat.categories.tolist()
        annotation_map = {}
        annotation_details = []

        for cluster in clusters:
            try:
                # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                genes_df = sc.get.rank_genes_groups_df(
                    adata,
                    group=str(cluster)
                )
                cluster_markers = set(genes_df['names'].head(10).tolist())

                # 查找最佳匹配
                best_match = None
                best_score = 0
                best_overlap = set()

                for cell_type, markers in marker_dict.items():
                    marker_set = set(markers)
                    overlap = cluster_markers & marker_set
                    score = len(overlap) / len(marker_set) if marker_set else 0

                    if score > best_score:
                        best_score = score
                        best_match = cell_type
                        best_overlap = overlap

                if best_score >= 0.1:  # 至少10%的标记基因匹配
                    annotation_map[cluster] = best_match
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": best_match,
                        "score": best_score,
                        "overlap_genes": list(best_overlap),
                    })
                else:
                    annotation_map[cluster] = f"Cluster_{cluster}"
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": f"Cluster_{cluster}",
                        "score": best_score,
                        "overlap_genes": [],
                    })
            except Exception as e:
                logger.warning(f"注释cluster {cluster} 失败: {e}")
                annotation_map[cluster] = f"Cluster_{cluster}"

        # 应用注释
        cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
        if 'cell_type' in adata.obs.columns:
            del adata.obs['cell_type']
        adata.obs['cell_type'] = cluster_values.astype(str)

        if 'pred_celltype' in adata.obs.columns:
            del adata.obs['pred_celltype']
        adata.obs['pred_celltype'] = adata.obs['cell_type'].astype(str)

        # 统计
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()

        result = ToolResult(
            status="success",
            message=f"注释完成，{len(annotation_map)}个clusters",
            data={
                "cluster_key": cluster_key,
                "species": species,
                "n_annotated": len(annotation_map),
                "annotations": annotation_map,
                "cell_type_counts": cell_type_counts,
                "annotation_details": annotation_details,
                "method": "simple_markers",
                "available_cell_types": list(marker_dict.keys()),
            }
        )

        # 保存结果
        if save_result:
            from src.tools.base import _save_result
            result_path = _save_result(adata, "annotated_simple", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

        logger.info(f"简单标记基因注释完成: {len(annotation_map)} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"简单标记基因注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"注释失败: {str(e)}",
            error=str(e)
        )


@tool("annotate_with_cima_markers", return_direct=False)
def annotate_with_cima_markers(
    file_path: str,
    cluster_key: Optional[str] = None,
    reference_level: int = 3,
    min_overlap: int = 1,
    save_result: bool = True,
) -> str:
    """使用 CIMA 参考数据库进行细胞类型注释

    CIMA (Cell Identity Marker Atlas) 是一个高质量的细胞标记数据库。
    支持多层级注释 (L3/L4)，适用于精确的细胞类型鉴定。

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        reference_level: 参考层级 (3 或 4)
        min_overlap: 最小重叠基因数阈值
        save_result: 是否保存结果

    Returns:
        注释结果摘要

    Example:
        >>> annotate_with_cima_markers("data.h5ad", reference_level=3)
        '{"annotations": {"0": "CD8_Tem", ...}, "n_annotated": 15}'
    """
    try:
        logger.info(f"开始 CIMA 标记基因注释: {file_path}")

        # 验证参数
        reference_level = int(validator.validate_choices(reference_level, [3, 4], 'reference_level'))
        min_overlap = int(validator.validate_positive_number(min_overlap, 'min_overlap', 0, 100))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 加载 CIMA 标记基因
        cima_markers = load_cima_markers(reference_level)

        if not cima_markers:
            return create_tool_result(
                status="error",
                message=f"CIMA L{reference_level} 标记基因数据未找到",
                error="标记基因数据为空"
            )

        # 转换为简单格式
        marker_dict = {
            ct: data["markers"]
            for ct, data in cima_markers.items()
            if data.get("markers")
        }

        logger.info(f"加载了 {len(marker_dict)} 种 CIMA L{reference_level} 细胞类型")

        # 获取聚类的标记基因
        if "rank_genes_groups" not in adata.uns:
            sc.tl.rank_genes_groups(
                adata,
                groupby=cluster_key,
                method='wilcoxon',
                n_genes=100
            )

        # 为每个聚类进行注释
        clusters = adata.obs[cluster_key].cat.categories.tolist()
        annotation_map = {}
        annotation_details = []

        for cluster in clusters:
            try:
                # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                genes_df = sc.get.rank_genes_groups_df(
                    adata,
                    group=str(cluster)
                )
                cluster_markers = set(genes_df['names'].head(50).tolist())

                # 查找最佳匹配
                best_match = None
                best_score = 0
                best_overlap = set()
                candidates = []

                for cell_type, markers in marker_dict.items():
                    marker_set = set(markers)
                    overlap = cluster_markers & marker_set
                    score = len(overlap) / len(marker_set) if marker_set else 0

                    if len(overlap) >= min_overlap:
                        candidates.append({
                            "cell_type": cell_type,
                            "score": score,
                            "overlap_count": len(overlap),
                            "overlap_genes": list(overlap)
                        })

                    if score > best_score and len(overlap) >= min_overlap:
                        best_score = score
                        best_match = cell_type
                        best_overlap = overlap

                # 按分数排序候选
                candidates.sort(key=lambda x: x["score"], reverse=True)

                if best_match:
                    annotation_map[cluster] = best_match
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": best_match,
                        "score": best_score,
                        "overlap_genes": list(best_overlap),
                        "n_candidates": len(candidates),
                        "top_candidates": candidates[:3],
                    })
                else:
                    annotation_map[cluster] = f"Cluster_{cluster}"
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": f"Cluster_{cluster}",
                        "score": 0.0,
                        "overlap_genes": [],
                        "n_candidates": 0,
                        "top_candidates": [],
                    })
            except Exception as e:
                logger.warning(f"注释cluster {cluster} 失败: {e}")
                annotation_map[cluster] = f"Cluster_{cluster}"

        # 应用注释
        cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
        if 'cell_type' in adata.obs.columns:
            del adata.obs['cell_type']
        adata.obs['cell_type'] = cluster_values.astype(str)

        if 'pred_celltype' in adata.obs.columns:
            del adata.obs['pred_celltype']
        adata.obs['pred_celltype'] = adata.obs['cell_type'].astype(str)

        # 统计
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()

        result = ToolResult(
            status="success",
            message=f"CIMA L{reference_level} 注释完成，{len(annotation_map)}个clusters",
            data={
                "cluster_key": cluster_key,
                "reference_level": reference_level,
                "min_overlap": min_overlap,
                "n_annotated": len(annotation_map),
                "annotations": annotation_map,
                "cell_type_counts": cell_type_counts,
                "annotation_details": annotation_details,
                "method": f"cima_l{reference_level}",
                "available_cell_types": list(marker_dict.keys()),
            }
        )

        # 保存结果
        if save_result:
            from src.tools.base import _save_result
            result_path = _save_result(adata, f"annotated_cima_l{reference_level}", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 保存注释详情
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_path = tables_dir / f"annotation_details_cima_l{reference_level}_{timestamp}.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(annotation_details, f, ensure_ascii=False, indent=2)
            result.artifacts["detail_path"] = str(detail_path)

        logger.info(f"CIMA 标记基因注释完成: {len(annotation_map)} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"CIMA 标记基因注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"注释失败: {str(e)}",
            error=str(e)
        )


@tool("annotate_with_blood_markers", return_direct=False)
def annotate_with_blood_markers(
    file_path: str,
    cluster_key: Optional[str] = None,
    min_overlap: int = 1,
    save_result: bool = True,
) -> str:
    """使用血液细胞标记进行细胞类型注释

    专门用于血液/免疫细胞的注释，使用血液细胞参考数据库。
    适用于外周血单核细胞 (PBMC) 等样本。

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        min_overlap: 最小重叠基因数阈值
        save_result: 是否保存结果

    Returns:
        注释结果摘要

    Example:
        >>> annotate_with_blood_markers("data.h5ad")
        '{"annotations": {"0": "CD8_Tem", ...}, "n_annotated": 15}'
    """
    try:
        logger.info(f"开始血液细胞标记注释: {file_path}")

        # 验证参数
        min_overlap = int(validator.validate_positive_number(min_overlap, 'min_overlap', 0, 100))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 加载血液细胞标记
        blood_df = marker_loader.load_blood_markers()

        if blood_df.empty:
            return create_tool_result(
                status="error",
                message="血液细胞标记基因数据未找到",
                error="标记基因数据为空"
            )

        # 解析血液细胞标记
        marker_dict = {}
        for _, row in blood_df.iterrows():
            cell_type = row.iloc[0]
            markers_str = row.get("markergene", "")
            markers = [m.strip() for m in str(markers_str).split(",") if m.strip()]
            if markers:
                marker_dict[cell_type] = markers

        logger.info(f"加载了 {len(marker_dict)} 种血液细胞类型")

        # 获取聚类的标记基因
        if "rank_genes_groups" not in adata.uns:
            sc.tl.rank_genes_groups(
                adata,
                groupby=cluster_key,
                method='wilcoxon',
                n_genes=100
            )

        # 为每个聚类进行注释
        clusters = adata.obs[cluster_key].cat.categories.tolist()
        annotation_map = {}
        annotation_details = []

        for cluster in clusters:
            try:
                # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                genes_df = sc.get.rank_genes_groups_df(
                    adata,
                    group=str(cluster)
                )
                cluster_markers = set(genes_df['names'].head(50).tolist())

                # 查找最佳匹配
                best_match = None
                best_score = 0
                best_overlap = set()
                candidates = []

                for cell_type, markers in marker_dict.items():
                    marker_set = set(markers)
                    overlap = cluster_markers & marker_set
                    score = len(overlap) / len(marker_set) if marker_set else 0

                    if len(overlap) >= min_overlap:
                        candidates.append({
                            "cell_type": cell_type,
                            "score": score,
                            "overlap_count": len(overlap),
                            "overlap_genes": list(overlap)
                        })

                    if score > best_score and len(overlap) >= min_overlap:
                        best_score = score
                        best_match = cell_type
                        best_overlap = overlap

                # 按分数排序候选
                candidates.sort(key=lambda x: x["score"], reverse=True)

                if best_match:
                    annotation_map[cluster] = best_match
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": best_match,
                        "score": best_score,
                        "overlap_genes": list(best_overlap),
                        "n_candidates": len(candidates),
                        "top_candidates": candidates[:3],
                    })
                else:
                    annotation_map[cluster] = f"Cluster_{cluster}"
                    annotation_details.append({
                        "cluster": cluster,
                        "cell_type": f"Cluster_{cluster}",
                        "score": 0.0,
                        "overlap_genes": [],
                        "n_candidates": 0,
                        "top_candidates": [],
                    })
            except Exception as e:
                logger.warning(f"注释cluster {cluster} 失败: {e}")
                annotation_map[cluster] = f"Cluster_{cluster}"

        # 应用注释
        cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
        if 'cell_type' in adata.obs.columns:
            del adata.obs['cell_type']
        adata.obs['cell_type'] = cluster_values.astype(str)

        if 'pred_celltype' in adata.obs.columns:
            del adata.obs['pred_celltype']
        adata.obs['pred_celltype'] = adata.obs['cell_type'].astype(str)

        # 统计
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()

        result = ToolResult(
            status="success",
            message=f"血液细胞标记注释完成，{len(annotation_map)}个clusters",
            data={
                "cluster_key": cluster_key,
                "min_overlap": min_overlap,
                "n_annotated": len(annotation_map),
                "annotations": annotation_map,
                "cell_type_counts": cell_type_counts,
                "annotation_details": annotation_details,
                "method": "blood_markers",
                "available_cell_types": list(marker_dict.keys()),
            }
        )

        # 保存结果
        if save_result:
            from src.tools.base import _save_result
            result_path = _save_result(adata, "annotated_blood", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 保存注释详情
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_path = tables_dir / f"annotation_details_blood_{timestamp}.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(annotation_details, f, ensure_ascii=False, indent=2)
            result.artifacts["detail_path"] = str(detail_path)

        logger.info(f"血液细胞标记注释完成: {len(annotation_map)} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"血液细胞标记注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"注释失败: {str(e)}",
            error=str(e)
        )


# 导出所有工具
__all__ = [
    "annotate_with_simple_markers",
    "annotate_with_cima_markers",
    "annotate_with_blood_markers",
]
