"""单细胞核心分析工具库

提供基础的单细胞数据分析功能：
- 数据加载
- 数据质量控制 (QC)
- 降维和聚类
- 标记基因识别
- 细胞类型注释
- 差异表达分析
- 分析报告生成
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.cluster import KMeans
from langchain_core.tools import tool

# 设置 matplotlib 后端为非交互式
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.web.config import settings
from src.tools.base import (
    ToolResult,
    WORK_DIR,
    REFERENCE_DIR,
    OUTPUT_DIR,
    fix_adata_var_index_name,
    detect_cluster_key,
    create_tool_result,
    _save_result,
)
from src.tools.utils.path_resolver import PathResolver
from src.tools.utils.validation import ToolValidator

logger = logging.getLogger(__name__)

# 初始化工具
path_resolver = PathResolver()
validator = ToolValidator()

# 设置 matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _generate_plot_metadata(
    plot_name: str,
    plot_path: Path,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """生成图表元数据

    Args:
        plot_name: 图表名称
        plot_path: 图表文件路径
        run_id: 运行ID

    Returns:
        图表元数据字典
    """
    relative_path = None
    if run_id:
        relative_path = f"/api/artifacts/{run_id}/plots/{plot_path.name}"

    return {
        "name": plot_name,
        "title": _get_plot_title(plot_name),
        "path": relative_path or str(plot_path),
        "local_path": str(plot_path),
        "interpretation": _get_plot_interpretation(plot_name),
    }


def _get_plot_title(plot_name: str) -> str:
    """获取图表标题"""
    titles = {
        "qc_violin": "质控指标分布 (QC Violin Plot)",
        "umap_cluster": "UMAP 聚类可视化 (UMAP Cluster Plot)",
        "umap_annotated": "UMAP 细胞类型注释 (UMAP Annotated Plot)",
        "marker_heatmap": "标记基因热图 (Marker Gene Heatmap)",
        "pca_variance": "PCA 方差解释 (PCA Variance Plot)",
        "volcano": "差异分析火山图 (Volcano Plot)",
    }
    return titles.get(plot_name, plot_name.replace("_", " ").title())


def _get_plot_interpretation(plot_name: str) -> Dict[str, Any]:
    """获取图表解读信息"""
    interpretations = {
        "qc_violin": {
            "title": "质控指标分布",
            "description": "展示每个细胞的基因数、UMI数和线粒体基因比例分布，用于识别低质量细胞。",
            "what_to_look": [
                "基因数和UMI数分布：大多数细胞应集中在相似范围",
                "线粒体基因比例：高比例（>20%）可能表示细胞损伤",
                "离群细胞：考虑过滤极端值"
            ]
        },
        "umap_cluster": {
            "title": "UMAP 聚类可视化",
            "description": "展示细胞在二维UMAP空间中的分布，相似细胞的聚集表示转录组相似性。",
            "what_to_look": [
                "聚类分离度：不同cluster应有明显分离",
                "细胞类型分布：观察是否有明显的细胞类型分层",
                "批次效应：如果样本按batch聚集可能存在批次效应"
            ]
        },
        "umap_annotated": {
            "title": "UMAP 细胞类型注释",
            "description": "在UMAP图上标注细胞类型，直观展示不同细胞类型的空间分布。",
            "what_to_look": [
                "注释连续性：相同细胞类型的细胞应聚集",
                "注释准确性：检查是否有明显的错误注释",
                "稀有细胞类型：小群体可能是稀有细胞类型"
            ]
        },
        "marker_heatmap": {
            "title": "标记基因热图",
            "description": "展示各cluster的top标记基因表达模式，红色表示高表达，蓝色表示低表达。",
            "what_to_look": [
                "cluster特异性标记：每个cluster应有独特的标记基因",
                "表达模式：相似表达模式的cluster可能是同一细胞类型",
                "标记基因强度：高logFC表示强marker"
            ]
        },
        "pca_variance": {
            "title": "PCA 方差解释",
            "description": "展示各主成分解释的方差比例，帮助确定使用多少个PC进行下游分析。",
            "what_to_look": [
                "肘部位置：方差解释率明显下降的点",
                "累积方差：前30-50个PC通常解释大部分方差",
                "选择PC数：建议选择累积方差>80%的PC数"
            ]
        },
        "volcano": {
            "title": "差异分析火山图",
            "description": "展示基因表达的log2 fold change与统计显著性关系。",
            "what_to_look": [
                "显著上调基因：右上角（高logFC，低p值）",
                "显著下调基因：左上角（低logFC，低p值）",
                "关键标记基因：已知细胞类型标记的位置"
            ]
        },
    }
    return interpretations.get(plot_name, {
        "title": _get_plot_title(plot_name),
        "description": "分析结果可视化",
        "what_to_look": []
    })


def _save_qc_plot(adata, plots_dir: Path, timestamp: str) -> Optional[Dict[str, Any]]:
    """生成QC小提琴图"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 检查是否有QC指标
        qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        available_metrics = [m for m in qc_metrics if m in adata.obs.columns]

        if not available_metrics:
            logger.warning("没有可用的QC指标，跳过绘图")
            return None

        for i, metric in enumerate(available_metrics[:3]):
            if i >= 3:
                break
            ax = axes[i] if len(available_metrics) > 1 else axes
            sc.pl.violin(adata, keys=metric, ax=ax, show=False)
            ax.set_title(metric.replace('_', ' ').title())

        plt.tight_layout()

        plot_name = f"qc_violin_{timestamp}.png"
        plot_path = plots_dir / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return _generate_plot_metadata("qc_violin", plot_path)

    except Exception as e:
        logger.warning(f"生成QC图失败: {e}")
        return None


def _save_umap_plot(adata, plots_dir: Path, timestamp: str, color_key: str = 'leiden',
                    run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """生成UMAP聚类图"""
    try:
        if 'X_umap' not in adata.obsm:
            logger.warning("没有UMAP坐标，跳过绘图")
            return None

        if color_key not in adata.obs.columns:
            logger.warning(f"没有 {color_key} 列，跳过绘图")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color=color_key, ax=ax, show=False,
                   frameon=False, legend_loc='on data')
        ax.set_title(f"UMAP - {color_key}")

        plot_name = f"umap_{color_key}_{timestamp}.png"
        plot_path = plots_dir / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        plot_type = "umap_annotated" if color_key in ['cell_type', 'pred_celltype', 'l2'] else "umap_cluster"
        return _generate_plot_metadata(plot_type, plot_path, run_id)

    except Exception as e:
        logger.warning(f"生成UMAP图失败: {e}")
        return None


def _save_marker_heatmap(adata, plots_dir: Path, timestamp: str, cluster_key: str = 'leiden',
                         n_genes: int = 10, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """生成标记基因热图"""
    try:
        if "rank_genes_groups" not in adata.uns:
            logger.warning("没有标记基因分析结果，跳过绘图")
            return None

        # 获取cluster数量
        clusters = adata.obs[cluster_key].cat.categories.tolist() if cluster_key in adata.obs else []

        if not clusters:
            logger.warning("没有聚类信息，跳过绘图")
            return None

        # 使用scanpy的heatmap函数
        fig = sc.pl.rank_genes_groups_heatmap(
            adata,
            n_genes=n_genes,
            groupby=cluster_key,
            show=False,
            cmap='RdBu_r'
        )

        plot_name = f"marker_heatmap_{timestamp}.png"
        plot_path = plots_dir / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return _generate_plot_metadata("marker_heatmap", plot_path, run_id)

    except Exception as e:
        logger.warning(f"生成标记基因热图失败: {e}")
        return None


def _save_pca_variance_plot(adata, plots_dir: Path, timestamp: str,
                            run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """生成PCA方差解释图"""
    try:
        if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
            logger.warning("没有PCA结果，跳过绘图")
            return None

        variance_ratio = adata.uns["pca"]["variance_ratio"]
        cumulative = np.cumsum(variance_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 方差条形图
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio * 100)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Variance Explained by PC')

        # 累积方差折线图
        ax2.plot(range(1, len(cumulative) + 1), cumulative * 100, 'b-')
        ax2.axhline(y=80, color='r', linestyle='--', label='80% threshold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()

        plt.tight_layout()

        plot_name = f"pca_variance_{timestamp}.png"
        plot_path = plots_dir / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return _generate_plot_metadata("pca_variance", plot_path, run_id)

    except Exception as e:
        logger.warning(f"生成PCA方差图失败: {e}")
        return None


def _resolve_artifact_dirs(input_path: Path) -> tuple[Path, Path, Path]:
    """根据输入路径解析产物目录（Job模式优先）"""
    return path_resolver.resolve_all_output_dirs(input_path)


def _resolve_input_path(file_path: str) -> Path:
    """解析输入文件路径"""
    return path_resolver.resolve_input_path(file_path, None)


@tool("load_h5ad_data", return_direct=False)
def load_h5ad_data(
    file_path: Optional[str] = None,
    filepath: Optional[str] = None,
    cache: bool = True,
) -> str:
    """加载 .h5ad 格式的单细胞数据文件

    Args:
        file_path: .h5ad 文件路径
        filepath: 文件路径的别名（兼容性参数）
        cache: 是否缓存数据

    Returns:
        数据加载结果摘要，包含细胞数、基因数等信息

    Example:
        >>> load_h5ad_data("data.h5ad")
        '{"n_cells": 5000, "n_genes": 20000, "status": "loaded"}'
    """
    try:
        # 处理参数别名
        if file_path is None and filepath is not None:
            file_path = filepath
        elif file_path is None and filepath is None:
            return create_tool_result(
                status="error",
                message="必须提供 'file_path' 或 'filepath' 参数",
                error="缺少必需参数"
            )
        elif file_path is not None and filepath is not None and file_path != filepath:
            logger.warning(f"同时提供了 file_path 和 filepath，使用 file_path: {file_path}")

        logger.info(f"加载数据文件: {file_path}")

        # 解析文件路径
        path = _resolve_input_path(file_path)

        if not path.exists():
            return create_tool_result(
                status="error",
                message=f"文件不存在: {file_path}",
                error=f"文件未找到: {file_path}"
            )

        # 加载数据
        adata = sc.read_h5ad(path)

        # 基础信息
        n_obs, n_vars = adata.n_obs, adata.n_vars

        result = ToolResult(
            status="success",
            message=f"成功加载 {n_obs} 个细胞和 {n_vars} 个基因",
            data={
                "n_cells": n_obs,
                "n_genes": n_vars,
                "file_path": str(path),
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
                "obsm_keys": list(adata.obsm.keys()),
                "uns_keys": list(adata.uns.keys()),
            },
            artifacts={"result_path": str(path)}
        )

        # 检查是否有基本聚类信息
        has_clustering = any(col in adata.obs.columns for col in ['leiden', 'louvain', 'clusters', 'cluster'])
        result.data["has_clustering"] = has_clustering

        # 检查是否有UMAP/TSNE信息
        has_embedding = any(key in adata.obsm for key in ['X_umap', 'X_tsne', 'X_pca'])
        result.data["has_embedding"] = has_embedding

        logger.info(f"数据加载成功: {n_obs} cells x {n_vars} genes")

        return result.to_json()

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return create_tool_result(
            status="error",
            message=f"加载数据失败: {str(e)}",
            error=str(e)
        )


@tool("calculate_qc_metrics", return_direct=False)
def calculate_qc_metrics(
    file_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    mt_prefix: str = "MT-",
    save_result: bool = True,
) -> str:
    """计算单细胞数据的质控指标

    包括：
    - 每个细胞检测到的基因数 (n_genes_by_counts)
    - 每个细胞检测到的UMI数 (total_counts)
    - 线粒体基因比例 (pct_counts_mt)

    Args:
        file_path: .h5ad 文件路径
        min_genes: 最小基因数阈值
        min_cells: 最小细胞数阈值
        mt_prefix: 线粒体基因前缀
        save_result: 是否保存结果

    Returns:
        质控统计摘要

    Example:
        >>> calculate_qc_metrics("data.h5ad")
        '{"n_cells_before": 5000, "n_cells_after": 4800, ...}'
    """
    try:
        logger.info(f"开始质控分析: {file_path}")

        # 验证参数
        min_genes = validator.validate_positive_number(min_genes, "min_genes", 1)
        min_cells = validator.validate_positive_number(min_cells, "min_cells", 1)

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, _, plots_dir = _resolve_artifact_dirs(path)

        # 获取run_id用于API路径
        runs_root = Path(settings.RUNS_DIR).resolve()
        run_id = None
        try:
            if path.is_relative_to(runs_root):
                run_id = path.relative_to(runs_root).parts[0]
        except ValueError:
            pass

        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars

        # 修复 index name 与列名冲突问题
        fix_adata_var_index_name(adata)

        # 计算质控指标
        adata.var['mt'] = adata.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # 基本统计
        qc_stats = {
            "n_cells_before": n_cells_before,
            "n_genes_before": n_genes_before,
            "mean_genes_per_cell": float(adata.obs['n_genes_by_counts'].mean()),
            "median_genes_per_cell": float(adata.obs['n_genes_by_counts'].median()),
            "mean_counts_per_cell": float(adata.obs['total_counts'].mean()),
            "median_counts_per_cell": float(adata.obs['total_counts'].median()),
            "mean_mt_percent": float(adata.obs['pct_counts_mt'].mean()),
        }

        # 过滤建议和统计
        high_mt = (adata.obs['pct_counts_mt'] > 20).sum()
        low_genes = (adata.obs['n_genes_by_counts'] < min_genes).sum()
        high_genes = (adata.obs['n_genes_by_counts'] > 6000).sum()

        qc_stats["high_mt_cells"] = int(high_mt)
        qc_stats["low_gene_cells"] = int(low_genes)
        qc_stats["high_gene_cells"] = int(high_genes)

        # 执行过滤
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)

        n_cells_after = adata.n_obs
        n_genes_after = adata.n_vars

        qc_stats.update({
            "n_cells_after": n_cells_after,
            "n_genes_after": n_genes_after,
            "filtered_cells": n_cells_before - n_cells_after,
            "filtered_genes": n_genes_before - n_genes_after,
            "filter_rate": f"{(n_cells_before - n_cells_after) / n_cells_before * 100:.2f}%",
        })

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "qc", "quality_control", output_dir=data_dir)
            qc_stats["result_path"] = result_path

            # 生成QC图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_metadata = _save_qc_plot(adata, plots_dir, timestamp)
            if plot_metadata:
                qc_stats["plot"] = plot_metadata
                if run_id:
                    plot_metadata["run_id"] = run_id

        logger.info(f"质控完成: {n_cells_before} -> {n_cells_after} cells")

        return ToolResult(
            status="success",
            message=f"质控完成: {n_cells_before} -> {n_cells_after} 细胞",
            data=qc_stats,
            artifacts={"result_path": qc_stats.get("result_path")} if save_result else {}
        ).to_json()

    except Exception as e:
        logger.error(f"质控分析失败: {e}")
        return create_tool_result(
            status="error",
            message=f"质控分析失败: {str(e)}",
            error=str(e)
        )


@tool("normalize_and_hvg", return_direct=False)
def normalize_and_hvg(
    file_path: str,
    target_sum: int = 10000,
    n_top_genes: int = 2000,
    save_result: bool = True,
) -> str:
    """数据标准化和鉴定高变基因

    Steps:
    1. 对每个细胞的UMI计数进行标准化
    2. 对数转换
    3. 鉴定高变基因 (Highly Variable Genes)

    Args:
        file_path: .h5ad 文件路径
        target_sum: 标准化目标UMI数
        n_top_genes: 高变基因数量
        save_result: 是否保存结果

    Returns:
        标准化和HVG结果摘要

    Example:
        >>> normalize_and_hvg("data.h5ad", n_top_genes=2000)
        '{"n_hvg": 2000, "hvg_genes": ["Gene1", ...]}'
    """
    try:
        logger.info(f"开始标准化和HVG分析: {file_path}")

        # 验证参数
        target_sum = validator.validate_positive_number(target_sum, "target_sum", 100)
        n_top_genes = validator.validate_n_top_genes(n_top_genes)

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)

        # 标准化
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # 鉴定高变基因
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                flavor='seurat_v3',
                batch_key=None
            )
        except Exception as exc:
            if "scikit-misc" in str(exc) or "skmisc" in str(exc):
                logger.warning("缺少 scikit-misc，改用 seurat 风格 HVG")
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_top_genes,
                    flavor='seurat',
                    batch_key=None
                )
            else:
                raise

        n_hvg = adata.var['highly_variable'].sum()
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()

        result = ToolResult(
            status="success",
            message=f"鉴定到 {n_hvg} 个高变基因",
            data={
                "n_hvg": int(n_hvg),
                "target_sum": target_sum,
                "hvg_genes_sample": hvg_genes[:50],
                "mean_var": float(adata.var['highly_variable'].mean()),
            }
        )

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "hvg", "normalization", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

        logger.info(f"HVG分析完成: {n_hvg} highly variable genes")

        return result.to_json()

    except Exception as e:
        logger.error(f"标准化和HVG分析失败: {e}")
        return create_tool_result(
            status="error",
            message=f"标准化和HVG分析失败: {str(e)}",
            error=str(e)
        )


@tool("pca_reduction", return_direct=False)
def pca_reduction(
    file_path: str,
    n_comps: int = 50,
    save_result: bool = True,
) -> str:
    """主成分分析 (PCA) 降维

    Args:
        file_path: .h5ad 文件路径
        n_comps: 主成分数量
        save_result: 是否保存结果

    Returns:
        PCA结果摘要

    Example:
        >>> pca_reduction("data.h5ad", n_comps=50)
        '{"variance_ratio": [0.12, 0.08, ...], "cumulative_ratio": 0.85}'
    """
    try:
        logger.info(f"开始PCA降维: {file_path}")

        # 验证参数
        n_comps = int(validator.validate_positive_number(n_comps, "n_comps", 2, 200))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)

        # 使用高变基因进行PCA
        if 'highly_variable' in adata.var.columns:
            adata_hvg = adata[:, adata.var['highly_variable']]
        else:
            adata_hvg = adata

        # 运行PCA
        sc.tl.pca(adata_hvg, n_comps=n_comps, svd_solver='arpack')

        # 复制PCA结果到原始对象
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        adata.uns['pca'] = adata_hvg.uns['pca']

        # 处理 varm['PCs']
        if 'highly_variable' in adata.var.columns:
            full_pcs = np.full((adata.n_vars, n_comps), np.nan)
            hvg_mask = adata.var['highly_variable'].values
            full_pcs[hvg_mask, :] = adata_hvg.varm['PCs']
            adata.varm['PCs'] = pd.DataFrame(
                full_pcs,
                index=adata.var_names,
                columns=[f'PC{i+1}' for i in range(n_comps)]
            )

        # 提取方差解释比例
        variance_ratio = adata.uns['pca']['variance_ratio'].tolist()
        cumulative_ratio = float(sum(variance_ratio[:30]))

        result = ToolResult(
            status="success",
            message=f"PCA完成，前30个成分解释{cumulative_ratio*100:.1f}%方差",
            data={
                "n_comps": n_comps,
                "variance_ratio_sample": variance_ratio[:10],
                "cumulative_variance_30pc": cumulative_ratio,
                "pca_shape": adata.obsm['X_pca'].shape,
            }
        )

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "pca", "dimensionality_reduction", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

        logger.info(f"PCA完成: {n_comps} components, {cumulative_ratio*100:.1f}% variance")

        return result.to_json()

    except Exception as e:
        logger.error(f"PCA降维失败: {e}")
        return create_tool_result(
            status="error",
            message=f"PCA降维失败: {str(e)}",
            error=str(e)
        )


@tool("cluster_and_umap", return_direct=False)
def cluster_and_umap(
    file_path: str,
    resolution: float = 0.5,
    n_neighbors: int = 15,
    save_result: bool = True,
) -> str:
    """聚类分析和UMAP降维

    使用 Leiden 算法进行聚类，并计算UMAP嵌入

    Args:
        file_path: .h5ad 文件路径
        resolution: 聚类分辨率（控制聚类数量）
        n_neighbors: UMAP近邻数
        save_result: 是否保存结果

    Returns:
        聚类和UMAP结果摘要

    Example:
        >>> cluster_and_umap("data.h5ad", resolution=0.5)
        '{"n_clusters": 15, "cluster_sizes": {"0": 500, ...}}'
    """
    try:
        logger.info(f"开始聚类和UMAP分析: {file_path}")

        # 验证参数
        resolution = validator.validate_resolution(resolution)
        n_neighbors = validator.validate_n_neighbors(n_neighbors)

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, plots_dir = _resolve_artifact_dirs(path)

        # 获取run_id用于API路径
        runs_root = Path(settings.RUNS_DIR).resolve()
        run_id = None
        try:
            if path.is_relative_to(runs_root):
                run_id = path.relative_to(runs_root).parts[0]
        except ValueError:
            pass

        # 检查是否有PCA
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

        # 计算邻接图
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)

        # Leiden 聚类
        try:
            sc.tl.leiden(adata, resolution=resolution)
            leiden_key = 'leiden'
        except Exception as exc:
            if "igraph" in str(exc).lower() or "leidenalg" in str(exc).lower():
                logger.warning("缺少 igraph/leidenalg，回退到 KMeans 聚类")
                n_clusters = max(2, int(round(resolution * 10)))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                labels = kmeans.fit_predict(adata.obsm['X_pca'])
                # 保存到 'leiden' 列以保持 API 一致性
                adata.obs['leiden'] = pd.Categorical(labels.astype(str))
                leiden_key = 'leiden'
            else:
                raise

        # UMAP
        sc.tl.umap(adata)

        # 提取聚类统计（leiden_key 已在上面的 try/except 块中设置）
        if leiden_key in adata.obs.columns:
            cluster_counts = adata.obs[leiden_key].value_counts().to_dict()
            n_clusters = len(cluster_counts)
        else:
            cluster_counts = {}
            n_clusters = 0

        result = ToolResult(
            status="success",
            message=f"聚类完成，识别到{n_clusters}个cluster",
            data={
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_counts,
                "n_neighbors": n_neighbors,
                "resolution": resolution,
                "umap_shape": adata.obsm['X_umap'].shape,
                "cluster_key": leiden_key,
            }
        )

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "cluster_umap", "clustering", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 导出UMAP坐标
            try:
                umap_df = pd.DataFrame(
                    adata.obsm['X_umap'],
                    columns=['UMAP_1', 'UMAP_2'],
                )
                umap_df['cluster'] = adata.obs[leiden_key].astype(str).values
                if 'cell_type' in adata.obs.columns:
                    umap_df['cell_type'] = adata.obs['cell_type'].astype(str).values
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                umap_path = tables_dir / f"umap_coords_{timestamp}.csv"
                umap_df.to_csv(umap_path, index=False)
                result.artifacts["umap_coords_path"] = str(umap_path)

                # 生成UMAP聚类图
                plot_metadata = _save_umap_plot(adata, plots_dir, timestamp, leiden_key, run_id)
                if plot_metadata:
                    result.artifacts["umap_plot"] = plot_metadata
                    result.data["plot"] = plot_metadata
                    if run_id:
                        plot_metadata["run_id"] = run_id
            except Exception as exc:
                logger.warning("UMAP坐标导出失败: %s", exc)

        logger.info(f"聚类和UMAP完成: {n_clusters} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"聚类和UMAP分析失败: {e}")
        return create_tool_result(
            status="error",
            message=f"聚类和UMAP分析失败: {str(e)}",
            error=str(e)
        )


@tool("find_marker_genes", return_direct=False)
def find_marker_genes(
    file_path: str,
    cluster_key: Optional[str] = None,
    method: str = "wilcoxon",
    n_genes: int = 25,
    save_result: bool = True,
) -> str:
    """寻找每个cluster的标记基因

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        method: 差异分析方法 ('wilcoxon', 't-test', 'rank')
        n_genes: 每个cluster返回的标记基因数
        save_result: 是否保存结果

    Returns:
        标记基因结果摘要

    Example:
        >>> find_marker_genes("data.h5ad", n_genes=10)
        '{"cluster_0_markers": {"CD3D": {"pval": 1e-50, ...}}, ...}'
    """
    try:
        logger.info(f"开始标记基因分析: {file_path}")

        # 验证参数
        method = validator.validate_choices(method, ['wilcoxon', 't-test', 'rank'], 'method')
        n_genes = int(validator.validate_positive_number(n_genes, 'n_genes', 1, 500))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, plots_dir = _resolve_artifact_dirs(path)

        # 获取run_id用于API路径
        runs_root = Path(settings.RUNS_DIR).resolve()
        run_id = None
        try:
            if path.is_relative_to(runs_root):
                run_id = path.relative_to(runs_root).parts[0]
        except ValueError:
            pass

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 寻找标记基因
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            method=method,
            corr_method='bonferroni',
            n_genes=n_genes,
        )

        # 提取结果
        result = ToolResult(
            status="success",
            message=f"标记基因分析完成",
            data={
                "cluster_key": cluster_key,
                "method": method,
                "n_genes_per_group": n_genes,
                "clusters": {},
            }
        )

        # 获取每个cluster的top标记基因
        groups = adata.obs[cluster_key].cat.categories.tolist()
        for group in groups:
            try:
                # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                genes_df = sc.get.rank_genes_groups_df(adata, group=group)
                top_genes = {}
                for _, row in genes_df.head(n_genes).iterrows():
                    gene_name = row['names']
                    top_genes[gene_name] = {
                        "logfoldchanges": float(row['logfoldchanges']) if 'logfoldchanges' in row else None,
                        "pval": float(row['pvals_adj']) if 'pvals_adj' in row else None,
                        "scores": float(row['scores']) if 'scores' in row else None,
                    }

                result.data["clusters"][group] = {
                    "top_genes": top_genes,
                    "n_genes": len(top_genes),
                }
            except Exception as e:
                logger.warning(f"提取cluster {group} 的标记基因失败: {e}")
                result.data["clusters"][group] = {"error": str(e)}

        result.data["n_clusters"] = len(groups)
        result.message = f"标记基因分析完成，{len(groups)}个clusters"

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "markers", "marker_genes", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 额外保存标记基因CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = tables_dir / f"marker_genes_{timestamp}.csv"
            try:
                # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                all_markers = sc.get.rank_genes_groups_df(adata)
                # 保存每个cluster的前 n_genes 个标记基因
                if 'group' in all_markers.columns:
                    all_markers = all_markers.groupby('group').head(n_genes)
                all_markers.to_csv(csv_path, index=False)
                result.artifacts["csv_path"] = str(csv_path)
            except Exception as e:
                logger.warning(f"保存标记基因CSV失败: {e}")

            # 生成标记基因热图
            try:
                plot_metadata = _save_marker_heatmap(adata, plots_dir, timestamp, cluster_key, min(n_genes, 10), run_id)
                if plot_metadata:
                    result.artifacts["heatmap_plot"] = plot_metadata
                    result.data["heatmap_plot"] = plot_metadata
                    if run_id:
                        plot_metadata["run_id"] = run_id
            except Exception as e:
                logger.warning(f"生成标记基因热图失败: {e}")

        logger.info(f"标记基因分析完成: {len(groups)} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"标记基因分析失败: {e}")
        return create_tool_result(
            status="error",
            message=f"标记基因分析失败: {str(e)}",
            error=str(e)
        )


@tool("annotate_cells", return_direct=False)
def annotate_cells(
    file_path: str,
    cluster_key: Optional[str] = None,
    annotations: Optional[Dict[str, str]] = None,
    marker_based: bool = True,
    save_result: bool = True,
) -> str:
    """细胞类型注释

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名
        annotations: 手动注释映射 {cluster: cell_type}
        marker_based: 是否基于标记基因自动注释
        save_result: 是否保存结果

    Returns:
        注释结果摘要

    Example:
        >>> annotate_cells("data.h5ad", annotations={"0": "T cells", "1": "B cells"})
        '{"annotations": {"0": "T cells", ...}, "n_annotated": 15}'
    """
    try:
        logger.info(f"开始细胞注释: {file_path}")

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, _, plots_dir = _resolve_artifact_dirs(path)

        # 获取run_id用于API路径
        runs_root = Path(settings.RUNS_DIR).resolve()
        run_id = None
        try:
            if path.is_relative_to(runs_root):
                run_id = path.relative_to(runs_root).parts[0]
        except ValueError:
            pass

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 获取clusters
        clusters = adata.obs[cluster_key].cat.categories.tolist()

        result = ToolResult(
            status="success",
            message="开始细胞注释",
            data={
                "cluster_key": cluster_key,
                "annotations": {},
                "n_annotated": 0,
            }
        )

        # 常见标记基因（简化版）
        common_markers = {
            "T cells": ["CD3D", "CD3E", "CD8A", "CD4"],
            "B cells": ["CD79A", "CD79B", "MS4A1", "CD19"],
            "NK cells": ["NCAM1", "NKG7", "GNLY"],
            "Monocytes": ["CD14", "LYZ", "S100A8", "S100A9"],
            "Dendritic cells": ["FCER1A", "CST3"],
            "Megakaryocytes": ["PPBP", "PF4"],
            "Erythrocytes": ["HBB", "HBA1"],
            "Fibroblasts": ["COL1A1", "DCN"],
            "Endothelial": ["VWF", "PECAM1"],
        }

        # 如果提供了手动注释，使用手动注释
        if annotations:
            annotation_map = {}
            for cluster, cell_type in annotations.items():
                if str(cluster) in clusters:
                    annotation_map[str(cluster)] = cell_type

            cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
            if 'cell_type' in adata.obs.columns:
                del adata.obs['cell_type']
            adata.obs['cell_type'] = cluster_values.astype(str)
            result.data["annotations"] = annotation_map
            result.data["n_annotated"] = len(annotation_map)
            result.data["method"] = "manual"

        # 基于标记基因的自动注释
        elif marker_based and 'rank_genes_groups' in adata.uns:
            annotation_map = {}

            for cluster in clusters:
                try:
                    # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                    genes_df = sc.get.rank_genes_groups_df(adata, group=cluster)
                    top_genes = set(genes_df['names'].head(10).tolist())

                    best_match = None
                    best_score = 0

                    for cell_type, markers in common_markers.items():
                        score = len(top_genes & set(markers))
                        if score > best_score:
                            best_score = score
                            best_match = cell_type

                    if best_score >= 1:
                        annotation_map[cluster] = best_match
                    else:
                        annotation_map[cluster] = f"Cluster_{cluster}"

                except Exception as e:
                    annotation_map[cluster] = f"Cluster_{cluster}"

            # 修复 Categorical 类型问题：先转换为字符串再映射
            cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
            if 'cell_type' in adata.obs.columns:
                del adata.obs['cell_type']
            adata.obs['cell_type'] = cluster_values.astype(str)
            result.data["annotations"] = annotation_map
            result.data["n_annotated"] = len(annotation_map)
            result.data["method"] = "marker_based"

        else:
            # 默认使用cluster名称
            if 'cell_type' in adata.obs.columns:
                del adata.obs['cell_type']
            adata.obs['cell_type'] = adata.obs[cluster_key].astype(str)
            for cluster in clusters:
                result.data["annotations"][cluster] = f"Cluster_{cluster}"
            result.data["n_annotated"] = len(clusters)
            result.data["method"] = "cluster_name"

        # 添加 pred_celltype 列
        if "pred_celltype" not in adata.obs.columns:
            if 'pred_celltype' in adata.obs.columns:
                del adata.obs['pred_celltype']
            adata.obs["pred_celltype"] = adata.obs["cell_type"].astype(str)

        # 统计每种细胞类型的数量
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()
        result.data["cell_type_counts"] = cell_type_counts
        result.message = f"注释完成，{len(result.data['annotations'])}个clusters"

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "annotated", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 生成注释后的UMAP图
            if 'X_umap' in adata.obsm:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_metadata = _save_umap_plot(adata, plots_dir, timestamp, 'cell_type', run_id)
                    if plot_metadata:
                        result.artifacts["annotated_umap_plot"] = plot_metadata
                        result.data["annotated_umap_plot"] = plot_metadata
                        if run_id:
                            plot_metadata["run_id"] = run_id
                except Exception as e:
                    logger.warning(f"生成注释UMAP图失败: {e}")

        logger.info(f"细胞注释完成: {result.data['n_annotated']} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"细胞注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"细胞注释失败: {str(e)}",
            error=str(e)
        )


@tool("differential_expression", return_direct=False)
def differential_expression(
    file_path: str,
    group1: Optional[str] = None,
    group2: Optional[str] = None,
    groupby: str = "cell_type",
    method: str = "wilcoxon",
    n_genes: int = 100,
    save_result: bool = True,
) -> str:
    """差异表达分析

    比较两组细胞之间的基因表达差异

    Args:
        file_path: .h5ad 文件路径
        group1: 第一组名称（如 'T cells'，可选，默认使用最大的细胞群）
        group2: 第二组名称（可选，默认是其余所有细胞）
        groupby: 分组依据的obs列名
        method: 差异分析方法
        n_genes: 返回的差异基因数量
        save_result: 是否保存结果

    Returns:
        差异表达结果摘要

    Example:
        >>> differential_expression("data.h5ad", group1="T cells", group2="B cells")
        '{"up_genes": {"CD3D": {...}}, "down_genes": {"CD79A": {...}}}'
    """
    try:
        logger.info(f"开始差异表达分析: {file_path}")

        # 验证参数
        method = validator.validate_choices(method, ['wilcoxon', 't-test', 'rank'], 'method')
        n_genes = int(validator.validate_positive_number(n_genes, 'n_genes', 1, 1000))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 验证分组列
        groupby = validator.validate_groupby(adata, groupby)

        # 自动选择 group1（如果未提供）
        if group1 is None:
            # 获取可用的组
            if groupby in adata.obs.columns:
                groups = adata.obs[groupby].cat.categories if hasattr(adata.obs[groupby], 'cat') else adata.obs[groupby].unique()
                # 选择细胞数量最多的组
                group_counts = adata.obs[groupby].value_counts()
                if len(group_counts) > 0:
                    group1 = group_counts.index[0]
                    logger.info(f"自动选择 group1: {group1} (细胞数: {group_counts.iloc[0]})")
                else:
                    return create_tool_result(
                        status="error",
                        message=f"未找到可用的细胞类型，无法进行差异分析"
                    ).to_json()
            else:
                return create_tool_result(
                    status="error",
                    message=f"数据中没有 {groupby} 列"
                ).to_json()

        logger.info(f"差异表达分析: {file_path}, {group1} vs {group2 or '其余细胞'}")

        # 验证参数
        method = validator.validate_choices(method, ['wilcoxon', 't-test', 'rank'], 'method')
        n_genes = int(validator.validate_positive_number(n_genes, 'n_genes', 1, 1000))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 验证分组列
        groupby = validator.validate_groupby(adata, groupby)

        # 执行差异表达分析
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            groups=[group1] if group2 else None,
            reference=group2,
            method=method,
            n_genes=n_genes,
        )

        # 提取结果
        try:
            # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
            genes_df = sc.get.rank_genes_groups_df(adata, group=group1)
            # 限制返回的基因数量
            genes_df = genes_df.head(n_genes)

            # 分为上调和下调基因
            up_genes = {}
            down_genes = {}

            for _, row in genes_df.iterrows():
                gene_name = row['names']
                logfc = row.get('logfoldchanges', 0)
                pval = row.get('pvals_adj', 1.0)

                gene_info = {
                    "logfoldchanges": float(logfc) if not pd.isna(logfc) else 0,
                    "pvals_adj": float(pval) if not pd.isna(pval) else 1.0,
                    "scores": float(row['scores']) if 'scores' in row and not pd.isna(row['scores']) else 0,
                }

                if logfc > 0:
                    up_genes[gene_name] = gene_info
                else:
                    down_genes[gene_name] = gene_info

            result = ToolResult(
                status="success",
                message=f"差异表达分析完成：{len(up_genes)} 上调，{len(down_genes)} 下调",
                data={
                    "group1": group1,
                    "group2": group2 or "rest",
                    "n_up_genes": len(up_genes),
                    "n_down_genes": len(down_genes),
                    "up_genes_sample": dict(list(up_genes.items())[:20]),
                    "down_genes_sample": dict(list(down_genes.items())[:20]),
                }
            )

            # 保存结果
            if save_result:
                de_key = f"de_{group1}_vs_{group2 or 'rest'}"
                adata.uns[de_key] = {
                    "up_genes": up_genes,
                    "down_genes": down_genes,
                    "genes_df": genes_df.to_dict(),
                }

                result_path = _save_result(adata, de_key, "differential_expression", output_dir=data_dir)
                result.artifacts["result_path"] = result_path

                # 保存CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = tables_dir / f"diff_expression_{de_key}_{timestamp}.csv"
                genes_df.to_csv(csv_path, index=False)
                result.artifacts["csv_path"] = str(csv_path)

            logger.info(f"差异表达分析完成: {len(up_genes)} up, {len(down_genes)} down")

            return result.to_json()

        except Exception as e:
            return create_tool_result(
                status="error",
                message=f"差异表达分析失败: {str(e)}",
                error=str(e)
            )

    except Exception as e:
        logger.error(f"差异表达分析失败: {e}")
        return create_tool_result(
            status="error",
            message=f"差异表达分析失败: {str(e)}",
            error=str(e)
        )


@tool("generate_analysis_report", return_direct=False)
def generate_analysis_report(
    file_path: str,
    include_plots: bool = False,
) -> str:
    """生成分析报告摘要

    综合当前所有分析结果生成报告

    Args:
        file_path: .h5ad 文件路径
        include_plots: 是否包含可视化信息

    Returns:
        分析报告摘要

    Example:
        >>> generate_analysis_report("data.h5ad")
        '{"summary": {...}, "clusters": {...}, "recommendations": [...]}'
    """
    try:
        logger.info(f"生成分析报告: {file_path}")

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        path = path.resolve()
        adata = sc.read_h5ad(path)

        report = ToolResult(
            status="success",
            message="生成分析报告中",  # 初始化 message 字段
            data={
                "data_overview": {
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "n_obs_columns": len(adata.obs.columns),
                    "n_obsm_keys": len(adata.obsm.keys()),
                    "n_uns_keys": len(adata.uns.keys()),
                },
                "analysis_status": {},
                "clusters": {},
                "annotations": {},
                "marker_genes": {},
                "recommendations": [],
                "report_content": "",
            }
        )

        # 检查各种分析状态
        analysis_checks = {
            "has_qc_metrics": any(col in adata.obs.columns for col in [
                'n_genes_by_counts', 'total_counts', 'pct_counts_mt',
                'n_genes', 'total_counts', 'pct_counts_mt'
            ]),
            "has_hvg": "highly_variable" in adata.var.columns,
            "has_pca": "X_pca" in adata.obsm,
            "has_umap": "X_umap" in adata.obsm,
            "has_clustering": any(col in adata.obs.columns for col in ['leiden', 'louvain', 'clusters', 'cluster']),
            "has_markers": "rank_genes_groups" in adata.uns,
            "has_cell_type": "cell_type" in adata.obs.columns,
        }

        report.data["analysis_status"] = analysis_checks

        # 聚类信息
        cluster_key = detect_cluster_key(adata)
        if cluster_key:
            cluster_counts = adata.obs[cluster_key].value_counts().to_dict()
            report.data["clusters"] = {
                "key": cluster_key,
                "n_clusters": len(cluster_counts),
                "sizes": cluster_counts,
            }

        # 细胞类型注释信息
        if "cell_type" in adata.obs.columns:
            cell_type_counts = adata.obs["cell_type"].value_counts().to_dict()
            report.data["annotations"] = {
                "n_types": len(cell_type_counts),
                "type_counts": cell_type_counts,
            }

        # 提取标记基因信息
        if "rank_genes_groups" in adata.uns:
            try:
                marker_info = {}
                for cluster in adata.obs[cluster_key].cat.categories if cluster_key else []:
                    try:
                        # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
                        genes_df = sc.get.rank_genes_groups_df(adata, group=cluster)
                        top_genes = genes_df['names'].head(5).tolist()
                        marker_info[str(cluster)] = top_genes
                    except:
                        marker_info[str(cluster)] = []
                report.data["marker_genes"] = marker_info
            except Exception as e:
                logger.warning(f"Failed to extract marker genes: {e}")

        # 生成建议
        if not analysis_checks["has_qc_metrics"]:
            report.data["recommendations"].append("建议运行质量控制分析 (calculate_qc_metrics)")
        if not analysis_checks["has_hvg"]:
            report.data["recommendations"].append("建议运行高变基因鉴定 (normalize_and_hvg)")
        if not analysis_checks["has_pca"]:
            report.data["recommendations"].append("建议运行主成分分析 (pca_reduction)")
        if not analysis_checks["has_clustering"]:
            report.data["recommendations"].append("建议运行聚类分析 (cluster_and_umap)")
        if not analysis_checks["has_markers"]:
            report.data["recommendations"].append("建议运行标记基因分析 (find_marker_genes)")
        if not analysis_checks["has_cell_type"]:
            report.data["recommendations"].append("建议运行细胞注释 (annotate_cells)")

        if not report.data["recommendations"]:
            report.data["recommendations"].append("分析已完成！可以进行下游分析或生成可视化。")

        report.message = f"分析报告生成完成，{len(report.data['recommendations'])} 条建议"

        # 生成格式化的报告内容（用于前端显示）
        # 使用 try-except 保护 markdown 生成过程
        try:
            md_lines = [
                "# 单细胞数据分析报告",
                "",
                f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 📊 数据概览",
                f"- **细胞数量**: {report.data['data_overview']['n_cells']:,}",
                f"- **基因数量**: {report.data['data_overview']['n_genes']:,}",
                "",
                "## ✅ 分析状态",
            ]

            for key, value in report.data["analysis_status"].items():
                status_icon = "✅" if value else "❌"
                status_text = "已完成" if value else "未完成"
                md_lines.append(f"- {status_icon} **{key}**: {status_text}")

            if report.data.get("clusters"):
                md_lines.append("")
                md_lines.append("## 🎯 聚类分析")
                md_lines.append(f"- **聚类方法**: {report.data['clusters'].get('key')}")
                md_lines.append(f"- **聚类数量**: {report.data['clusters'].get('n_clusters')}")
                md_lines.append("- **各cluster大小**:")
                for cluster_id, size in sorted(report.data['clusters']['sizes'].items(), key=lambda x: int(x[0]) if x[0].isdigit() else x):
                    pct = size / sum(report.data['clusters']['sizes'].values()) * 100
                    md_lines.append(f"  - Cluster {cluster_id}: {size} cells ({pct:.1f}%)")

            if report.data.get("marker_genes"):
                md_lines.append("")
                md_lines.append("## 🧬 标记基因")
                for cluster_id, genes in report.data["marker_genes"].items():
                    if genes:
                        md_lines.append(f"- **Cluster {cluster_id}**: {', '.join(genes[:5])}")

            if report.data.get("annotations"):
                md_lines.append("")
                md_lines.append("## 🏷️ 细胞类型注释")
                md_lines.append(f"- **细胞类型数量**: {report.data['annotations'].get('n_types')}")
                md_lines.append("- **各类型细胞数**:")
                for cell_type, count in report.data['annotations']['type_counts'].items():
                    md_lines.append(f"  - {cell_type}: {count}")

            if report.data.get("recommendations"):
                md_lines.append("")
                md_lines.append("## 💡 建议")
                for rec in report.data["recommendations"]:
                    md_lines.append(f"- {rec}")

            report_content = "\n".join(md_lines)
            report.data["report_content"] = report_content
            report.data["report_markdown"] = report_content
        except Exception as e:
            logger.warning(f"Markdown report generation failed: {e}")
            report.data["report_content"] = "# 分析报告\n\n分析已完成。"
            report.data["report_markdown"] = report.data["report_content"]

        # 产物落盘
        runs_root = Path(settings.RUNS_DIR).resolve()
        if path.is_relative_to(runs_root):
            run_id = path.relative_to(runs_root).parts[0]
            report_dir = runs_root / run_id / "artifacts" / "reports"
            plots_dir = runs_root / run_id / "artifacts" / "plots"
        else:
            run_id = None
            report_dir = Path(settings.UPLOAD_DIR) / "analysis_results" / "reports"
            plots_dir = Path(settings.UPLOAD_DIR) / "analysis_results" / "plots"
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_json_path = report_dir / f"analysis_report_{timestamp}.json"
        report_md_path = report_dir / f"analysis_report_{timestamp}.md"

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report.data, f, ensure_ascii=False, indent=2, default=str)

        # 写入Markdown报告文件
        report_md_path.write_text(report_content, encoding="utf-8")

        report.artifacts["result_path"] = str(report_md_path)
        report.artifacts["report_json_path"] = str(report_json_path)

        # 收集所有生成的图表
        plots_metadata = []
        if plots_dir.exists():
            for plot_file in sorted(plots_dir.glob("*.png")):
                plot_name = plot_file.stem
                # 解析图表名称以获取类型
                plot_type = "unknown"
                if "qc_violin" in plot_name:
                    plot_type = "qc_violin"
                elif "umap_cluster" in plot_name or "umap_leiden" in plot_name:
                    plot_type = "umap_cluster"
                elif "umap_annotated" in plot_name or "umap_cell_type" in plot_name:
                    plot_type = "umap_annotated"
                elif "marker_heatmap" in plot_name:
                    plot_type = "marker_heatmap"
                elif "pca_variance" in plot_name:
                    plot_type = "pca_variance"

                plot_metadata = {
                    "name": plot_name,
                    "title": _get_plot_title(plot_type),
                    "type": plot_type,
                    "path": f"/api/artifacts/{run_id}/plots/{plot_file.name}" if run_id else str(plot_file),
                    "local_path": str(plot_file),
                    "interpretation": _get_plot_interpretation(plot_type),
                }
                plots_metadata.append(plot_metadata)

        report.data["plots"] = plots_metadata
        report.data["n_plots"] = len(plots_metadata)
        if run_id:
            report.data["run_id"] = run_id

        logger.info("分析报告生成完成: %s, 包含 %d 个图表", report_md_path, len(plots_metadata))

        # 安全检查：确保 message 已正确设置
        if not report.message:
            report.message = f"分析报告生成完成，包含 {len(plots_metadata)} 个图表"

        # 安全检查：确保 report_content 已设置
        if not report.data.get("report_content"):
            report.data["report_content"] = "分析报告已生成"

        return report.to_json()

    except Exception as e:
        logger.error(f"生成分析报告失败: {e}")
        return create_tool_result(
            status="error",
            message=f"生成分析报告失败: {str(e)}",
            error=str(e)
        )


# 导出所有工具
__all__ = [
    "load_h5ad_data",
    "calculate_qc_metrics",
    "normalize_and_hvg",
    "pca_reduction",
    "cluster_and_umap",
    "find_marker_genes",
    "annotate_cells",
    "differential_expression",
    "generate_analysis_report",
]
