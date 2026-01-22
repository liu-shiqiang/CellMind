"""单细胞核心分析工具库

提供基础的单细胞数据分析功能：
- 数据质量控制 (QC)
- 降维和聚类
- 标记基因识别
- 细胞类型注释
- 差异表达分析
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.cluster import KMeans
from langchain_core.tools import tool

from src.web.config import settings

logger = logging.getLogger(__name__)

# 工作目录配置
WORK_DIR = Path(settings.UPLOAD_DIR) / "analysis_results"
WORK_DIR.mkdir(exist_ok=True, parents=True)

# 参考数据目录
REFERENCE_DIR = Path(settings.DATA_DIR) / "references"
REFERENCE_DIR.mkdir(exist_ok=True, parents=True)

# 输出目录（非Job场景兜底）
OUTPUT_DIR = Path(settings.UPLOAD_DIR) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _resolve_artifact_dirs(input_path: Path) -> tuple[Path, Path, Path]:
    """根据输入路径解析产物目录（Job模式优先）"""
    runs_root = Path(settings.RUNS_DIR).resolve()
    try:
        if input_path.is_relative_to(runs_root):
            run_id = input_path.relative_to(runs_root).parts[0]
            artifacts_dir = runs_root / run_id / "artifacts"
            data_dir = artifacts_dir / "data"
            tables_dir = artifacts_dir / "tables"
            plots_dir = artifacts_dir / "plots"
            for dir_path in (data_dir, tables_dir, plots_dir):
                dir_path.mkdir(parents=True, exist_ok=True)
            return data_dir, tables_dir, plots_dir
    except ValueError:
        pass

    return OUTPUT_DIR, OUTPUT_DIR, OUTPUT_DIR


def _save_result(
    adata: ad.AnnData,
    result_key: str,
    analysis_type: str,
    output_dir: Optional[Path] = None,
) -> str:
    """保存分析结果到文件

    Args:
        adata: AnnData 对象
        result_key: 结果键名
        analysis_type: 分析类型
        output_dir: 输出目录（默认 OUTPUT_DIR）

    Returns:
        保存的文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{analysis_type}_{result_key}_{timestamp}.h5ad"
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename

    adata.write(filepath)
    logger.info(f"结果已保存到: {filepath}")

    # 保存元数据为JSON（便于查看）
    metadata_path = target_dir / f"{analysis_type}_{result_key}_{timestamp}_metadata.json"
    metadata = {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "result_key": result_key,
        "analysis_type": analysis_type,
        "timestamp": timestamp,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    return str(filepath)


def _extract_summary(adata: ad.AnnData, result_key: str) -> Dict[str, Any]:
    """提取分析结果摘要

    Args:
        adata: AnnData 对象
        result_key: 结果键名

    Returns:
        结果摘要字典
    """
    summary = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
    }

    if result_key in adata.obs.columns:
        obs_data = adata.obs[result_key]
        if pd.api.types.is_categorical_dtype(obs_data):
            summary["categories"] = obs_data.cat.categories.tolist()
            summary["category_counts"] = obs_data.value_counts().to_dict()
        else:
            summary["min"] = float(obs_data.min()) if hasattr(obs_data, 'min') else None
            summary["max"] = float(obs_data.max()) if hasattr(obs_data, 'max') else None
            summary["mean"] = float(obs_data.mean()) if hasattr(obs_data, 'mean') else None

    if result_key in adata.var.columns:
        var_data = adata.var[result_key]
        summary["var_min"] = float(var_data.min()) if hasattr(var_data, 'min') else None
        summary["var_max"] = float(var_data.max()) if hasattr(var_data, 'max') else None

    if f"{result_key}_names" in adata.uns:
        summary["feature_names"] = adata.uns[f"{result_key}_names"][:50].tolist()  # 只返回前50个

    return summary


@tool(
    "load_h5ad_data",
    return_direct=False,
)
def load_h5ad_data(
    file_path: Optional[str] = None,
    filepath: Optional[str] = None, # Added to handle potential LLM argument name mismatch
    cache: bool = True,
) -> str:
    """加载 .h5ad 格式的单细胞数据文件

    Args:
        file_path: .h5ad 文件路径
        cache: 是否缓存数据

    Returns:
        数据加载结果摘要，包含细胞数、基因数等信息

    Example:
        >>> load_h5ad_data("data.h5ad")
        '{"n_cells": 5000, "n_genes": 20000, "status": "loaded"}'
    """
    try:
        # Resolve the correct file_path from either file_path or filepath
        if file_path is None and filepath is not None:
            file_path = filepath
        elif file_path is None and filepath is None:
            return json.dumps({
                "status": "error",
                "message": "必须提供 'file_path' 或 'filepath' 参数",
            }, ensure_ascii=False)
        elif file_path is not None and filepath is not None and file_path != filepath:
            logger.warning(f"Both file_path and filepath were provided and are different. Using file_path: {file_path}")
        
        logger.info(f"加载数据文件: {file_path}")

        # 检查文件路径
        path = Path(file_path)
        if not path.is_absolute():
            # 尝试在上传目录中查找
            upload_path = Path(settings.UPLOAD_DIR) / file_path
            if upload_path.exists():
                path = upload_path
            else:
                # 尝试添加 .h5ad 扩展名
                if not file_path.endswith('.h5ad'):
                    h5ad_path = Path(settings.UPLOAD_DIR) / f"{file_path}.h5ad"
                    if h5ad_path.exists():
                        path = h5ad_path
                    else:
                        # 尝试直接使用文件名
                        path = Path(settings.UPLOAD_DIR) / Path(file_path).name
                        if not path.exists() and not str(path).endswith('.h5ad'):
                            path = Path(settings.UPLOAD_DIR) / f"{Path(file_path).name}.h5ad"
                else:
                    path = Path(settings.UPLOAD_DIR) / Path(file_path).name

        if not path.exists():
            return json.dumps({
                "status": "error",
                "message": f"文件不存在: {file_path}",
            }, ensure_ascii=False)

        # 加载数据
        adata = sc.read_h5ad(path)

        # 基础信息
        n_obs, n_vars = adata.n_obs, adata.n_vars

        result = {
            "status": "success",
            "n_cells": n_obs,
            "n_genes": n_vars,
            "file_path": str(path),
            "result_path": str(path),
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys()),
            "message": f"成功加载 {n_obs} 个细胞和 {n_vars} 个基因",
        }

        # 检查是否有基本聚类信息
        has_clustering = any(col in adata.obs.columns for col in ['leiden', 'louvain', 'clusters', 'cluster'])
        result["has_clustering"] = has_clustering

        # 检查是否有UMAP/TSNE信息
        has_embedding = any(key in adata.obsm for key in ['X_umap', 'X_tsne', 'X_pca'])
        result["has_embedding"] = has_embedding

        logger.info(f"数据加载成功: {n_obs} cells x {n_vars} genes")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"加载数据失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "calculate_qc_metrics",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

        adata = sc.read_h5ad(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)

        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars

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

        # 过滤建议
        # 常用阈值：基因数 < 200 或 > 6000 可能是低质量细胞
        # 线粒体比例 > 20% 可能是死亡细胞
        high_mt = (adata.obs['pct_counts_mt'] > 20).sum()
        low_genes = (adata.obs['n_genes_by_counts'] < min_genes).sum()
        high_genes = (adata.obs['n_genes_by_counts'] > 6000).sum()

        qc_stats["high_mt_cells"] = int(high_mt)
        qc_stats["low_gene_cells"] = int(low_genes)
        qc_stats["high_gene_cells"] = int(high_genes)

        # 过滤数据
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

        logger.info(f"质控完成: {n_cells_before} -> {n_cells_after} cells")

        return json.dumps(qc_stats, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"质控分析失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"质控分析失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "normalize_and_hvg",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

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

        # 获取高变基因列表
        hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()

        result = {
            "status": "success",
            "n_hvg": int(n_hvg),
            "target_sum": target_sum,
            "hvg_genes_sample": hvg_genes[:50],  # 返回前50个作为示例
            "mean_var": float(adata.var['highly_variable'].mean()),
            "message": f"鉴定到 {n_hvg} 个高变基因",
        }

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "hvg", "normalization", output_dir=data_dir)
            result["result_path"] = result_path

        logger.info(f"HVG分析完成: {n_hvg} highly variable genes")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"标准化和HVG分析失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"标准化和HVG分析失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "pca_reduction",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

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
        adata.varm['PCs'] = adata_hvg.varm['PCs']
        adata.uns['pca'] = adata_hvg.uns['pca']

        # 提取方差解释比例
        variance_ratio = adata.uns['pca']['variance_ratio'].tolist()
        cumulative_ratio = float(sum(variance_ratio[:30]))  # 前30个成分的累积方差

        result = {
            "status": "success",
            "n_comps": n_comps,
            "variance_ratio_sample": variance_ratio[:10],
            "cumulative_variance_30pc": cumulative_ratio,
            "pca_shape": adata.obsm['X_pca'].shape,
            "message": f"PCA完成，前30个成分解释{cumulative_ratio*100:.1f}%方差",
        }

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "pca", "dimensionality_reduction", output_dir=data_dir)
            result["result_path"] = result_path

        logger.info(f"PCA完成: {n_comps} components, {cumulative_ratio*100:.1f}% variance")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"PCA降维失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"PCA降维失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "cluster_and_umap",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 检查是否有PCA
        if 'X_pca' not in adata.obsm:
            # 如果没有PCA，先运行
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

        # 计算邻接图
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)

        # Leiden 聚类
        try:
            sc.tl.leiden(adata, resolution=resolution)
        except Exception as exc:
            if "igraph" in str(exc).lower() or "leidenalg" in str(exc).lower():
                logger.warning("缺少 igraph/leidenalg，回退到 KMeans 聚类")
                if 'X_pca' not in adata.obsm:
                    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
                n_clusters = max(2, int(round(resolution * 10)))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                labels = kmeans.fit_predict(adata.obsm['X_pca'])
                adata.obs['kmeans'] = pd.Categorical(labels.astype(str))
            else:
                raise

        # UMAP
        sc.tl.umap(adata)

        # 提取聚类统计
        leiden_key = 'leiden'
        if leiden_key not in adata.obs.columns:
            # 尝试其他可能的键名
            for key in ['kmeans', 'louvain', 'clusters', 'cluster']:
                if key in adata.obs.columns:
                    leiden_key = key
                    break

        if leiden_key in adata.obs.columns:
            cluster_counts = adata.obs[leiden_key].value_counts().to_dict()
            n_clusters = len(cluster_counts)
        else:
            cluster_counts = {}
            n_clusters = 0

        result = {
            "status": "success",
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_counts,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
            "umap_shape": adata.obsm['X_umap'].shape,
            "cluster_key": leiden_key,
            "message": f"聚类完成，识别到{n_clusters}个cluster",
        }

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "cluster_umap", "clustering", output_dir=data_dir)
            result["result_path"] = result_path

            try:
                umap_df = pd.DataFrame(
                    adata.obsm['X_umap'],
                    columns=['UMAP_1', 'UMAP_2'],
                )
                if leiden_key in adata.obs.columns:
                    umap_df['cluster'] = adata.obs[leiden_key].astype(str).values
                if 'cell_type' in adata.obs.columns:
                    umap_df['cell_type'] = adata.obs['cell_type'].astype(str).values
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                umap_path = tables_dir / f"umap_coords_{timestamp}.csv"
                umap_df.to_csv(umap_path, index=False)
                result["umap_coords_path"] = str(umap_path)
            except Exception as exc:
                logger.warning("UMAP坐标导出失败: %s", exc)

        logger.info(f"聚类和UMAP完成: {n_clusters} clusters")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"聚类和UMAP分析失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"聚类和UMAP分析失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "find_marker_genes",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        if cluster_key is None:
            for key in ['leiden', 'louvain', 'clusters', 'cluster']:
                if key in adata.obs.columns:
                    cluster_key = key
                    break

        if cluster_key is None or cluster_key not in adata.obs.columns:
            return json.dumps({
                "status": "error",
                "message": "未找到聚类信息，请先运行聚类分析",
            }, ensure_ascii=False)

        # 寻找标记基因
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            method=method,
            corr_method='bonferroni',
            n_genes=n_genes,
        )

        # 提取结果
        result = {
            "status": "success",
            "cluster_key": cluster_key,
            "method": method,
            "n_genes_per_group": n_genes,
            "clusters": {},
        }

        # 获取每个cluster的top标记基因
        groups = adata.obs[cluster_key].cat.categories.tolist()
        for group in groups:
            try:
                genes_df = sc.get.rank_genes_groups_df(
                    adata,
                    group=group,
                    n_genes=n_genes
                )
                top_genes = {}
                for _, row in genes_df.head(n_genes).iterrows():
                    gene_name = row['names']
                    top_genes[gene_name] = {
                        "logfoldchanges": float(row['logfoldchanges']) if 'logfoldchanges' in row else None,
                        "pval": float(row['pvals_adj']) if 'pvals_adj' in row else None,
                        "scores": float(row['scores']) if 'scores' in row else None,
                    }

                result["clusters"][group] = {
                    "top_genes": top_genes,
                    "n_genes": len(top_genes),
                }
            except Exception as e:
                logger.warning(f"提取cluster {group} 的标记基因失败: {e}")
                result["clusters"][group] = {"error": str(e)}

        result["n_clusters"] = len(groups)
        result["message"] = f"标记基因分析完成，{len(groups)}个clusters"

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "markers", "marker_genes", output_dir=data_dir)
            result["result_path"] = result_path

            # 额外保存标记基因CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = tables_dir / f"marker_genes_{timestamp}.csv"
            try:
                all_markers = sc.get.rank_genes_groups_df(adata, n_genes=n_genes)
                all_markers.to_csv(csv_path, index=False)
                result["csv_path"] = str(csv_path)
            except Exception as e:
                logger.warning(f"保存标记基因CSV失败: {e}")

        logger.info(f"标记基因分析完成: {len(groups)} clusters")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"标记基因分析失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"标记基因分析失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "annotate_cells",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

        adata = sc.read_h5ad(path)
        data_dir, _, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        if cluster_key is None:
            for key in ['leiden', 'louvain', 'clusters', 'cluster']:
                if key in adata.obs.columns:
                    cluster_key = key
                    break

        if cluster_key is None or cluster_key not in adata.obs.columns:
            return json.dumps({
                "status": "error",
                "message": "未找到聚类信息，请先运行聚类分析",
            }, ensure_ascii=False)

        # 获取clusters
        clusters = adata.obs[cluster_key].cat.categories.tolist()

        result = {
            "status": "success",
            "cluster_key": cluster_key,
            "annotations": {},
            "n_annotated": 0,
        }

        # 如果提供了手动注释，使用手动注释
        if annotations:
            # 创建注释映射
            annotation_map = {}
            for cluster, cell_type in annotations.items():
                # 支持多种cluster格式
                if str(cluster) in clusters:
                    annotation_map[str(cluster)] = cell_type

            # 应用注释
            adata.obs['cell_type'] = adata.obs[cluster_key].map(annotation_map).fillna('Unknown')
            result["annotations"] = annotation_map
            result["n_annotated"] = len(annotation_map)
            result["method"] = "manual"

        # 基于标记基因的自动注释（简化版）
        elif marker_based and 'rank_genes_groups' in adata.uns:
            # 简化的自动注释逻辑
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

            annotation_map = {}

            for cluster in clusters:
                try:
                    genes_df = sc.get.rank_genes_groups_df(adata, group=cluster, n_genes=10)
                    top_genes = set(genes_df['names'].head(10).tolist())

                    best_match = None
                    best_score = 0

                    for cell_type, markers in common_markers.items():
                        score = len(top_genes & set(markers))
                        if score > best_score:
                            best_score = score
                            best_match = cell_type

                    if best_score >= 1:  # 至少有一个匹配
                        annotation_map[cluster] = best_match
                    else:
                        annotation_map[cluster] = f"Cluster_{cluster}"

                except Exception as e:
                    annotation_map[cluster] = f"Cluster_{cluster}"

            adata.obs['cell_type'] = adata.obs[cluster_key].map(annotation_map).fillna('Unknown')
            result["annotations"] = annotation_map
            result["n_annotated"] = len(annotation_map)
            result["method"] = "marker_based"

        else:
            # 默认使用cluster名称
            adata.obs['cell_type'] = adata.obs[cluster_key].astype(str)
            for cluster in clusters:
                result["annotations"][cluster] = f"Cluster_{cluster}"
            result["n_annotated"] = len(clusters)
            result["method"] = "cluster_name"

        if "pred_celltype" not in adata.obs.columns:
            adata.obs["pred_celltype"] = adata.obs["cell_type"]

        # 统计每种细胞类型的数量
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()
        result["cell_type_counts"] = cell_type_counts

        result["message"] = f"注释完成，{len(result['annotations'])}个clusters"

        # 保存结果
        if save_result:
            result_path = _save_result(adata, "annotated", "cell_annotation", output_dir=data_dir)
            result["result_path"] = result_path

        logger.info(f"细胞注释完成: {result['n_annotated']} clusters")

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"细胞注释失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"细胞注释失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "differential_expression",
    return_direct=False,
)
def differential_expression(
    file_path: str,
    group1: str,
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
        group1: 第一组名称（如 'T cells'）
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
        logger.info(f"开始差异表达分析: {file_path}, {group1} vs {group2}")

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name

        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        if groupby not in adata.obs.columns:
            return json.dumps({
                "status": "error",
                "message": f"未找到分组列 '{groupby}'",
            }, ensure_ascii=False)

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
            genes_df = sc.get.rank_genes_groups_df(adata, group=group1, n_genes=n_genes)

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

            result = {
                "status": "success",
                "group1": group1,
                "group2": group2 or "rest",
                "n_up_genes": len(up_genes),
                "n_down_genes": len(down_genes),
                "up_genes_sample": dict(list(up_genes.items())[:20]),
                "down_genes_sample": dict(list(down_genes.items())[:20]),
                "message": f"差异表达分析完成：{len(up_genes)} 上调，{len(down_genes)} 下调",
            }

            # 保存结果
            if save_result:
                # 保存到adata.uns
                de_key = f"de_{group1}_vs_{group2 or 'rest'}"
                adata.uns[de_key] = {
                    "up_genes": up_genes,
                    "down_genes": down_genes,
                    "genes_df": genes_df.to_dict(),
                }

                result_path = _save_result(adata, de_key, "differential_expression", output_dir=data_dir)
                result["result_path"] = result_path

                # 保存CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = tables_dir / f"diff_expression_{de_key}_{timestamp}.csv"
                genes_df.to_csv(csv_path, index=False)
                result["csv_path"] = str(csv_path)

            logger.info(f"差异表达分析完成: {len(up_genes)} up, {len(down_genes)} down")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"差异表达分析失败: {str(e)}",
            }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"差异表达分析失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"差异表达分析失败: {str(e)}",
        }, ensure_ascii=False)


@tool(
    "generate_analysis_report",
    return_direct=False,
)
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

        # 加载数据
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(settings.UPLOAD_DIR) / path.name
        path = path.resolve()

        adata = sc.read_h5ad(path)

        report = {
            "status": "success",
            "data_overview": {
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_obs_columns": len(adata.obs.columns),
                "n_obsm_keys": len(adata.obsm.keys()),
                "n_uns_keys": len(adata.uns.keys),
            },
            "analysis_status": {},
            "clusters": {},
            "annotations": {},
            "recommendations": [],
        }

        # 检查各种分析状态
        analysis_checks = {
            "has_qc_metrics": any(col in adata.obs.columns for col in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']),
            "has_hvg": "highly_variable" in adata.var.columns,
            "has_pca": "X_pca" in adata.obsm,
            "has_umap": "X_umap" in adata.obsm,
            "has_clustering": any(col in adata.obs.columns for col in ['leiden', 'louvain', 'clusters', 'cluster']),
            "has_markers": "rank_genes_groups" in adata.uns,
            "has_cell_type": "cell_type" in adata.obs.columns,
        }

        report["analysis_status"] = analysis_checks

        # 聚类信息
        cluster_key = None
        for key in ['leiden', 'louvain', 'clusters', 'cluster']:
            if key in adata.obs.columns:
                cluster_key = key
                break

        if cluster_key:
            cluster_counts = adata.obs[cluster_key].value_counts().to_dict()
            report["clusters"] = {
                "key": cluster_key,
                "n_clusters": len(cluster_counts),
                "sizes": cluster_counts,
            }

        # 细胞类型注释信息
        if "cell_type" in adata.obs.columns:
            cell_type_counts = adata.obs["cell_type"].value_counts().to_dict()
            report["annotations"] = {
                "n_types": len(cell_type_counts),
                "type_counts": cell_type_counts,
            }

        # 生成建议
        if not analysis_checks["has_qc_metrics"]:
            report["recommendations"].append("建议运行质量控制分析 (calculate_qc_metrics)")
        if not analysis_checks["has_hvg"]:
            report["recommendations"].append("建议运行高变基因鉴定 (normalize_and_hvg)")
        if not analysis_checks["has_pca"]:
            report["recommendations"].append("建议运行主成分分析 (pca_reduction)")
        if not analysis_checks["has_clustering"]:
            report["recommendations"].append("建议运行聚类分析 (cluster_and_umap)")
        if not analysis_checks["has_markers"]:
            report["recommendations"].append("建议运行标记基因分析 (find_marker_genes)")
        if not analysis_checks["has_cell_type"]:
            report["recommendations"].append("建议运行细胞注释 (annotate_cells)")

        if not report["recommendations"]:
            report["recommendations"].append("分析已完成！可以进行下游分析或生成可视化。")

        report["message"] = f"分析报告生成完成，{len(report['recommendations'])} 条建议"

        # 产物落盘
        runs_root = Path(settings.RUNS_DIR).resolve()
        if path.is_relative_to(runs_root):
            run_id = path.relative_to(runs_root).parts[0]
            report_dir = runs_root / run_id / "artifacts" / "reports"
        else:
            report_dir = Path(settings.UPLOAD_DIR) / "analysis_results" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_json_path = report_dir / f"analysis_report_{timestamp}.json"
        report_md_path = report_dir / f"analysis_report_{timestamp}.md"

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        md_lines = [
            "# 单细胞数据分析报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 数据概览",
            f"- 细胞数: {report['data_overview']['n_cells']}",
            f"- 基因数: {report['data_overview']['n_genes']}",
            "",
            "## 分析状态",
        ]
        for key, value in report["analysis_status"].items():
            md_lines.append(f"- {key}: {value}")

        if report.get("clusters"):
            md_lines.append("")
            md_lines.append("## 聚类概览")
            md_lines.append(f"- 聚类键: {report['clusters'].get('key')}")
            md_lines.append(f"- 聚类数量: {report['clusters'].get('n_clusters')}")

        if report.get("annotations"):
            md_lines.append("")
            md_lines.append("## 细胞类型注释")
            md_lines.append(f"- 细胞类型数: {report['annotations'].get('n_types')}")

        if report.get("recommendations"):
            md_lines.append("")
            md_lines.append("## 建议")
            for rec in report["recommendations"]:
                md_lines.append(f"- {rec}")

        report_md_path.write_text("\n".join(md_lines), encoding="utf-8")

        report["result_path"] = str(report_md_path)
        report["report_json_path"] = str(report_json_path)

        logger.info("分析报告生成完成: %s", report_md_path)

        return json.dumps(report, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"生成分析报告失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"生成分析报告失败: {str(e)}",
        }, ensure_ascii=False)
