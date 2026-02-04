"""工具参数验证

提供统一的参数验证功能：
- 文件路径验证
- h5ad 文件验证
- 聚类键名验证
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import scanpy as sc

from src.web.config import settings

logger = logging.getLogger(__name__)


class ToolValidator:
    """工具参数验证器"""

    def __init__(self):
        """初始化验证器"""
        self.upload_dir = Path(settings.UPLOAD_DIR)

    def validate_file_path(
        self,
        file_path: str,
        must_exist: bool = True
    ) -> Path:
        """验证文件路径

        Args:
            file_path: 文件路径
            must_exist: 文件是否必须存在

        Returns:
            验证后的 Path 对象

        Raises:
            ValueError: 验证失败
        """
        if not file_path:
            raise ValueError("file_path 不能为空")

        path = Path(file_path)

        # 如果是相对路径，尝试解析为绝对路径
        if not path.is_absolute():
            upload_path = self.upload_dir / file_path
            if upload_path.exists():
                path = upload_path
            elif must_exist:
                raise ValueError(f"文件不存在: {file_path}")

        if must_exist and not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        return path

    def validate_h5ad_file(
        self,
        file_path: str,
        min_cells: int = 0,
        min_genes: int = 0
    ) -> sc.AnnData:
        """验证并加载 h5ad 文件

        Args:
            file_path: h5ad 文件路径
            min_cells: 最小细胞数要求
            min_genes: 最小基因数要求

        Returns:
            加载的 AnnData 对象

        Raises:
            ValueError: 验证失败
        """
        path = self.validate_file_path(file_path, must_exist=True)

        if path.suffix not in [".h5ad", ".h5"]:
            raise ValueError(f"文件格式错误，期望 .h5ad 文件: {file_path}")

        try:
            adata = sc.read_h5ad(path)
        except Exception as e:
            raise ValueError(f"无法读取 h5ad 文件: {e}")

        # 验证细胞数
        if min_cells > 0 and adata.n_obs < min_cells:
            raise ValueError(f"细胞数不足: {adata.n_obs} < {min_cells}")

        # 验证基因数
        if min_genes > 0 and adata.n_vars < min_genes:
            raise ValueError(f"基因数不足: {adata.n_vars} < {min_genes}")

        logger.info(f"成功加载 h5ad 文件: {adata.n_obs} cells x {adata.n_vars} genes")

        return adata

    def validate_cluster_key(
        self,
        adata: sc.AnnData,
        cluster_key: Optional[str] = None
    ) -> str:
        """验证并检测聚类键名

        Args:
            adata: AnnData 对象
            cluster_key: 指定的聚类键名（可选）

        Returns:
            有效的聚类键名

        Raises:
            ValueError: 未找到聚类信息
        """
        # 如果指定了键名，验证其存在
        if cluster_key:
            if cluster_key not in adata.obs.columns:
                available = [col for col in adata.obs.columns
                            if col in ['leiden', 'louvain', 'kmeans', 'clusters', 'cluster', 'scGPT_clusters']]
                raise ValueError(
                    f"聚类列 '{cluster_key}' 不存在。"
                    f"可用的聚类列: {available or '无'}"
                )
            return cluster_key

        # 自动检测聚类键名
        for key in ['leiden', 'louvain', 'kmeans', 'clusters', 'cluster', 'scGPT_clusters']:
            if key in adata.obs.columns:
                logger.info(f"自动检测到聚类列: {key}")
                return key

        raise ValueError(
            "未找到聚类信息，请先运行聚类分析。"
            "可用的聚类方法: cluster_and_umap"
        )

    def validate_groupby(
        self,
        adata: sc.AnnData,
        groupby: str
    ) -> str:
        """验证分组列名

        Args:
            adata: AnnData 对象
            groupby: 分组列名

        Returns:
            有效的分组列名

        Raises:
            ValueError: 列不存在
        """
        if groupby not in adata.obs.columns:
            raise ValueError(
                f"分组列 '{groupby}' 不存在。"
                f"可用的列: {list(adata.obs.columns)[:20]}"
            )
        return groupby

    def validate_positive_number(
        self,
        value: Any,
        name: str = "参数",
        min_value: float = 0,
        max_value: Optional[float] = None
    ) -> float:
        """验证正数参数

        Args:
            value: 参数值
            name: 参数名称
            min_value: 最小值
            max_value: 最大值

        Returns:
            验证后的浮点数值

        Raises:
            ValueError: 验证失败
        """
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} 必须是数字: {value}")

        if num_value < min_value:
            raise ValueError(f"{name} 不能小于 {min_value}: {value}")

        if max_value is not None and num_value > max_value:
            raise ValueError(f"{name} 不能大于 {max_value}: {value}")

        return num_value

    def validate_choices(
        self,
        value: Any,
        choices: List[Any],
        name: str = "参数"
    ) -> Any:
        """验证参数值在允许的选项中

        Args:
            value: 参数值
            choices: 允许的选项列表
            name: 参数名称

        Returns:
            验证后的参数值

        Raises:
            ValueError: 验证失败
        """
        if value not in choices:
            raise ValueError(
                f"{name} 必须是以下值之一: {choices}，实际值: {value}"
            )
        return value

    def validate_resolution(
        self,
        resolution: float
    ) -> float:
        """验证聚类分辨率参数

        Args:
            resolution: 分辨率值

        Returns:
            验证后的分辨率值

        Raises:
            ValueError: 验证失败
        """
        return self.validate_positive_number(
            resolution,
            name="resolution",
            min_value=0.01,
            max_value=10.0
        )

    def validate_n_neighbors(
        self,
        n_neighbors: int
    ) -> int:
        """验证近邻数参数

        Args:
            n_neighbors: 近邻数

        Returns:
            验证后的近邻数

        Raises:
            ValueError: 验证失败
        """
        return int(self.validate_positive_number(
            n_neighbors,
            name="n_neighbors",
            min_value=2,
            max_value=200
        ))

    def validate_n_top_genes(
        self,
        n_top_genes: int
    ) -> int:
        """验证高变基因数参数

        Args:
            n_top_genes: 高变基因数

        Returns:
            验证后的高变基因数

        Raises:
            ValueError: 验证失败
        """
        return int(self.validate_positive_number(
            n_top_genes,
            name="n_top_genes",
            min_value=10,
            max_value=10000
        ))


# 全局单例
_validator: Optional[ToolValidator] = None


def get_tool_validator() -> ToolValidator:
    """获取全局工具验证器单例"""
    global _validator
    if _validator is None:
        _validator = ToolValidator()
    return _validator


# 便捷函数
def validate_file_path(
    file_path: str,
    must_exist: bool = True
) -> Path:
    """验证文件路径"""
    return get_tool_validator().validate_file_path(file_path, must_exist)


def validate_h5ad_file(
    file_path: str,
    min_cells: int = 0,
    min_genes: int = 0
) -> sc.AnnData:
    """验证并加载 h5ad 文件"""
    return get_tool_validator().validate_h5ad_file(file_path, min_cells, min_genes)


def validate_cluster_key(
    adata: sc.AnnData,
    cluster_key: Optional[str] = None
) -> str:
    """验证并检测聚类键名"""
    return get_tool_validator().validate_cluster_key(adata, cluster_key)


# 导出
__all__ = [
    "ToolValidator",
    "get_tool_validator",
    "validate_file_path",
    "validate_h5ad_file",
    "validate_cluster_key",
]
