"""单细胞分析工具基类

提供统一的工具基础类，包含：
- 统一的错误处理
- 统一的日志格式
- 统一的返回格式
- 统一的参数验证
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic

from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator

from src.web.config import settings

logger = logging.getLogger(__name__)

# 工作目录配置
WORK_DIR = Path(settings.UPLOAD_DIR) / "analysis_results"
WORK_DIR.mkdir(exist_ok=True, parents=True)

# 参考数据目录
REFERENCE_DIR = Path(settings.DATA_DIR) / "references"
REFERENCE_DIR.mkdir(exist_ok=True, parents=True)

# 输出目录
OUTPUT_DIR = Path(settings.UPLOAD_DIR) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class ToolResult(BaseModel):
    """统一的工具返回格式"""

    status: str = Field(description="执行状态: success, error, warning")
    message: str = Field(description="执行结果消息")
    data: Dict[str, Any] = Field(default_factory=dict, description="返回数据")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="产物文件路径")
    error: Optional[str] = Field(default=None, description="错误详情（仅当status为error时）")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        data = self.dict()
        # 确保 message 字段存在（防御性编程）
        if "message" not in data or not data["message"]:
            data["message"] = "处理完成"
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def is_success(self) -> bool:
        """是否执行成功"""
        return self.status == "success"

    def is_error(self) -> bool:
        """是否执行失败"""
        return self.status == "error"


class SingleCellToolArgs(BaseModel):
    """单细胞工具的基础参数模型"""

    file_path: str = Field(..., description="输入文件路径 (.h5ad)")
    save_result: bool = Field(default=True, description="是否保存结果文件")
    work_dir: Optional[str] = Field(default=None, description="工作目录（可选）")

    @validator("file_path")
    def validate_file_path(cls, v):
        """验证文件路径"""
        if not v:
            raise ValueError("file_path 不能为空")
        return v


class BaseSingleCellTool(ABC):
    """单细胞分析工具基类

    提供通用的工具功能：
    - 文件路径解析
    - 参数验证
    - 错误处理
    - 结果保存
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def resolve_input_path(self, file_path: str) -> Path:
        """解析输入文件路径

        Args:
            file_path: 文件路径（可以是相对路径或绝对路径）

        Returns:
            解析后的绝对路径
        """
        path = Path(file_path)

        # 如果是相对路径，尝试在上传目录中查找
        if not path.is_absolute():
            upload_path = Path(settings.UPLOAD_DIR) / file_path
            if upload_path.exists():
                return upload_path

            # 尝试添加 .h5ad 扩展名
            if not file_path.endswith('.h5ad'):
                h5ad_path = Path(settings.UPLOAD_DIR) / f"{file_path}.h5ad"
                if h5ad_path.exists():
                    return h5ad_path

            # 尝试使用文件名
            path = Path(settings.UPLOAD_DIR) / Path(file_path).name
            if path.exists():
                return path

        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        return path

    def resolve_output_dirs(self, input_path: Path) -> Tuple[Path, Path, Path]:
        """解析输出目录（Job模式优先）

        Args:
            input_path: 输入文件路径

        Returns:
            (data_dir, tables_dir, plots_dir) 元组
        """
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

        # 非 Job 模式，使用通用输出目录
        return OUTPUT_DIR, OUTPUT_DIR, OUTPUT_DIR

    def execute_safe(self, **kwargs) -> str:
        """安全执行工具，捕获异常并返回统一格式

        Args:
            **kwargs: 工具参数

        Returns:
            JSON 格式的结果字符串
        """
        try:
            result = self.execute(**kwargs)
            return result
        except FileNotFoundError as e:
            logger.error(f"文件错误: {e}")
            return ToolResult(
                status="error",
                message=f"文件未找到: {e}",
                error=str(e)
            ).to_json()
        except ValueError as e:
            logger.error(f"参数错误: {e}")
            return ToolResult(
                status="error",
                message=f"参数错误: {e}",
                error=str(e)
            ).to_json()
        except Exception as e:
            logger.exception(f"工具执行失败: {self.name}")
            return ToolResult(
                status="error",
                message=f"{self.name} 执行失败: {e}",
                error=str(e)
            ).to_json()

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具的具体逻辑

        Args:
            **kwargs: 工具参数

        Returns:
            JSON 格式的结果字符串
        """
        pass


def create_tool_result(
    status: str = "success",
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
) -> str:
    """创建标准工具返回结果

    Args:
        status: 执行状态
        message: 执行消息
        data: 返回数据
        artifacts: 产物文件路径
        error: 错误详情

    Returns:
        JSON 格式的结果字符串
    """
    return ToolResult(
        status=status,
        message=message,
        data=data or {},
        artifacts=artifacts or {},
        error=error
    ).to_json()


def fix_adata_var_index_name(adata) -> None:
    """修复 AnnData var DataFrame 的 index name 与列名冲突问题

    当 var.index.name 与 var 中的列名相同时，scanpy 的 calculate_qc_metrics
    会抛出错误。此函数将 index.name 重置为 None 以避免冲突。

    Args:
        adata: AnnData 对象，会就地修改
    """
    if adata.var.index.name is not None:
        index_name = adata.var.index.name
        # 检查 index.name 是否与列名冲突
        if index_name in adata.var.columns:
            # 如果冲突，检查列值是否与 index 相同
            if not (adata.var[index_name] == adata.var.index).all():
                # 值不同，需要重置 index.name
                logger.warning(f"var.index.name '{index_name}' conflicts with column name and has different values. Resetting index.name to None.")
                adata.var.index.name = None
            else:
                # 值相同，可以直接删除该列
                logger.warning(f"var.index.name '{index_name}' conflicts with column name but values are identical. Dropping column.")
                adata.var = adata.var.drop(columns=[index_name])
        else:
            # 没有直接冲突，但 scanpy 可能有其他问题，重置 index.name 更安全
            adata.var.index.name = None

    # 同时检查 obs 的 index.name
    if adata.obs.index.name is not None:
        adata.obs.index.name = None


def detect_cluster_key(adata) -> Optional[str]:
    """自动检测聚类键名

    Args:
        adata: AnnData 对象

    Returns:
        检测到的聚类键名，如果未找到返回 None
    """
    for key in ['leiden', 'louvain', 'kmeans', 'clusters', 'cluster', 'scGPT_clusters']:
        if key in adata.obs.columns:
            return key
    return None


def _save_result(
    adata: Any,
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
    import scanpy as sc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{analysis_type}_{result_key}_{timestamp}.h5ad"
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename

    adata.write(filepath)
    logger.info(f"结果已保存到: {filepath}")

    # 保存元数据
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
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    return str(filepath)


# 导出
__all__ = [
    "ToolResult",
    "SingleCellToolArgs",
    "BaseSingleCellTool",
    "create_tool_result",
    "fix_adata_var_index_name",
    "detect_cluster_key",
    "_save_result",
    "WORK_DIR",
    "REFERENCE_DIR",
    "OUTPUT_DIR",
]
