"""路径解析工具

统一处理单细胞分析中的文件路径解析：
- 输入文件路径解析
- 产物文件路径获取
- 输出目录解析
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.web.config import settings

logger = logging.getLogger(__name__)

# 输出目录
OUTPUT_DIR = Path(settings.UPLOAD_DIR) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class PathResolver:
    """路径解析器

    提供统一的路径解析功能：
    - 解析输入文件路径
    - 获取最新产物文件
    - 解析输出目录
    """

    # 工具到其依赖的前置工具的映射
    TOOL_DEPENDENCY_MAP = {
        "calculate_qc_metrics": ["load_h5ad_data"],
        "normalize_and_hvg": ["calculate_qc_metrics"],
        "pca_reduction": ["normalize_and_hvg"],
        "cluster_and_umap": ["pca_reduction"],
        "find_marker_genes": ["cluster_and_umap"],
        "annotate_cells": ["find_marker_genes"],
        "annotate_with_simple_markers": ["find_marker_genes"],
        "annotate_with_cima_markers": ["find_marker_genes"],
        "annotate_with_blood_markers": ["find_marker_genes"],
        "differential_expression": ["cluster_and_umap"],
        "generate_analysis_report": ["annotate_cells"],
    }

    # 工具结果路径在 dataset_entry 中的字段名
    PATH_FIELD_MAP = {
        "load_h5ad_data": "loaded_path",
        "calculate_qc_metrics": "qc_path",
        "normalize_and_hvg": "normalized_path",
        "pca_reduction": "pca_path",
        "cluster_and_umap": "clustered_path",
        "find_marker_genes": "markers_path",
        "annotate_cells": "annotated_path",
        "annotate_with_simple_markers": "annotated_path",
        "annotate_with_cima_markers": "annotated_path",
        "annotate_with_blood_markers": "annotated_path",
        "differential_expression": "de_path",
    }

    # 需要文件路径参数的工具
    FILE_PATH_TOOLS = {
        "load_h5ad_data",
        "calculate_qc_metrics",
        "normalize_and_hvg",
        "pca_reduction",
        "cluster_and_umap",
        "find_marker_genes",
        "annotate_cells",
        "differential_expression",
        "generate_analysis_report",
        "annotate_with_simple_markers",
        "annotate_with_cima_markers",
        "annotate_with_blood_markers",
    }

    def __init__(self):
        """初始化路径解析器"""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.runs_dir = Path(settings.RUNS_DIR)

    def resolve_input_path(
        self,
        file_path: str,
        state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """解析输入文件路径

        Args:
            file_path: 文件路径（可以是相对路径或绝对路径）
            state: Agent 状态（包含 input_files, work_dir 等）

        Returns:
            解析后的绝对路径
        """
        path = Path(file_path)

        # 如果已经是绝对路径且存在，直接返回
        if path.is_absolute() and path.exists():
            return path

        # 1. 尝试直接使用原始路径
        if path.exists():
            return path

        # 2. 尝试在上传目录中查找
        upload_path = self.upload_dir / file_path
        if upload_path.exists():
            return upload_path

        # 3. 尝试添加 .h5ad 扩展名
        if not file_path.endswith('.h5ad'):
            h5ad_path = self.upload_dir / f"{file_path}.h5ad"
            if h5ad_path.exists():
                return h5ad_path

        # 4. 尝试使用文件名
            path = self.upload_dir / Path(file_path).name
            if path.exists():
                return path

        # 5. 从 state 的 input_files 中查找
        if state:
            input_files = state.get("input_files", [])
            for f in input_files:
                f_path = Path(f)
                if f_path.exists():
                    return f_path

                # 尝试在上传目录中查找
                upload_f = self.upload_dir / Path(f).name
                if upload_f.exists():
                    return upload_f

        # 6. 从 state 的 work_dir 中查找最新的 h5ad 文件
        if state and state.get("work_dir"):
            work_dir = Path(state["work_dir"])
            if work_dir.exists():
                h5ad_files = list(work_dir.glob("**/*.h5ad"))
                if h5ad_files:
                    # 按修改时间排序，返回最新的
                    h5ad_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return h5ad_files[0]

        # 如果都找不到，返回原始路径（让调用者处理错误）
        logger.warning(f"无法找到文件: {file_path}")
        return Path(file_path)

    def get_latest_artifact(
        self,
        state: Dict[str, Any],
        tool_name: str
    ) -> Optional[str]:
        """获取工具应该使用的最新结果路径

        Args:
            state: Agent 状态
            tool_name: 工具名称

        Returns:
            最新结果文件路径，如果未找到返回 None
        """
        # 获取当前数据集条目
        dataset_entry = self._get_active_dataset_entry(state)
        if not dataset_entry:
            return None

        # 查找依赖工具的结果路径
        dependency_tools = self.TOOL_DEPENDENCY_MAP.get(tool_name, [])
        for dep_tool in reversed(dependency_tools):  # 反向查找，取最接近的
            field_name = self.PATH_FIELD_MAP.get(dep_tool)
            if field_name and field_name in dataset_entry:
                result_path = dataset_entry[field_name]
                if result_path:
                    logger.info(f"[{tool_name}] Using result from {dep_tool}: {result_path}")
                    return result_path

        # 如果没有找到依赖工具的结果，尝试使用最近修改的文件
        if state.get("work_dir"):
            work_dir = Path(state["work_dir"])
            artifacts_dir = work_dir / "artifacts" / "data"
            if artifacts_dir.exists():
                h5ad_files = sorted(
                    artifacts_dir.glob("*.h5ad"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if h5ad_files:
                    logger.info(f"[{tool_name}] Using latest file from artifacts: {h5ad_files[0]}")
                    return str(h5ad_files[0])

        return None

    def _get_active_dataset_entry(self, state: Dict[str, Any]) -> Optional[Dict]:
        """获取当前活动的数据集条目"""
        project_state = state.get("project_state") or {}
        datasets = project_state.get("datasets") or {}
        dataset_id = project_state.get("active_dataset") or project_state.get("last_dataset")

        if dataset_id and isinstance(datasets.get(dataset_id), dict):
            return datasets[dataset_id]

        return None

    def resolve_output_dir(
        self,
        run_id: str,
        artifact_type: str = "data"
    ) -> Path:
        """解析输出目录

        Args:
            run_id: 运行ID
            artifact_type: 产物类型 (data, tables, plots, reports)

        Returns:
            输出目录路径
        """
        runs_root = self.runs_dir.resolve()
        artifacts_dir = runs_root / run_id / "artifacts"

        if artifact_type == "data":
            output_dir = artifacts_dir / "data"
        elif artifact_type == "tables":
            output_dir = artifacts_dir / "tables"
        elif artifact_type == "plots":
            output_dir = artifacts_dir / "plots"
        elif artifact_type == "reports":
            output_dir = artifacts_dir / "reports"
        else:
            output_dir = artifacts_dir / artifact_type

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def resolve_all_output_dirs(self, input_path: Path, run_id: Optional[str] = None) -> Tuple[Path, Path, Path]:
        """根据输入路径解析所有产物目录（Job模式优先）

        Args:
            input_path: 输入文件路径
            run_id: 可选的运行ID（优先使用此值）

        Returns:
            (data_dir, tables_dir, plots_dir) 元组
        """
        runs_root = self.runs_dir.resolve()

        # 如果直接提供了 run_id，使用它
        if run_id:
            artifacts_dir = runs_root / run_id / "artifacts"
            data_dir = artifacts_dir / "data"
            tables_dir = artifacts_dir / "tables"
            plots_dir = artifacts_dir / "plots"
            for dir_path in (data_dir, tables_dir, plots_dir):
                dir_path.mkdir(parents=True, exist_ok=True)
            return data_dir, tables_dir, plots_dir

        # 尝试从输入路径推断 run_id
        try:
            if input_path.is_relative_to(runs_root):
                rel_path = input_path.relative_to(runs_root)
                # 检查是否在某个 run_id 目录下 (runs/{run_id}/...)
                if len(rel_path.parts) > 1 and rel_path.parts[0] not in ['artifacts', 'uploads']:
                    # 看起来像是一个有效的 run_id
                    potential_run_id = rel_path.parts[0]
                    artifacts_dir = runs_root / potential_run_id / "artifacts"
                    data_dir = artifacts_dir / "data"
                    tables_dir = artifacts_dir / "tables"
                    plots_dir = artifacts_dir / "plots"
                    for dir_path in (data_dir, tables_dir, plots_dir):
                        dir_path.mkdir(parents=True, exist_ok=True)
                    return data_dir, tables_dir, plots_dir

                # 文件直接在 runs 目录下，不是在 run_id 子目录中
                # 创建新的 run_id (使用当前时间戳)
                from datetime import datetime
                new_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                artifacts_dir = runs_root / new_run_id / "artifacts"
                data_dir = artifacts_dir / "data"
                tables_dir = artifacts_dir / "tables"
                plots_dir = artifacts_dir / "plots"
                for dir_path in (data_dir, tables_dir, plots_dir):
                    dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"输入文件直接在 runs 目录下，创建新的运行目录: {new_run_id}")
                return data_dir, tables_dir, plots_dir
        except ValueError:
            pass

        # 非 Job 模式，使用通用输出目录
        return OUTPUT_DIR, OUTPUT_DIR, OUTPUT_DIR

    def build_tool_args(
        self,
        state: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """根据工具名称构建工具调用参数

        Args:
            state: Agent 状态
            tool_name: 工具名称

        Returns:
            工具参数字典
        """
        tool_args = {}

        # 获取最新的文件路径（从project_state中获取上一步的结果）
        latest_file_path = self.get_latest_artifact(state, tool_name)

        # 从 state 获取文件路径
        if tool_name in self.FILE_PATH_TOOLS:
            if latest_file_path:
                tool_args["file_path"] = latest_file_path
            else:
                input_files = state.get("input_files", [])
                if input_files:
                    tool_args["file_path"] = input_files[0]

        # 添加 work_dir 参数
        if state.get("work_dir"):
            tool_args["work_dir"] = state["work_dir"]

        return tool_args


# 全局单例
_resolver: Optional[PathResolver] = None


def get_path_resolver() -> PathResolver:
    """获取全局路径解析器单例"""
    global _resolver
    if _resolver is None:
        _resolver = PathResolver()
    return _resolver


# 便捷函数
def resolve_input_path(
    file_path: str,
    state: Optional[Dict[str, Any]] = None
) -> Path:
    """解析输入文件路径"""
    return get_path_resolver().resolve_input_path(file_path, state)


def get_latest_artifact(
    state: Dict[str, Any],
    tool_name: str
) -> Optional[str]:
    """获取最新产物文件路径"""
    return get_path_resolver().get_latest_artifact(state, tool_name)


def resolve_output_dir(run_id: str, artifact_type: str = "data") -> Path:
    """解析输出目录"""
    return get_path_resolver().resolve_output_dir(run_id, artifact_type)


# 导出
__all__ = [
    "PathResolver",
    "get_path_resolver",
    "resolve_input_path",
    "get_latest_artifact",
    "resolve_output_dir",
    "FILE_PATH_TOOLS",
    "TOOL_DEPENDENCY_MAP",
]
