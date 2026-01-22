"""Agent 工具注册表

管理所有可用的生物信息学工具，为 Agent 提供行动空间。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Callable, Any, Optional
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

# 导入单细胞核心工具
from src.tools.single_cell_core import (
    load_h5ad_data,
    calculate_qc_metrics,
    normalize_and_hvg,
    pca_reduction,
    cluster_and_umap,
    find_marker_genes,
    annotate_cells,
    differential_expression,
    generate_analysis_report,
)
try:
    from src.tools.extract_embeddings_scgpt_client import extract_embeddings_with_scgpt
except Exception as exc:  # pragma: no cover - optional dependency
    extract_embeddings_with_scgpt = None
    logger.warning("extract_embeddings_with_scgpt unavailable: %s", exc)

# 导入高级分析工具（可选依赖）
try:
    from src.tools.cellphoneDB import run_cellphonedb_core
except Exception as exc:  # pragma: no cover - optional dependency
    run_cellphonedb_core = None
    logger.warning("run_cellphonedb_core unavailable: %s", exc)

try:
    from src.tools.pseudotime_analysis import run_pseudotime_analysis
except Exception as exc:  # pragma: no cover - optional dependency
    run_pseudotime_analysis = None
    logger.warning("run_pseudotime_analysis unavailable: %s", exc)

try:
    from src.tools.enrichment_analysis.ora import run_ora_enrichment
except Exception as exc:  # pragma: no cover - optional dependency
    run_ora_enrichment = None
    logger.warning("run_ora_enrichment unavailable: %s", exc)

try:
    from src.tools.report_generator import generate_comprehensive_report
except Exception as exc:  # pragma: no cover - optional dependency
    generate_comprehensive_report = None
    logger.warning("generate_comprehensive_report unavailable: %s", exc)


class ToolRegistry:
    """工具注册表

    管理所有可用工具，提供：
    1. 工具描述给 LLM
    2. 工具执行能力
    3. 工具依赖关系
    4. 工具结果缓存
    """

    # 工具依赖关系（某些工具需要先运行其他工具）
    TOOL_DEPENDENCIES = {
        "cluster_and_umap": ["normalize_and_hvg", "pca_reduction"],
        "find_marker_genes": ["cluster_and_umap"],
        "annotate_cells": ["find_marker_genes"],
        "differential_expression": ["cluster_and_umap"],
        "generate_analysis_report": [],  # 不依赖其他工具
    }

    # 推荐的分析流程
    RECOMMENDED_WORKFLOW = [
        "load_h5ad_data",
        "calculate_qc_metrics",
        "normalize_and_hvg",
        "pca_reduction",
        "cluster_and_umap",
        "find_marker_genes",
        "annotate_cells",
        "generate_analysis_report",
    ]

    # 工具分类
    TOOL_CATEGORIES = {
        "data_loading": ["load_h5ad_data"],
        "quality_control": ["calculate_qc_metrics"],
        "preprocessing": ["normalize_and_hvg", "pca_reduction"],
        "clustering": ["cluster_and_umap"],
        "annotation": ["find_marker_genes", "annotate_cells"],
        "analysis": ["differential_expression"],
        "advanced": ["extract_embeddings_with_scgpt"],
        "communication": ["run_cellphonedb_core"],
        "trajectory": ["run_pseudotime_analysis"],
        "enrichment": ["run_ora_enrichment"],
        "reporting": ["generate_analysis_report", "generate_comprehensive_report"],
    }

    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_descriptions: Dict[str, str] = {}
        self._register_all_tools()

    def _register_all_tools(self):
        """注册所有工具"""
        # 单细胞核心工具
        self.register(load_h5ad_data)
        self.register(calculate_qc_metrics)
        self.register(normalize_and_hvg)
        self.register(pca_reduction)
        self.register(cluster_and_umap)
        self.register(find_marker_genes)
        self.register(annotate_cells)
        self.register(differential_expression)
        self.register(generate_analysis_report)

        # 高级工具（可选）
        if extract_embeddings_with_scgpt is not None:
            self.register(extract_embeddings_with_scgpt)

        # 细胞通讯分析工具
        if run_cellphonedb_core is not None:
            self.register(run_cellphonedb_core)

        # 伪时间轨迹分析工具
        if run_pseudotime_analysis is not None:
            self.register(run_pseudotime_analysis)

        # 富集分析工具
        if run_ora_enrichment is not None:
            self.register(run_ora_enrichment)

        # 综合报告生成工具
        if generate_comprehensive_report is not None:
            self.register(generate_comprehensive_report)

        logger.info(f"工具注册完成: {len(self._tools)} 个工具")

    def register(self, tool: BaseTool) -> None:
        """注册工具

        Args:
            tool: LangChain BaseTool 实例
        """
        tool_name = tool.name
        self._tools[tool_name] = tool
        self._tool_descriptions[tool_name] = tool.description

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具

        Args:
            name: 工具名称

        Returns:
            工具实例，如果不存在返回 None
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有工具

        Returns:
            所有工具列表
        """
        return list(self._tools.values())

    def get_tool_description_for_llm(self) -> str:
        """获取工具描述（供 LLM 使用）

        Returns:
            工具描述字符串
        """
        descriptions = []
        for name, desc in self._tool_descriptions.items():
            category = self._get_tool_category(name)
            descriptions.append(f"- **{name}** [{category}]: {desc}")

        # 添加推荐流程
        workflow_desc = "\\n\\n**推荐分析流程**:\\n"
        workflow_desc += " -> ".join(self.RECOMMENDED_WORKFLOW)

        # 添加依赖说明
        deps_desc = "\\n\\n**工具依赖关系**:\\n"
        for tool, deps in self.TOOL_DEPENDENCIES.items():
            if deps:
                deps_str = ", ".join(deps)
                deps_desc += f"- {tool} 需要: {deps_str}\\n"

        return "## 可用工具\\n\\n" + "\\n".join(descriptions) + workflow_desc + deps_desc

    def _get_tool_category(self, tool_name: str) -> str:
        """获取工具分类"""
        for category, tools in self.TOOL_CATEGORIES.items():
            if tool_name in tools:
                return category
        return "other"

    def get_required_tools(self, tool_name: str) -> List[str]:
        """获取工具的前置依赖

        Args:
            tool_name: 工具名称

        Returns:
            依赖的工具列表
        """
        return self.TOOL_DEPENDENCIES.get(tool_name, [])

    def get_workflow_for_intent(self, intent: str) -> List[str]:
        """根据意图获取推荐工作流

        Args:
            intent: 用户意图

        Returns:
            推荐的工具列表
        """
        # 基础工作流（包含所有必要步骤）
        base_workflow = [
            "load_h5ad_data",
            "calculate_qc_metrics",
            "normalize_and_hvg",
            "pca_reduction",
            "cluster_and_umap",
            "find_marker_genes",
            "generate_analysis_report",
        ]

        # 根据意图调整工作流
        intent_workflows = {
            "full_analysis": self.RECOMMENDED_WORKFLOW,
            "clustering_analysis": [
                "load_h5ad_data",
                "calculate_qc_metrics",
                "normalize_and_hvg",
                "pca_reduction",
                "cluster_and_umap",
                "generate_analysis_report",
            ],
            "cell_annotation": [
                "load_h5ad_data",
                "calculate_qc_metrics",
                "normalize_and_hvg",
                "pca_reduction",
                "cluster_and_umap",
                "find_marker_genes",
                "annotate_cells",
                "generate_analysis_report",
            ],
            "marker_gene_analysis": [
                "load_h5ad_data",
                "cluster_and_umap",
                "find_marker_genes",
                "generate_analysis_report",
            ],
            "differential_expression": [
                "load_h5ad_data",
                "cluster_and_umap",
                "differential_expression",
                "generate_analysis_report",
            ],
            "quality_control": [
                "load_h5ad_data",
                "calculate_qc_metrics",
                "generate_analysis_report",
            ],
        }

        return intent_workflows.get(intent, base_workflow)

    def validate_plan(self, plan: List[str]) -> Dict[str, Any]:
        """验证计划的有效性

        Args:
            plan: 计划的工具列表

        Returns:
            验证结果
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_deps": {},
        }

        # 检查工具是否存在
        for tool in plan:
            if tool not in self._tools:
                result["valid"] = False
                result["errors"].append(f"工具不存在: {tool}")

        # 检查依赖关系
        executed = set()
        for tool in plan:
            deps = self.get_required_tools(tool)
            missing = [d for d in deps if d not in executed]
            if missing:
                result["warnings"].append(f"{tool} 缺少前置步骤: {missing}")
                result["missing_deps"][tool] = missing
            executed.add(tool)

        return result


# 全局工具注册表实例
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表实例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_available_tools() -> List[BaseTool]:
    """获取所有可用工具（供 LangChain 使用）"""
    return get_tool_registry().get_all_tools()


def get_tools_description() -> str:
    """获取工具描述（供 LLM prompt 使用）"""
    return get_tool_registry().get_tool_description_for_llm()


# 向后兼容的导出
TOOLS = get_available_tools()


# 便捷函数：获取工具注册器和持久化管理器
def get_agent_components():
    """获取Agent所需的核心组件"""
    from src.utils.result_persistence import get_persistence_manager

    return {
        "tool_registry": get_tool_registry(),
        "persistence_manager": get_persistence_manager(),
        "tools": get_available_tools(),
    }
