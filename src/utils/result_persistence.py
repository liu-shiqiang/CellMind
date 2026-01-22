"""工具结果持久化管理器

负责管理Agent执行过程中产生的中间结果和最终报告。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionRecord:
    """工具执行记录"""
    tool_name: str
    timestamp: str
    status: str  # success, error, skipped
    input_params: Dict[str, Any]
    output_summary: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    result_path: Optional[str] = None


class ResultPersistenceManager:
    """结果持久化管理器

    功能：
    1. 保存工具执行记录
    2. 缓存中间结果（避免重复计算）
    3. 生成分析报告
    4. 导出结果供前端展示
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """初始化管理器

        Args:
            base_dir: 基础存储目录
        """
        if base_dir is None:
            from src.web.config import settings
            base_dir = Path(settings.RUNS_DIR)

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 全局缓存目录（跨run复用）
        self.cache_dir = self.base_dir / "_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, run_id: str) -> Path:
        """获取运行目录"""
        run_dir = self.base_dir / run_id / "artifacts"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_run_subdir(self, run_id: str, name: str) -> Path:
        """获取运行子目录"""
        subdir = self._get_run_dir(run_id) / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def _get_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 对参数进行排序后生成hash
        param_str = json.dumps(params, sort_keys=True, default=str)
        content = f"{tool_name}:{param_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def save_execution_record(
        self,
        run_id: str,
        record: ToolExecutionRecord,
    ) -> str:
        """保存工具执行记录

        Args:
            run_id: Agent运行ID
            record: 执行记录

        Returns:
            记录文件路径
        """
        record_dir = self._get_run_subdir(run_id, "records")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        record_path = record_dir / f"record_{record.tool_name}_{timestamp}.json"

        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"保存执行记录: {record_path}")

        return str(record_path)

    def save_tool_result(
        self,
        run_id: str,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        execution_time: Optional[float] = None,
    ) -> str:
        """保存工具结果

        Args:
            run_id: Agent运行ID
            tool_name: 工具名称
            params: 输入参数
            result: 输出结果
            execution_time: 执行时间（秒）

        Returns:
            结果文件路径
        """
        record = ToolExecutionRecord(
            tool_name=tool_name,
            timestamp=datetime.now().isoformat(),
            status=result.get("status", "unknown"),
            input_params=params,
            output_summary=result,
            error_message=result.get("message") if result.get("status") == "error" else None,
            execution_time=execution_time,
            result_path=result.get("result_path"),
        )

        return self.save_execution_record(run_id, record)

    def get_cached_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的结果

        Args:
            tool_name: 工具名称
            params: 输入参数

        Returns:
            缓存的结果，如果不存在返回None
        """
        cache_key = self._get_cache_key(tool_name, params)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)

                logger.info(f"缓存命中: {tool_name} -> {cache_key}")
                return cached
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")

        return None

    def save_cache(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """保存结果到缓存

        Args:
            tool_name: 工具名称
            params: 输入参数
            result: 输出结果
        """
        cache_key = self._get_cache_key(tool_name, params)
        cache_path = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "tool_name": tool_name,
            "params": params,
            "result": result,
            "cached_at": datetime.now().isoformat(),
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"保存缓存: {tool_name} -> {cache_key}")

    def generate_analysis_report(
        self,
        run_id: str,
        objective: str,
        records: List[ToolExecutionRecord],
        final_summary: str,
    ) -> str:
        """生成分析报告

        Args:
            run_id: Agent运行ID
            objective: 分析目标
            records: 执行记录列表
            final_summary: 最终摘要

        Returns:
            报告文件路径
        """
        report_dir = self._get_run_subdir(run_id, "reports")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"report_{timestamp}.md"

        # 构建报告内容
        report_lines = [
            "# 单细胞数据分析报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**运行ID**: {run_id}",
            "",
            "## 分析目标",
            "",
            objective,
            "",
            "## 执行步骤",
            "",
        ]

        # 添加执行步骤
        for i, record in enumerate(records, 1):
            status_icon = {
                "success": "✅",
                "error": "❌",
                "skipped": "⏭️",
            }.get(record.status, "❓")

            report_lines.append(f"### {i}. {record.tool_name} {status_icon}")
            report_lines.append("")

            if record.input_params:
                report_lines.append("**输入参数**:")
                for key, value in record.input_params.items():
                    report_lines.append(f"- `{key}`: {value}")
                report_lines.append("")

            if record.status == "error" and record.error_message:
                report_lines.append(f"**错误**: {record.error_message}")
                report_lines.append("")
            elif record.output_summary:
                # 提取关键信息
                summary = record.output_summary
                if "message" in summary:
                    report_lines.append(f"**结果**: {summary['message']}")
                if "n_clusters" in summary:
                    report_lines.append(f"**聚类数**: {summary['n_clusters']}")
                if "n_cells" in summary:
                    report_lines.append(f"**细胞数**: {summary['n_cells']}")
                report_lines.append("")

            if record.result_path:
                report_lines.append(f"**结果文件**: `{record.result_path}`")
                report_lines.append("")

        # 添加最终摘要
        report_lines.extend([
            "## 分析摘要",
            "",
            final_summary,
            "",
            "## 数据文件",
            "",
        ])

        # 列出所有生成的文件
        run_dir = self._get_run_dir(run_id)
        if run_dir.exists():
            for file_path in run_dir.glob("*.h5ad"):
                report_lines.append(f"- `{file_path.name}`")
            for file_path in run_dir.glob("*.csv"):
                report_lines.append(f"- `{file_path.name}`")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*本报告由 CellMind 自动生成*")

        # 写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"生成报告: {report_path}")

        return str(report_path)

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """获取运行摘要

        Args:
            run_id: Agent运行ID

        Returns:
            运行摘要
        """
        run_dir = self._get_run_dir(run_id)

        if not run_dir.exists():
            return {
                "run_id": run_id,
                "status": "not_found",
                "records": [],
            }

        # 读取所有记录
        records = []
        for record_path in run_dir.glob("record_*.json"):
            try:
                with open(record_path, "r", encoding="utf-8") as f:
                    record_data = json.load(f)
                    records.append(ToolExecutionRecord(**record_data))
            except Exception as e:
                logger.warning(f"读取记录失败 {record_path}: {e}")

        # 统计
        n_success = sum(1 for r in records if r.status == "success")
        n_error = sum(1 for r in records if r.status == "error")
        n_skipped = sum(1 for r in records if r.status == "skipped")

        return {
            "run_id": run_id,
            "status": "completed" if records else "empty",
            "n_records": len(records),
            "n_success": n_success,
            "n_error": n_error,
            "n_skipped": n_skipped,
            "records": [
                {
                    "tool_name": r.tool_name,
                    "timestamp": r.timestamp,
                    "status": r.status,
                    "result_path": r.result_path,
                }
                for r in records
            ],
        }

    def export_for_frontend(self, run_id: str) -> Dict[str, Any]:
        """导出适合前端展示的结果

        Args:
            run_id: Agent运行ID

        Returns:
            前端可用的结果数据
        """
        summary = self.get_run_summary(run_id)

        # 提取分析结果
        analysis_results = {
            "run_id": run_id,
            "status": summary["status"],
            "tools_executed": [],
            "clusters": {},
            "markers": {},
            "annotations": {},
            "plots": [],
        }

        for record in summary["records"]:
            tool_result = {
                "name": record["tool_name"],
                "status": record["status"],
                "timestamp": record["timestamp"],
            }

            # 根据工具类型提取特定结果
            tool_name = record["tool_name"]

            if tool_name == "cluster_and_umap":
                # 聚类结果
                record_path = record.get("result_path")
                if record_path:
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(record_path)

                        # 提取聚类信息
                        for key in ['leiden', 'louvain', 'clusters', 'cluster']:
                            if key in adata.obs.columns:
                                cluster_counts = adata.obs[key].value_counts().to_dict()
                                analysis_results["clusters"] = {
                                    "key": key,
                                    "counts": cluster_counts,
                                    "n_clusters": len(cluster_counts),
                                }
                                break

                        # 提取UMAP坐标
                        if "X_umap" in adata.obsm:
                            umap_coords = adata.obsm["X_umap"].tolist()
                            analysis_results["umap"] = {
                                "coordinates": umap_coords[:100],  # 限制数量
                            }
                    except Exception as e:
                        logger.warning(f"读取聚类结果失败: {e}")

            elif tool_name == "find_marker_genes":
                # 标记基因结果
                record_path = record.get("result_path")
                if record_path:
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(record_path)

                        if "rank_genes_groups" in adata.uns:
                            # 提取每个cluster的top标记基因
                            groups = adata.uns["rank_genes_groups"]["names"].dtype.names
                            for group in groups:
                                genes_df = sc.get.rank_genes_groups_df(adata, group=group, n_genes=10)
                                analysis_results["markers"][group] = {
                                    "top_genes": genes_df["names"].tolist(),
                                    "logfc": genes_df.get("logfoldchanges", []).tolist(),
                                }
                    except Exception as e:
                        logger.warning(f"读取标记基因结果失败: {e}")

            elif tool_name == "annotate_cells":
                # 细胞注释结果
                record_path = record.get("result_path")
                if record_path:
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(record_path)

                        if "cell_type" in adata.obs.columns:
                            cell_type_counts = adata.obs["cell_type"].value_counts().to_dict()
                            analysis_results["annotations"] = {
                                "counts": cell_type_counts,
                                "n_types": len(cell_type_counts),
                            }
                    except Exception as e:
                        logger.warning(f"读取注释结果失败: {e}")

            analysis_results["tools_executed"].append(tool_result)

        return analysis_results


# 全局实例
_persistence_manager: Optional[ResultPersistenceManager] = None


def get_persistence_manager() -> ResultPersistenceManager:
    """获取持久化管理器实例"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = ResultPersistenceManager()
    return _persistence_manager
