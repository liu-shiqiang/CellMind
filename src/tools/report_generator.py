"""综合分析报告生成器

生成包含所有分析结果的综合报告，支持多种格式：
- Markdown (.md)
- HTML (.html)
- PDF (.pdf)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSection:
    """分析报告的一个章节"""
    title: str
    content: str
    status: str = "success"  # success, warning, error
    data: Dict[str, Any] = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)  # base64 encoded plots


@dataclass
class ReportMetadata:
    """报告元数据"""
    run_id: str
    timestamp: str
    data_file: str
    n_cells: int
    n_genes: int
    objective: str = ""


class ReportGenerator:
    """综合分析报告生成器

    整合所有分析工具的结果，生成包含以下内容的综合报告：
    1. 数据概览
    2. 质控分析结果
    3. 降维分析 (PCA, UMAP)
    4. 聚类分析
    5. 标记基因
    6. 细胞类型注释
    7. 差异表达分析
    8. 细胞通讯分析 (CellPhoneDB)
    9. 轨迹分析
    10. 富集分析
    11. 总结与建议
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        if output_dir is None:
            from src.web.config import settings
            output_dir = Path(settings.RUNS_DIR) / "reports"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sections: List[AnalysisSection] = []
        self.metadata: Optional[ReportMetadata] = None

    def add_section(self, section: AnalysisSection) -> None:
        """添加报告章节"""
        self.sections.append(section)

    def set_metadata(self, metadata: ReportMetadata) -> None:
        """设置报告元数据"""
        self.metadata = metadata

    def generate_markdown(self) -> str:
        """生成 Markdown 格式报告

        Returns:
            报告内容
        """
        if not self.metadata:
            raise ValueError("报告元数据未设置，请先调用 set_metadata()")

        lines = []

        # 标题和元数据
        lines.extend([
            "# 单细胞数据分析综合报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**运行ID**: {self.metadata.run_id}",
            f"**数据文件**: {self.metadata.data_file}",
            "",
            "---",
            "",
        ])

        # 目录
        lines.extend([
            "## 目录",
            "",
        ])
        for i, section in enumerate(self.sections, 1):
            status_icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(section.status, "")
            lines.append(f"{i}. [{section.title}](#{section.title.lower().replace(' ', '-')}) {status_icon}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 各章节内容
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")

            # 状态标识
            if section.status == "error":
                lines.append(f"> **注意**: 此步骤执行失败")
                lines.append("")
            elif section.status == "warning":
                lines.append(f"> **警告**: 此步骤有潜在问题")
                lines.append("")

            # 章节内容
            lines.append(section.content)
            lines.append("")

            # 数据表格
            if section.data:
                lines.extend(self._format_data_as_markdown(section.data))
                lines.append("")

            # 图片
            if section.plots:
                for plot_path in section.plots:
                    lines.append(f"![图表]({plot_path}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # 总结
        lines.extend([
            "## 总结与建议",
            "",
            self._generate_summary(),
            "",
            "---",
            "",
            "*本报告由 [CellMind](https://github.com/your-repo) 自动生成*",
        ])

        return "\n".join(lines)

    def generate_html(self) -> str:
        """生成 HTML 格式报告

        Returns:
            HTML 内容
        """
        md_content = self.generate_markdown()

        # 转换为 HTML (使用内联样式)
        html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>单细胞数据分析报告 - {run_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .success-icon::before {{ content: "✅ "; }}
        .warning-icon::before {{ content: "⚠️ "; }}
        .error-icon::before {{ content: "❌ "; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .alert {{
            padding: 15px;
            border-left: 4px solid;
            margin: 15px 0;
            background: #f8f9fa;
        }}
        .alert-error {{ border-color: #e74c3c; background: #fadbd8; }}
        .alert-warning {{ border-color: #f39c12; background: #fef5e7; }}
        .alert-success {{ border-color: #27ae60; background: #d5f4e6; }}
        .section-card {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .toc a {{
            text-decoration: none;
            color: #3498db;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .metric {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-size: 14px;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>单细胞数据分析综合报告</h1>
        <div class="metadata">
            <p><strong>生成时间</strong>: {timestamp}</p>
            <p><strong>运行ID</strong>: {run_id}</p>
            <p><strong>数据文件</strong>: {data_file}</p>
            <p><strong>细胞数量</strong>: {n_cells:,}</p>
            <p><strong>基因数量</strong>: {n_genes:,}</p>
        </div>

        {content}

        <div class="footer">
            <p>本报告由 <strong>CellMind</strong> 自动生成</p>
            <p>{timestamp}</p>
        </div>
    </div>
</body>
</html>"""

        # 将 Markdown 内容转换为简化的 HTML
        html_content = self._markdown_to_html(md_content)

        return html_template.format(
            run_id=self.metadata.run_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data_file=self.metadata.data_file,
            n_cells=self.metadata.n_cells,
            n_genes=self.metadata.n_genes,
            content=html_content
        )

    def save_report(self, format: str = "both") -> Tuple[str, ...]:
        """保存报告到文件

        Args:
            format: 输出格式 ("md", "html", "both")

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = self.metadata.run_id if self.metadata else "unknown"

        paths = []

        if format in ("md", "both"):
            md_path = self.output_dir / f"{run_id}_report_{timestamp}.md"
            md_content = self.generate_markdown()
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            paths.append(str(md_path))
            logger.info(f"Markdown 报告已保存: {md_path}")

        if format in ("html", "both"):
            html_path = self.output_dir / f"{run_id}_report_{timestamp}.html"
            html_content = self.generate_html()
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            paths.append(str(html_path))
            logger.info(f"HTML 报告已保存: {html_path}")

        return tuple(paths)

    def _format_data_as_markdown(self, data: Dict[str, Any]) -> List[str]:
        """将数据格式化为 Markdown 表格"""
        lines = []

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"### {key}")
                lines.append("")
                lines.append("| 键 | 值 |")
                lines.append("|-----|-----|")
                for k, v in value.items():
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            elif isinstance(value, (list, tuple)):
                lines.append(f"### {key}")
                lines.append("")
                if value and isinstance(value[0], dict):
                    # 列表包含字典，转为表格
                    headers = list(value[0].keys())
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("|" + "|".join(["----"] * len(headers)) + "|")
                    for item in value[:20]:  # 限制行数
                        lines.append("| " + " | ".join(str(item.get(h, "")) for h in headers) + " |")
                    if len(value) > 20:
                        lines.append(f"| ... ({len(value) - 20} more rows) |")
                else:
                    for item in value[:50]:
                        lines.append(f"- {item}")
                    if len(value) > 50:
                        lines.append(f"- ... ({len(value) - 50} more items)")
                lines.append("")
            elif isinstance(value, (int, float)):
                if isinstance(value, int):
                    formatted = f"{value:,}"
                else:
                    formatted = f"{value:.4f}"
                lines.append(f"- **{key}**: {formatted}")
            else:
                lines.append(f"- **{key}**: {value}")

        return lines

    def _markdown_to_html(self, md_content: str) -> str:
        """将 Markdown 转换为简化的 HTML

        这是一个简化的转换器，处理常用语法
        """
        html = md_content

        # 转换标题
        import re
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # 转换粗体
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # 转换代码
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # 转换链接
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        # 转换水平线
        html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)

        # 转换段落
        paragraphs = html.split('\n\n')
        html_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                if p.startswith('> '):
                    # 引用块
                    p = f'<div class="alert alert-success">{p[2:]}</div>'
                elif p.startswith('- '):
                    # 列表
                    items = [li.strip().replace('- ', '') for li in p.split('\n') if li.strip().startswith('- ')]
                    p = '<ul>' + ''.join(f'<li>{li}</li>' for li in items) + '</ul>'
                else:
                    p = f'<p>{p}</p>'
            html_paragraphs.append(p)

        html = '\n'.join(html_paragraphs)

        return html

    def _generate_summary(self) -> str:
        """生成分析总结"""
        if not self.sections:
            return "暂无分析结果。"

        lines = [
            "### 分析概览",
            "",
            f"- **总分析步骤**: {len(self.sections)}",
            f"- **成功完成**: {sum(1 for s in self.sections if s.status == 'success')}",
            f"- **警告**: {sum(1 for s in self.sections if s.status == 'warning')}",
            f"- **失败**: {sum(1 for s in self.sections if s.status == 'error')}",
            "",
        ]

        # 根据分析内容生成具体总结
        for section in self.sections:
            if "质控" in section.title:
                lines.extend(self._summarize_qc(section))
            elif "聚类" in section.title:
                lines.extend(self._summarize_clustering(section))
            elif "标记基因" in section.title:
                lines.extend(self._summarize_markers(section))
            elif "细胞类型" in section.title:
                lines.extend(self._summarize_annotation(section))

        return "\n".join(lines)

    def _summarize_qc(self, section: AnalysisSection) -> List[str]:
        """总结质控结果"""
        lines = [
            "### 质控分析",
            "",
        ]
        if "n_cells_after_qc" in section.data:
            lines.append(f"- 过滤后细胞数: {section.data['n_cells_after_qc']:,}")
        if "n_genes_after_qc" in section.data:
            lines.append(f"- 过滤后基因数: {section.data['n_genes_after_qc']:,}")
        lines.append("")
        return lines

    def _summarize_clustering(self, section: AnalysisSection) -> List[str]:
        """总结聚类结果"""
        lines = [
            "### 聚类分析",
            "",
        ]
        if "n_clusters" in section.data:
            lines.append(f"- 识别到 {section.data['n_clusters']} 个细胞簇")
        if "cluster_sizes" in section.data:
            sizes = section.data["cluster_sizes"]
            if isinstance(sizes, dict):
                lines.append(f"- 最大簇: {max(sizes.values()):,} 细胞")
                lines.append(f"- 最小簇: {min(sizes.values()):,} 细胞")
        lines.append("")
        return lines

    def _summarize_markers(self, section: AnalysisSection) -> List[str]:
        """总结标记基因"""
        lines = [
            "### 标记基因",
            "",
        ]
        if "n_clusters_with_markers" in section.data:
            lines.append(f"- {section.data['n_clusters_with_markers']} 个簇找到标记基因")
        lines.append("")
        return lines

    def _summarize_annotation(self, section: AnalysisSection) -> List[str]:
        """总结细胞注释"""
        lines = [
            "### 细胞类型注释",
            "",
        ]
        if "n_cell_types" in section.data:
            lines.append(f"- 识别出 {section.data['n_cell_types']} 种细胞类型")
        if "cell_types" in section.data:
            types = section.data["cell_types"]
            if isinstance(types, (list, dict)):
                lines.append(f"- 细胞类型: {', '.join(list(types)[:10])}")
        lines.append("")
        return lines


def create_report_from_results(
    run_id: str,
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Tuple[str, ...]:
    """从分析结果创建报告

    Args:
        run_id: 运行ID
        results: 包含所有工具执行结果的字典
        output_dir: 输出目录

    Returns:
        生成的报告文件路径
    """
    generator = ReportGenerator(output_dir)

    # 设置元数据
    metadata = ReportMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        data_file=results.get("data_file", "unknown"),
        n_cells=results.get("n_cells", 0),
        n_genes=results.get("n_genes", 0),
        objective=results.get("objective", ""),
    )
    generator.set_metadata(metadata)

    # 添加各分析章节
    for tool_name, tool_result in results.get("tools", {}).items():
        section = _create_section_from_tool_result(tool_name, tool_result)
        if section:
            generator.add_section(section)

    # 保存报告
    return generator.save_report(format="both")


def _create_section_from_tool_result(tool_name: str, result: Dict[str, Any]) -> Optional[AnalysisSection]:
    """从工具结果创建报告章节"""
    sections_map = {
        "load_h5ad_data": ("数据加载", "成功加载数据"),
        "calculate_qc_metrics": ("质量控制分析", "质控指标统计"),
        "normalize_and_hvg": ("标准化与高变基因", "数据标准化完成"),
        "pca_reduction": ("PCA降维分析", "主成分分析结果"),
        "cluster_and_umap": ("聚类与UMAP分析", "细胞聚类结果"),
        "find_marker_genes": ("标记基因鉴定", "差异表达基因分析"),
        "annotate_cells": ("细胞类型注释", "自动注释结果"),
        "differential_expression": ("差异表达分析", "组间差异基因"),
        "run_cellphonedb_core": ("细胞通讯分析", "CellPhoneDB 分析结果"),
        "run_pseudotime_analysis": ("伪时间轨迹分析", "发育轨迹推断"),
        "run_ora_enrichment": ("富集分析", "ORA 分析结果"),
        "generate_analysis_report": ("分析报告", "综合分析摘要"),
    }

    if tool_name not in sections_map:
        return None

    title, default_desc = sections_map[tool_name]

    status = "success"
    if result.get("status") == "error":
        status = "error"
    elif "warning" in result.get("message", "").lower():
        status = "warning"

    # 构建内容
    content = result.get("message", default_desc)

    # 提取数据
    data = {}
    if result.get("status") == "success":
        if tool_name == "calculate_qc_metrics":
            data = {
                "过滤前细胞数": result.get("n_cells_before", 0),
                "过滤后细胞数": result.get("n_cells_after", 0),
                "过滤前基因数": result.get("n_genes_before", 0),
                "过滤后基因数": result.get("n_genes_after", 0),
            }
        elif tool_name == "cluster_and_umap":
            data = {
                "聚类数量": result.get("n_clusters", 0),
                "聚类分辨率": result.get("resolution", 0),
                "UMAP维度": result.get("umap_shape", []),
            }
        elif tool_name == "find_marker_genes":
            data = {
                "分析聚类数": result.get("n_clusters_tested", 0),
                "每簇标记基因数": result.get("n_genes", 0),
            }
        elif tool_name == "annotate_cells":
            data = {
                "识别细胞类型数": result.get("n_cell_types", 0),
                "注释方法": result.get("method", "unknown"),
            }

    return AnalysisSection(
        title=title,
        content=content,
        status=status,
        data=data,
    )


# LangChain 工具
from langchain_core.tools import tool


@tool("generate_comprehensive_report")
def generate_comprehensive_report(
    run_id: str,
    data_file: str,
    n_cells: int,
    n_genes: int,
    analysis_results: str,
    output_format: str = "both",
) -> str:
    """生成综合分析报告

    整合所有分析结果，生成包含以下内容的综合报告：
    - 数据概览
    - 质控分析
    - 降维分析 (PCA, UMAP)
    - 聚类分析
    - 标记基因
    - 细胞类型注释
    - 差异表达分析
    - 细胞通讯分析
    - 轨迹分析
    - 富集分析
    - 总结与建议

    Args:
        run_id: 分析运行ID
        data_file: 原始数据文件路径
        n_cells: 细胞数量
        n_genes: 基因数量
        analysis_results: 分析结果JSON字符串
        output_format: 输出格式 ("md", "html", "both")

    Returns:
        生成的报告文件路径
    """
    try:
        from src.web.config import settings

        # 解析分析结果
        results = json.loads(analysis_results) if isinstance(analysis_results, str) else analysis_results

        # 创建报告
        output_dir = Path(settings.RUNS_DIR) / run_id / "artifacts" / "reports"
        paths = create_report_from_results(
            run_id=run_id,
            results={
                "data_file": data_file,
                "n_cells": n_cells,
                "n_genes": n_genes,
                **results,
            },
            output_dir=output_dir,
        )

        return json.dumps({
            "status": "success",
            "report_paths": list(paths),
            "message": f"报告生成成功，共 {len(paths)} 个文件",
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return json.dumps({
            "status": "error",
            "message": f"报告生成失败: {str(e)}",
        }, ensure_ascii=False)
