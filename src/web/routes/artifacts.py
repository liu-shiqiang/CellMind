"""分析产物API路由

提供访问分析过程中生成的图表、表格等产物的API。
"""
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from src.web.config import settings

router = APIRouter(tags=["Artifacts"])


def _validate_run_id(run_id: str) -> Path:
    """验证run_id并返回对应的运行目录

    Args:
        run_id: 运行ID

    Returns:
        运行目录路径

    Raises:
        HTTPException: 如果run_id无效
    """
    runs_dir = Path(settings.RUNS_DIR).resolve()
    run_dir = runs_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return run_dir


@router.get("/api/artifacts/{run_id}/plots/{filename}")
async def get_plot_file(run_id: str, filename: str):
    """获取分析生成的图表文件

    Args:
        run_id: 运行ID
        filename: 图表文件名

    Returns:
        图表文件
    """
    run_dir = _validate_run_id(run_id)
    plot_path = run_dir / "artifacts" / "plots" / filename

    if not plot_path.exists():
        # 尝试不区分大小写查找
        plots_dir = run_dir / "artifacts" / "plots"
        if plots_dir.exists():
            for existing_file in plots_dir.iterdir():
                if existing_file.name.lower() == filename.lower():
                    plot_path = existing_file
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
        else:
            raise HTTPException(status_code=404, detail=f"Plots directory not found for run {run_id}")

    # 根据文件扩展名设置MIME类型
    media_type = "image/png"
    if filename.endswith(".svg"):
        media_type = "image/svg+xml"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".pdf"):
        media_type = "application/pdf"

    return FileResponse(plot_path, media_type=media_type)


@router.get("/api/artifacts/{run_id}/tables/{filename}")
async def get_table_file(run_id: str, filename: str):
    """获取分析生成的表格文件

    Args:
        run_id: 运行ID
        filename: 表格文件名

    Returns:
        表格文件
    """
    run_dir = _validate_run_id(run_id)
    table_path = run_dir / "artifacts" / "tables" / filename

    if not table_path.exists():
        raise HTTPException(status_code=404, detail=f"Table {filename} not found")

    # 根据文件扩展名设置MIME类型
    media_type = "text/csv"
    if filename.endswith(".json"):
        media_type = "application/json"
    elif filename.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif filename.endswith(".tsv"):
        media_type = "text/tab-separated-values"

    return FileResponse(table_path, media_type=media_type)


@router.get("/api/artifacts/{run_id}/reports/{filename}")
async def get_report_file(run_id: str, filename: str):
    """获取分析报告文件

    Args:
        run_id: 运行ID
        filename: 报告文件名

    Returns:
        报告文件
    """
    run_dir = _validate_run_id(run_id)
    report_path = run_dir / "artifacts" / "reports" / filename

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {filename} not found")

    # 根据文件扩展名设置MIME类型
    media_type = "text/markdown"
    if filename.endswith(".json"):
        media_type = "application/json"
    elif filename.endswith(".html"):
        media_type = "text/html"
    elif filename.endswith(".pdf"):
        media_type = "application/pdf"

    return FileResponse(report_path, media_type=media_type)


@router.get("/api/artifacts/{run_id}/data/{filename}")
async def get_data_file(run_id: str, filename: str):
    """获取分析数据文件（h5ad）

    Args:
        run_id: 运行ID
        filename: 数据文件名

    Returns:
        数据文件
    """
    run_dir = _validate_run_id(run_id)
    data_path = run_dir / "artifacts" / "data" / filename

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data file {filename} not found")

    media_type = "application/octet-stream"
    if filename.endswith(".h5ad"):
        media_type = "application/h5ad"

    return FileResponse(data_path, media_type=media_type)


@router.get("/api/artifacts/{run_id}/list")
async def list_artifacts(run_id: str) -> Dict[str, List[str]]:
    """列出指定run的所有产物文件

    Args:
        run_id: 运行ID

    Returns:
        按类型分类的文件列表
    """
    run_dir = _validate_run_id(run_id)
    artifacts_dir = run_dir / "artifacts"

    result: Dict[str, List[str]] = {
        "plots": [],
        "tables": [],
        "reports": [],
        "data": [],
    }

    for artifact_type in result.keys():
        type_dir = artifacts_dir / artifact_type
        if type_dir.exists():
            for file_path in type_dir.iterdir():
                if file_path.is_file():
                    result[artifact_type].append(file_path.name)

    return result


@router.get("/api/artifacts/{run_id}/plots")
async def list_plots(run_id: str) -> List[Dict[str, Any]]:
    """列出指定run的所有图表及其元数据

    Args:
        run_id: 运行ID

    Returns:
        图表元数据列表
    """
    from src.web.services.plot_interpretation import get_plot_interpretation, get_plot_title

    run_dir = _validate_run_id(run_id)
    plots_dir = run_dir / "artifacts" / "plots"

    if not plots_dir.exists():
        return []

    plots = []
    for plot_file in sorted(plots_dir.glob("*.png")):
        # 解析图表类型
        plot_type = "unknown"
        if "qc_violin" in plot_file.name:
            plot_type = "qc_violin"
        elif "umap_cluster" in plot_file.name or "umap_leiden" in plot_file.name:
            plot_type = "umap_cluster"
        elif "umap_annotated" in plot_file.name or "umap_cell_type" in plot_file.name:
            plot_type = "umap_annotated"
        elif "marker_heatmap" in plot_file.name:
            plot_type = "marker_heatmap"
        elif "pca_variance" in plot_file.name:
            plot_type = "pca_variance"
        elif "volcano" in plot_file.name:
            plot_type = "volcano"

        plots.append({
            "name": plot_file.stem,
            "filename": plot_file.name,
            "url": f"/api/artifacts/{run_id}/plots/{plot_file.name}",
            "title": get_plot_title(plot_type),
            "type": plot_type,
            "interpretation": get_plot_interpretation(plot_type),
        })

    return plots


__all__ = ["router"]
