"""
可视化数据API路由
提供UMAP、火山图等可视化数据接口
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _latest_csv(path: Path, pattern: str) -> Optional[Path]:
    candidates = list(path.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@router.get("/{run_id}/umap")
async def get_umap_data(
    run_id: str,
    sample_size: int = Query(10000, ge=100, le=100000, description="采样数量")
):
    """
    获取UMAP可视化数据

    从Agent执行结果中提取UMAP降维数据
    支持大数据集采样以提高前端性能

    - **run_id**: Agent运行ID
    - **sample_size**: 最大返回点数（默认10000）
    """
    from src.web.config import settings
    runs_dir = Path(settings.RUNS_DIR) / run_id / "artifacts" / "tables"

    if not runs_dir.exists():
        return {
            "run_id": run_id,
            "points": [],
            "metadata": {"status": "no_data", "message": "UMAP数据尚未生成"}
        }

    umap_file = _latest_csv(runs_dir, "umap_coords_*.csv")
    if not umap_file:
        return {
            "run_id": run_id,
            "points": [],
            "metadata": {"status": "no_data", "message": "UMAP数据尚未生成"}
        }

    import pandas as pd

    df = pd.read_csv(umap_file)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    points = []
    for _, row in df.iterrows():
        point = {
            "x": float(row.get("UMAP_1", row.get("x", 0))),
            "y": float(row.get("UMAP_2", row.get("y", 0))),
        }
        if "cluster" in row:
            point["cluster"] = str(row["cluster"])
        if "cell_type" in row:
            point["cellType"] = str(row["cell_type"])
        points.append(point)

    return {
        "run_id": run_id,
        "points": points,
        "total_points": len(df),
        "sampled": len(points),
        "metadata": {"status": "success", "source": str(umap_file.name)}
    }


@router.get("/{run_id}/clustering")
async def get_clustering_data(run_id: str):
    """
    获取聚类分析数据
    """
    return {
        "run_id": run_id,
        "clusters": {},
        "marker_genes": {},
        "status": "not_implemented"
    }


@router.get("/{run_id}/markers")
async def get_marker_genes(
    run_id: str,
    cluster: Optional[int] = None,
    top_n: int = Query(10, ge=1, le=50)
):
    """
    获取标记基因数据
    """
    from src.web.config import settings
    tables_dir = Path(settings.RUNS_DIR) / run_id / "artifacts" / "tables"
    if not tables_dir.exists():
        return {"run_id": run_id, "markers": [], "status": "no_data"}

    marker_file = _latest_csv(tables_dir, "marker_genes_*.csv")
    if not marker_file:
        return {"run_id": run_id, "markers": [], "status": "no_data"}

    import pandas as pd

    df = pd.read_csv(marker_file)
    if cluster is not None and "group" in df.columns:
        df = df[df["group"] == str(cluster)]
    if top_n and "group" in df.columns:
        df = df.groupby("group").head(top_n)

    return {
        "run_id": run_id,
        "markers": df.to_dict(orient="records"),
        "status": "success",
        "source": str(marker_file.name),
    }


@router.get("/{run_id}/diff_expression")
async def get_diff_expression_data(
    run_id: str,
    cluster_a: Optional[int] = None,
    cluster_b: Optional[int] = None
):
    """
    获取差异表达数据
    """
    from src.web.config import settings
    tables_dir = Path(settings.RUNS_DIR) / run_id / "artifacts" / "tables"
    if not tables_dir.exists():
        return {"run_id": run_id, "genes": [], "status": "no_data"}

    de_file = _latest_csv(tables_dir, "diff_expression_*.csv")
    if not de_file:
        return {"run_id": run_id, "genes": [], "status": "no_data"}

    import pandas as pd

    df = pd.read_csv(de_file)
    df = df.head(200)

    return {
        "run_id": run_id,
        "genes": df.to_dict(orient="records"),
        "status": "success",
        "source": str(de_file.name),
    }


@router.get("/{run_id}/export")
async def export_visualization(
    run_id: str,
    format: str = Query("json", description="导出格式: json, csv, png")
):
    """
    导出可视化数据
    """
    if format == "json":
        umap_data = await get_umap_data(run_id)
        return umap_data

    raise HTTPException(status_code=400, detail=f"不支持的导出格式: {format}")
