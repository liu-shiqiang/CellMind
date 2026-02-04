"""
文件上传API路由
使用file_service管理文件生命周期
"""
from uuid import uuid4
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from src.services.file_service import (
    get_file_service,
    FileUploadResponse,
    ValidationResult,
)
from src.web.config import settings

router = APIRouter()

# 确保上传目录存在
from pathlib import Path
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    validate: bool = Query(True, description="是否验证H5AD文件")
):
    """
    上传文件
    支持.h5ad单细胞数据文件，自动验证文件格式

    - **file**: 上传的文件对象
    - **session_id**: 关联的会话ID
    - **validate**: 是否验证H5AD文件（默认true）
    """
    # 验证文件类型
    if not file.filename.lower().endswith('.h5ad'):
        raise HTTPException(
            status_code=400,
            detail="仅支持.h5ad格式的文件"
        )

    # 读取文件内容
    contents = await file.read()

    # 检查文件大小
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"文件大小不能超过{max_mb}MB"
        )

    # 生成文件ID
    file_id = f"file_{uuid4().hex[:12]}"

    # 使用文件服务保存
    file_service = get_file_service()
    metadata = await file_service.save_upload(
        file_id=file_id,
        filename=file.filename,
        content=contents,
        session_id=session_id
    )

    # 可选：验证H5AD文件
    validation_result = None
    if validate and file.filename.lower().endswith('.h5ad'):
        validation_result = await file_service.validate_h5ad(file_id)
        if not validation_result.valid:
            # 验证失败，删除文件
            await file_service.delete_file(file_id)
            raise HTTPException(
                status_code=400,
                detail=f"无效的H5AD文件: {validation_result.error}"
            )

    return FileUploadResponse(
        file_id=metadata.file_id,
        filename=metadata.original_name,
        size=metadata.size,
        content_type=metadata.content_type,
        created_at=metadata.created_at,
        validation=validation_result
    )


@router.get("/{file_id}")
async def get_file_info(file_id: str):
    """
    获取文件信息

    返回文件的元数据和验证结果
    """
    file_service = get_file_service()
    metadata = await file_service.get_file(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="文件不存在")

    return {
        "file_id": metadata.file_id,
        "exists": True,
        "metadata": metadata
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    删除文件

    同时删除文件和元数据
    """
    file_service = get_file_service()
    success = await file_service.delete_file(file_id)

    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")

    return {"success": True, "message": "文件已删除", "file_id": file_id}


@router.get("/")
async def list_files(
    session_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    列出文件

    - **session_id**: 过滤特定会话的文件
    - **limit**: 返回数量限制（默认20）
    """
    file_service = get_file_service()
    files = await file_service.list_files(session_id=session_id, limit=limit)

    return {
        "files": files,
        "total": len(files)
    }


@router.post("/{file_id}/validate")
async def validate_file(file_id: str):
    """
    重新验证H5AD文件
    """
    file_service = get_file_service()
    result = await file_service.validate_h5ad(file_id)

    if not result.valid:
        raise HTTPException(status_code=400, detail=result.error)

    return result


@router.post("/cleanup")
async def cleanup_expired_files():
    """
    手动触发清理过期文件

    通常由后台任务自动执行，此接口用于手动触发
    """
    file_service = get_file_service()
    expired = await file_service.cleanup_expired()

    return {
        "cleaned": len(expired),
        "file_ids": expired
    }


@router.get("/{file_id}/preview")
async def get_file_preview(file_id: str):
    """
    获取H5AD文件详细预览

    返回数据的统计信息和预览内容，包括：
    - 细胞数量 (n_obs)
    - 基因数量 (n_vars)
    - 细胞元数据列 (obs_columns)
    - 基因信息列 (var_columns)
    - obs前5行预览
    - var前5行预览
    - layers列表
    - obsm列表
    """
    file_service = get_file_service()
    metadata = await file_service.get_file(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="文件不存在")

    preview_data = await file_service.get_h5ad_preview(file_id)

    if not preview_data:
        raise HTTPException(status_code=400, detail="无法读取H5AD文件预览")

    return preview_data
