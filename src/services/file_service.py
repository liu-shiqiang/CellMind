"""
文件服务
管理上传文件的生命周期，包括保存、验证、检索和清理
"""
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
    print('[Warning] aiofiles not installed, using fallback async I/O')

from src.web.config import settings


# ============== 数据模型 ==============

class FileMetadata(BaseModel):
    """文件元数据"""
    file_id: str
    original_name: str
    filepath: str
    size: int
    content_type: str
    created_at: datetime
    session_id: Optional[str] = None
    validated: bool = False
    validation_result: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    """H5AD文件验证结果"""
    valid: bool
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    cell_types: Optional[List[str]] = None
    error: Optional[str] = None


# ============== 文件服务 ==============

class FileService:
    """文件服务 - 管理上传文件的生命周期"""

    def __init__(
        self,
        upload_dir: Optional[str] = None,
        max_age_hours: int = 24,
        metadata_dir: Optional[str] = None
    ):
        self.upload_dir = Path(upload_dir or settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # 元数据存储目录
        self.metadata_dir = Path(metadata_dir or self.upload_dir / 'metadata')
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.max_age = timedelta(hours=max_age_hours)
        self._lock = asyncio.Lock()

    def _get_metadata_path(self, file_id: str) -> Path:
        """获取元数据文件路径"""
        return self.metadata_dir / f"{file_id}.json"

    def _get_content_type(self, filename: str) -> str:
        """根据文件名获取Content-Type"""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.h5ad': 'application/h5ad',
            '.h5': 'application/hdf5',
            '.csv': 'text/csv',
            '.json': 'application/json',
        }
        return content_types.get(ext, 'application/octet-stream')

    async def save_upload(
        self,
        file_id: str,
        filename: str,
        content: bytes,
        session_id: Optional[str] = None
    ) -> FileMetadata:
        """
        保存上传文件

        Args:
            file_id: 文件ID
            filename: 原始文件名
            content: 文件内容
            session_id: 会话ID

        Returns:
            FileMetadata: 文件元数据
        """
        # 保存文件
        file_ext = Path(filename).suffix
        filepath = self.upload_dir / f"{file_id}{file_ext}"

        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(content)

        # 创建元数据
        metadata = FileMetadata(
            file_id=file_id,
            original_name=filename,
            filepath=str(filepath),
            size=len(content),
            content_type=self._get_content_type(filename),
            created_at=datetime.utcnow(),
            session_id=session_id,
        )

        # 保存元数据
        await self._save_metadata(metadata)

        return metadata

    async def _save_metadata(self, metadata: FileMetadata) -> None:
        """保存文件元数据"""
        metadata_path = self._get_metadata_path(metadata.file_id)
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(metadata.model_dump_json(indent=2))

    async def get_file(self, file_id: str) -> Optional[FileMetadata]:
        """
        获取文件元数据

        Args:
            file_id: 文件ID

        Returns:
            FileMetadata or None
        """
        metadata_path = self._get_metadata_path(file_id)
        if not metadata_path.exists():
            return None

        async with aiofiles.open(metadata_path, 'r') as f:
            content = await f.read()
            return FileMetadata.model_validate_json(content)

    async def get_file_content(self, file_id: str) -> Optional[bytes]:
        """
        获取文件内容

        Args:
            file_id: 文件ID

        Returns:
            文件内容 or None
        """
        metadata = await self.get_file(file_id)
        if not metadata:
            return None

        filepath = Path(metadata.filepath)
        if not filepath.exists():
            return None

        async with aiofiles.open(filepath, 'rb') as f:
            return await f.read()

    async def delete_file(self, file_id: str) -> bool:
        """
        删除文件及其元数据

        Args:
            file_id: 文件ID

        Returns:
            是否成功删除
        """
        metadata = await self.get_file(file_id)
        if not metadata:
            return False

        # 删除实际文件
        filepath = Path(metadata.filepath)
        if filepath.exists():
            filepath.unlink()

        # 删除元数据
        metadata_path = self._get_metadata_path(file_id)
        if metadata_path.exists():
            metadata_path.unlink()

        return True

    async def cleanup_expired(self) -> List[str]:
        """
        清理过期文件

        Returns:
            被删除的文件ID列表
        """
        expired = []
        cutoff = datetime.utcnow() - self.max_age

        async with self._lock:
            for metadata_file in self.metadata_dir.glob('*.json'):
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        content = await f.read()
                        metadata = FileMetadata.model_validate_json(content)

                    if metadata.created_at < cutoff:
                        await self.delete_file(metadata.file_id)
                        expired.append(metadata.file_id)
                except Exception as e:
                    print(f"[FileService] Error cleaning up {metadata_file}: {e}")

        return expired

    async def validate_h5ad(self, file_id: str) -> ValidationResult:
        """
        验证H5AD文件

        Args:
            file_id: 文件ID

        Returns:
            ValidationResult
        """
        try:
            import anndata

            # 获取文件
            file_content = await self.get_file_content(file_id)
            if not file_content:
                return ValidationResult(valid=False, error='File not found')

            # 由于anndata.read_h5ad需要文件路径，我们需要先保存到临时文件
            import tempfile

            metadata = await self.get_file(file_id)
            if not metadata:
                return ValidationResult(valid=False, error='Metadata not found')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp:
                tmp.write(file_content)
                tmp.flush()

                adata = anndata.read_h5ad(tmp.name)

                # 更新元数据
                metadata.validated = True
                metadata.validation_result = {
                    'n_obs': adata.n_obs,
                    'n_vars': adata.n_vars,
                    'cell_types': list(adata.obs.columns) if adata is not None else [],
                }
                await self._save_metadata(metadata)

                return ValidationResult(
                    valid=True,
                    n_obs=adata.n_obs,
                    n_vars=adata.n_vars,
                    cell_types=list(adata.obs.columns) if adata is not None else [],
                )

        except Exception as e:
            return ValidationResult(
                valid=False,
                error=str(e)
            )

    async def list_files(
        self,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[FileMetadata]:
        """
        列出文件

        Args:
            session_id: 过滤会话ID
            limit: 返回数量限制

        Returns:
            文件元数据列表
        """
        files = []

        async with self._lock:
            for metadata_file in self.metadata_dir.glob('*.json'):
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        content = await f.read()
                        metadata = FileMetadata.model_validate_json(content)

                    if session_id is None or metadata.session_id == session_id:
                        files.append(metadata)
                        if len(files) >= limit:
                            break
                except Exception:
                    continue

        # 按创建时间倒序排列
        files.sort(key=lambda m: m.created_at, reverse=True)

        return files

    async def update_session_id(self, file_id: str, session_id: str) -> bool:
        """
        更新文件的会话ID

        Args:
            file_id: 文件ID
            session_id: 新的会话ID

        Returns:
            是否成功更新
        """
        metadata = await self.get_file(file_id)
        if not metadata:
            return False

        metadata.session_id = session_id
        await self._save_metadata(metadata)
        return True

    async def get_h5ad_preview(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取H5AD文件详细预览

        Args:
            file_id: 文件ID

        Returns:
            包含预览数据的字典或None
        """
        try:
            import anndata
            import pandas as pd
            import numpy as np

            # 获取文件
            file_content = await self.get_file_content(file_id)
            if not file_content:
                return None

            metadata = await self.get_file(file_id)
            if not metadata:
                return None

            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp:
                tmp.write(file_content)
                tmp.flush()

                adata = anndata.read_h5ad(tmp.name)

                # 构建预览数据
                preview_data = {
                    'n_obs': adata.n_obs,
                    'n_vars': adata.n_vars,
                    'obs_columns': list(adata.obs.columns),
                    'var_columns': list(adata.var.columns),
                    'obs_dtypes': {col: str(dtype) for col, dtype in adata.obs.dtypes.items()},
                    'var_dtypes': {col: str(dtype) for col, dtype in adata.var.dtypes.items()},
                    'layers': list(adata.layers.keys()) if hasattr(adata, 'layers') else [],
                    'obsm_keys': list(adata.obsm.keys()) if hasattr(adata, 'obsm') else [],
                    'obsm_shapes': {
                        k: v.shape for k, v in adata.obsm.items()
                    } if hasattr(adata, 'obsm') else {},
                    # 前5行预览
                    'obs_preview': adata.head(5).obs.to_dict(orient='records') if hasattr(adata, 'head') else adata.obs.iloc[:5].to_dict(orient='records'),
                    'var_preview': adata.var.iloc[:5].to_dict(orient='records'),
                    # 是否有原始数据
                    'has_raw': adata.raw is not None if hasattr(adata, 'raw') else False,
                }

                return preview_data

        except Exception as e:
            print(f"[FileService] Error getting H5AD preview: {e}")
            return None


# ============== 全局单例 ==============

_file_service: Optional[FileService] = None


def get_file_service() -> FileService:
    """获取文件服务单例"""
    global _file_service
    if _file_service is None:
        _file_service = FileService()
    return _file_service


# ============== 后台任务 ==============

async def periodic_file_cleanup():
    """定期清理过期文件的后台任务"""
    service = get_file_service()
    while True:
        try:
            expired = await service.cleanup_expired()
            if expired:
                print(f"[FileCleanup] Cleaned up {len(expired)} expired files")
        except Exception as e:
            print(f"[FileCleanup] Error during cleanup: {e}")

        # 每小时执行一次
        await asyncio.sleep(3600)


# ============== Pydantic模型（用于API） ==============

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str
    filename: str
    size: int
    content_type: str
    created_at: datetime
    validation: Optional[ValidationResult] = None


class FileInfoResponse(BaseModel):
    """文件信息响应"""
    file_id: str
    exists: bool
    metadata: Optional[FileMetadata] = None
