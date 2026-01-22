"""scGPT Microservice

独立的 scGPT 服务,避免与主服务的依赖冲突。
运行在独立的进程和端口上。
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="scGPT Service",
    description="独立的 scGPT 嵌入提取服务",
    version="1.0.0",
)


# ============== Request/Response Models ==============

class EmbeddingRequest(BaseModel):
    """scGPT embedding 提取请求"""
    file_path: str = Field(..., description="输入 .h5ad 文件路径")
    model_name: str = Field(default="scgpt", description="模型名称")
    output_dir: Optional[str] = Field(None, description="输出目录")


class EmbeddingResponse(BaseModel):
    """scGPT embedding 提取响应"""
    status: str
    embeddings_path: Optional[str] = None
    message: str
    execution_time: Optional[float] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    service: str
    version: str
    scgpt_available: bool


# ============== Helper Functions ==============

def check_scgpt_available() -> bool:
    """检查 scGPT 是否可用"""
    try:
        import torch
        import torchtext
        from scgpt import Preprocessor
        return True
    except ImportError as e:
        logger.warning(f"scGPT 依赖检查失败: {e}")
        return False


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        service="scgpt",
        version="1.0.0",
        scgpt_available=check_scgpt_available()
    )


@app.post("/embeddings", response_model=EmbeddingResponse)
async def extract_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks
):
    """
    提取 scGPT embeddings

    这个端点会在独立进程中处理 scGPT 相关计算,
    避免阻塞主服务。
    """
    import time
    start_time = time.time()

    # 验证文件存在
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {request.file_path}"
        )

    try:
        # 导入 scGPT 相关模块 (仅在需要时)
        from src.bio_pretrained_model.data_prep import ScGPTDataProcessor
        import scanpy as sc

        logger.info(f"开始处理: {request.file_path}")

        # 读取数据
        adata = sc.read_h5ad(request.file_path)
        logger.info(f"数据形状: {adata.shape}")

        # 初始化 processor
        processor = ScGPTDataProcessor(model_name=request.model_name)

        # 提取 embeddings
        logger.info("提取 embeddings...")
        embeddings = processor.extract_embeddings(adata)

        # 确定输出路径
        if request.output_dir is None:
            output_dir = Path(request.file_path).parent
        else:
            output_dir = Path(request.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "scgpt_embeddings.h5ad"

        # 保存结果
        embeddings.write_h5ad(output_path)
        logger.info(f"Embeddings 保存到: {output_path}")

        execution_time = time.time() - start_time

        return EmbeddingResponse(
            status="success",
            embeddings_path=str(output_path),
            message=f"成功提取 {embeddings.shape[0]} 个细胞的 embeddings",
            execution_time=execution_time
        )

    except ImportError as e:
        logger.error(f"scGPT 依赖缺失: {e}")
        return EmbeddingResponse(
            status="error",
            message=f"scGPT 依赖缺失: {str(e)}。请确保安装了兼容的 torch 和 torchtext 版本。"
        )

    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"scGPT 处理失败: {str(e)}"
        )


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "scGPT Microservice",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "embeddings": "/embeddings (POST)",
            "docs": "/docs"
        }
    }


# ============== Main ==============

if __name__ == "__main__":
    # 从环境变量读取端口
    port = int(os.environ.get("SCGPT_SERVICE_PORT", 8001))

    logger.info(f"🚀 启动 scGPT 服务在端口 {port}")
    logger.info(f"📖 API 文档: http://localhost:{port}/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
