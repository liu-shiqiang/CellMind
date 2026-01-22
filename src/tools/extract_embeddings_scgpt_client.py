"""scGPT 客户端工具

通过 HTTP 调用 scGPT 微服务,避免直接导入导致的依赖冲突
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# scGPT 服务地址 (可通过环境变量配置)
SCGPT_SERVICE_URL = os.environ.get(
    "SCGPT_SERVICE_URL",
    "http://localhost:8001"
)


@tool(
    "extract_embeddings_with_scgpt",
    return_direct=False,
)
def extract_embeddings_with_scgpt(
    file_path: str,
    model_name: str = "scgpt",
    output_dir: Optional[str] = None,
) -> str:
    """
    使用 scGPT 提取单细胞数据的 embeddings

    Args:
        file_path: 输入 .h5ad 文件路径
        model_name: scGPT 模型名称 (默认: scgpt)
        output_dir: 输出目录 (可选)

    Returns:
        embeddings 文件路径

    Example:
        >>> extract_embeddings_with_scgpt("data.h5ad")
        "/path/to/scgpt_embeddings.h5ad"
    """
    from pathlib import Path
    from src.tools.artifact_paths import resolve_artifact_dir

    logger.info(f"调用 scGPT 服务处理: {file_path}")
    if output_dir is None:
        input_path = Path(file_path).expanduser().resolve()
        output_dir = str(resolve_artifact_dir(input_path=input_path, subdir="embeddings"))

    try:
        # 调用 scGPT 微服务
        response = requests.post(
            f"{SCGPT_SERVICE_URL}/embeddings",
            json={
                "file_path": file_path,
                "model_name": model_name,
                "output_dir": output_dir,
            },
            timeout=600,  # scGPT 处理可能需要较长时间
        )

        response.raise_for_status()
        result = response.json()

        if result["status"] == "success":
            logger.info(f"✅ scGPT 处理成功: {result['embeddings_path']}")
            if result.get("execution_time"):
                logger.info(f"⏱️ 执行时间: {result['execution_time']:.2f}秒")
            return json.dumps(
                {
                    "status": "success",
                    "embeddings_path": result["embeddings_path"],
                    "output_dir": output_dir,
                },
                ensure_ascii=False,
            )
        else:
            error_msg = result.get("message", "Unknown error")
            logger.error(f"❌ scGPT 处理失败: {error_msg}")
            raise RuntimeError(f"scGPT service error: {error_msg}")

    except requests.exceptions.ConnectionError:
        # scGPT 服务不可用时，返回友好提示
        error_msg = (
            f"scGPT 服务 ({SCGPT_SERVICE_URL}) 不可用。"
            f"Embeddings 提取是可选的，您可以继续使用其他分析功能。"
        )
        logger.warning(error_msg)
        return json.dumps(
            {
                "status": "skipped",
                "message": error_msg,
            },
            ensure_ascii=False,
        )

    except requests.exceptions.Timeout:
        error_msg = "scGPT 处理超时 (10分钟)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    except requests.exceptions.HTTPError as e:
        # scGPT 服务返回错误时，返回友好提示而不是抛出异常
        if "502" in str(e) or "503" in str(e) or "504" in str(e):
            error_msg = (
                f"scGPT 服务 ({SCGPT_SERVICE_URL}) 暂时不可用。"
                f"Embeddings 提取是可选的，已跳过。"
            )
            logger.warning(error_msg)
            return json.dumps(
                {
                    "status": "skipped",
                    "message": error_msg,
                },
                ensure_ascii=False,
            )
        else:
            error_msg = f"scGPT 服务 HTTP 错误: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"scGPT 调用失败: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def check_scgpt_service() -> bool:
    """检查 scGPT 服务是否可用"""
    try:
        response = requests.get(f"{SCGPT_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    # 测试脚本
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_embeddings_scgpt_client.py <file_path>")
        sys.exit(1)

    # 检查服务
    if not check_scgpt_service():
        print(f"❌ scGPT 服务不可用 ({SCGPT_SERVICE_URL})")
        print("请先启动: uvicorn src.scgpt_service.main:app --port 8001")
        sys.exit(1)

    # 提取 embeddings
    file_path = sys.argv[1]
    try:
        result = extract_embeddings_with_scgpt(file_path)
        print(f"✅ Embeddings 保存到: {result}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)
