"""
CellMind 应用配置
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """应用配置"""

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./cellmind.db"

    # JWT 认证配置
    JWT_SECRET_KEY: str = "cellmind-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 密码策略配置
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_NUMBER: bool = True

    # 登录限流配置
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # LLM配置
    LLM_PROVIDER: str = "openai"  # openai, anthropic, ollama
    LLM_API_KEY: str = ""  # 从环境变量读取
    LLM_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
    LLM_MODEL: str = "glm-4"
    LLM_TEMPERATURE: float = 0.7

    # RAG配置
    CHROMA_PERSIST_DIR: str = "./data/chroma_data"
    KNOWLEDGE_BASE_PATH: str = "./data/knowledge"
    RETRIEVE_TOP_K: int = 3

    # Embedding模型配置
    LIT_RAG_EMBEDDING_MODEL: str = "./data/pretrained_model/all-MiniLM-L6-v2"
    CELL_RAG_EMBEDDING_MODEL: str = "./data/pretrained_model/scgpt/scgpt_human"

    # CellPhoneDB数据库配置
    CELLPHONEDB_DB_PATH: str = "./data/cellphonedb/cellphonedb.zip"

    # 存储配置
    UPLOAD_DIR: str = "./uploads"
    RUNS_DIR: str = "./runs"
    DATA_DIR: str = "./data"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    # Agent配置
    MAX_REPLAN_ATTEMPTS: int = 4
    RECURSION_LIMIT: int = 50

    # scGPT配置（可选）
    SCGPT_SERVICE_URL: str = "http://localhost:8001"
    SCGPT_ENABLED: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def reference_dir(self) -> Path:
        """参考数据目录"""
        return Path(self.DATA_DIR) / "references"

    @property
    def cell_marker_dir(self) -> Path:
        """细胞标记基因目录"""
        return self.reference_dir / "cell_markers"

    @property
    def literature_dir(self) -> Path:
        """文献知识目录"""
        return self.reference_dir / "literature"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()

# 确保目录存在
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.RUNS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
settings.reference_dir.mkdir(parents=True, exist_ok=True)
settings.cell_marker_dir.mkdir(parents=True, exist_ok=True)
settings.literature_dir.mkdir(parents=True, exist_ok=True)

