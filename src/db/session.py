"""
数据库会话管理
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.web.config import settings

# 创建异步引擎
engine = create_async_engine(
    settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://"),
    echo=False,
)

# 创建会话工厂
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """获取数据库会话（依赖注入）"""
    async with async_session_maker() as session:
        yield session


async def init_db():
    """初始化数据库，创建所有表"""
    # 导入所有模型以确保它们被注册
    from src.db import models  # noqa: F401

    from src.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """关闭数据库连接"""
    await engine.dispose()
