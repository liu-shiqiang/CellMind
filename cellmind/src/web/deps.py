"""
依赖注入
"""
from typing import AsyncGenerator
from fastapi import Depends


async def get_db():
    """获取数据库会话"""
    from src.db.session import async_session_maker

    async with async_session_maker() as session:
        yield session
