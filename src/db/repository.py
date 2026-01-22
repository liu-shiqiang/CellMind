"""
数据访问层 (Repository)
提供数据库操作的抽象接口
"""
from typing import List, Optional
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Session, Message, File, AgentRun


class SessionRepository:
    """会话仓储"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> Session:
        """创建会话"""
        session = Session(**kwargs)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        result = await self.db.execute(
            select(Session).where(Session.id == session_id)
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        user_id: str = "default",
        limit: int = 50
    ) -> List[Session]:
        """列出会话"""
        result = await self.db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update(
        self,
        session_id: str,
        **kwargs
    ) -> Optional[Session]:
        """更新会话"""
        session = await self.get(session_id)
        if session:
            for key, value in kwargs.items():
                setattr(session, key, value)
            session.updated_at = datetime.utcnow()
            await self.db.commit()
            await self.db.refresh(session)
        return session

    async def delete(self, session_id: str) -> bool:
        """删除会话"""
        session = await self.get(session_id)
        if session:
            await self.db.delete(session)
            await self.db.commit()
            return True
        return False

    async def get_message_count(self, session_id: str) -> int:
        """获取会话的消息数量"""
        result = await self.db.execute(
            select(func.count(Message.id)).where(Message.session_id == session_id)
        )
        return result.scalar() or 0


class MessageRepository:
    """消息仓储"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> Message:
        """创建消息"""
        message = Message(**kwargs)
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get(self, message_id: str) -> Optional[Message]:
        """获取消息"""
        result = await self.db.execute(
            select(Message).where(Message.id == message_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Message]:
        """获取会话消息"""
        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_session_after(
        self,
        session_id: str,
        after_timestamp: datetime,
        limit: int = 100
    ) -> List[Message]:
        """获取指定时间之后的会话消息"""
        result = await self.db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .where(Message.timestamp > after_timestamp)
            .order_by(Message.timestamp.asc())
            .limit(limit)
        )
        return list(result.scalars().all())


class FileRepository:
    """文件仓储"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> File:
        """创建文件记录"""
        file_record = File(**kwargs)
        self.db.add(file_record)
        await self.db.commit()
        await self.db.refresh(file_record)
        return file_record

    async def get(self, file_id: str) -> Optional[File]:
        """获取文件"""
        result = await self.db.execute(
            select(File).where(File.id == file_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(
        self,
        session_id: str
    ) -> List[File]:
        """获取会话文件列表"""
        result = await self.db.execute(
            select(File)
            .where(File.session_id == session_id)
            .order_by(File.upload_time.desc())
        )
        return list(result.scalars().all())

    async def delete(self, file_id: str) -> bool:
        """删除文件记录"""
        file_record = await self.get(file_id)
        if file_record:
            await self.db.delete(file_record)
            await self.db.commit()
            return True
        return False


class AgentRunRepository:
    """Agent运行记录仓储"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, **kwargs) -> AgentRun:
        """创建运行记录"""
        run = AgentRun(**kwargs)
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)
        return run

    async def get(self, run_id: str) -> Optional[AgentRun]:
        """获取运行记录"""
        result = await self.db.execute(
            select(AgentRun).where(AgentRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        run_id: str,
        **kwargs
    ) -> Optional[AgentRun]:
        """更新运行记录"""
        run = await self.get(run_id)
        if run:
            for key, value in kwargs.items():
                setattr(run, key, value)
            await self.db.commit()
            await self.db.refresh(run)
        return run

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[AgentRun]:
        """获取会话的运行记录"""
        result = await self.db.execute(
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .order_by(AgentRun.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_status(
        self,
        status: str,
        limit: int = 50
    ) -> List[AgentRun]:
        """获取指定状态的运行记录"""
        result = await self.db.execute(
            select(AgentRun)
            .where(AgentRun.status == status)
            .order_by(AgentRun.started_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
