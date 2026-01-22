"""
会话管理API路由
使用数据库持久化存储
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Header, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.web.schemas import SessionCreate, SessionUpdate, SessionMessageCreate
from src.db.session import get_db
from src.db.models import Session as DBSession, Message as DBMessage
from src.core.security import decode_token

router = APIRouter()


async def get_optional_user_id(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[str]:
    """
    获取当前用户ID（可选）
    如果有有效的认证 token 则返回 user_id，否则返回 None
    """
    if not authorization:
        return None

    # 提取 Bearer token
    if not authorization.startswith("Bearer "):
        return None

    token = authorization[7:]  # 移除 "Bearer " 前缀

    # 解码 token 获取 user_id
    payload = decode_token(token)
    if payload:
        return payload.get("sub")

    return None


@router.get("/")
async def get_sessions(
    limit: int = 20,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """获取会话列表 - 只返回已登录用户的会话"""
    # 未登录时不返回任何会话
    if not current_user_id:
        return {"sessions": [], "total": 0}

    user_id = current_user_id

    # 获取会话及其消息数量
    result = await db.execute(
        select(
            DBSession.id,
            DBSession.title,
            DBSession.created_at,
            DBSession.updated_at,
            DBSession.agent_mode,
            func.count(DBMessage.id).label("message_count")
        )
        .outerjoin(DBMessage, DBSession.id == DBMessage.session_id)
        .where(DBSession.user_id == user_id)
        .group_by(DBSession.id)
        .order_by(DBSession.updated_at.desc())
        .limit(limit)
    )

    sessions = []
    for row in result:
        sessions.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "agent_mode": row[4],
            "message_count": row[5] or 0
        })

    # 获取总数
    count_result = await db.execute(
        select(func.count(DBSession.id)).where(DBSession.user_id == user_id)
    )
    total = count_result.scalar() or 0

    return {"sessions": sessions, "total": total}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取会话详情"""
    result = await db.execute(
        select(DBSession).where(
            DBSession.id == session_id,
            DBSession.user_id == current_user_id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 获取消息数量
    count_result = await db.execute(
        select(func.count(DBMessage.id)).where(DBMessage.session_id == session_id)
    )
    message_count = count_result.scalar() or 0

    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "agent_mode": session.agent_mode,
        "message_count": message_count,
        "user_id": session.user_id,
    }


@router.post("/")
async def create_session(
    request: SessionCreate,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """创建新会话"""
    session_id = f"session_{uuid4().hex[:12]}"
    title = request.title or f"New Analysis {datetime.now().strftime('%H:%M')}"

    new_session = DBSession(
        id=session_id,
        user_id=current_user_id,
        title=title,
        agent_mode=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    return {
        "id": new_session.id,
        "title": new_session.title,
        "created_at": new_session.created_at.isoformat() if new_session.created_at else None,
        "updated_at": new_session.updated_at.isoformat() if new_session.updated_at else None,
        "message_count": 0,
        "agent_mode": new_session.agent_mode
    }


@router.put("/{session_id}")
async def update_session(
    session_id: str,
    request: SessionUpdate,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """更新会话"""
    result = await db.execute(
        select(DBSession).where(
            DBSession.id == session_id,
            DBSession.user_id == current_user_id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.title:
        session.title = request.title
    if request.agent_mode is not None:
        session.agent_mode = request.agent_mode

    session.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(session)

    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "agent_mode": session.agent_mode
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """删除会话"""
    result = await db.execute(
        select(DBSession).where(
            DBSession.id == session_id,
            DBSession.user_id == current_user_id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()

    # 清理匿名用户
    if session.user_id:
        remaining_result = await db.execute(
            select(func.count(DBSession.id)).where(DBSession.user_id == session.user_id)
        )
        remaining = remaining_result.scalar() or 0
        if remaining == 0:
            user_result = await db.execute(select(User).where(User.id == session.user_id))
            user = user_result.scalar_one_or_none()
            if user and is_anonymous_user(user):
                await db.delete(user)
                await db.commit()

    return {"message": "Session deleted successfully"}


@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = 100,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取会话的所有消息"""
    # 未登录无法访问
    if not current_user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    # 验证会话是否存在且属于当前用户
    session_result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 获取消息列表
    result = await db.execute(
        select(DBMessage)
        .where(DBMessage.session_id == session_id)
        .order_by(DBMessage.timestamp.asc())
        .limit(limit)
    )

    messages = []
    for msg in result.scalars():
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "metadata": msg.user_metadata
        })

    return {"messages": messages}


@router.post("/{session_id}/messages")
async def create_message(
    session_id: str,
    message: SessionMessageCreate,
    current_user_id: Optional[str] = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """创建新消息"""
    # 未登录无法访问
    if not current_user_id:
        raise HTTPException(status_code=401, detail="需要登录")

    # 验证会话是否存在
    session_result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 创建消息
    new_message = DBMessage(
        id=f"msg_{uuid4().hex[:16]}",
        session_id=session_id,
        role=message.role,
        content=message.content,
        user_metadata=message.metadata or {},
        timestamp=datetime.utcnow()
    )

    db.add(new_message)

    # 更新会话的 updated_at 时间
    session.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(new_message)

    return {
        "id": new_message.id,
        "role": new_message.role,
        "content": new_message.content,
        "timestamp": new_message.timestamp.isoformat() if new_message.timestamp else None,
        "metadata": new_message.user_metadata
    }
