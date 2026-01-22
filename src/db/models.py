"""
数据库模型定义
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, DateTime,
    ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base

# 创建 Base 基类
Base = declarative_base()


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # 关系
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """会话表"""
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, default=None)
    title = Column(String, nullable=False)
    agent_mode = Column(Boolean, default=False)
    project_state = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    files = relationship("File", back_populates="session", cascade="all, delete-orphan")
    agent_runs = relationship("AgentRun", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    user_metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # 关系
    session = relationship("Session", back_populates="messages")


class File(Base):
    """文件表"""
    __tablename__ = "files"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_size = Column(Integer)
    upload_time = Column(DateTime, default=datetime.utcnow)

    # 关系
    session = relationship("Session", back_populates="files")


class AgentRun(Base):
    """Agent运行记录表"""
    __tablename__ = "agent_runs"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    objective = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending|running|completed|failed
    steps = Column(JSON, nullable=True)
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # 关系
    session = relationship("Session", back_populates="agent_runs")
