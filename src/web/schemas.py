from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    objective: str = Field(..., min_length=1)
    thread_id: Optional[str] = Field(default=None)
    stream_mode: Optional[str] = Field(default="updates", pattern="^(updates|messages|debug)$")

    model_config = {"extra": "forbid"}


class JobCreateResponse(BaseModel):
    job_id: str
    thread_id: str

    model_config = {"extra": "forbid"}


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    current_node: Optional[str] = None
    objective: str
    input_files: List[str] = []
    stream_mode: str
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    model_config = {"extra": "forbid"}


class SSEvent(BaseModel):
    type: str
    ts: str
    job_id: str
    step_id: Optional[str] = None
    payload: Dict[str, Any]

    model_config = {"extra": "allow"}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None

    model_config = {"extra": "forbid"}


class ChatResponse(BaseModel):
    message: str

    model_config = {"extra": "forbid"}


# ============= 新增的Agent相关schemas =============

class AgentRunRequest(BaseModel):
    """Agent运行请求"""
    objective: str = Field(..., min_length=1)
    files: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    stream_mode: str = Field(default="updates", pattern="^(updates|messages|debug)$")

    model_config = {"extra": "forbid"}

    def normalized_files(self) -> List[str]:
        """返回规范化的文件列表"""
        return [f for f in self.files if f]


class AgentRunResponse(BaseModel):
    """Agent运行响应"""
    run_id: str
    session_id: str
    status: str

    model_config = {"extra": "forbid"}


class SessionCreate(BaseModel):
    """创建会话"""
    title: Optional[str] = None

    model_config = {"extra": "forbid"}


class SessionUpdate(BaseModel):
    """更新会话"""
    title: Optional[str] = None
    agent_mode: Optional[bool] = None

    model_config = {"extra": "forbid"}


class SessionMessageCreate(BaseModel):
    """创建会话消息"""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str
    filename: str
    size: int
    path: str
    uploaded_at: Optional[datetime] = None

    model_config = {"extra": "forbid"}
