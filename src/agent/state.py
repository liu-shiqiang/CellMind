"""
Agent状态定义
完整的多Agent状态，包含所有必要字段
"""
from typing_extensions import NotRequired, TypedDict
from typing import Any, List, Optional, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class Intent(BaseModel):
    """意图识别模式"""

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="标准化意图标签")
    description: str = Field(default="", description="任务详细描述")
    confidence: float = Field(default=0.8, ge=0, le=1, description="置信度 (0-1)")
    dependencies: List[str] = Field(
        default_factory=list,
        description="前置依赖意图"
    )
    justification: str = Field(default="", description="选择该标签的理由")

    def __hash__(self):
        return hash((self.intent, self.description, self.confidence))

    def __eq__(self, other):
        if not isinstance(other, Intent):
            return False
        return (
            self.intent == other.intent
            and self.description == other.description
            and self.confidence == other.confidence
        )


class IntentResponse(BaseModel):
    """意图识别响应"""
    model_config = ConfigDict(extra="ignore")
    intents: List[Intent] = Field(description="识别到的意图列表")
    is_task: bool = Field(default=True, description="是否需要执行分析任务")


class Plan(BaseModel):
    """执行计划"""
    steps: List[str] = Field(
        description="按顺序执行的不同步骤"
    )


class AgentState(TypedDict):
    """完整的多Agent状态定义"""

    # === 基础信息 ===
    objective: str                          # 用户目标
    messages: List[BaseMessage]            # 消息历史
    input_files: List[str]                 # 输入文件列表

    # === 意图与计划 ===
    intents: List[Any]                     # 识别到的意图 (List[Intent])
    plan: List[str]                        # 执行计划

    # === 执行控制 ===
    next_step: Optional[str]               # 下一步
    execution_status: str                  # 执行状态: in_progress, completed, failed, waiting_for_input
    replan_attempts: int                   # 重规划次数
    max_replan_attempts: int               # 最大重规划次数

    # === 工作上下文 ===
    work_dir: Optional[str]                # 工作目录
    tool_history: List[Dict[str, Any]]     # 工具调用历史
    analysis_notes: Dict[str, Any]         # 分析笔记

    # === 会话信息 ===
    thread_id: str                         # 线程ID
    session_id: str                        # 会话ID
    run_id: str                            # 运行ID

    # === 内存管理 ===
    memory_summary: str                    # 记忆摘要
    memory_records: List[Dict[str, Any]]   # 记忆记录
    project_state: Dict[str, Any]          # 项目状态

    # === 意图追踪 ===
    intent_trace: Dict[str, Any]           # 意图追踪信息
    recognized_intents: NotRequired[List[Dict[str, Any]]]  # 原始识别的意图

    # === 扩展字段 ===
    input_file_info: NotRequired[List[Dict[str, Any]]]  # 输入文件信息
