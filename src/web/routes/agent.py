"""
Agent API路由
Agent模式的分析接口，支持SSE流式输出
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.utils.langgraph_stream import run_agent_stream
from src.web.schemas import AgentRunRequest

router = APIRouter()


class _RunStreamState:
    """运行状态管理"""
    def __init__(self, run_id: str, thread_id: str):
        self.run_id = run_id
        self.thread_id = thread_id
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.done = asyncio.Event()
        self.error_info: Optional[Any] = None


# 全局运行状态存储
_run_states: Dict[str, _RunStreamState] = {}
_run_states_lock = asyncio.Lock()


def _format_sse(event: Dict[str, Any]) -> str:
    """格式化SSE事件"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _execute_run(
    state: _RunStreamState,
    objective: str,
    input_files: Optional[List[str]],
    stream_mode: str,
) -> None:
    """执行Agent运行"""
    async def handler(event: Dict[str, Any]) -> None:
        await state.queue.put(event)

    try:
        from src.utils.langgraph_stream import run_agent_stream

        await run_agent_stream(
            objective=objective,
            input_files=input_files,
            thread_id=state.thread_id,
            run_id=state.run_id,
            event_handler=handler,
            stream_mode=stream_mode,
        )
    except Exception as exc:
        error_event = {
            "type": "error",
            "run_id": state.run_id,
            "thread_id": state.thread_id,
            "data": {"error": str(exc)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await state.queue.put(error_event)
        state.error_info = {"detail": str(exc)}
    finally:
        state.done.set()


@router.post("/run")
async def run_agent(request: AgentRunRequest) -> Dict[str, str]:
    """
    启动Agent分析

    返回run_id用于流式获取状态
    """
    objective = request.objective
    input_files = request.normalized_files()

    run_id = str(uuid4())
    thread_id = request.thread_id or run_id
    state = _RunStreamState(run_id, thread_id)

    async with _run_states_lock:
        _run_states[run_id] = state

    # 启动后台任务
    asyncio.create_task(_execute_run(state, objective, input_files, request.stream_mode))

    return {"run_id": run_id, "session_id": thread_id}


@router.get("/stream/{run_id}")
async def stream_events(run_id: str) -> StreamingResponse:
    """
    流式获取Agent执行事件 (SSE)
    """
    async with _run_states_lock:
        state = _run_states.get(run_id)

    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        """生成SSE事件"""
        try:
            while True:
                if state.done.is_set() and state.queue.empty():
                    # 发送结束事件
                    end_event = {
                        "type": "end",
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {"status": "completed"}
                    }
                    yield _format_sse(end_event)
                    break

                try:
                    event = await asyncio.wait_for(state.queue.get(), timeout=1.0)
                    yield _format_sse(event)
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    heartbeat = {
                        "type": "heartbeat",
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    yield _format_sse(heartbeat)

        finally:
            # 清理
            async with _run_states_lock:
                _run_states.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    """获取运行状态"""
    # 简化实现，返回基本信息
    return {"run_id": run_id, "status": "running"}
