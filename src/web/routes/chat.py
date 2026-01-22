"""
聊天API路由
非Agent模式的对话接口
"""
import json
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.llm_manager import get_llm
from src.web.schemas import ChatRequest, ChatResponse

router = APIRouter()

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def send_message(request: ChatRequest) -> ChatResponse:
    """
    发送消息 (非流式)
    """
    system_prompt = (
        "You are CellMind, a biomedical assistant. Provide concise, well-supported answers. "
        "If you do not know, say so."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.message)
    ]

    try:
        llm = get_llm()
        result = llm.invoke(messages)
        content = getattr(result, "content", str(result))
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
    except Exception as exc:
        content = f"LLM temporarily unavailable: {exc}"

    return ChatResponse(message=str(content))


@router.get("/stream")
async def stream_message(
    message: str = Query(..., min_length=1),
    thread_id: Optional[str] = Query(default=None),
) -> StreamingResponse:
    """
    流式聊天 (SSE)
    """
    system_prompt = (
        "You are CellMind, a biomedical assistant. Provide concise, well-supported answers. "
        "If you do not know, say so."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    async def event_generator() -> AsyncGenerator[str, None]:
        full_content: list[str] = []
        try:
            llm = get_llm()
            async for chunk in llm.astream(messages):
                content = getattr(chunk, "content", None)
                if not content:
                    continue
                if isinstance(content, list):
                    for part in content:
                        token = str(part)
                        if not token:
                            continue
                        full_content.append(token)
                        yield _format_sse({
                            "type": "token",
                            "thread_id": thread_id,
                            "payload": {"token": token},
                            "ts": _iso_now(),
                        })
                else:
                    token = str(content)
                    if token:
                        full_content.append(token)
                        yield _format_sse({
                            "type": "token",
                            "thread_id": thread_id,
                            "payload": {"token": token},
                            "ts": _iso_now(),
                        })

            final_message = "".join(full_content).strip()
            yield _format_sse({
                "type": "end",
                "thread_id": thread_id,
                "payload": {"message": final_message},
                "ts": _iso_now(),
            })
        except Exception as exc:
            yield _format_sse({
                "type": "error",
                "thread_id": thread_id,
                "payload": {"message": str(exc)},
                "ts": _iso_now(),
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
