from __future__ import annotations

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.llm_manager import get_llm
from src.web.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Lightweight chat endpoint (non-Agent) using LLM (+ optional RAG hook)."""
    system_prompt = (
        "You are CellMind, a biomedical assistant. Provide concise, well-supported answers. "
        "If you do not know, say so."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]
    try:
        llm = get_llm()
        result = llm.invoke(messages)
        content = getattr(result, "content", str(result))
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
    except Exception as exc:  # fallback for offline/local errors
        content = f"LLM temporarily unavailable: {exc}"

    return ChatResponse(message=str(content))
