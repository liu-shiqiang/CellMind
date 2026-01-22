"""
聊天服务
非Agent模式的对话服务
"""
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.llm_manager import get_llm
from src.services.rag_service import RAGService


class ChatService:
    """聊天服务（非Agent模式）"""

    def __init__(self):
        self.rag_service = RAGService()

    async def chat(self, message: str, session_id: str = "new") -> str:
        """
        处理聊天消息

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            AI回复
        """
        system_prompt = """You are CellMind, a biomedical research assistant.

Provide concise, accurate answers. If you don't know something, say so.
Answer in Chinese unless the user explicitly requests another language.
"""

        # RAG增强（可选）
        context = await self.rag_service.retrieve(message, top_k=3)
        if context:
            augmented_message = f"""参考以下资料回答问题:

{context}

问题: {message}

请基于参考资料给出专业回答。如果资料不足，请结合专业知识补充。
"""
        else:
            augmented_message = message

        # 调用LLM
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=augmented_message)
        ]

        response = await llm.ainvoke(messages)
        return response.content
