"""
RAG检索服务
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.web.config import settings


class RAGService:
    """RAG检索服务"""

    def __init__(self):
        # 初始化ChromaDB客户端
        persist_dir = settings.CHROMA_PERSIST_DIR
        import os
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="literature",
            metadata={"description": "生物医学文献知识库"}
        )

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        检索相关文献

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            拼接的上下文文本
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            if not results["documents"][0]:
                return ""

            # 格式化结果
            contexts = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                source = metadata.get("source", "未知来源")
                contexts.append(f"[来源: {source}]\\n{doc}")

            return "\\n\\n---\\n\\n".join(contexts)

        except Exception as e:
            print(f"RAG检索错误: {e}")
            return ""
