from typing import List
import chromadb
from langchain.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.embeddings import Embeddings
from langchain_core.documents.base import Document
from src.scripts.pubmed_rag.rag_pipeline.interface import RagWorkflow
from config.logging_config import get_logger


class ChromaDbRag(RagWorkflow):
    """ 
    使用 Chroma 作为向量存储的简单 RAG 工作流
    """

    def __init__(self, persist_directory: str, embeddings: Embeddings):
        self.persist_directory = persist_directory  # 持久化目录
        self.embeddings = embeddings  # 嵌入向量
        self.client = self._create_chromadb_client()  # 创建数据库客户端
        self.logger = get_logger(__name__)  # 初始化日志
    
    def _create_chromadb_client(self):
        # 创建 ChromaDB 持久客户端
        return chromadb.PersistentClient(path=self.persist_directory)
    
    def create_vector_index_for_user_query(self, documents: List[Document], query_id: str) -> VectorStore:
        """
        为用户查询创建 Chroma 向量索引并设置查询 ID 作为集合名称。
        """
        self.logger.info(f'为 {query_id} 创建向量索引')
        documents = filter_complex_metadata(documents)  # 过滤复杂的元数据
        try:
            index = Chroma.from_documents(
                documents, 
                self.embeddings,
                client=self.client, 
                collection_name=query_id
            )
            return index
        except Exception as e:
            self.logger.error(f'创建 {query_id} 的向量索引时遇到问题。问题：{e}')
            raise
    
    def get_vector_index_by_user_query(self, query_id: str) -> VectorStore:
        """
        通过设置为查询 ID 的集合名称检索现有的 Chroma 索引。
        """
        self.logger.info(f'加载查询 {query_id} 的向量索引')
        try:
            index = Chroma(
                client=self.client,
                collection_name=query_id,
                embedding_function=self.embeddings,
            )
            return index
        except Exception as e:
            self.logger.error(f'检索查询 {query_id} 的向量索引时遇到问题。问题：{e}')
            raise
