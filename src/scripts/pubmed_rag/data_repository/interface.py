from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_core.documents.base import Document
from src.scripts.pubmed_rag.data_repository.models import ScientificAbstract


class UserQueryDataStore(ABC):
    """与文摘数据库交互的仓储类"""

    @abstractmethod
    def save_dataset(self, abstracts_data: List[ScientificAbstract], user_query: str) -> str:
        """
        将文摘数据和查询细节保存到数据存储中。
        返回一个字符串，表示新分配的查询ID。
        """
        raise NotImplementedError

    @abstractmethod
    def read_dataset(self, query_id: str) -> List[ScientificAbstract]:
        """
        从数据存储中检索指定查询ID的文摘数据。
        """
        raise NotImplementedError

    @abstractmethod
    def delete_dataset(self, query_id: str) -> None:
        """
        从数据库中删除指定查询ID的所有数据。
        """
        raise NotImplementedError

    @abstractmethod 
    def get_list_of_queries(self) -> Dict[str, str]:
        """
        检索查询ID和用户查询的字典。用于在UI上显示查询列表，并供查找使用。
        """
        raise NotImplementedError

    def create_document_list(self, abstracts_data: List[ScientificAbstract]) -> List[Document]:
        """
        将文摘数据转换为 LangChain 文档对象的列表。
        每个文档包含文摘内容及其元数据。
        """
        return [
            Document(
                page_content=entry.abstract_content, metadata={
                    "source": entry.doi, "title": entry.title, 
                    "authors": entry.authors, "year_of_publication": entry.year
                }
            )
            for entry in abstracts_data
        ]

    def read_documents(self, query_id: str) -> List[Document]:
        """ 
        读取数据集并将其转换为所需的 List[Document] 类型。
        """
        query_record = self.read_dataset(query_id)
        return self.create_document_list(query_record)
