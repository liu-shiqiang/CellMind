import json
import os
import shutil
from typing import Dict, List
from src.scripts.pubmed_rag.data_repository.models import UserQueryRecord, ScientificAbstract
from src.scripts.pubmed_rag.data_repository.interface import UserQueryDataStore
from config.logging_config import get_logger


class LocalJSONStore(UserQueryDataStore):
    """ 
    用于本地测试，通过本地JSON文件模拟数据库存储。
    """

    def __init__(self, storage_folder_path: str):
        self.storage_folder_path = storage_folder_path
        self.index_file_path = os.path.join(storage_folder_path, 'index.json')
        self.logger = get_logger(__name__)
        self.metadata_index = None

    def get_new_query_id(self) -> str:
        """
        通过递增上一个查询ID的整数后缀来生成新的查询ID。
        """
        try:
            with open(self.index_file_path, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        keys = [k for k in data.keys() if k.startswith('query_')]
        if not keys:
            return 'query_1'
        numbers = [int(k.split('_')[-1]) for k in keys]
        max_number = max(numbers)
        return f'query_{max_number + 1}'

    def read_dataset(self, query_id: str) -> List[ScientificAbstract]:
        """ 
        从本地存储读取包含文摘的数据集。
        """
        try:
            with open(f'{self.storage_folder_path}/{query_id}/abstracts.json', 'r') as file:
                data = json.load(file)
                return [ScientificAbstract(**abstract_record) for abstract_record in data]
        except FileNotFoundError:
            self.logger.error(f'未找到查询 {query_id} 的JSON文件。')
            raise FileNotFoundError('未找到JSON文件。')

    def save_dataset(self, abstracts_data: List[ScientificAbstract], user_query: str) -> str:
        """ 
        将文摘数据集和查询元数据保存到本地存储，重建索引，并返回查询ID。
        """
        try:
            query_id = self.get_new_query_id()
            user_query_details = UserQueryRecord(
                user_query_id=query_id, 
                user_query=user_query
            )

            os.makedirs(f'{self.storage_folder_path}/{query_id}', exist_ok=True)
            
            with open(f"{self.storage_folder_path}/{query_id}/abstracts.json", "w") as file:
                list_of_abstracts = [model.model_dump() for model in abstracts_data]
                json.dump(list_of_abstracts, file, indent=4)

            with open(f"{self.storage_folder_path}/{query_id}/query_details.json", "w") as file:
                json_data = user_query_details.model_dump_json(indent=4)
                file.write(json_data)

            self.logger.info(f"查询ID {query_id} 的数据保存成功。")
            self._rebuild_index()  # 数据保存后重建索引

            return query_id

        except Exception as e:
            self.logger.error(f"保存查询ID {query_id} 数据集失败: {e}")
            raise RuntimeError(f"由于错误导致保存数据集失败: {e}")
        
    def delete_dataset(self, query_id: str) -> None:
        """ 
        从本地存储中删除文摘数据集和查询元数据。
        """
        path_to_data = f'{self.storage_folder_path}/{query_id}'
        if os.path.exists(path_to_data):
            shutil.rmtree(path_to_data)
            self.logger.info(f"目录 '{path_to_data}' 已删除。")
            self._rebuild_index()  # 删除数据后重建索引
        else:
            self.logger.warning(f"目录 '{path_to_data}' 不存在，无法删除。")

    def get_list_of_queries(self) -> Dict[str, str]:
        """ 
        从索引中获取包含查询ID（作为键）和原始用户查询（作为值）的字典。
        """
        return self.metadata_index

    def _rebuild_index(self) -> Dict[str, str]:
        """ 
        从所有查询详情文件重建索引，供查询使用。
        """
        index = {}
        query_data_paths = [os.path.join(self.storage_folder_path, name) for name in os.listdir(self.storage_folder_path)
                            if os.path.isdir(os.path.join(self.storage_folder_path, name))]
        
        for query_data_path in query_data_paths:
            metadata_path = os.path.join(query_data_path, 'query_details.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as file:
                    metadata = json.load(file)
                    index[metadata['user_query_id']] = metadata['user_query']
            else:
                self.logger.warning(f"在 {query_data_path} 中未找到 query_details.json 文件")
        
        with open(self.index_file_path, 'w') as file:
            json.dump(index, file, indent=4)
        self.metadata_index = index
        return index
