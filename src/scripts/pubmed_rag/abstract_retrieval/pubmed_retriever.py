from typing import List
import time
import re
import random
from metapub import PubMedFetcher
from src.scripts.pubmed_rag.data_repository.models import ScientificAbstract
from src.scripts.pubmed_rag.abstract_retrieval.interface import AbstractRetriever
from src.scripts.pubmed_rag.abstract_retrieval.pubmed_query_simplification import simplify_pubmed_query
from src.scripts.pubmed_rag.abstract_retrieval.translation_query import translation_chain
from config.logging_config import get_logger


class PubMedAbstractRetriever(AbstractRetriever):
    def __init__(self, pubmed_fetch_object: PubMedFetcher):
        # 初始化 PubMedFetch 对象和日志记录器
        self.pubmed_fetch_object = pubmed_fetch_object
        self.logger = get_logger(__name__)

    def _simplify_pubmed_query(self, query: str, simplification_function: callable = simplify_pubmed_query) -> str:
        # 使用简化函数简化查询
        return simplification_function(query)

    def _translation_chain(self, query: str, translation_function: callable = translation_chain) -> str:
        ret = bool(re.search('[\u4e00-\u9fff]', query))
        if ret:
            trans_query = translation_function(query)
            self.logger.info(f'输入是中文，翻译的英文是：{trans_query}')
            return trans_query
        else:
            self.logger.info('输入是英文，不需要翻译')
            return query

    def _get_abstract_list(self, query: str, simplify_query: bool = True) -> List[str]:
        # 获取给定查询的 PubMed ID 列表
        if simplify_query:
            # 如果需要简化查询，则简化查询
            self.logger.info(f'尝试简化使用人员查询 {query}')
            query_simplified = self._simplify_pubmed_query(query)

            if query_simplified != query:
                self.logger.info(f'初始查询已简化为: {query_simplified}')
                query = query_simplified
            else:
                self.logger.info('初始查询已经足够简单，无需简化')

        self.logger.info(f'正在搜索查询: {query}')
        return self.pubmed_fetch_object.pmids_for_query(query)

    def _get_abstracts(self, pubmed_ids: List[str]) -> List[ScientificAbstract]:
        # 获取 PubMed 文摘
        self.logger.info(f'正在获取以下 PubMed ID 的文摘数据: {pubmed_ids}')
        scientific_abstracts = []

        for id in pubmed_ids:
            initial_delay = 1  # 初始延迟时间（秒）
            max_attempts = 3  # 最大尝试次数
            success = False  # 标记是否成功获取文摘

            for attempt in range(max_attempts):
                try:
                    # 尝试获取文摘
                    abstract = self.pubmed_fetch_object.article_by_pmid(id)

                    # 如果文摘内容为 None，跳过当前 PubMed ID
                    if abstract.abstract is None:
                        self.logger.warning(f'PubMed ID {id} 未找到文摘，跳过...')
                        continue

                    # 处理 authors 字段，确保它是一个列表
                    # 将 authors 转换为逗号分隔的字符串
                    authors = abstract.authors
                    if isinstance(authors, list):
                        authors = ", ".join(authors)

                    # 创建 ScientificAbstract 对象
                    abstract_formatted = ScientificAbstract(
                        doi=abstract.doi,
                        title=abstract.title,
                        authors=authors,  # 传递作者列表
                        year=abstract.year,
                        abstract_content=abstract.abstract
                    )

                    scientific_abstracts.append(abstract_formatted)
                    success = True
                    break

                except Exception as e:
                    # 如果请求失败，进行指数退避和随机延时
                    wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(f'PubMed ID {id} 的重试 {attempt + 1} 失败. 错误信息: {e}. {wait_time:.2f} 秒后重试...')
                    time.sleep(wait_time)

            if not success:
                # 如果达到最大尝试次数仍未成功，记录错误
                self.logger.error(f'在尝试 {max_attempts} 次后，仍未成功获取 PubMed ID {id} 的文摘')

        self.logger.info(f'共获取到 {len(scientific_abstracts)} 条文摘数据')
        return scientific_abstracts

    def get_abstract_data(self, scientist_question: str, simplify_query: bool = True) -> List[ScientificAbstract]:
        # 获取使用人员查询的文摘列表
        translation_question = self._translation_chain(scientist_question)
        pmids = self._get_abstract_list(translation_question, simplify_query)  # 获取 PubMed ID 列表
        abstracts = self._get_abstracts(pmids)  # 获取对应的文摘
        return abstracts
