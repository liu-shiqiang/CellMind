from abc import ABC, abstractmethod
from typing import List
from .data_repository import ScientificAbstract


class AbstractRetriever(ABC):

    @abstractmethod
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """ Retrieve a list of scientific abstracts based on a given query. """
        raise NotImplementedError