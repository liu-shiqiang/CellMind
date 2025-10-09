from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EnrichmentAnalysiszer(ABC):
    @abstractmethod
    def run(self, gene_list: List[str]) -> Dict[str, Any]:
        pass

class EnrichmentVisualizer(ABC):
    @abstractmethod
    def plot(self, results: Dict[str, Any]) -> str:
        pass

class EnrichmentEvaluator(ABC):
    @abstractmethod
    def evaluate(self, results: Dict[str, Any]) -> str:
        pass

class EnrichmentFactory(ABC):
    @abstractmethod
    def create_analyzer(self) -> EnrichmentAnalysiszer:
        pass

    @abstractmethod
    def create_visualizer(self) -> EnrichmentVisualizer:
        pass

    @abstractmethod
    def create_evaluator(self) -> EnrichmentEvaluator:
        pass
