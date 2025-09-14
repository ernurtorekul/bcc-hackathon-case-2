from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseModule(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        pass