from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseFeatureExtractor(ABC):
    """
    Abstraktná trieda pre všetky extraktory príznakov.
    """
    @abstractmethod
    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extrahuje príznaky z jedného segmentu.

        :param segment: 1D pole (vyrezaný signál).
        :param fs: Vzorkovacia frekvencia.
        :return: Slovník {nazov_priznaku: hodnota}.
        """
        pass
