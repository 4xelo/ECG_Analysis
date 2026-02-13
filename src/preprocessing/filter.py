import numpy as np
from scipy.signal import butter, sosfiltfilt
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseFilter(ABC):

    @abstractmethod
    def apply(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Každý filter musí implementovať túto metódu."""
        pass


class ButterworthFilter(BaseFilter):
    """
    Pásmový Butterworthov filter s nulovým fázovým posunom.
    Odstraňuje blúdenie základnej čiary a vysokofrekvenčný šum.
    """
    def __init__(self, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def apply(self, signal: np.ndarray, fs: int) -> np.ndarray:
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq

        try:
            # Návrh filtra vo formáte sekcií druhého rádu (SOS) pre stabilitu
            sos = butter(self.order, [low, high], btype='band', output='sos')

            # Aplikácia filtra v oboch smeroch (nulový fázový posun)
            filtered_signal = sosfiltfilt(sos, signal)

            return filtered_signal

        except Exception as e:
            logger.error(f"Chyba pri aplikácii Butterworthovho filtra: {e}")
            raise

