import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 1. Abstraktná trieda
# ==========================================
class BaseSegmenter(ABC):
    """
    Abstraktná trieda definujúca spoločné rozhranie pre extrakciu EKG segmentov.
    """

    @abstractmethod
    def extract_segments(self, signal: np.ndarray, r_peaks: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Vyberie segmenty zo signálu na základe R-vrcholov.

        :param signal: Prefiltrovaný EKG signál (1D Numpy array).
        :param r_peaks: Indexy R-vrcholov v signáli.
        :return: Zoznam entít (segment_signálu, index_pôvodného_R_vrcholu).
        """
        pass


# ==========================================
# 2. Konkrétne implementácie
# ==========================================
class RCenteredSegmenter(BaseSegmenter):
    """
    Vyrezáva segmenty tak, že R-vrchol je presne v zadanom bode
    medzi 'pre_window' a 'post_window'. (Vhodné pre morfologickú analýzu).
    """

    def __init__(self, pre_window_s: float, post_window_s: float, fs: int):
        self.pre_samples = int(pre_window_s * fs)
        self.post_samples = int(post_window_s * fs)
        logger.debug(f"Inicializovaný RCenteredSegmenter: -{self.pre_samples} až +{self.post_samples} vzoriek.")

    def extract_segments(self, signal: np.ndarray, r_peaks: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        segments = []
        for r_peak in r_peaks:
            start = r_peak - self.pre_samples
            end = r_peak + self.post_samples

            # Kontrola hraníc signálu
            if start >= 0 and end <= len(signal):
                segment = signal[start:end]
                segments.append((segment, r_peak))

        return segments


class RAlignedSegmenter(BaseSegmenter):
    """
    Vyrezáva segmenty tak, že segment ZAČÍNA presne na R-vrchole
    (alebo malý kúsok pred ním) a pokračuje zadanú dobu.
    (Pôvodný prístup z tvojej diplomovky pre 10s okná).
    """

    def __init__(self, window_s: float, fs: int, offset_s: float = 0.0):
        """
        :param window_s: Celková dĺžka okna v sekundách.
        :param fs: Vzorkovacia frekvencia.
        :param offset_s: Ak chceš začať kúsok pred R-vrcholom (napr. 0.2s), zadaj kladné číslo.
        """
        self.offset_samples = int(offset_s * fs)
        self.window_samples = int(window_s * fs)
        logger.debug(f"Inicializovaný RAlignedSegmenter: dĺžka {self.window_samples} vzoriek.")

    def extract_segments(self, signal: np.ndarray, r_peaks: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        segments = []
        for r_peak in r_peaks:
            start = r_peak - self.offset_samples
            end = start + self.window_samples

            # Kontrola hraníc signálu
            if start >= 0 and end <= len(signal):
                segment = signal[start:end]
                segments.append((segment, r_peak))

        return segments