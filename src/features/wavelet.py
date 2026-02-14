import numpy as np
import pywt
from typing import Dict
from .base import BaseFeatureExtractor


class WaveletFeatures(BaseFeatureExtractor):
    """
    Extrahuje energiu waveletových koeficientov (DWT).
    Vykonáva Z-score normalizáciu pred transformáciou, aby boli hodnoty
    konzistentné bez ohľadu na zosilnenie signálu.
    """

    def __init__(self, wavelet_name: str = "db4", level: int = 3):
        self.wavelet_name = wavelet_name
        self.level = level

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        features = {}

        # ==========================================
        # 1. NORMALIZÁCIA (Kritická pre zhodu s diplomovkou)
        # ==========================================
        mean_val = np.mean(segment)
        std_val = np.std(segment)

        # Ochrana proti deleniu nulou
        if std_val == 0:
            std_val = 1e-8

        # Vytvoríme normalizovanú verziu signálu (strip_norm)
        segment_norm = (segment - mean_val) / std_val
        # ==========================================

        try:
            # DWT dekompozícia na NORMALIZOVANOM signáli
            coeffs = pywt.wavedec(segment_norm, self.wavelet_name, level=self.level)

            # Výpočet energie pre každú úroveň
            for i, c in enumerate(coeffs):
                energy = np.sum(c ** 2)

                # Level 0 = Aproximácia, Level 1..N = Detaily
                key_name = f"wav_energy_level_{i}"
                features[key_name] = float(energy)

        except Exception as e:
            print(f"Wavelet Error: {e}")
            features = {f"wav_energy_level_{i}": 0.0 for i in range(self.level + 1)}

        return features