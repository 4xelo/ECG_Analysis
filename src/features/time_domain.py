import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict
from .base import BaseFeatureExtractor


class TimeDomainFeatures(BaseFeatureExtractor):
    """
    Extrahuje štatistické príznaky v časovej oblasti.
    Signál je pred výpočtom normalizovaný (Z-score), aby sa eliminoval vplyv amplitúdy.
    """
    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Vypočíta: mean, std, var, min, max, ptp, skew, kurtosis, energy.
        """

        # 1. Normalizácia (Z-score standardization)
        # Toto je kritický krok z diplomovky - features sa počítajú z 'strip_norm'
        mean_raw = np.mean(segment)
        std_raw = np.std(segment)

        # Ochrana proti deleniu nulou (ak by bol signál konštantná čiara)
        if std_raw == 0:
            std_raw = 1e-8

        strip_norm = (segment - mean_raw) / std_raw

        # 2. Výpočet príznakov na normalizovanom signáli
        # Poznámka: Pri Z-score normalizácii budú mean ~ 0 a std ~ 1,
        # ale skew, kurtosis, min, max a energy nesú dôležitú informáciu o tvare.
        return {
            "time_mean": float(np.mean(strip_norm)),
            "time_std": float(np.std(strip_norm)),
            "time_var": float(np.var(strip_norm)),
            "time_min": float(np.min(strip_norm)),
            "time_max": float(np.max(strip_norm)),
            "time_ptp": float(np.ptp(strip_norm)),  # Peak-to-peak (Max - Min)
            "time_skew": float(skew(strip_norm)),  # Šikmosť rozdelenia
            "time_kurtosis": float(kurtosis(strip_norm)),  # Špicatosť rozdelenia
            "time_energy": float(np.sum(strip_norm ** 2))  # Energia signálu
        }
    