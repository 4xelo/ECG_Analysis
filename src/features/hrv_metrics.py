import numpy as np
from scipy.signal import find_peaks
from typing import Dict
from .base import BaseFeatureExtractor


class HRVFeatures(BaseFeatureExtractor):
    """
    Extrahuje lokálne metriky variability srdcového rytmu (HRV).
    Implementácia presne kopíruje logiku z diplomovej práce.
    """

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # 1. Lokálna detekcia R-vrcholov v rámci segmentu
        # Potrebujeme ich nájsť znova, lebo 'segment' je len pole hodnôt.
        # Parametre: min vzdialenosť 200ms, výška aspoň 50% maxima segmentu.
        distance = int(0.200 * fs)
        height_thr = 0.5 * np.max(segment) if np.max(segment) > 0 else None

        r_peaks, _ = find_peaks(segment, distance=distance, height=height_thr)

        if len(r_peaks) < 2:
            return {
                "hrv_mean_rr": np.nan,
                "hrv_sdnn": np.nan,
                "hrv_rmssd": np.nan,
                "hrv_pnn50": np.nan
            }

        # Výpočet RR intervalov v sekundách
        rr = np.diff(r_peaks) / fs

        # RMSSD a pNN50 (vyžadujú aspoň 2 RR intervaly, t.j. 3 vrcholy)
        if len(rr) > 1:
            diff_rr = np.diff(rr)
            rmssd = np.sqrt(np.mean(diff_rr ** 2))
            pnn50 = np.mean(np.abs(diff_rr) > 0.05)
        else:
            rmssd = np.nan
            pnn50 = 0.0  # Tvoja logika: ak je len 1 interval, pnn50 je 0

        # Návrat hodnôt (s prefixom 'hrv_' pre poriadok v datasete)
        return {
            "hrv_mean_rr": float(np.mean(rr)),
            "hrv_sdnn": float(np.std(rr, ddof=1)),  # ddof=1 pre výberovú smerodajnú odchýlku
            "hrv_rmssd": float(rmssd),
            "hrv_pnn50": float(pnn50)
        }
