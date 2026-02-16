import numpy as np
from scipy.signal import find_peaks
from typing import Dict
from .base import BaseFeatureExtractor


class HRVFeatures(BaseFeatureExtractor):
    """
    Extrahuje lokálne metriky variability srdcového rytmu (HRV).
    """

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # 1. Lokálna detekcia R-vrcholov v rámci segmentu
        distance = int(0.200 * fs)
        # Ochrana: ak je signál plochý (max=0), height_thr musí byť None alebo 0
        seg_max = np.max(segment)
        height_thr = 0.5 * seg_max if seg_max > 0 else None

        r_peaks, _ = find_peaks(segment, distance=distance, height=height_thr)

        # Ak máme menej ako 2 vrcholy, nemáme žiadny interval
        if len(r_peaks) < 2:
            return {
                "hrv_mean_rr": np.nan,
                "hrv_sdnn": np.nan,
                "hrv_rmssd": np.nan,
                "hrv_pnn50": np.nan
            }

        # Výpočet RR intervalov v sekundách
        rr = np.diff(r_peaks) / fs

        # Inicializácia premenných
        sdnn = 0.0
        rmssd = np.nan
        pnn50 = 0.0

        # Výpočet metrík, len ak máme dostatok dát
        # Na SDNN (s ddof=1) potrebujeme aspoň 2 intervaly (3 vrcholy)
        if len(rr) > 1:
            diff_rr = np.diff(rr)
            sdnn = np.std(rr, ddof=1)
            rmssd = np.sqrt(np.mean(diff_rr ** 2))
            pnn50 = np.mean(np.abs(diff_rr) > 0.05)
        else:
            # Ak máme len 1 interval (2 vrcholy):
            # SDNN je 0 (žiadna variabilita)
            sdnn = 0.0
            # RMSSD sa nedá vypočítať (potrebuje rozdiel dvoch intervalov)
            rmssd = np.nan
            pnn50 = 0.0

        return {
            "hrv_mean_rr": float(np.mean(rr)),
            "hrv_sdnn": float(sdnn),
            "hrv_rmssd": float(rmssd),
            "hrv_pnn50": float(pnn50)
        }