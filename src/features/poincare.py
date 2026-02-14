import numpy as np
from scipy.signal import find_peaks
from typing import Dict
from .base import BaseFeatureExtractor


class PoincareFeatures(BaseFeatureExtractor):
    """
    Extrahuje geometrické príznaky z Poincaré plotu (SD1, SD2).

    Tieto príznaky vyžadujú sériu RR intervalov.
    Upozornenie: Na krátkych segmentoch (4s) často nebude dostatok R-vrcholov
    na výpočet. V takom prípade vracia -1.0.
    """

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # 1. Lokálna detekcia R-vrcholov (rovnako ako v HRVFeatures)
        # Hľadáme vrcholy s minimálnym odstupom 200ms
        peaks, _ = find_peaks(segment, distance=int(0.2 * fs), height=np.mean(segment))

        # 2. Kontrola dostatku dát
        # Potrebujeme aspoň 4 vrcholy -> 3 intervaly -> 2 body do grafu
        # Ak máme menej, štatistika (std) by bola nezmyselná.
        if len(peaks) < 4:
            return {
                "poincare_sd1": -1.0,
                "poincare_sd2": -1.0,
                "poincare_ratio": -1.0
            }

        # 3. Výpočet RR intervalov v milisekundách
        rr_intervals = np.diff(peaks) / fs * 1000

        # 4. Vytvorenie vektorov pre Poincaré plot
        # x = RR_n (aktuálny interval)
        # y = RR_n+1 (nasledujúci interval)
        x = rr_intervals[:-1]
        y = rr_intervals[1:]

        # 5. Výpočet SD1 a SD2 podľa geometrických vzorcov
        # SD1: Šírka elipsy (krátkodobá variabilita - kolmo na identitu)
        sd1 = np.std(np.subtract(x, y) / np.sqrt(2), ddof=1)

        # SD2: Dĺžka elipsy (dlhodobá variabilita - pozdĺž identity)
        sd2 = np.std(np.add(x, y) / np.sqrt(2), ddof=1)

        # Pomer SD1/SD2 (balans autonómneho nervového systému)
        ratio = sd1 / sd2 if sd2 > 0 else 0.0

        return {
            "poincare_sd1": float(sd1),
            "poincare_sd2": float(sd2),
            "poincare_ratio": float(ratio)
        }
