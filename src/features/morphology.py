import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import Dict
from .base import BaseFeatureExtractor


class MorphologyFeatures(BaseFeatureExtractor):
    """
    Extrahuje tvarové vlastnosti QRS komplexov (šírka, amplitúda).
    """

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # 1. Nájdeme vrcholy v segmente (lokálne)
        # Hľadáme dominantné R-vrcholy
        peaks, properties = find_peaks(segment, height=0, prominence=0.5)

        if len(peaks) == 0:
            return {"r_amp_mean": 0.0, "qrs_width_mean": 0.0}

        # 2. Priemerná amplitúda (výška) R-vrcholov
        # properties['peak_heights'] obsahuje výšky nájdených píkov
        r_amp_mean = np.mean(properties['peak_heights'])

        # 3. Priemerná šírka QRS komplexov
        # Scipy vypočíta šírku v polovici prominencie píku (rel_height=0.5)
        # Výstup je vo vzorkách, musíme deliť fs pre sekundy alebo * 1000 pre ms
        widths_samples = peak_widths(segment, peaks, rel_height=0.5)[0]

        # Prevod na milisekundy (v EKG sa šírka udáva v ms)
        # Ak chceš v sekundách, vymaž '* 1000'
        widths_ms = (widths_samples / fs) * 1000

        qrs_width_mean = np.mean(widths_ms)

        return {
            "r_amp_mean": float(r_amp_mean),
            "qrs_width_mean": float(qrs_width_mean)
        }