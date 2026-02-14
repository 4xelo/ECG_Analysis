import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from typing import Dict
from .base import BaseFeatureExtractor


class FrequencyDomainFeatures(BaseFeatureExtractor):
    """
    Extrahuje príznaky z frekvenciálnej oblasti.
    Používa Welchov metódu pre výpočet PSD (Power Spectral Density).
    """
    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # Výpočet PSD pomocou Welchovej metódy
        # nperseg nastavíme na dĺžku segmentu pre maximálne rozlíšenie
        freqs, psd = welch(segment, fs=fs, nperseg=len(segment))

        # Normalizácia PSD pre výpočet entropie (suma musí byť 1)
        psd_norm = psd / (np.sum(psd) + 1e-12)

        # 1. Spektrálna Entropia (Shannonova entropia z normalizovaného PSD)
        spec_entropy = entropy(psd_norm)

        # Normalizácia PSD (aby suma bola 1 -> pre výpočet spektrálnej entropie)
        total_power = np.sum(psd)
        if total_power == 0:
            return {
                "spectral_entropy": 0.0,
                "power_lf": 0.0,
                "power_hf": 0.0,
                "dominant_freq": 0.0
            }

        # Definícia pásiem (podľa štandardov HRV/EKG)
        # LF: 0.04 - 0.15 Hz, HF: 0.15 - 0.40 Hz
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        power_lf = np.sum(psd[lf_mask]) / total_power
        power_hf = np.sum(psd[hf_mask]) / total_power

        # Dominantná frekvencia
        dom_freq = freqs[np.argmax(psd)]

        return {
            "spectral_entropy": float(spec_entropy),
            "power_lf": float(power_lf),
            "power_hf": float(power_hf),
            "dominant_freq": float(dom_freq)
        }
