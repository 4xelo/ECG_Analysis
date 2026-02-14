import numpy as np
from typing import Dict
from numba import jit
from .base import BaseFeatureExtractor


# --- Numba JIT Funkcia (Static, mimo triedy pre kompiláciu) ---
@jit(nopython=True)
def _calculate_sample_entropy(signal: np.ndarray, m: int, r: float) -> float:
    """
    Rýchly výpočet Sample Entropy pomocou Numba JIT (Just-In-Time) kompilácie.
    Zložitosť O(N^2) je v čistom Pythone pomalá, Numba to zrýchľuje na úroveň C.
    """
    N = len(signal)
    if N < m + 1:
        return 0.0

    # Helper function inside JIT
    def _phi(m_len):
        count = 0.0
        num_vectors = N - m_len + 1
        for i in range(num_vectors):
            for j in range(num_vectors):
                if i == j: continue
                # Chebyshev distance (max rozdiel po zložkách)
                dist = 0.0
                for k in range(m_len):
                    d = abs(signal[i+k] - signal[j+k])
                    if d > dist: dist = d
                    if dist > r: break
                if dist <= r:
                    count += 1.0
        return count / (num_vectors * (num_vectors - 1))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m > 0 and phi_m1 > 0:
        return -np.log(phi_m1 / phi_m)
    else:
        return 0.0


class EntropyFeatures(BaseFeatureExtractor):
    """
    Extrahuje Výberovú entropiu (Sample Entropy/SampEn) z EKG signálu.

    Sample Entropy je miera zložitosti a nepravidelnosti časového radu.
    Nižšie hodnoty znamenajú, že signál je pravidelnejší a predvídateľnejší.
    Vyššie hodnoty indikujú vyššiu mieru chaosu a nepravidelnosti (často spojené so zdravým srdcom,
    zatiaľ čo príliš nízka entropia môže indikovať patológiu alebo starnutie).

    Táto implementácia využíva 'Numba' pre extrémne rýchly výpočet.

    Attributes:
        m (int): Dĺžka porovnávaných vzorov (Embedding dimension). Štandard pre EKG je 2.
        r_factor (float): Koeficient tolerancie. Prahová hodnota 'r' sa vypočíta dynamicky
                          pre každý segment ako r = r_factor * std(segment).
                          Štandardná hodnota je 0.2 (20% smerodajnej odchýlky).
    """
    def __init__(self, m: int = 2, r_factor: float = 0.2):
        self.m = m
        self.r_factor = r_factor  # Tolerancia r = r_factor * std

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Vypočíta Sample Entropy pre zadaný segment.

        :param segment: 1D Numpy pole so signálom (odporúča sa dĺžka aspoň pár stoviek vzoriek).
        :param fs: Vzorkovacia frekvencia (tu sa nepoužíva, ale je súčasťou rozhrania).
        :return: Slovník {'entropy_sample': hodnota}.
        """
        # Normalizácia
        std_val = np.std(segment)
        if std_val == 0: return {"entropy_sample": 0.0}

        # Vstup do entropie by mal byť ideálne normalizovaný
        # Ale SampleEn sa často počíta s r = 0.2 * std pôvodného signálu
        r = self.r_factor * std_val

        # Volanie Numba funkcie
        sampen = _calculate_sample_entropy(segment, self.m, r)

        return {
            "entropy_sample": float(sampen)
        }

