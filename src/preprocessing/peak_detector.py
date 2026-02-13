import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BasePeakDetector(ABC):
    """Abstraktná trieda pre všetky algoritmy na detekciu R-vrcholov."""

    @abstractmethod
    def detect(self, ecg_clean: np.ndarray, fs: int) -> np.ndarray:
        pass


class PanTompkinsDetector(BasePeakDetector):
    """
    Adaptívny detektor QRS komplexov inšpirovaný Pan-Tompkinsovým algoritmom.
    """

    def __init__(self, window_s: int = 10, overlap: float = 0.5,
                 search_ms: int = 100, refractory_ms: int = 250):
        self.window_s = window_s
        self.overlap = overlap
        self.search_ms = search_ms
        self.refractory_ms = refractory_ms

    def _enhance_qrs(self, ecg: np.ndarray, fs: int) -> np.ndarray:
        """Krok 1: Zvýraznenie energie QRS komplexu pomocou transformácií."""
        nyq = 0.5 * fs
        # Úzkopásmová filtrácia 5-15 Hz pre maximalizáciu energie QRS
        sos = butter(2, [5 / nyq, 15 / nyq], btype='band', output='sos')
        y = sosfiltfilt(sos, ecg)

        # Derivácia na výpočet sklonu vĺn
        b = np.array([1, 2, 0, -2, -1]) * (1 / (8 * fs))
        y = np.convolve(y, b, mode='same')

        # Umocnenie a vyhladenie posuvným oknom (150ms)
        y = y ** 2
        N = max(1, int(0.150 * fs))
        y = np.convolve(y, np.ones(N) / N, mode='same')
        return y

    def _relocalize_r(self, ecg_clean: np.ndarray, cand_idx: np.ndarray, fs: int) -> np.ndarray:
        """Krok 3: Spresnenie polohy R-vrcholu v pôvodnom filtrovanom signáli."""
        win = int(self.search_ms / 1000 * fs)
        r_idx = []
        for i in cand_idx:
            a = max(0, i - win)
            b = min(len(ecg_clean), i + win + 1)
            if b > a:
                loc = a + np.argmax(ecg_clean[a:b])
                r_idx.append(loc)
        return np.array(sorted(set(r_idx)))

    def _enforce_refractory(self, idxs: np.ndarray, fs: int) -> np.ndarray:
        """Krok 4: Odstránenie fyziologicky nemožných duplikátov (< 250ms)."""
        if len(idxs) == 0:
            return idxs
        keep = [idxs[0]]
        min_dist = int(self.refractory_ms / 1000 * fs)
        for i in idxs[1:]:
            if i - keep[-1] >= min_dist:
                keep.append(i)
        return np.array(keep)

    def detect(self, ecg_clean: np.ndarray, fs: int) -> np.ndarray:
        """
        Krok 2: Hlavná detekčná metóda aplikujúca dynamický prah na okná.

        :param ecg_clean: Prefiltrovaný EKG signál.
        :param fs: Vzorkovacia frekvencia.
        :return: Pole (Numpy array) s indexmi R-vrcholov.
        """
        try:
            qrs_signal = self._enhance_qrs(ecg_clean, fs)
            step = int(self.window_s * fs * (1 - self.overlap))
            win_len = int(self.window_s * fs)
            candidates = []

            for i in range(0, len(ecg_clean), step):
                seg = qrs_signal[i:i + win_len]
                if len(seg) == 0:
                    continue

                # Adaptívny prahovací mechanizmus pomocou MAD
                med = np.median(seg)
                mad = np.median(np.abs(seg - med)) + 1e-12
                thr = med + 4.0 * mad

                distance = int(250 / 1000 * fs)
                prom = 0.2 * (np.percentile(seg, 99) - med)

                p, _ = find_peaks(seg, height=thr, distance=distance,
                                  prominence=prom if prom > 0 else None)
                candidates.extend(i + p)

            candidates = np.array(sorted(set(candidates)))

            # Postprocessing: relokalizácia a refraktórna perióda
            r1 = self._relocalize_r(ecg_clean, candidates, fs)
            r_idx = self._enforce_refractory(r1, fs)

            return r_idx

        except Exception as e:
            logger.error(f"Chyba pri detekcii R-vrcholov: {e}")
            raise