import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
import logging


logger = logging.getLogger(__name__)


class ECGPlotter:
    """Trieda na vizualizáciu EKG signálov a výsledkov predspracovania."""

    def __init__(self, fs: int):
        self.fs = fs

    def plot_raw_vs_filtered(self, raw_signal: np.ndarray, filtered_signal: np.ndarray,
                             start_s: float = 0.0, end_s: float = 5.0,
                             title: str = "Pôvodný vs. Filtrovaný signál",
                             save_path: Optional[Union[str, Path]] = None):
        """
        Vykreslí pod seba surový a prefiltrovaný signál pre vizuálne porovnanie.
        """
        # Prepočet sekúnd na indexy poľa
        start_idx = int(start_s * self.fs)
        end_idx = int(end_s * self.fs)

        # Ochrana proti indexom mimo poľa
        end_idx = min(end_idx, len(raw_signal))

        raw_slice = raw_signal[start_idx:end_idx]
        filt_slice = filtered_signal[start_idx:end_idx]
        time_axis = np.arange(start_idx, end_idx) / self.fs

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"{title} ({start_s}s - {end_s}s)", fontsize=14)

        # 1. Pôvodný signál (Raw)
        axes[0].plot(time_axis, raw_slice, color='#B0BEC5', linewidth=1.5, label='Raw ECG')
        axes[0].set_title("Pôvodný signál (obsahuje šum a baseline wander)", fontsize=11)
        axes[0].set_ylabel("Amplitúda")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(loc="upper right")

        # 2. Filtrovaný signál
        axes[1].plot(time_axis, filt_slice, color='#2196F3', linewidth=1.5, label='Filtered ECG')
        axes[1].set_title("Prefiltrovaný signál (0.5 - 40 Hz)", fontsize=11)
        axes[1].set_xlabel("Čas [s]")
        axes[1].set_ylabel("Amplitúda")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graf uložený do: {save_path}")
        else:
            plt.show()

        # Zatvorenie figúry, aby sme uvoľnili RAM pamäť
        plt.close(fig)

    def plot_peaks_on_signal(self, signal: np.ndarray, r_peaks: np.ndarray,
                             start_s: float = 0.0, end_s: float = 5.0,
                             title: str = "Detegované R-vrcholy"):
        """
        Vykreslí signál a označí na ňom nájdené R-vrcholy bodkami.
        """
        start_idx = int(start_s * self.fs)
        end_idx = int(end_s * self.fs)
        end_idx = min(end_idx, len(signal))

        sig_slice = signal[start_idx:end_idx]
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # Vyberieme len tie R-vrcholy, ktoré padnú do nášho zobrazeného okna
        peaks_in_window = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
        peaks_time = peaks_in_window / self.fs
        peaks_amps = signal[peaks_in_window]

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, sig_slice, color='#2196F3', linewidth=1.5, label='Filtered ECG')

        # Vykreslenie R-vrcholov ako červených bodiek (scatter)
        plt.scatter(peaks_time, peaks_amps, color='#F44336', s=50, zorder=5, label='R-peaks')

        plt.title(f"{title} ({start_s}s - {end_s}s)", fontsize=14)
        plt.xlabel("Čas [s]")
        plt.ylabel("Amplitúda")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_segment(self, segment: np.ndarray, title: str = "Vyrezaný EKG Segment",
                     r_peak_idx_in_segment: Optional[int] = None,
                     start_time_s: float = 0.0,
                     save_path: Optional[Union[str, Path]] = None):
        """
        Vykreslí jeden konkrétny vyrezaný EKG segment.

        :param segment: Numpy pole obsahujúce hodnoty segmentu (napr. 2048 vzoriek).
        :param title: Názov grafu.
        :param r_peak_idx_in_segment: Relatívny index R-vrcholu v rámci tohto segmentu.
        :param start_time_s: Čas (v sekundách), kedy tento segment začína v pôvodnom zázname.
        :param save_path: Cesta pre uloženie grafu (voliteľné).
        """
        # Vytvoríme časovú os POSUNUTÚ o reálny čas začiatku segmentu
        time_axis = start_time_s + (np.arange(len(segment)) / self.fs)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_axis, segment, color='#2196F3', linewidth=1.5, label='EKG Signál')

        # Ak sme zadali, kde je R-vrchol, nakreslíme naň 'x'
        if r_peak_idx_in_segment is not None:
            # Vypočítame reálny čas R-vrcholu
            r_time = start_time_s + (r_peak_idx_in_segment / self.fs)
            r_amplitude = segment[r_peak_idx_in_segment]

            ax.plot(r_time, r_amplitude, marker='x', color='#F44336', markersize=10,
                    markeredgewidth=2, linestyle='None', label='Stred (R-vrchol)')
            ax.legend(loc="upper right")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Reálny čas záznamu [s]")  # Zmena popisku osi X
        ax.set_ylabel("Amplitúda")
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # logger.info(f"Segment uložený do: {save_path}")
        else:
            plt.show()

        plt.close(fig)