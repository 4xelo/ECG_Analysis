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

    # ==========================================
    # METÓDA NA PREHLIADANIE GRAFU
    # ==========================================
    def browse_signal(self, signal: np.ndarray, r_peaks: Optional[np.ndarray] = None,
                      start_s: float = 0.0, end_s: Optional[float] = None,
                      window_s: float = 5.0):
        """
        Interaktívne prechádza signál po segmentoch (napr. po 5 sekundách).

        :param signal: Celý (filtrovaný) signál.
        :param r_peaks: (Voliteľné) Indexy R-vrcholov pre vizualizáciu.
        :param start_s: Čas začiatku prehliadania (napr. 260).
        :param end_s: Čas konca prehliadania (napr. 300). Ak None, ide až do konca signálu.
        :param window_s: Dĺžka jedného zobrazeného okna v sekundách.
        """
        # Ak nie je zadaný koniec, nastavíme ho na koniec signálu
        total_duration = len(signal) / self.fs
        if end_s is None or end_s > total_duration:
            end_s = total_duration

        # Ak nemáme r_peaks, vyrobíme prázdne pole, aby funkcia nepadla
        if r_peaks is None:
            r_peaks = np.array([])

        current_start = start_s

        print(f"\n=== Spúšťam prehliadač signálu ({start_s}s - {end_s}s) ===")
        print(f"Inštrukcie: Zatvor okno grafu pre posun na ďalší segment.")
        print(f"            V konzole napíš 'q' a stlač Enter pre ukončenie.\n")

        while current_start < end_s:
            current_end = current_start + window_s

            # Orezanie konca, aby sme neprešli za požadovaný end_s
            if current_end > end_s:
                current_end = end_s

            # Využijeme existujúcu metódu na vykreslenie
            self.plot_peaks_on_signal(
                signal=signal,
                r_peaks=r_peaks,
                start_s=current_start,
                end_s=current_end,
                title=f"Prehliadanie úseku {current_start:.1f}s - {current_end:.1f}s"
            )

            # Interakcia v konzole
            if current_end >= end_s:
                print("Dosiahli ste koniec požadovaného úseku.")
                break

            # Tu sa program zastaví a čaká na Enter (po zatvorení grafu)
            user_input = input(
                f"Zobrazený úsek {current_start:.1f}-{current_end:.1f}s. [Enter] pre ďalší, [q] pre koniec: ")

            if user_input.lower() == 'q':
                print("Prehliadanie ukončené používateľom.")
                break

            # Posun na ďalšie okno
            current_start += window_s
        # showcase:
        # print(f" [*] Hľadám segment pre R-vrchol s indexom: {TARGET_R_PEAK}...")
        #
        # plotter.browse_signal(
        #     signal=clean_signal,  # Celý filtrovaný signál
        #     r_peaks=r_peaks,  # Všetky R-vrcholy
        #     start_s=250.0,  # Začiatok (napr. kde sa ti niečo nezdá)
        #     end_s=271.0,  # Koniec
        #     window_s=3.0  # Veľkosť kroku (zoom)
        # )
        #
        # print(" [*] Prehliadanie dokončené, pokračujem v pipeline...")