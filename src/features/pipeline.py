import numpy as np
import logging
from typing import List, Dict, Any

from .base import BaseFeatureExtractor
# Importujeme konkrétne implementácie pre potreby továrenskej metódy
from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from .entropy import EntropyFeatures
from .hrv_metrics import HRVFeatures
from .wavelet import WaveletFeatures
from .morphology import MorphologyFeatures

# Nastavenie loggera
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """
    Orchestrátor, ktorý spravuje a spúšťa kolekciu extraktorov príznakov.
    """

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        """
        :param extractors: Zoznam inštancií tried dediacich od BaseFeatureExtractor.
        """
        self.extractors = extractors

    def process_segment(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Spustí všetky zaregistrované extraktory nad daným segmentom a zlúči výsledky.

        :param segment: EKG signál (1D pole).
        :param fs: Vzorkovacia frekvencia.
        :return: Jeden plochý slovník obsahujúci všetky vypočítané príznaky.
        """
        combined_features = {}

        for extractor in self.extractors:
            try:
                # 1. Extrakcia príznakov z konkrétneho extraktora
                features = extractor.extract(segment, fs)

                # 2. Aktualizácia hlavného slovníka (merge)
                combined_features.update(features)

            except Exception as e:
                # Ak nastane chyba (napr. v HRV alebo Wavelete), zalogujeme ju,
                # ale NEZASTAVÍME program. Ostatné features sa vypočítajú.
                extractor_name = extractor.__class__.__name__
                logger.error(f"Chyba v extraktore {extractor_name}: {e}")

                # Voliteľne: Môžeme sem vložiť NaN hodnoty, ak poznáme kľúče,
                # ale jednoduchšie je nechať ich chýbať a Pandas to potom vyrieši.

        return combined_features

    @staticmethod
    def create_default_pipeline() -> 'FeatureExtractionPipeline':
        """
        Továrenská metóda, ktorá vytvorí pipeline s konfiguráciou z diplomovej práce.
        Obsahuje:
          - Time Domain (Mean, Std, Skew, Kurtosis...)
          - Frequency Domain (Welch PSD, LF/HF)
          - Entropy (Sample Entropy, m=2, r=0.2)
          - HRV (SDNN, RMSSD, pNN50)
          - Wavelet (db4, level 3)
          - Morphology (R-amp, QRS-width)
        """
        return FeatureExtractionPipeline(extractors=[
            TimeDomainFeatures(),
            FrequencyDomainFeatures(),
            EntropyFeatures(m=2, r_factor=0.2),
            HRVFeatures(),
            WaveletFeatures(wavelet_name="db4", level=3),
            MorphologyFeatures()
        ])