import numpy as np
from typing import Dict
from .base import BaseFeatureExtractor


class HjorthFeatures(BaseFeatureExtractor):
    """
    Extrahuje Hjorthove parametre (Activity, Mobility, Complexity).
    Tieto parametre popisujú spektrálne vlastnosti signálu v časovej oblasti.
    Sú robustné a vhodné aj pre krátke segmenty.
    """

    def extract(self, segment: np.ndarray, fs: int) -> Dict[str, float]:
        # 1. Výpočet prvej a druhej derivácie signálu
        # (Diskretna derivácia = rozdiel susedných vzoriek)
        first_deriv = np.diff(segment)
        second_deriv = np.diff(first_deriv)

        # 2. Výpočet variancií (rozptylov)
        var_zero = np.var(segment)      # Variancia pôvodného signálu
        var_first = np.var(first_deriv) # Variancia 1. derivácie
        var_second = np.var(second_deriv) # Variancia 2. derivácie

        # Ochrana proti deleniu nulou (ak by bol signál flatline)
        if var_zero == 0:
            return {
                "hjorth_activity": 0.0,
                "hjorth_mobility": 0.0,
                "hjorth_complexity": 0.0
            }

        # 3. Hjorth Activity (Aktivita)
        # Reprezentuje celkový výkon (energiu) signálu (= variancia)
        activity = var_zero

        # 4. Hjorth Mobility (Mobilita)
        # Odhad strednej frekvencie signálu
        mobility = np.sqrt(var_first / var_zero)

        # 5. Hjorth Complexity (Komplexita)
        # Miera podobnosti signálu s čistou sínusoidou.
        # Hodnota 1 = čistá sínusoida. Vyššie hodnoty = komplexnejší/šumovejší signál.
        if mobility > 0 and var_first > 0:
            mobility_first_deriv = np.sqrt(var_second / var_first)
            complexity = mobility_first_deriv / mobility
        else:
            complexity = 0.0

        return {
            "hjorth_activity": float(activity),
            "hjorth_mobility": float(mobility),
            "hjorth_complexity": float(complexity)
        }
