import numpy as np

class QualityControl:
    """
    Trieda na validáciu kvality EKG segmentov.
    Slúži na odfiltrovanie segmentov, ktoré obsahujú:
    1. Pohybové artefakty (extrémne vysoká amplitúda).
    2. Odpojené elektródy (nulová čiara / flatline).
    3. Matematické chyby (NaN, Infinity).
    """

    def __init__(self, max_amplitude: float = 3000.0, min_std: float = 5.0):
        """
        :param max_amplitude: Prahová hodnota pre artefakty.
                              Bežné EKG má < 1000. Artefakty majú > 5000.
                              Nastavenie na 3000 je bezpečná hranica.
        :param min_std: Minimálna smerodajná odchýlka.
                        Ak je signál "mŕtvy" (rovná čiara), std bude blízko 0.
        """
        self.max_amplitude = max_amplitude
        self.min_std = min_std

    def is_valid(self, segment: np.ndarray) -> bool:
        """
        Skontroluje segment a vráti True, ak je vhodný na analýzu.
        """
        # 1. Kontrola na prázdny segment
        if segment is None or len(segment) == 0:
            return False

        # 2. Kontrola na NaN (Not a Number) a Inf (nekonečno)
        if np.isnan(segment).any() or np.isinf(segment).any():
            return False

        # 3. Kontrola extrémnej amplitúdy (Pohybové artefakty)
        # Ak maximum absolútnej hodnoty prekročí limit, je to šum.
        if np.max(np.abs(segment)) > self.max_amplitude:
            return False

        # 4. Kontrola Flatline (Odpojená elektróda)
        # Ak sa signál takmer nehýbe, nemá zmysel ho analyzovať.
        if np.std(segment) < self.min_std:
            return False

        return True
