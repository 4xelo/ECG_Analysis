import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Union, List
from dataclasses import dataclass
import logging


# Nastavenie jednoducheho loggera
logger = logging.getLogger(__name__)

@dataclass
class ECGRecord:
    """Class representing one loaded ECG record"""
    filename: str
    signal: np.ndarray
    label: int  # 0 -> healthy / 1 -> risk


class ECGLoader:
    """
    Trieda zodpovedná za načítavanie surových EKG dát z disku.
    """
    def __init__(self, data_dir: Union[str, Path], label: int):
        """
        Inicializuje loader s cestou k priečinku s dátami.
        :param data_dir: Cesta k priečinku (napr. "data/raw/Positive")
        :param label: Label (0 -> healthy / 1 -> risk)
        """
        self.data_dir = Path(data_dir)
        self.label = label

        # Rýchla validácia pri inicializácii
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Priečinok neexistuje: {self.data_dir}")

    def _load_csv(self, file_path: Path) -> np.ndarray:
        """
        Súkromná metóda na načítanie čistého .csv súboru.
        """
        try:
            # Načítanie súboru
            df = pd.read_csv(file_path)

            # Vezmeme prvý stĺpec a konvertujeme na float32 pre úsporu pamäte (RAM)
            signal = df.iloc[:, 0].values.astype(np.float32)
            return signal

        except Exception as e:
            logger.error(f"Chyba pri čítaní CSV súboru {file_path.name}: {e}")
            raise

    def _load_crv(self, file_path: Path, skip_rows: int = 5) -> np.ndarray:
        """
        Súkromná metóda na načítanie .crv súboru s hlavičkou.
        (Užitočná, pri použití nových dát -> dáta, ktoré neprešli transformáciou na .csv).
        """
        try:
            # Preskočíme prvé riadky s metadátami (hlavičku)
            df = pd.read_csv(file_path, skiprows=skip_rows)
            signal = df.iloc[:, 0].values.astype(np.float32)
            return signal

        except Exception as e:
            logger.error(f"Chyba pri čítaní CRV súboru {file_path.name}: {e}")
            raise

    def load_single_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Načíta jeden konkrétny súbor na základe jeho prípony.
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() == '.crv':
            return self._load_crv(file_path)
        else:
            raise ValueError(f"Nepodporovaný formát súboru: {file_path.suffix}")

    def get_file_paths(self, extension: str = "*.csv") -> List[Path]:
        """
        Vráti zotriedený zoznam všetkých ciest k súborom s danou príponou.
        """
        # rglob hľadá rekurzívne aj v podpriečinkoch
        return sorted(self.data_dir.rglob(extension))

    def load_all(self, extension: str = "*.csv") -> Generator[ECGRecord, None, None]:
        """
        Generátor, ktorý načíta súbory a vytvorí ECGRecord s prednastaveným labelom.
        """
        files = self.get_file_paths(extension)
        logger.info(f"Nájdených {len(files)} súborov v {self.data_dir} (Label: {self.label})")

        for file_path in files:
            signal = self.load_single_file(file_path)

            record = ECGRecord(
                filename=file_path.name,
                signal=signal,
                label=self.label  # Použije label zadaný pri inicializácii loadra
            )

            yield record