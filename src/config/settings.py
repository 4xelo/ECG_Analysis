from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    FS: int = 512
    PRE_WINDOW_S: float = 2.0
    POST_WINDOW_S: float = 2.0
    RAW_DATA_PATH: Path = Path("data/raw")
    PROCESSED_DATA_PATH: Path = Path("data/processed")
    RANDOM_STATE: int = 42

    # Odvodené parametre (vypočítajú sa automaticky, netreba ich zadávať)
    # Použijeme init=False, aby ich konštruktor od používateľa nepýtal
    PRE_SAMPLES: int = field(init=False)
    POST_SAMPLES: int = field(init=False)
    TOTAL_SAMPLES: int = field(init=False)

    def __post_init__(self):
        self.PRE_SAMPLES = int(self.PRE_WINDOW_S * self.FS)
        self.POST_SAMPLES = int(self.POST_WINDOW_S * self.FS)
        self.TOTAL_SAMPLES = self.PRE_SAMPLES + self.POST_SAMPLES


config = Config()


