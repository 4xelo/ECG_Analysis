from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
from .base_model import BaseModel
from scipy.stats import randint


class RandomForestModel(BaseModel):  # <--- Dedičnosť
    """
    Konkrétna implementácia pre Random Forest.
    """

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)  # Inicializácia rodiča
        self.name = "RandomForest"

    @property
    def requires_scaling(self) -> bool:
        return False  # RF nepotrebuje škálovanie

    def get_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_preliminary_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_hyperparameter_grid(self) -> Dict[str, Any]:
        return {
            # Počet stromov: 100 až 500 je zlatý stred.
            # Menej ako 100 je nestabilné, viac ako 500 zbytočne spomaľuje tréning.
            'n_estimators': randint(100, 501),  # (horná hranica je exkluzívna, takže do 500)

            # Hĺbka stromu: Príliš hlboký strom (napr. 100) sa naučí šum (Overfitting).
            # Príliš plytký (napr. 2) sa nenaučí nič (Underfitting).
            # Rozsah 5-50 je pre EKG ideálny.
            'max_depth': randint(5, 50),

            # Koľko vzoriek musí ostať na konci vetvy.
            # Väčšie číslo = robustnejší model (menej citlivý na šum).
            # 1-10 je bezpečný rozsah.
            'min_samples_leaf': randint(2, 11),

            # Koľko vzoriek je potrebných na rozdelenie uzla.
            # Musí to byť viac ako min_samples_leaf.
            'min_samples_split': randint(10, 30),

            'max_features': ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }
