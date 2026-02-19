from scipy.stats import loguniform, uniform
from sklearn.svm import SVC, LinearSVC
from typing import Dict, Any
from .base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.name = "SVM"

    @property
    def requires_scaling(self) -> bool:
        return True  # SVM ZLYHÁ bez škálovania!

    def get_estimator(self):
        # Finálny model bude nelineárny (RBF kernel), lebo EKG je zložité
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,  # Nutné pre ROC-AUC
            random_state=self.random_state
        )

    def get_preliminary_estimator(self):
        # Na výber príznakov použijeme LINEÁRNE SVM,
        # pretože to má atribút 'coef_', z ktorého vieme určiť dôležitosť.
        # RBF SVM tento atribút nemá.
        return LinearSVC(
            class_weight='balanced',
            dual=False,  # Odporúčané pre n_samples > n_features
            random_state=self.random_state,
            max_iter=2000
        )

    def get_hyperparameter_grid(self) -> Dict[str, Any]:
        """
        Definícia priestoru pre RandomizedSearchCV pomocou spojitých distribúcií.
        Umožňuje modelu nájsť presnejšie hodnoty (napr. C=23.4 namiesto C=10).
        """
        return {
            # C: Regularizácia. Hľadáme v rozsahu 0.1 až 1000.
            # loguniform zabezpečí, že malé hodnoty (0.1-1) budú testované
            # rovnako často ako veľké (100-1000).
            'C': loguniform(1e-1, 1e3),  # 0.1 až 1000

            # Gamma: Koeficient jadra. Hľadáme v rozsahu 0.0001 až 1.
            'gamma': loguniform(1e-4, 1e0), # 0.0001 až 1

            # Kernel ostáva fixný, lebo to nie je číslo
            'kernel': ['rbf'],

            # Class weight môžeme tiež skúsiť meniť (voliteľné)
            # Buď 'balanced', alebo None (žiadne váženie)
            'class_weight': ['balanced', None]
        }
