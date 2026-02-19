from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator


class BaseModel(ABC):
    """
    Abstraktná trieda pre všetky modely strojového učenia.
    Vynucuje implementáciu metód pre získanie estimátora a mriežky parametrov.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.name = "BaseModel"

    @property
    def requires_scaling(self) -> bool:
        """
        Vráti True, ak model vyžaduje normalizáciu dát (napr. SVM, KNN, MLP).
        Vráti False pre stromové modely (RF, XGBoost).
        Predvolene False.
        """
        return False

    @abstractmethod
    def get_estimator(self) -> BaseEstimator:
        """
        Musí vrátiť čistú inštanciu modelu (napr. RandomForestClassifier)
        pripravenú na trénovanie.
        """
        pass

    @abstractmethod
    def get_preliminary_estimator(self) -> BaseEstimator:
        """
        Musí vrátiť inštanciu modelu s fixnými parametrami, ktorá sa použije
        na výber príznakov (Feature Selection).
        """
        pass

    @abstractmethod
    def get_hyperparameter_grid(self) -> Dict[str, Any]:
        """
        Musí vrátiť slovník parametrov pre RandomizedSearchCV.
        """
        pass