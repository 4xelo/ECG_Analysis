import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from typing import Dict, Any
from sklearn.base import BaseEstimator


class HyperparameterTuner:
    """
    Zodpovedá za Ladenie hyperparametrov.
    Používa RandomizedSearchCV na nájdenie najlepšej kombinácie.
    """

    def __init__(self, n_iter: int = 100, n_splits: int = 10, random_state: int = 42):
        """
        :param n_iter: Počet náhodných kombinácií, ktoré sa vyskúšajú.
                       Pre rýchle testovanie daj 10-20. Pre finálny beh 100-1000.
        :param n_splits: Počet foldov krížovej validácie (10 je štandard).
        """
        self.n_iter = n_iter
        self.random_state = random_state
        # StratifiedKFold zabezpečí, že v každom folde bude rovnaký pomer zdravých/chorých
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def tune(self, estimator: BaseEstimator, param_grid: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[
        str, Any]:
        """
        Spustí proces ladenia.

        :param estimator: Čistá inštancia modelu (napr. SVC() alebo RandomForestClassifier()).
        :param param_grid: Slovník s rozsahmi parametrov (z get_hyperparameter_grid).
        :return: Slovník s najlepšími parametrami.
        """
        print(f" [*] Tuning: Spúšťam RandomizedSearchCV ({self.n_iter} iterácií) pre {estimator.__class__.__name__}...")

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=self.n_iter,
            scoring='roc_auc',  # Optimalizujeme podľa ROC-AUC (v medicíne kľúčové)
            cv=self.cv,
            verbose=1,  # Vypisuje progress bar
            random_state=self.random_state,
            n_jobs=-1,  # Využije všetky jadrá CPU
            error_score='raise'  # Ak nejaká kombinácia zlyhá, vyhodí chybu (neschová ju)
        )

        search.fit(X, y)

        print(f" [*] Najlepšie ROC-AUC (val): {search.best_score_:.4f}")
        print(f" [*] Najlepšie parametre: {search.best_params_}")

        return search.best_params_
