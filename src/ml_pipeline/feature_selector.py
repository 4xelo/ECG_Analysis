import pandas as pd
import numpy as np


class FeatureSelector:
    def __init__(self, estimator, n_features: int = 40):
        self.estimator = estimator
        self.n_features = n_features
        self.selected_features_ = []

    def fit(self, x: pd.DataFrame, y: pd.Series):
        print(f" [*] Feature Selection: Trénujem {self.estimator.__class__.__name__}...")
        self.estimator.fit(x, y)

        # LOGIKA PRE RÔZNE MODELY
        if hasattr(self.estimator, 'feature_importances_'):
            # Pre Random Forest, XGBoost
            importances = self.estimator.feature_importances_
        elif hasattr(self.estimator, 'coef_'):
            # Pre Linear SVM, Logistic Regression
            # Berieme absolútnu hodnotu koeficientov (signifikancia)
            importances = np.abs(self.estimator.coef_[0])
        else:
            raise ValueError(f"Model {self.estimator.__class__.__name__} nemá atribút pre dôležitosť príznakov!")

        indices = np.argsort(importances)[::-1]
        top_indices = indices[:self.n_features]
        self.selected_features_ = x.columns[top_indices].tolist()

        print(f" [*] Vybraných {self.n_features} príznakov.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]