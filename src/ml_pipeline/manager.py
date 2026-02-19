import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

from .feature_selector import FeatureSelector
from .tuner import HyperparameterTuner
from .validator import ModelValidator
from ..models.base_model import BaseModel


class PipelineManager:
    """
    Orchestrátor (Facade), ktorý riadi celý proces trénovania modelu:
    Preprocessing -> Feature Selection -> Tuning -> Validation -> Saving.
    """

    def __init__(self,
                 output_dir: str = "models",
                 n_iter: int = 50,
                 n_splits: int = 10,
                 random_state: int = 42):
        """
        :param output_dir: Kam sa budú ukladať .joblib súbory.
        :param n_iter: Počet iterácií pre Model.
        :param n_splits: Počet foldov pre Cross Validation.
        """
        self.output_dir = output_dir
        self.n_iter = n_iter
        self.n_splits = n_splits
        self.random_state = random_state

        # Vytvoríme priečinok pre modely, ak neexistuje
        os.makedirs(self.output_dir, exist_ok=True)

    def run_pipeline(self, model_def: BaseModel, X: pd.DataFrame, y: pd.Series):
        print(f"\n{'=' * 50}")
        print(f" SPÚŠŤAM PIPELINE PRE: {model_def.name}")
        print(f"{'=' * 50}")

        # --- KROK 1: PREPROCESSING (Škálovanie) ---
        # Ak model (napr. SVM) vyžaduje škálovanie, urobíme to tu.
        if model_def.requires_scaling:
            print(" [1/5] Preprocessing: Aplikujem StandardScaler...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Konverzia späť na DataFrame, aby sme zachovali názvy stĺpcov
            X = pd.DataFrame(X_scaled, columns=X.columns)

            # Uloženie scalera pre budúce použitie v aplikácii
            scaler_path = os.path.join(self.output_dir, f"{model_def.name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            print(f"       Scaler uložený: {scaler_path}")
        else:
            print(" [1/5] Preprocessing: Model nevyžaduje škálovanie.")

        # --- KROK 2: FEATURE SELECTION ---
        print(" [2/5] Feature Selection...")
        selector = FeatureSelector(
            estimator=model_def.get_preliminary_estimator(),
            n_features=40
        )
        selector.fit(X, y)
        X_selected = selector.transform(X)

        # --- KROK 3: HYPERPARAMETER TUNING ---
        print(f" [3/5] Hyperparameter Tuning ({self.n_iter} iterácií)...")
        tuner = HyperparameterTuner(
            n_iter=self.n_iter,
            n_splits=self.n_splits,
            random_state=self.random_state
        )
        best_params = tuner.tune(
            estimator=model_def.get_estimator(),
            param_grid=model_def.get_hyperparameter_grid(),
            X=X_selected,
            y=y
        )

        # --- KROK 4: VALIDÁCIA ---
        print(" [4/5] Finálna Validácia...")
        # Vytvoríme inštanciu modelu s najlepšími parametrami
        final_model_candidate = model_def.get_estimator()
        final_model_candidate.set_params(**best_params)

        validator = ModelValidator(n_splits=self.n_splits)
        validator.evaluate(final_model_candidate, X_selected, y)

        # --- KROK 5: PRODUKCIA (Uloženie) ---
        print(" [5/5] Trénovanie finálneho modelu a uloženie...")
        final_model_candidate.fit(X_selected, y)

        model_path = os.path.join(self.output_dir, f"{model_def.name}_model.joblib")
        features_path = os.path.join(self.output_dir, f"{model_def.name}_features.joblib")

        joblib.dump(final_model_candidate, model_path)
        joblib.dump(selector.selected_features_, features_path)

        print(f" [OK] Pipeline úspešne dokončená.")
        print(f"      Model:    {model_path}")
        print(f"      Features: {features_path}")
        