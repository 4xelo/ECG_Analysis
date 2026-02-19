import pandas as pd
import os
from src.ml_pipeline.manager import PipelineManager
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel


def main():
    # 1. Načítanie dát
    data_path = os.path.join("data", "processed", "ecg_features_patient_level.csv")
    if not os.path.exists(data_path):
        data_path = "ecg_features_patient_level.csv"  # Fallback

    print(f"Načítavam dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Oddelenie features a labelu
    drop_cols = ['label', 'filename', 'segment_count', 'original_r_peak']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['label']

    # 2. Inicializácia Manažéra
    # Tu nastavíš globálne parametre (napr. 50 iterácií pre tuning)
    manager = PipelineManager(output_dir="models", n_iter=50, n_splits=10)

    # 3. Spustenie pre Random Forest
    # Stačí vytvoriť definíciu modelu a odovzdať manažérovi
    rf_def = RandomForestModel(random_state=42)
    manager.run_pipeline(rf_def, X, y)

    # 4. (Voliteľné) Spustenie pre SVM
    # Ak chceš skúsiť aj SVM, stačí odkomentovať tieto dva riadky:
    # svm_def = SVMModel(random_state=42)
    # manager.run_pipeline(svm_def, X, y)


if __name__ == "__main__":
    main()