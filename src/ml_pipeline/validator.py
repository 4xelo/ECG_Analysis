import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.base import BaseEstimator


class ModelValidator:
    """
    Zodpovedá za Finálne vyhodnotenie a metriky.
    """

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate(self, estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series):
        """
        Vykoná 10-fold Cross Validation a vypočíta metriky.
        """
        model_name = estimator.__class__.__name__
        print(f" [*] Validácia: Spúšťam {self.cv.get_n_splits()}-fold CV pre {model_name}...")

        # 1. Získanie pravdepodobností (pre ROC-AUC)
        # method='predict_proba' vráti maticu [P(0), P(1)]. Nás zaujíma stĺpec 1.
        try:
            y_probas = cross_val_predict(estimator, X, y, cv=self.cv, method='predict_proba', n_jobs=-1)[:, 1]
        except AttributeError:
            # Fallback pre modely, ktoré nemajú predict_proba (napr. LinearSVC bez kalibrácie)
            print(" [!] Model nepodporuje predict_proba, používam decision_function.")
            y_probas = cross_val_predict(estimator, X, y, cv=self.cv, method='decision_function', n_jobs=-1)

        # 2. Získanie tvrdých predikcií 0/1 (pre Confusion Matrix)
        y_preds = cross_val_predict(estimator, X, y, cv=self.cv, method='predict', n_jobs=-1)

        # 3. Výpočet metrík
        auc = roc_auc_score(y, y_probas)
        cm = confusion_matrix(y, y_preds)

        # Rozklad Confusion Matrixu na TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()

        # Senzitivita (Recall pre Positive) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Špecificita (Recall pre Negative) = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 4. Výpis výsledkov
        print("\n" + "=" * 40)
        print(f" VÝSLEDKY VALIDÁCIE ({model_name})")
        print("=" * 40)
        print(f" ROC-AUC:      {auc:.4f}")
        print(f" Sensitivity:  {sensitivity:.4f}  (Schopnosť odhaliť chorých)")
        print(f" Specificity:  {specificity:.4f}  (Schopnosť nepopliesť zdravých)")
        print("-" * 40)
        print(" Classification Report:")
        print(classification_report(y, y_preds))

        # 5. Vizualizácia
        self._plot_results(y, y_preds, y_probas, cm, auc_score=auc)

    def _plot_results(self, y_true, y_pred, y_prob, cm, auc_score):
        """
        Pomocná metóda na vykreslenie grafov.
        """
        plt.figure(figsize=(12, 5))

        # Graf 1: Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('Skutočnosť (Ground Truth)', fontsize=12)
        plt.xlabel('Predikcia modelu', fontsize=12)
        plt.xticks([0.5, 1.5], ['Zdravý (0)', 'Chorý (1)'])
        plt.yticks([0.5, 1.5], ['Zdravý (0)', 'Chorý (1)'])

        # Graf 2: ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.show()