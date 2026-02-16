import pandas as pd
import numpy as np
from typing import List


class PatientAggregator:
    """
    Agreguje segment-level dataset (kde 1 riadok = 1 úder/okno)
    na patient-level dataset (kde 1 riadok = 1 pacient).
    """

    def __init__(self, aggregations: List[str] = ['mean', 'std', 'min', 'max']):
        """
        :param aggregations: Aké štatistiky sa majú počítať.
                             Odporúčam: ['mean', 'std', 'min', 'max'].
                             Tým zachytíš celkový stav (mean) aj extrémy/variabilitu (std, min, max).
        """
        self.aggregations = aggregations

    def aggregate(self, df: pd.DataFrame, id_col: str = 'filename', label_col: str = 'label') -> pd.DataFrame:
        print(f"Agregujem dataset: {len(df)} segmentov -> úroveň pacienta...")

        # 1. Definícia stĺpcov
        # Tieto stĺpce NECHCEME priemerovať
        # 'original_r_peak' je index segmentu, jeho priemer nedáva klinický zmysel.
        exclude_cols = [id_col, label_col, 'original_r_peak']

        # Všetky ostatné stĺpce sú features (time_mean, hrv_sdnn, wav_energy...)
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Pre istotu overíme, či sú numerické
        feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        # 2. GroupBy operácia podľa pacienta (filename)
        grouped = df.groupby(id_col)

        # 3. Výpočet štatistík
        # Vytvorí to DataFrame s MultiIndexom (napr. stĺpec 'hrv_sdnn' bude mať podstĺpce 'mean', 'std'...)
        agg_df = grouped[feature_cols].agg(self.aggregations)

        # 4. Sploštenie názvov stĺpcov (Flattening)
        # Zmena z ('hrv_sdnn', 'mean') na 'hrv_sdnn_mean'
        agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]

        # 5. Pridanie metadát naspäť
        # Label je pre pacienta vždy rovnaký, vezmeme prvú hodnotu
        agg_df[label_col] = grouped[label_col].first()

        # BONUS: Pridáme 'segment_count' - koľko segmentov prežilo filtráciu?
        # Málo segmentov môže znamenať zašumený záznam.
        agg_df['segment_count'] = grouped[label_col].count()

        # Reset indexu, aby 'filename' bol bežný stĺpec
        agg_df = agg_df.reset_index()

        print(f"Hotovo. Pôvodné features: {len(feature_cols)}. Nové features: {len(agg_df.columns) - 2}.")
        print(f"Výsledný dataset má {len(agg_df)} pacientov.")

        return agg_df