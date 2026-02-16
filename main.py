import itertools
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import config
from src.data_loader import ECGLoader
from src.preprocessing import ButterworthFilter, PanTompkinsDetector, RCenteredSegmenter, QualityControl, PatientAggregator
from src.visualizations import ECGPlotter
from src.features import FeatureExtractionPipeline
from src.utils import reduce_mem_usage



# ==========================================
# WORKER FUNKCIA (Beží na každom jadre CPU)
# ==========================================
def process_single_record(record):
    """
    Spracuje jeden celý záznam (súbor) a vráti zoznam riadkov pre dataset.
    """
    try:
        # 1. Lokálna inicializácia nástrojov (aby sa nebili medzi procesmi)
        # Inicializácia je veľmi rýchla, takže nevadí, že sa robí pre každého pacienta.
        ecg_filter = ButterworthFilter(lowcut=0.5, highcut=40.0, order=4)
        peak_detector = PanTompkinsDetector()
        segmenter = RCenteredSegmenter(config.PRE_WINDOW_S, config.POST_WINDOW_S, config.FS)
        qc = QualityControl(max_amplitude=3000.0, min_std=5.0)

        # Factory method pre features (vytvorí pipeline so všetkým: Time, Freq, Hjorth, Poincare...)
        feature_pipeline = FeatureExtractionPipeline.create_default_pipeline()

        dataset_rows = []

        # 2. Preprocessing
        clean_signal = ecg_filter.apply(record.signal, config.FS)
        r_peaks = peak_detector.detect(clean_signal, config.FS)
        segments = segmenter.extract_segments(clean_signal, r_peaks)

        # 3. Spracovanie segmentov
        for segment_signal, r_idx in segments:

            # A) Quality Control (Zahodíme artefakty)
            if not qc.is_valid(segment_signal):
                continue

                # B) Feature Extraction
            features = feature_pipeline.process_segment(segment_signal, config.FS)

            # C) Metadáta
            features["filename"] = record.filename
            features["label"] = record.label
            features["original_r_peak"] = r_idx

            dataset_rows.append(features)

        return dataset_rows

    except Exception as e:
        print(f"Chyba pri spracovaní súboru {record.filename}: {e}")
        return []  # Vrátime prázdny list, aby proces nespadol


def main():
    # 1. Vytvorenie dvoch explicitných loaderov
    loader_pos = ECGLoader(data_dir=config.RAW_DATA_PATH_POSITIVE, label=1)
    loader_neg = ECGLoader(data_dir=config.RAW_DATA_PATH_NEGATIVE, label=0)
    extracted_df = pd.read_csv("data/processed/ecg_features_final.csv")

    plotter = ECGPlotter(fs=config.FS)

    # Musíme vytvoriť list, lebo Parallel potrebuje poznať dĺžku pre progress bar
    # A tiež generátory sa ťažko picklujú.
    all_records = list(itertools.chain(
        loader_pos.load_all(extension="*.csv"),
        loader_neg.load_all(extension="*.csv")
    ))

    print(f"Spúšťam paralelné spracovanie pre {len(all_records)} súborov.")
    print(f"Využívam všetky dostupné jadrá CPU (n_jobs=-1)...")

    # PARALELNÉ SPRACOVANIE
    # n_jobs=-1 znamená "použi všetky jadrá".
    # backend="loky" je najstabilnejší pre Python multiprocessing.
    # results_lists = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(process_single_record)(record)
    #     for record in tqdm(all_records, desc="Extrakcia príznakov")
    # )

    #  Spájanie výsledkov
    # results_lists je zoznam zoznamov [[row1, row2], [row3, row4], ...], musíme to "sploštiť"
    # print("Spájam výsledky do jedného Datasetu...")
    # flat_results = [item for sublist in results_lists for item in sublist]
    #
    # # Tvorba DataFrame a Optimalizácia pamäte
    # df = pd.DataFrame(flat_results)
    #
    # print(f"Pôvodná veľkosť v pamäti: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
    #
    # # Ak máš funkciu reduce_mem_usage v utils.py (odporúčam!)
    # df = reduce_mem_usage(df)
    #
    # # Uloženie
    # output_path = "data/processed/ecg_features_final.csv"
    # df.to_csv(output_path, index=False)
    #
    # print(f"Hotovo! Dataset uložený do: {output_path}")
    # print(f"Počet segmentov: {len(df)}")

    print("\nSpúšťam agregáciu na úroveň pacienta...")

    # Inicializácia
    aggregator = PatientAggregator(aggregations=['mean', 'std', 'min', 'max'])

    # Vstupuje: df (segment level)
    # Vystupuje: df_patient (patient level - 1 riadok na súbor)
    df_patient = aggregator.aggregate(extracted_df, id_col='filename', label_col='label')

    # Uloženie
    output_patient = "data/processed/ecg_features_patient_level.csv"
    df_patient.to_csv(output_patient, index=False)

    print(f"Patient-level dataset uložený do: {output_patient}")

if __name__ == "__main__":
    main()
