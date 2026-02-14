import itertools
import pandas as pd
from src.config import config
from src.data_loader import ECGLoader
from src.preprocessing import ButterworthFilter, PanTompkinsDetector, RCenteredSegmenter
from src.visualizations import ECGPlotter
from src.features import FeatureExtractionPipeline


def main():
    # 1. Vytvorenie dvoch explicitných loaderov
    loader_pos = ECGLoader(data_dir=config.RAW_DATA_PATH_POSITIVE, label=1)
    loader_neg = ECGLoader(data_dir=config.RAW_DATA_PATH_NEGATIVE, label=0)

    # 3. Inicializácia preprocessing komponentov (VŽDY PRED CYKLOM!)
    ecg_filter = ButterworthFilter(lowcut=0.5, highcut=40.0, order=4)
    peak_detector = PanTompkinsDetector()
    segmenter = RCenteredSegmenter(
        pre_window_s=config.PRE_WINDOW_S,
        post_window_s=config.POST_WINDOW_S,
        fs=config.FS
    )
    plotter = ECGPlotter(fs=config.FS)

    # 4. Spojenie generátorov do jedného súvislého prúdu (pipeline)
    all_records = itertools.chain(
        loader_pos.load_all(extension="*.csv"),
        loader_neg.load_all(extension="*.csv")
    )

    # Vytvorenie pipeline na extrakciu priznakov
    feature_pipeline = FeatureExtractionPipeline.create_default_pipeline()

    # 5. Jednotný cyklus pre spracovanie VŠETKÝCH dát
    print("Spúšťam pipeline na spracovanie EKG záznamov...\n")

    # Pomocné premenné pre testovanie
    test_limit = 3
    iter = 0
    dataset_rows = []

    for record in all_records:
        print(f"--- Spracúvam: {record.filename} | Label: {record.label} ---")

        # KROK A: Filtrácia signálu
        clean_signal = ecg_filter.apply(record.signal, config.FS)

        # KROK B: Detekcia R-vrcholov
        r_peaks = peak_detector.detect(clean_signal, config.FS)

        # KROK C: Segmentácia (vyrezanie okien okolo R-vrcholov)
        segments = segmenter.extract_segments(clean_signal, r_peaks)

        # Výpis výsledkov pre daný záznam
        print(f" [*] Signál prefiltrovaný (dĺžka: {len(clean_signal)} vzoriek)")
        print(f" [*] Detegovaných R-vrcholov: {len(r_peaks)}")
        print(f" [*] Úspešne vyrezaných segmentov: {len(segments)}")


        for segment_signal, r_idx in segments:
        # 2. Extrakcia VŠETKÝCH príznakov naraz
            features = feature_pipeline.process_segment(segment_signal, config.FS)

            # Pridáme metadáta (aby sme vedeli, koho to je a aký má label)
            features["filename"] = record.filename
            features["label"] = record.label
            features["original_r_peak"] = r_idx

            dataset_rows.append(features)

        # Ak chceme vidieť, čo presne segment obsahuje (ukážeme prvý segment z pacienta)
        if segments:
            prvy_segment_signal, prvy_r_peak_idx = segments[0]
            print(f"  -> Ukážka 1. segmentu: stred v R-vrchole index {prvy_r_peak_idx}, "
                  f"dĺžka poľa: {len(prvy_segment_signal)} vzoriek\n")

            # ==========================================
            # VIZUALIZÁCIA KONKRÉTNEHO R-VRCHOLU (PODĽA ID)
            # ==========================================

            if iter == 2:

                # --- TU ZADAJ HODNOTU 'original_r_peak', KTORÚ CHCEŠ VIDIEŤ ---
                TARGET_R_PEAK = 132467
                # --------------------------------------------------------------

                print(f" [*] Hľadám segment pre R-vrchol s indexom: {TARGET_R_PEAK}...")

                plotter.browse_signal(
                    signal=clean_signal,  # Celý filtrovaný signál
                    r_peaks=r_peaks,  # Všetky R-vrcholy
                    start_s=250.0,  # Začiatok (napr. kde sa ti niečo nezdá)
                    end_s=271.0,  # Koniec
                    window_s=3.0  # Veľkosť kroku (zoom)
                )

                print(" [*] Prehliadanie dokončené, pokračujem v pipeline...")
            # ==========================================



        # Predčasné ukončenie pre účely testovania
        iter += 1
        if iter >= test_limit:
            print("=== Testovanie úspešne ukončené (dosiahnutý limit 3 záznamov). ===")
            break

    # Na konci budeš mať v 'dataset_rows' kompletnú tabuľku pre Pandas
    print(f"Hotovo. Extrahovaných {len(dataset_rows)} riadkov dát.")
    df = pd.DataFrame(dataset_rows)
    df.to_csv("ecg_features.csv", index=False)

if __name__ == "__main__":
    main()
