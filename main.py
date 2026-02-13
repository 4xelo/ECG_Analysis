import itertools
from src.config import config
from src.data_loader import ECGLoader
# 1. Importujeme naše nové preprocessing triedy
from src.preprocessing import ButterworthFilter, PanTompkinsDetector, RCenteredSegmenter


def main():
    # 1. Vytvorenie dvoch explicitných loaderov
    loader_pos = ECGLoader(data_dir=config.RAW_DATA_PATH_POSITIVE, label=1)
    loader_neg = ECGLoader(data_dir=config.RAW_DATA_PATH_NEGATIVE, label=0)

    # 3. Inicializácia preprocessing komponentov (VŽDY PRED CYKLOM!)
    ecg_filter = ButterworthFilter(lowcut=0.5, highcut=40.0, order=4)
    peak_detector = PanTompkinsDetector()

    # Segmenter si berie nastavenie okien priamo z configu
    segmenter = RCenteredSegmenter(
        pre_window_s=config.PRE_WINDOW_S,
        post_window_s=config.POST_WINDOW_S,
        fs=config.FS
    )

    # 4. Spojenie generátorov do jedného súvislého prúdu (pipeline)
    all_records = itertools.chain(
        loader_pos.load_all(extension="*.csv"),
        loader_neg.load_all(extension="*.csv")
    )

    # 5. Jednotný cyklus pre spracovanie VŠETKÝCH dát
    print("Spúšťam pipeline na spracovanie EKG záznamov...\n")

    # Pomocné premenné pre testovanie
    test_limit = 3
    iter = 0

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

        # Ak chceme vidieť, čo presne segment obsahuje (ukážeme prvý segment z pacienta)
        if segments:
            prvy_segment_signal, prvy_r_peak_idx = segments[0]
            print(f"  -> Ukážka 1. segmentu: stred v R-vrchole index {prvy_r_peak_idx}, "
                  f"dĺžka poľa: {len(prvy_segment_signal)} vzoriek\n")

        # Predčasné ukončenie pre účely testovania
        iter += 1
        if iter >= test_limit:
            print("=== Testovanie úspešne ukončené (dosiahnutý limit 3 záznamov). ===")
            break

if __name__ == "__main__":
    main()
