import itertools
from src.config import config
from src.data_loader import ECGLoader


def main():
    # 1. Vytvorenie dvoch explicitných loaderov
    loader_pos = ECGLoader(data_dir=config.RAW_DATA_PATH_POSITIVE, label=1)
    loader_neg = ECGLoader(data_dir=config.RAW_DATA_PATH_NEGATIVE, label=0)

    # 2. Spojenie generátorov do jedného súvislého prúdu (pipeline)
    # Najprv sa spracujú všetky pozitívne, potom plynule prejde na negatívne
    all_records = itertools.chain(
        loader_pos.load_all(extension="*.csv"),
        loader_neg.load_all(extension="*.csv")
    )

    # 3. Jednotný cyklus pre spracovanie VŠETKÝCH dát
    print("Spúšťam spracovanie EKG záznamov...")
    for record in all_records:
        print(f"Spracúvam: {record.filename} | Label: {record.label} | Dĺžka: {len(record.signal)}")


if __name__ == "__main__":
    main()
