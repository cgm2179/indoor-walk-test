from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "CSV"
OUTPUT_FILE = ROOT / "Concat_Indoor_Walk_Test_from_csv.csv"


def main() -> None:
    csv_files = sorted(CSV_DIR.glob("*.CSV"))
    if not csv_files:
        raise SystemExit("No .CSV files found in CSV/")

    fieldnames: list[str] = []
    fieldname_set: set[str] = set()
    total_rows = 0

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                continue

            for fieldname in reader.fieldnames:
                if fieldname not in fieldname_set:
                    fieldname_set.add(fieldname)
                    fieldnames.append(fieldname)

            for _ in reader:
                total_rows += 1

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        written_rows = 0
        for csv_file in csv_files:
            with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    continue

                for row in reader:
                    writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
                    written_rows += 1

    print(f"files={len(csv_files)}")
    print(f"columns={len(fieldnames)}")
    print(f"input_rows={total_rows}")
    print(f"output_rows={written_rows}")

    if written_rows != total_rows:
        raise SystemExit("Output row count does not match input row count")


if __name__ == "__main__":
    main()