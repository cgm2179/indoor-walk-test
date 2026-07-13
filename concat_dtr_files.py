from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DTR_DIR = ROOT / "DTR"
OUTPUT_FILE = ROOT / "Concat_Indoor_Walk_Test.csv"


def main() -> None:
    dtr_files = sorted(DTR_DIR.glob("*.DTR"))
    if not dtr_files:
        raise SystemExit("No .DTR files found in DTR/")

    total_input_lines = 0

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as destination:
        for index, dtr_file in enumerate(dtr_files):
            content = dtr_file.read_text(encoding="utf-8", errors="replace")
            total_input_lines += len(content.splitlines())
            destination.write(content)

            if content and not content.endswith(("\n", "\r")):
                destination.write("\n")

            if index != len(dtr_files) - 1 and content and content.endswith("\r"):
                destination.write("\n")

    output_line_count = len(OUTPUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines())

    print(f"files={len(dtr_files)}")
    print(f"input_lines={total_input_lines}")
    print(f"output_lines={output_line_count}")

    if output_line_count < total_input_lines:
        raise SystemExit("Output line count is smaller than input line count")


if __name__ == "__main__":
    main()