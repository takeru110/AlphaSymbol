import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s",
)


def unify_expr_csv(input_file: Path, output_file: Path):
    logging.info("Started removing duplicated expressions from %s", input_file)
    seen = set()  # 既に出現した行を記録するためのset
    with (
        open(input_file, mode="r", newline="", encoding="utf-8") as infile,
        open(output_file, mode="w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # add header
        writer.writerow(next(reader))

        for row in reader:
            # 行がすでに出現したことがない場合のみ書き込む
            if tuple(row) not in seen:
                writer.writerow(row)
                seen.add(tuple(row))
    logging.info("Saved unique exprs to %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicate exprs from a CSV file"
    )
    parser.add_argument("--input_file", type=str, help="Input CSV file")
    parser.add_argument("--output_file", type=str, help="Output CSV file")
    args = parser.parse_args()

    # 使用例
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    unify_expr_csv(input_file, output_file)
