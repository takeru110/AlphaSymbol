import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s",
)


def output_bytes_not_const(expr: Expr, eq_domain: list[NDArray]) -> bytes:
    assert expr.arity is not None
    ans: NDArray = np.array(tuple(expr.eval(*x) for x in eq_domain[expr.arity]))
    return ans.tobytes()


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[NDArray]
) -> bytes:
    assert input_size >= 1
    return np.array(
        expr.eval(0) for _ in range(len(eq_domain[input_size]))
    ).tobytes()


def expr_is_unseen(
    expr: Expr, seen: list[set[bytes]], max_arity: int, eq_domain: list[NDArray]
) -> bool:
    if expr.arity is not None:
        return output_bytes_not_const(expr, eq_domain) not in seen[expr.arity]
    for input_size in range(1, max_arity + 1):
        output_bytes = output_bytes_const(expr, input_size, eq_domain)
        if output_bytes not in seen[input_size]:
            return True
    return False


def update_seen(
    expr: Expr, seen: list[set[bytes]], max_arity: int, eq_domain: list[NDArray]
):
    if expr.arity is not None:
        seen[expr.arity].add(output_bytes_not_const(expr, eq_domain))
    for input_size in range(1, max_arity):
        output_bytes = output_bytes_const(expr, input_size, eq_domain)
        if output_bytes not in seen[input_size]:
            seen[input_size].add(output_bytes)


def init_seen(max_arity: int) -> list[set[bytes]]:
    ret: list[set[bytes]] = [
        set() for _ in range(max_arity + 1)
    ]  # 既に出現した行を記録するためのset
    return ret


def remove_duplicates_from_csv(
    input_file: Path, output_file: Path, max_arity, eq_domain
):
    logging.info(
        "Started removing output-duplicated expressions from %s", input_file
    )
    seen: list[set[bytes]] = init_seen(max_arity)

    with (
        open(input_file, mode="r", newline="", encoding="utf-8") as infile,
        open(output_file, mode="w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        writer.writerow(next(reader))  # copy and pass header

        unique_counter = 0

        for expr_counter, row in enumerate(reader, start=1):
            expr: Expr = eval(row[0])
            try:
                if expr_is_unseen(expr, seen, max_arity, eq_domain):
                    writer.writerow(row)
                    update_seen(expr, seen, max_arity, eq_domain)
                    unique_counter += 1
            except OverflowError:
                pass
            if unique_counter % 1000 == 0:
                seen = init_seen(max_arity)
            if expr_counter % 10000 == 0:
                logging.info(f"{expr_counter}th expr is processed")
    logging.info("Saved output file to %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicate output exprs"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=30,
        help="nummer of samples to define two exprs are equal",
    )
    parser.add_argument(
        "--max_arity",
        type=int,
        help="maximum arity of exprs in the csv file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="input csv file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="output csv file",
    )

    args = parser.parse_args()
    sample_num = args.sample_num
    max_arity: int = args.max_arity
    input_file: str = Path(args.input_file)
    output_file: str = Path(args.output_file)

    eq_domain: list[NDArray] = [np.zeros((1))] + [
        np.random.randint(0, 10, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]

    remove_duplicates_from_csv(input_file, output_file, max_arity, eq_domain)
