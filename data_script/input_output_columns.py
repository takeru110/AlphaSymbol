import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z


def generate_eq_domain(sample_max, sample_num, max_arity) -> list[npt.NDArray]:
    eq_domain = [np.zeros((sample_num, 1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]
    eq_domain[1] = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    return eq_domain


def get_output(expr: Expr, eq_domain: list[npt.NDArray]) -> list[int]:
    arity = 0 if expr.arity is None else expr.arity
    ans = []
    for x in eq_domain[arity]:
        try:
            ans.append(int(expr.eval(*x)))
        except OverflowError:
            ans.append("Overflowed")
    return ans


def get_input(expr: Expr, eq_domain: list[npt.NDArray]) -> list[tuple[int]]:
    arity = 0 if expr.arity is None else expr.arity
    ans = [tuple(int(x) for x in input) for input in eq_domain[int(arity)]]
    return ans


def input_output_columns(df: pd.DataFrame, eq_domain) -> pd.DataFrame:
    df["input"] = df["expr"].apply(lambda x: get_input(eval(x), eq_domain))
    df["output"] = df["expr"].apply(lambda x: get_output(eval(x), eq_domain))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-o", "--output_file")
    parser.parse_args()
    args = parser.parse_args()

    path = args.input_file
    output_path = args.output_file

    df = pd.read_csv(path)
    eq_domain = generate_eq_domain(sample_max=10, sample_num=10, max_arity=5)
    df_in_out = input_output_columns(df, eq_domain)
    df_in_out.to_csv(output_path, index=False)
