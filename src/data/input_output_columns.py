import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data.prfndim_utils import (
    expr_eval_safe,
    generate_eq_domain,
    get_input,
    get_output,
)
from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z


def input_output_columns(df: pd.DataFrame, eq_domain) -> pd.DataFrame:
    df["input"] = df["expr"].apply(
        lambda x: get_input(expr_eval_safe(x).arity, eq_domain)
    )
    df["output"] = df["expr"].apply(
        lambda x: get_output(expr_eval_safe(x), eq_domain)
    )
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
