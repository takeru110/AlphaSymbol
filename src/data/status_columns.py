"""add some columns of status to csv file which have expr column"""

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


def status_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add status len, arity, and depth columns to df"""
    df["expr_Expr"] = df["expr"].apply(lambda x: expr_eval_safe(x))
    df["arity"] = df["expr_Expr"].apply(
        lambda x: 0 if x.arity is None else x.arity
    )
    df["len"] = df["expr"].apply(len)
    df["depth"] = df["expr_Expr"].apply(lambda x: x.depth)

    df = df.drop(columns=["expr_Expr"])
    return df


def input_output_columns(df: pd.DataFrame, eq_domain) -> pd.DataFrame:
    df.fillna(0)
    df["input"] = df["arity"].apply(lambda x: get_input(x, eq_domain))
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
    df_status = status_columns(df)
    df_status.to_csv(output_path, index=False)
