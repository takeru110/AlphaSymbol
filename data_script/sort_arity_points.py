import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_script.prfndim_utils import expr_eval_safe
from prfndim.prfndim import C, P, R, S, Z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort the data by arity, len, and depth"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="path to the data file (csv)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="path to the output file (csv)",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df["arity"] = df["expr"].apply(
        lambda x: expr_eval_safe(x).arity
        if expr_eval_safe(x).arity is not None
        else 0
    )
    df_arity_sorted = df.sort_values(by=["arity", "n_points"])
    df_arity_sorted.to_csv(args.output_file, index=False)
