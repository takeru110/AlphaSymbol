import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def generate_uniform_integer(n_min: int, n_max: int) -> int:
    """
    Generate a random integer n from a uniform distribution U(n_min, n_max).

    Args:
        n_min (int): Minimum value of the uniform distribution (inclusive).
        n_max (int): Maximum value of the uniform distribution (inclusive).

    Returns:
        int: Random integer sampled from U(n_min, n_max).
    """
    return int(np.random.randint(n_min, n_max + 1))


def generate_exp_input(
    n: int, rate=1.0, max_value: Optional[int] = None
) -> list[int]:
    """
    Generate input x according to an exponential distribution.

    Args:
        rate (float): Rate parameter λ for the exponential distribution.
        n (int): Number of elements to generate.

    Returns:
        list[float]: Input vector x.
    """
    ret = []
    for _ in range(n):
        sample = int(np.random.exponential(scale=1 / rate))
        while max_value is not None and sample > max_value:
            sample = int(np.random.exponential(scale=1 / rate))
        ret.append(sample)
    return ret


def input_output_columns_exp(
    df: pd.DataFrame,
    max_arity: int,
    rate=1.0,
    max_value: Optional[int] = None,
    n_points: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply exponential input generation row by row and generate input/output columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing expressions.
        rate (float): Rate parameter λ for the exponential distribution.

    Returns:
        pd.DataFrame: DataFrame with input and output columns.
    """

    inputs = []
    outputs = []
    n_points_list = []
    logging.info("Generating inputs and outputs")
    logging.info(f"Processing {len(df)} rows")
    for expr_str in tqdm(df["expr"], desc="Processing"):
        expr = eval(expr_str)
        input_dim = (
            expr.arity
            if expr.arity is not None
            else generate_uniform_integer(1, max_arity)
        )

        # Sample n from U[n_min, n_max]
        n_min = 5 * input_dim
        n_max = 20 * input_dim

        if n_points is None:
            n_points_sample = generate_uniform_integer(n_min, n_max)
        else:
            n_points_sample = n_points

        # Generate x n-times and evaluate output
        x_list = []
        y_list = []
        actual_n_points = 0

        for i in range(n_points_sample):
            lmbd = 1 / (rate * n_points_sample ** (1 / input_dim))
            x = generate_exp_input(n=input_dim, rate=lmbd, max_value=max_value)
            if x in x_list:
                continue
            try:
                y = expr.eval(*x)
            except OverflowError:
                continue

            actual_n_points += 1
            x_list.append(x)
            y_list.append(y)

        inputs.append(x_list)
        outputs.append(y_list)
        n_points_list.append(actual_n_points)

    df["input"] = inputs
    df["output"] = outputs
    df["n_points"] = n_points_list
    logging.info("Finished generating inputs and outputs")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-o", "--output_file")
    parser.add_argument("-a", "--max_arity", type=int)
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=0.2,
        help="Rate parameter for the exponential distribution",
    )
    parser.add_argument(
        "-m",
        "--max_value",
        type=int,
        default=None,
        help="The maximum value of the input",
    )
    parser.add_argument(
        "-p",
        "--n_point",
        type=int,
        default=None,
        help="The number of points to generate. if None, random.",
    )
    args = parser.parse_args()

    path = args.input_file
    output_path = args.output_file

    df = pd.read_csv(path)

    df_in_out = input_output_columns_exp(
        df,
        args.max_arity,
        rate=args.rate,
        max_value=args.max_value,
        n_points=args.n_point,
    )

    df_in_out.to_csv(output_path, index=False)
