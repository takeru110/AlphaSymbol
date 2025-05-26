import argparse
import logging
import multiprocessing as mp
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.prfndim_utils import expr_eval_safe
from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z


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


def process_row(
    expr_str: str,
    max_arity: int,
    rate: float,
    max_value: Optional[int],
    n_points: Optional[int],
) -> dict:
    """
    Process a single row: generate inputs and outputs for the given expression.

    Args:
        expr_str (str): String representation of the expression.
        max_arity (int): Maximum arity for the expression.
        rate (float): Rate parameter λ for the exponential distribution.
        max_value (Optional[int]): Maximum value of input.
        n_points (Optional[int]): Number of points to generate.

    Returns:
        dict: Processed data including inputs, outputs, and number of points.
    """
    np.random.seed()
    expr = expr_eval_safe(expr_str)
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

    for _ in range(n_points_sample):
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

    return {"input": x_list, "output": y_list, "n_points": actual_n_points}


def input_output_columns_exp_parallel(
    df: pd.DataFrame,
    max_arity: int,
    rate=1.0,
    max_value: Optional[int] = None,
    n_points: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply exponential input generation row by row using multiprocessing.

    Args:
        df (pd.DataFrame): Input DataFrame containing expressions.
        rate (float): Rate parameter λ for the exponential distribution.

    Returns:
        pd.DataFrame: DataFrame with input and output columns.
    """
    logging.info("Generating inputs and outputs using multiprocessing")
    logging.info(f"Processing {len(df)} rows")

    # Prepare partial function for multiprocessing
    process_func = partial(
        process_row,
        max_arity=max_arity,
        rate=rate,
        max_value=max_value,
        n_points=n_points,
    )

    # Use multiprocessing Pool
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_func, df["expr"]), total=len(df)))

    # Combine results into the DataFrame
    df["input"] = [result["input"] for result in results]
    df["output"] = [result["output"] for result in results]
    df["n_points"] = [result["n_points"] for result in results]
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
        default=1.0,
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

    df_in_out = input_output_columns_exp_parallel(
        df,
        args.max_arity,
        rate=args.rate,
        max_value=args.max_value,
        n_points=args.n_point,
    )

    df_in_out.to_csv(output_path, index=False)
