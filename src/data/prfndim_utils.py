import logging
from typing import Optional
import math

import numpy as np
import numpy.typing as npt

from prfndim.prfndim import C, Expr, P, R, S, Z


def generate_eq_domain(sample_max, sample_num, max_arity) -> list[npt.NDArray]:
    """
    Generate a list of NumPy array for function evaluation
    to detect the same functions.

    This function creates a set of input samples for function evaluation.
    Two functions f and g are equal if f(x) == g(x) for all x where x is
    an element of `return_value[f.arity]`.

    Args:
        sample_max (int): The maximum value for randomly generated inputs.
        sample_num (int): The number of inputs per function.
        max_arity (int): The maximum function arity of the program.

    Returns:
        list[npt.NDArray]: A list of NumPy arrays, where each array represents
        input values for a particular arity. The shape of each array is
        `(sample_num, dim)`, where `dim` corresponds to the arity.

    Example:
        >>> generate_eq_domain(10, 5, 3)
        [
            array([[0.], [0.], [0.], [0.], [0.]]),         # Arity 0 (all zeros)
            array([[3], [7], [2], [1], [9]]),              # Arity 1
            array([[1, 5], [4, 7], [3, 9], [6, 2], [8, 3]]),  # Arity 2
            array([[2, 5, 9], [1, 3, 4], [7, 8, 6], ...])  # Arity 3
        ]
    """
    eq_domain = [np.zeros((sample_num, 1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]
    return eq_domain


def get_input(arity: Optional[int], eq_domain: list[npt.NDArray]) -> list[tuple[int]]:
    """
    Generate input tuples for a given function arity from the provided domain.

    This function retrieves a set of input tuples corresponding to the given
    function arity from the `eq_domain`. If the provided `arity` is None,
    it defaults to 0.

    Args:
        arity (int): The arity (number of input variables) of the function.
        eq_domain (list[npt.NDArray]): Index represents the arity of the function.

    Returns:
        list[tuple[int]]: A list of input tuples, where each tuple contains
        integers representing input values for the given arity.

    Example:
        >>> eq_domain = [np.array([[0], [1], [2]]), np.array([[3, 4], [5, 6]])]
        >>> get_input(1, eq_domain)
        [(3, 4), (5, 6)]
    """
    if arity is None:
        arity = 0
    ans = [tuple(int(x) for x in input) for input in eq_domain[int(arity)]]
    return ans


def expr_eval_safe(expr_str: str) -> Expr:
    """evaluate string as Expr safely"""
    try:
        return eval(expr_str)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate expression: {expr_str}, Error: {e}"
        )


def get_output(expr: Expr, eq_domain: list[npt.NDArray]) -> list[int]:
    arity = 0 if expr.arity is None else expr.arity
    ans = []
    for x in eq_domain[arity]:
        try:
            ans.append(int(expr.eval(*x)))
        except OverflowError:
            ans.append("Overflowed")
    return ans


def handle_interrupt(place: str = "anywhere") -> int:
    logging.info(f"""Evaluation interrupted by user in {place}
    Evaluation interrupted by user
    Do you want to do? 
    [digits]: skip the number of samples
    [q]: Quit evaluation""")
    value = input("Press a keyboard: ")
    if value.isdigit:
        skip_num = int(value)
        logging.info(f"Skipping {skip_num} samples...")
        return skip_num
    elif value == "q":
        logging.info("Quitting program...")
        exit(0)
    else:
        logging.info("Invalid input. Automatically skipping...")
        return 1
