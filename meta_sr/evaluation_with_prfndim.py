import os
import sys

import numpy as np
import pandas as pd
import requests
import symbolicregression
import sympy as sp
import torch
from IPython.display import display
from sympy.parsing.sympy_parser import parse_expr

n_points = 100
csv_path = "/home/takeru/AlphaSymbol/data/prfndim/d5-a3-c2-r3-stopped-random-points-test10k-crop100-test-columns.csv"
tolerances = [0, 1, 2, 3, 4]


def compute_r2(y_true, y_pred):
    """
    Compute the R2 score.
    R2 = 1 - (Σ(y_i - ŷ_i)^2) / (Σ(y_i - ȳ)^2)

    Args:
    - y_true (list[float]): Ground truth values
    - y_pred (list[float]): Predicted values

    Returns:
    - float: R2 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle pathological cases
    if denominator == 0:
        return 0.0  # No variance in y_true
    return 1 - numerator / denominator


def inlier_rate(y_true, y_pred, tau):
    """
    Compute the accuracy within tolerance τ.
    Accτ = 1(max(|(ŷ_i - y_i) / y_i|) <= τ)

    Args:
    - y_true (list[float]): Ground truth values
    - y_pred (list[float]): Predicted values
    - tau (float): Tolerance threshold

    Returns:
    - int: 1 if the maximum relative error is within tolerance, else 0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero; filter out y_true == 0
    non_zero_indices = y_true != 0
    if not non_zero_indices.any():
        return 0  # If all y_true are zero, return 0

    relative_errors = np.abs(
        (y_pred[non_zero_indices] - y_true[non_zero_indices])
    )
    num_inlier = sum(relative_errors <= tau)
    return int(num_inlier) / len(y_true)


if __name__ == "__main__":
    model_path = "/home/takeru/AlphaSymbol/meta_sr/model.pt"
    try:
        if not os.path.isfile(model_path):
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, "wb").write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            model = torch.load(model_path)
            model = model.cuda()
        print(model.device)
        print("Model successfully loaded!")

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)

    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model, max_input_points=200, n_trees_to_refine=100, rescale=True
    )

    df = pd.read_csv(csv_path)
    # the index of lists are sample number
    acc_taus = {tau: [] for tau in tolerances}
    r2s = []
    for i, (
        input_str,
        output_str,
        correct_expr,
        test_input_str,
        test_output_str,
    ) in enumerate(
        zip(
            df["input"],
            df["output"],
            df["expr"],
            df["test_input"],
            df["test_output"],
        ),
    ):
        print("\n")
        print("Sample number: ", i + 1)
        x = np.array(eval(input_str)[:n_points])
        y = np.array(eval(output_str)[:n_points])
        print("The number of points for regression: ", len(x))
        test_input, test_output = eval(test_input_str), eval(test_output_str)
        print(f"Input points: {test_input}")
        print(f"Correct output: {test_output}")
        est.fit(x, y)
        replace_ops = {
            "add": "+",
            "mul": "*",
            "sub": "-",
            "pow": "**",
            "inv": "1/",
        }
        model_str = est.retrieve_tree(with_infos=True)[
            "relabed_predicted_tree"
        ].infix()
        for op, replace_op in replace_ops.items():
            model_str = model_str.replace(op, replace_op)
        # display(sp.parse_expr(model_str))

        # use function
        x_0, x_1, x_2, x_3, x_4 = sp.symbols("x_0 x_1 x_2 x_3 x_4")
        local_dict = {
            "e": sp.E,
            "pi": sp.pi,
            "euler_gamma": sp.EulerGamma,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "step": sp.Heaviside,
            "sign": sp.sign,
        }
        expr = parse_expr(model_str, evaluate=True, local_dict=local_dict)

        print(f"The number of points for testing: {len(test_input)}")
        pred_output = []
        for xs, y in zip(test_input, test_output):
            dict_vars = {f"x_{i}": x for i, x in enumerate(xs)}
            pred_output.append(expr.subs(dict_vars))
        print(f"Predicted output: {pred_output}")

        # calc r2
        r2_score = compute_r2(test_output, pred_output)
        r2s.append(r2_score)

        # calc accuracy per sample
        for tau in tolerances:
            acc_tau = inlier_rate(test_output, pred_output, tau)
            acc_taus[tau].append(acc_tau)

    r2 = np.mean(r2s)
    acc_tau = dict.fromkeys(tolerances, None)
    for tau in tolerances:
        acc_tau[tau] = np.mean(acc_taus[tau])
    print(r2)
    print(acc_tau)
