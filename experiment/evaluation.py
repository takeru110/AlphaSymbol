import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_script.input_output_columns_exp import generate_exp_input
from prfndim.prfndim import C, Expr, P, R, S, Z
from src.model_meta.beam_search import beam_search
from src.model_meta.data import PREDataModule
from src.model_meta.train import LitTransformer

logging.basicConfig(level=logging.INFO)


def id2token(id, vocab):
    """
    Args:
    - id (int): the id of the token
    - vocab (dict[str, int]): the vocabulary
    """
    for k, v in vocab.items():
        if v == id:
            return k
    return None


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


def evaluate_model(model, data_module, df, tolerances):
    """
    Evaluate the model using R2 score and Accτ metrics.

    Args:
    - model: Trained model
    - data_module: Data module with vocab and test data
    - test_loader: DataLoader for test data
    - tolerances (list[float]): List of tolerance values for Accτ

    Returns:
    - dict: Dictionary of metrics (R2, Accτ for each τ)
    """
    model.eval()
    r2s = []
    acc_taus = {tau: [] for tau in tolerances}
    error_counter = 0

    for i, (src_input_str, src_output_str, correct_expr) in enumerate(
        zip(df["input"], df["output"], df["expr"]),
    ):
        logging.info(f"\n\nSample: {i + 1}")
        correct_func: Expr = eval(correct_expr)
        logging.info(f"Correct Expression: {correct_func}")
        xs = eval(src_input_str)
        arity = len(xs[0])
        ys = eval(src_output_str)
        logging.info(f"x values for regression: {xs}")
        logging.info(f"y values for regression: {ys}")
        pred_func = beam_search(model, data_module, xs, ys, beam_width=3)

        n_points = 10
        rate = 0.1
        x_values = [
            generate_exp_input(n=arity, rate=rate) for _ in range(n_points)
        ]
        try:
            pred_y_values = [pred_func.eval(*x) for x in x_values]
            corredt_y_values = [correct_func.eval(*x) for x in x_values]
        except Exception:
            logging.info("Error in evaluating the expression")
            error_counter += 1
            continue

        logging.info(f"x values for test: {x_values}")
        logging.info(f"Correct y values : {corredt_y_values}")
        logging.info(f"Predicted expression: {pred_func}")
        logging.info(f"Predicted y values {pred_y_values}")
        r2_score = compute_r2(corredt_y_values, pred_y_values)
        r2s.append(r2_score)
        for tau in tolerances:
            acc_tau = inlier_rate(corredt_y_values, pred_y_values, tau)
            acc_taus[tau].append(acc_tau)

    r2 = np.mean(r2s)
    acc_tau = dict.fromkeys(tolerances, None)
    for tau in tolerances:
        acc_tau[tau] = np.mean(acc_taus[tau])
    return {"r2": r2, "acc": acc_tau, "n_errors": error_counter}


if __name__ == "__main__":
    # Paths to the model and data
    model_path = "/home/takeru/AlphaSymbol/logs/2025-0116-1213-41-lowest-val/best_model-epoch=16-val_loss=0.04.ckpt"
    data_module_path = "/home/takeru/AlphaSymbol/logs/2025-0116-1213-41-lowest-val/data_module.pkl"
    csv_path = "/home/takeru/AlphaSymbol/data/prfndim/d5-a3-c2-r3-stopped-random-points-test10.csv"

    # Load the trained model
    model = LitTransformer.load_from_checkpoint(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)

    with open(data_module_path, "rb") as f:
        data_module = pickle.load(f)

    # Define tolerances for Accτ
    tolerances = [0, 1, 2, 3, 4]

    # Evaluate the model
    metrics = evaluate_model(model, data_module, df, tolerances)

    # Print and save results
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    results_path = Path("/home/takeru/AlphaSymbol/logs/evaluation_results.txt")
    with open(results_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    print(f"Results saved to {results_path}")
