import argparse
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


def pred_output(model, data_module, xs_reg, ys_reg, test_input_points):
    """
    Predict the output using the trained model.
    Args:
    - model: Trained model
    - data_module: Data module with meta data like vocab
    - xs_reg (list[float]): x values for symbolic regression
    - ys_reg (list[float]): y values for symbolic regression
    - test_input_points (list[list[float]]): x values for test
    """
    pred_func = beam_search(model, data_module, xs_reg, ys_reg, beam_width=3)
    pred_outputs = [pred_func.eval(*x) for x in test_input_points]
    logging.info(f"Predicted function: {pred_func}")
    return pred_outputs


def evaluate_model(pred_output_hook, df, tolerances):
    """
    Evaluate the model using R2 score and Accτ metrics.

    Args:
    - pred_output_hook (function): Function to predict the output
        - Takes xs, ys, test_input as input and returns predicted output
    - df: DataLoader for test data
        - df must have columns "input", "output", "test_input", "test_output"
    - tolerances (list[float]): List of tolerance values for Accτ

    Returns:
    - summary (dict): Dictionary of metrics (R2, Accτ for each τ)
    - all_data (dict): Dictionary of all data for each sample
    """
    r2s = []
    acc_taus = {tau: [] for tau in tolerances}
    error_counter = 0

    all_data = {
        "expr": [],
        "x_reg": [],
        "y_reg": [],
        "n_reg": [],
        "dim_reg": [],
        "x_test": [],
        "y_test": [],
        "n_test": [],
        "y_pred": [],
        "r2": [],
    }
    for tau in tolerances:
        all_data[f"acc_{tau}"] = []

    for i, (
        expr,
        input_str,
        output_str,
        test_input_str,
        test_output_str,
    ) in enumerate(
        zip(
            df["expr"],
            df["input"],
            df["output"],
            df["test_input"],
            df["test_output"],
        ),
    ):
        test_input = eval(test_input_str)
        test_output = eval(test_output_str)
        logging.info(f"\n\nSample: {i + 1}")
        xs = eval(input_str)
        ys = eval(output_str)
        logging.info(f"x values for regression: {xs}")
        logging.info(f"y values for regression: {ys}")

        try:
            pred_output = pred_output_hook(xs, ys, test_input)
        except KeyboardInterrupt:
            logging.info("Evaluation interrupted by user")
            print("Evaluation interrupted by user")
            print("Do you want to do? ")
            print("1. Skip this sample")
            print("2. Quit evaluation")
            value = input()
            error_counter += 1
            if value == "1":
                logging.info(f"User skipped sample {i}")
                continue
            elif value == "2":
                logging.info(f"User terminated evaluation at sample {i}")
                break
        except Exception:
            logging.info("Error in evaluating the expression")
            error_counter += 1
            continue

        all_data["expr"].append(expr)
        all_data["x_reg"].append(xs)
        all_data["y_reg"].append(ys)
        all_data["n_reg"].append(len(xs))
        all_data["dim_reg"].append(len(xs[0]))
        all_data["x_test"].append(test_input)
        all_data["y_test"].append(test_output)
        all_data["n_test"].append(len(test_input))
        all_data["y_pred"].append(pred_output)
        logging.info(f"x values for test: {test_input}")
        logging.info(f"Correct y values : {test_output}")
        logging.info(f"Predicted y values {pred_output}")
        r2_score = compute_r2(test_output, pred_output)
        all_data["r2"].append(r2_score)
        logging.info(f"R2 score: {r2_score}")
        r2s.append(r2_score)
        for tau in tolerances:
            acc_tau = inlier_rate(test_output, pred_output, tau)
            all_data[f"acc_{tau}"].append(acc_tau)
            logging.info(f"Accuracy tau={tau}: {acc_tau}")
            acc_taus[tau].append(acc_tau)

    r2 = np.mean(r2s)
    acc_tau = dict.fromkeys(tolerances, None)
    for tau in tolerances:
        acc_tau[tau] = np.mean(acc_taus[tau])
    summary = {"r2": r2, "acc": acc_tau, "n_errors": error_counter}
    return summary, all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-d", "--data_module_path", type=str)
    parser.add_argument("-i", "--data_csv_path", type=str)
    parser.add_argument("-o", "--output_csv_path", type=str)
    args = parser.parse_args()
    model_path = args.model_path
    csv_path = args.data_csv_path
    output_csv_path = args.output_csv_path
    data_module_path = args.data_module_path

    # Load objects from files
    model = LitTransformer.load_from_checkpoint(model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with open(data_module_path, "rb") as f:
        data_module = pickle.load(f)
    df = pd.read_csv(csv_path)

    # Define tolerances for Accτ
    tolerances = [0, 1, 2, 3, 4]

    def pred_output_hook(xs, ys, test_input):
        return pred_output(model, data_module, xs, ys, test_input)

    # Evaluate the model
    summary, all_data = evaluate_model(pred_output_hook, df, tolerances)

    # Print and save results
    print("Evaluation Metrics:")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
