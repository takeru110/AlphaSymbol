import argparse
import logging
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

from experiment.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO)


def pred_output(est, xs_reg, ys_reg, xs_test):
    xs_reg_np, ys_reg_np = np.array(xs_reg), np.array(ys_reg)
    est.fit(xs_reg_np, ys_reg_np)
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
    ys_pred = []
    for xs in xs_test:
        dict_vars = {f"x_{i}": x for i, x in enumerate(xs)}
        val = expr.subs(dict_vars)
        assert val.is_real
        ys_pred.append(val)
    return ys_pred


def load_est():
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
        logging.info(model.device)
        logging.info("Model successfully loaded!")

    except Exception as e:
        logging.info(f"ERROR: model not loaded! path was: {format(model_path)}")
        logging.info(e)

    est = symbolicregression.model.SymbolicTransformerRegressor(
        model=model, max_input_points=200, n_trees_to_refine=100, rescale=True
    )
    return est


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_csv_path", type=str)
    parser.add_argument("-o", "--output_csv_path", type=str)
    args = parser.parse_args()
    data_csv_path = args.data_csv_path
    output_csv_path = args.output_csv_path

    tolerances = [0.5, 1.5, 2.5, 3.5, 4.5]
    est = load_est()

    def pred_output_hook(xs, ys, test_input):
        return pred_output(est, xs, ys, test_input)

    df = pd.read_csv(data_csv_path)
    summary, all_data = evaluate_model(pred_output_hook, df, tolerances)

    # Print and save results
    print("Evaluation Metrics:")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
