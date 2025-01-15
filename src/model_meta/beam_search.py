import pickle
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from prfndim.prfndim import C, Expr, P, R, S, Z
from src.model_meta.data import PREDataModule
from src.model_meta.train import LitTransformer


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


def beam_search(model, data_module, xs, ys, beam_width=3):
    """
    Perform beam search for generating valid expressions.
    Args:
    - model: The trained transformer model.
    - data_module: The data module containing vocabulary and helper functions.
    - xs: The input data for evaluating the expression.
    - beam_width (int): Number of candidates to keep at each step.
    - max_length (int): Maximum length of the generated sequence.
    """
    device = model.device

    # Initialize the beam with the start token
    beam = [([], 0)]  # List of (sequence, score)
    discovered: list[tuple[Expr, int]] = []

    for gen_token_idx in range(1, data_module.tgt_input_size):
        new_beam = []

        for current_token_list, score in beam:
            # Prepare inputs for the model
            src_input_list = [
                data_module.src_pad_point(data_module.src_add_ends([*x, y]))
                for x, y in zip(xs, ys)
            ]
            src_input = torch.tensor(src_input_list).to(device).unsqueeze(0)

            tgt_token_list = data_module.tgt_add_ends(current_token_list)
            tgt_token_list = data_module.tgt_pad(tgt_token_list)
            tgt_token_idx = [data_module.tgt_vocab[t] for t in tgt_token_list]
            tgt_input = torch.tensor(tgt_token_idx[:-1]).to(device).unsqueeze(0)

            # Generate predictions
            with torch.no_grad():
                output = model(src_input, tgt_input)
                logits = output[gen_token_idx - 1, 0, :]
                pred = logits.softmax(dim=0)
                topk_scores, topk_ids = torch.topk(pred, beam_width)

            # Expand the beam with the top-k candidates
            for k in range(beam_width):
                new_token = id2token(topk_ids[k].item(), data_module.tgt_vocab)
                new_score = score + topk_scores[k].item()

                # Stop expansion if <eos> is generated
                if new_token == "<eos>":
                    expr_str = "".join(current_token_list)
                    try:
                        expression = eval(expr_str)
                    except Exception:
                        continue

                    try:
                        if expression.is_valid and isinstance(
                            expression, Expr
                        ):  # Ensure the expression is valid
                            discovered.append((expression, new_score))
                        else:
                            continue
                    except Exception:
                        continue

                new_beam.append((current_token_list + [new_token], new_score))

        # Prune the beam to keep only the top-k candidates
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[
            :beam_width
        ]
        beam = new_beam

    # Return the best candidate after max_length iterations
    best_candidate = max(discovered, key=lambda x: x[1])
    return best_candidate[0]


# Load the model and data module
model_path = "/home/takeru/AlphaSymbol/logs/2025-0114-1041-37/best_model-epoch=213-val_loss=0.08.ckpt"
data_module_path = (
    "/home/takeru/AlphaSymbol/logs/2025-0114-1041-37/data_module.pkl"
)

model = LitTransformer.load_from_checkpoint(model_path)

with open(data_module_path, "rb") as f:
    data_module = pickle.load(f)

# Input data

xs = [(random.randint(0, 9),) for _ in range(20)]
ys = [3 * i[0] for i in xs]

print(xs)
print(ys)

# Perform beam search
expression = beam_search(
    model,
    data_module,
    xs,
    ys,
    beam_width=5,
)

# Evaluate the result
print(expression)
print([expression.eval(*x) for x in xs])
