import pickle
from pathlib import Path

import torch

from prfndim.prfndim import C, P, R, S, Z
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


model_path = "/home/takeru/AlphaSymbol/logs/2025-0109-1845-03/lightning_logs/version_0/checkpoints/epoch=4-step=60.ckpt"
data_module_path = (
    "/home/takeru/AlphaSymbol/logs/2025-0109-1845-03/data_module.pkl"
)

model = LitTransformer.load_from_checkpoint(model_path)

with open(data_module_path, "rb") as f:
    data_module = pickle.load(f)


xs = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
ys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


current_token_list: list[str] = []
with torch.no_grad():
    for gen_token_idx in range(1, data_module.tgt_input_size):
        src_input_list = [
            data_module.src_pad_point(data_module.src_add_ends([*x, y]))
            for x, y in zip(xs, ys)
        ]
        src_input = torch.tensor(src_input_list).to(model.device).unsqueeze(0)

        tgt_token_list = data_module.tgt_add_ends(current_token_list)
        tgt_token_list = data_module.tgt_pad(tgt_token_list)
        tgt_token_idx = [data_module.tgt_vocab[t] for t in tgt_token_list]
        tgt_input = (
            torch.tensor(tgt_token_idx[:-1]).to(model.device).unsqueeze(0)
        )

        output = model(src_input, tgt_input)
        pred = output[gen_token_idx - 1, 0, :]
        _, max_id = torch.max(pred, 0)
        new_token = id2token(max_id, data_module.tgt_vocab)
        if new_token == "<eos>":
            break
        current_token_list.append(new_token)
expression = "".join(current_token_list)
print(expression)
pre = eval(expression)
print([pre.eval(x) for x in xs])
