from typing import Iterable, Optional

import pandas as pd


def get_tgt_str(inputs: list[tuple[int]], outputs: list[int]) -> str:
    tgt_str_li = [
        f"{list(input)}:{output}," for input, output in zip(inputs, outputs)
    ]
    tgt_str = "".join(tgt_str_li)
    tgt_str = tgt_str.replace(" ", "")
    tgt_str = tgt_str[:-1]
    return tgt_str


def get_tgt(row: pd.Series) -> str:
    """
    Usage:
        row = {"input": "[(1, 10), (2, 20), (3, 30)]", "output": "[4, 5, 6]"}
        get_tgt(row) # [1,10]:4,[2,20]:5,[3,30]:6
    """
    inputs = eval(row["input"])
    outputs = eval(row["output"])
    return get_tgt_str(inputs, outputs)


def build_vocab(strings: Iterable[str]):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for sentence in strings:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def id2token(vocab: dict[int, str], id: int) -> Optional[str]:
    for token, idx in vocab.items():
        if idx == id:
            return token
    return None
