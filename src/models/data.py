from typing import Iterable, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def build_vocab(strings: Iterable[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for sentence in strings:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def id2token(vocab: dict[str, int], id: int) -> Optional[str]:
    for token, idx in vocab.items():
        if idx == id:
            return token
    return None


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


def tokenize(string: str, vocab: dict[str, int]) -> list[int]:
    return (
        [vocab["<sos>"]] + [vocab[token] for token in string] + [vocab["<eos>"]]
    )


class TransformerDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args
        df: Should have columns "expr", "input" and "output".
        """
        self.df = pd.read_csv(csv_path)
        self.df["tgt_str"] = self.df["expr"].apply(lambda x: x.replace(" ", ""))
        self.df["src_str"] = self.df.apply(get_tgt, axis=1)
        self.src_vocab = build_vocab(self.df["src_str"])
        self.tgt_vocab = build_vocab(self.df["tgt_str"])
        self.src_max_len = int(self.df["src_str"].apply(len).max() + 2)
        self.tgt_max_len = int(self.df["tgt_str"].apply(len).max() + 2)
        self.config = {
            "src_vocab": self.src_vocab,
            "tgt_vocab": self.tgt_vocab,
            "src_max_len": int(self.src_max_len),
            "tgt_max_len": int(self.tgt_max_len),
        }

    def __len__(self):
        return len(self.df)

    def pad_sequence(self, seq, vocab, max_len):
        assert max_len >= len(seq), "max_len should be greater than len(seq)"
        return seq + [vocab["<pad>"]] * (max_len - len(seq))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_tokens = tokenize(row["src_str"], self.src_vocab)
        tgt_tokens = tokenize(row["tgt_str"], self.tgt_vocab)
        src_padded = self.pad_sequence(
            src_tokens, self.src_vocab, self.src_max_len
        )
        tgt_padded = self.pad_sequence(
            tgt_tokens, self.tgt_vocab, self.tgt_max_len
        )
        return torch.tensor(src_padded), torch.tensor(tgt_padded)
