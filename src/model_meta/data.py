from typing import Iterable

import lightning as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class PREDataModule(pl.LightningDataModule):
    """
    This model is the data module for model of Meta AI.
    """

    def __init__(self, data_path, batch_size, max_value, num_workers=0):
        """
        Args:
        - data_path (str): path to the CSV file
        - batch_size (str): int
        - max_value (int): the maximum value of the input and output
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.src_pad_idx = max_value + 1
        self.src_sos_idx = max_value + 2
        self.src_eos_idx = max_value + 3
        self.src_token_num = max_value + 4
        self.num_workers = num_workers

    def prepare_data(self):
        self.df = pd.read_csv(self.data_path)

    def setup(self, stage=None):
        # process source data
        seq_idx = []
        for input, output in zip(self.df["input"], self.df["output"]):
            input, output = eval(input), eval(output)
            point_li = []
            for x, y in zip(input, output):
                point_li.append([self.src_sos_idx, *x, y, self.src_eos_idx])
            seq_idx.append(point_li)

        # pad source data
        self.max_input_size = max([len(seq[0]) for seq in seq_idx])
        for i, seq in enumerate(seq_idx):
            current_len = len(seq[0])
            for p in seq:
                p.extend(
                    [self.src_pad_idx] * (self.max_input_size - current_len)
                )
        seq_idx = torch.tensor(seq_idx)

        # process target data
        self.tgt_vocab = self.build_vocab(self.df["expr"])
        tgt_tokens = []
        for target in self.df["expr"]:
            tgt_tokens.append(["<sos>"] + list(target) + ["<eos>"])

        self.tgt_input_size = max([len(seq) for seq in tgt_tokens])

        for seq in tgt_tokens:
            seq.extend(["<pad>"] * (self.tgt_input_size - len(seq)))

        tgt_idx = [
            [self.tgt_vocab[token] for token in seq] for seq in tgt_tokens
        ]
        tgt_idx = torch.tensor(tgt_idx)

        # combine source and target data
        self.point_num = len(seq_idx[0])

        # Split the data into training, validation, and test sets
        dataset = TensorDataset(seq_idx, tgt_idx)

        train_val_seq, test_seq = train_test_split(
            dataset, test_size=0.2, random_state=42
        )
        train_seq, val_seq = train_test_split(
            train_val_seq, test_size=0.25, random_state=42
        )  # 0.25 * 0.8 = 0.2

        self.train_seq = train_seq
        self.val_seq = val_seq
        self.test_seq = test_seq

    def build_vocab(self, strings: Iterable[str]) -> dict[str, int]:
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        idx = 3
        for sentence in strings:
            for token in sentence:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def train_dataloader(self):
        return DataLoader(
            self.train_seq,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_seq,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_seq,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
