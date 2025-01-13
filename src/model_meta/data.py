from typing import Iterable

import lightning as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, seq_idx, tgt_idx):
        """
        Args:
            seq_idx (list): List of input sequences.
            tgt_idx (list): List of target sequences.
        """
        self.seq_idx = seq_idx
        self.tgt_idx = tgt_idx

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.seq_idx)

    def __getitem__(self, idx):
        """
        Dynamically loads and converts a single sample to tensor.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        input_data = torch.tensor(self.seq_idx[idx], dtype=torch.int64)
        target_data = torch.tensor(self.tgt_idx[idx], dtype=torch.int64)
        return input_data, target_data


class PREDataModule(pl.LightningDataModule):
    """
    This model is the data module for model of Meta AI.
    """

    def __init__(
        self,
        data_path,
        batch_size,
        max_value,
        num_workers=0,
        test_ratio=0.2,
        val_ratio=0.25,
    ):
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
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.df = pd.read_csv(self.data_path)
        self.setup_attrs()

    def src_add_ends(self, point: list[int]) -> list[int]:
        return [self.src_sos_idx] + point + [self.src_eos_idx]

    def src_pad_point(self, point: list[int]) -> list[int]:
        return point + [self.src_pad_idx] * (
            self.point_vector_size - len(point)
        )

    def tgt_add_ends(self, list_char: list[str]) -> list[str]:
        """add <sos> and <eos> to the target token list
        Args:
            list_char (list[str]): list of characters
        """
        return ["<sos>"] + list_char + ["<eos>"]

    def tgt_pad(self, list_char: list[str]) -> list[str]:
        """pad the target token list
        Args:
            list_char (list[str]): list of characters
            max_len (int): the maximum length of the target list
        """
        return list_char + ["<pad>"] * (self.tgt_input_size - len(list_char))

    def setup_attrs(self, stage=None):
        # process source data
        seq_idx = []
        for input_str, output_str in zip(self.df["input"], self.df["output"]):
            input, output = eval(input_str), eval(output_str)
            point_li = []
            for x, y in zip(input, output):
                point_li.append([*x, y])
            seq_idx.append(point_li)

        # pad source data
        # + 2 means the length of <sos> and <tgt>
        self.point_vector_size = max([len(seq[0]) for seq in seq_idx]) + 2
        for seq in seq_idx:
            for i, p in enumerate(seq):
                p_with_ends = self.src_add_ends(p)
                seq[i] = self.src_pad_point(p_with_ends)
        self.point_num = len(seq_idx[0])

        # process target data
        self.tgt_vocab = self.build_vocab(self.df["expr"])
        tgt_tokens = []
        for target in self.df["expr"]:
            tgt_tokens.append(list(target))

        # + 2 means the length of <sos> and <tgt>
        self.tgt_input_size = max([len(seq) for seq in tgt_tokens]) + 2
        for i, seq in enumerate(tgt_tokens):
            seq_ends = self.tgt_add_ends(seq)
            tgt_tokens[i] = self.tgt_pad(seq_ends)

        tgt_idx = [
            [self.tgt_vocab[token] for token in seq] for seq in tgt_tokens
        ]

        # Split the data into training, validation, and test sets
        dataset = CustomDataset(seq_idx, tgt_idx)
        train_val_seq, test_seq = train_test_split(
            dataset, test_size=self.test_ratio, random_state=42
        )
        train_seq, val_seq = train_test_split(
            train_val_seq, test_size=self.val_ratio, random_state=42
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
        data_loader = DataLoader(
            self.train_seq,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return data_loader

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
