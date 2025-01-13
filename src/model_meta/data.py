import random
from typing import Iterable

import lightning as pl
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(
        self, seq_idx, tgt_idx, src_pad_idx, tgt_pad_idx, few_pad_batches=False
    ):
        """
        Args:
            seq_idx (list): List of input sequences.
            tgt_idx (list): List of target sequences.
        """
        self.seq_idx, self.tgt_idx = (
            self.align_and_shuffle(seq_idx, tgt_idx, src_pad_idx, tgt_pad_idx)
            if few_pad_batches
            else (
                seq_idx,
                tgt_idx,
            )
        )

    def align_and_shuffle(self, seq, tgt, src_pad_idx, tgt_pad_idx):
        # Combine source and target into a single list of tuples
        data = list(zip(seq, tgt))

        # Sort by length of the source and shuffle within groups of the same length
        sorted_data = sorted(
            data, key=lambda x: len(x[0])
        )  # Sort by length of source
        grouped = {}
        for item in sorted_data:
            length = len(item[0])
            grouped.setdefault(length, []).append(item)

        # Shuffle each group randomly
        shuffled_data = []
        for group in grouped.values():
            random.shuffle(group)
            shuffled_data.extend(group)

        # Split the sorted and shuffled data back into source and target
        sorted_source, sorted_target = zip(*shuffled_data)

        # Convert back to lists
        sorted_source = list(sorted_source)
        sorted_target = list(sorted_target)
        return sorted_source, sorted_target

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

        data_list = list(zip(seq_idx, tgt_idx))
        random.shuffle(data_list)
        shuffled_seq_idx, shuffled_tgt_idx = zip(*data_list)

        # Split the data into training, validation, and test sets
        # devider means "train | val | test"
        data_len = len(shuffled_seq_idx)
        train_val_devider_idx = int(data_len - self.test_ratio * data_len)

        self.train_data = CustomDataset(
            shuffled_seq_idx[:train_val_devider_idx],
            shuffled_tgt_idx[:train_val_devider_idx],
            self.src_pad_idx,
            self.tgt_vocab["<pad>"],
            few_pad_batches=True,
        )

        val_test_seq_idx = shuffled_seq_idx[train_val_devider_idx:]
        val_test_tgt_idx = shuffled_tgt_idx[train_val_devider_idx:]
        val_test_len = len(val_test_seq_idx)
        test_val_devider_idx = int(self.val_ratio * val_test_len)
        self.val_data = CustomDataset(
            val_test_seq_idx[:test_val_devider_idx],
            val_test_tgt_idx[:test_val_devider_idx],
            self.src_pad_idx,
            self.tgt_vocab["<pad>"],
        )
        self.test_data = CustomDataset(
            val_test_seq_idx[test_val_devider_idx:],
            val_test_tgt_idx[test_val_devider_idx:],
            self.src_pad_idx,
            self.tgt_vocab["<pad>"],
        )

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
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_vocab["<pad>"]
            ),
        )
        return data_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_vocab["<pad>"]
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_vocab["<pad>"]
            ),
        )


def collate_fn(batch, src_pad_id, tgt_pad_id):
    """
    Custom collate function to pad sequences with <pad> token.
    """
    src_max_points = max(len(x) for x, _ in batch)

    x_padded = [
        torch.cat(
            [
                x,
                torch.ones((src_max_points - x.shape[0], x.shape[1]))
                * src_pad_id,
            ],
            dim=0,
        )
        for x, _ in batch
    ]
    y_padded = [y for _, y in batch]

    return torch.stack(x_padded).type(torch.int64), torch.stack(y_padded).type(
        torch.int64
    )
