from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from data import build_vocab, get_tgt, id2token
from models import TransformerModel


def tokenize(string: str, vocab: dict[str, int]) -> list[int]:
    return (
        [vocab["<sos>"]] + [vocab[token] for token in string] + [vocab["<eos>"]]
    )


class TransformerDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Args
        df: Should have columns "expr", "input" and "output.
        """
        self.df = df
        self.df["tgt_str"] = df["expr"].apply(lambda x: x.replace(" ", ""))
        self.df["src_str"] = df.apply(get_tgt, axis=1)
        self.src_vocab = build_vocab(df["src_str"])
        self.tgt_vocab = build_vocab(df["tgt_str"])
        self.src_max_len = df["src_str"].apply(len).max()
        self.tgt_max_len = df["tgt_str"].apply(len).max()
        self.config = {
            "src_vocab": self.src_vocab,
            "tgt_vocab": self.tgt_vocab,
            "src_max_len": int(self.src_max_len),
            "tgt_max_len": int(self.tgt_max_len),
        }

    def __len__(self):
        return len(self.df)

    def pad_sequence(self, seq, vocab, max_len):
        return seq + [vocab["<pad>"]] * (max_len - len(seq))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_tokens = tokenize(row["src_str"], self.src_vocab)
        tgt_tokens = tokenize(row["tgt_str"], self.tgt_vocab)
        src_padded = self.pad_sequence(
            src_tokens[: self.src_max_len], self.src_vocab, self.src_max_len
        )
        tgt_padded = self.pad_sequence(
            tgt_tokens[: self.tgt_max_len], self.tgt_vocab, self.tgt_max_len
        )
        return torch.tensor(src_padded), torch.tensor(tgt_padded)


def train(data_path: Path, models_output_dir: Path, config_dir: Path):
    # make dataset
    df = pd.read_csv(data_path)

    dataset = TransformerDataset(df)
    # self.src_str like [0]:0,[1]:0,[2]:0,[3]:1,[4]:2,[5]:3,[6]:4,[7]:5,[8]:6,[9]:7
    # self.tgt_str like C(R(P(1,1),P(2,1),Z()),R(P(1,1),P(2,1),Z()))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    config_path = config_dir / f"config_{timestamp}.yaml"
    with open(config_path, "w") as yaml_file:
        yaml.dump(dataset.config, yaml_file, default_flow_style=False)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TransformerModel(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        src_max_len=dataset.src_max_len,
        tgt_max_len=dataset.tgt_max_len,
        d_model=64,
        nhead=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.src_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    best_loss = float("inf")

    models_output = (
        models_output_dir / f"model_{num_epochs}epochs_{timestamp}.pth"
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for src_batch, tgt_batch in dataloader:
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            optimizer.zero_grad()
            output = model(src_batch, tgt_input)  # (T, N, C)
            output = output.permute(1, 2, 0)  # (N, C, T) for CrossEntropyLoss
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), models_output)


if __name__ == "__main__":
    # set parameters
    data_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
    models_output_dir = "./temp/"
    config_dir = "./temp/"
    train(Path(data_path), Path(models_output_dir), Path(config_dir))
