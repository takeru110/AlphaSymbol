from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, Dataset

from data import TransformerDataset
from models import PositionalEncoding


class LitTransformer(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_max_len,
        tgt_max_len,
        src_vocab,
        tgt_vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.src_pos_enc = PositionalEncoding(d_model, src_max_len)
        self.tgt_pos_enc = PositionalEncoding(d_model, tgt_max_len - 1)
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=src_vocab["<pad>"]
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=tgt_vocab["<pad>"]
        )
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])

    def forward(self, src, tgt):
        """
        Args:
        - src: Tensor of shape (N, S: seq len, Emb)
        - tgt: Tensor of shape (N, T: seq len, Emb)
        """
        src = self.src_embedding(src) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        tgt = self.tgt_embedding(tgt) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32)
        )
        src = src.permute(1, 0, 2)  # (S, N, E)
        tgt = tgt.permute(1, 0, 2)  # (T, N, E)
        src = src + self.src_pos_enc(src)
        tgt = tgt + self.tgt_pos_enc(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0))
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(output)  # (T, N, C)
        return output

    def training_step(self, batch, batch_idx):
        # src_batch: (N, S), tgt_batch: (N, T)
        src_batch, tgt_batch = batch
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        output = self(src_batch, tgt_input)  # (T, N, C)
        output = output.permute(1, 2, 0)  # (N, C, T)
        loss = self.loss_fn(output, tgt_output)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # src_batch: (N, S), tgt_batch: (N, T)
        src_batch, tgt_batch = batch
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        output = self(src_batch, tgt_input)  # (T, N, C)
        output = output.permute(1, 2, 0)  # (N, C, T)
        loss = self.loss_fn(output, tgt_output)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@hydra.main(version_base=None, config_path=".", config_name="training_config")
def main(cfg: DictConfig):
    csv_path = cfg.csv_path

    log_dir = Path(HydraConfig.get().run.dir)

    df = pd.read_csv(csv_path)
    raw_dataset = TransformerDataset(df)

    dataset: Dataset = utils.data.ConcatDataset([raw_dataset] * 100)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=cfg.batch_size
    )

    val_loader: DataLoader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=cfg.batch_size
    )

    lightning_module = LitTransformer(
        src_vocab_size=len(raw_dataset.src_vocab),
        tgt_vocab_size=len(raw_dataset.tgt_vocab),
        src_max_len=raw_dataset.src_max_len,
        tgt_max_len=raw_dataset.tgt_max_len,
        learning_rate=eval(cfg.learning_rate),
        src_vocab=raw_dataset.src_vocab,
        tgt_vocab=raw_dataset.tgt_vocab,
    )

    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(lightning_module, train_loader, val_loader)


if __name__ == "__main__":
    main()
