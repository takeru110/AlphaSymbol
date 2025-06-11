import logging
import os
import sys
import tempfile
from pathlib import Path

import lightning as L
import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.data import TransformerDataset
from src.models.train import LitTransformer, main


@pytest.fixture
def training_config():
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_train(training_config):
    """小さなデータセットでトレーニング機能をテスト"""
    root_dir = Path(__file__).parent.parent.parent

    # データセットの作成
    logging.info("csv_path: %s", training_config["csv_path"])
    dataset = TransformerDataset(training_config["csv_path"])

    # データ分割
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=training_config["batch_size"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config["batch_size"]
    )

    # モデルの初期化
    lightning_module = LitTransformer(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        src_max_len=dataset.src_max_len,
        tgt_max_len=dataset.tgt_max_len,
        learning_rate=eval(training_config["learning_rate"]),
        src_vocab=dataset.src_vocab,
        tgt_vocab=dataset.tgt_vocab,
        d_model=training_config["transformer"]["d_model"],
        nhead=training_config["transformer"]["nhead"],
        num_encoder_layers=training_config["transformer"]["num_encoder_layers"],
        num_decoder_layers=training_config["transformer"]["num_decoder_layers"],
        dim_feedforward=training_config["transformer"]["dim_feedforward"],
        dropout=training_config["transformer"]["dropout"],
    )

    trainer = L.Trainer(
        default_root_dir=root_dir / "logs",
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    # トレーニング実行
    trainer.fit(lightning_module, train_loader, val_loader)
