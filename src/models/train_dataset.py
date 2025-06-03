import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.dataset import PRFDataset
from src.models.train import LitTransformer


def setup_logging(level=None):
    """ログレベルを設定する関数"""
    if level is None:
        # 1. コマンドライン引数をチェック
        if "--log-level" in sys.argv:
            idx = sys.argv.index("--log-level")
            if idx + 1 < len(sys.argv):
                level = sys.argv[idx + 1].upper()
        # 2. 環境変数をチェック
        elif "LOG_LEVEL" in os.environ:
            level = os.environ["LOG_LEVEL"].upper()
        # 3. デフォルト値
        else:
            level = "INFO"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        force=True,  # 既存の設定を上書き
    )

    return level


# 早期にログレベルを設定
current_log_level = setup_logging()


def collate_fn(batch):
    """
    Custom collate function to convert PRFDataset output format
    to the format expected by LitTransformer.

    PRFDataset returns: {'src': array, 'tgt': array, 'original_expr': str}
    LitTransformer expects: (src_tensor, tgt_tensor)
    """
    src_list = []
    tgt_list = []

    for item in batch:
        # Convert numpy arrays to tensors
        src = torch.tensor(item["src"], dtype=torch.long)
        tgt = torch.tensor(item["tgt"], dtype=torch.long)

        # srcは既にトークン化済みなので、そのまま使用
        src_list.append(src)
        tgt_list.append(tgt)

    # Find maximum lengths for padding
    max_src_len = max(src.size(0) for src in src_list)
    max_tgt_len = max(tgt.size(0) for tgt in tgt_list)

    # Pad sequences to maximum length
    src_padded = []
    tgt_padded = []

    # Use padding ID 0 for both src and tgt (standard padding)
    src_pad_id = 0
    tgt_pad_id = 0

    for src, tgt in zip(src_list, tgt_list):
        # Pad src sequence
        if src.size(0) < max_src_len:
            padding_size = max_src_len - src.size(0)
            src_padded.append(
                torch.cat(
                    [
                        src,
                        torch.full(
                            (padding_size,), src_pad_id, dtype=torch.long
                        ),
                    ]
                )
            )
        else:
            src_padded.append(src)

        # Pad tgt sequence
        if tgt.size(0) < max_tgt_len:
            padding_size = max_tgt_len - tgt.size(0)
            tgt_padded.append(
                torch.cat(
                    [
                        tgt,
                        torch.full(
                            (padding_size,), tgt_pad_id, dtype=torch.long
                        ),
                    ]
                )
            )
        else:
            tgt_padded.append(tgt)

    # Stack tensors to create batch
    src_batch = torch.stack(src_padded)
    tgt_batch = torch.stack(tgt_padded)

    return src_batch, tgt_batch


def train(
    csv_path: str,
    metadata_path: str,
    batch_size: int = 32,
    max_epochs: int = 10,
    learning_rate: float = 3e-4,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    accelerator: str = "auto",
    log_dir: str = None,
    num_workers: int = 4,
    # 新しいstep-based設定パラメータを追加
    checkpoint_every_n_steps: int = 1000,
    val_check_interval: int = 2000,
    limit_val_batches: int = 100,
    log_every_n_steps: int = 50,
    save_top_k: int = 3,
    # WandB設定パラメータを追加
    wandb_project: str = "PrfSR-experiments",
    wandb_entity: str = None,
    wandb_name: str = None,
    wandb_tags: list = None,
    wandb_log_model: str = "all",
    wandb_log_gradients: bool = False,
    wandb_save_code: bool = True,
):
    """
    Train the transformer model using PRFDataset with separate src and tgt tokenizers.

    Args:
        csv_path: Path to the CSV file containing training data
        metadata_path: Path to the metadata.yaml file
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        accelerator: Device accelerator ('auto', 'cpu', 'gpu', etc.)
        log_dir: Directory for logging (if None, creates timestamp-based dir)

    Returns:
        Tuple of (trained_model, trainer, dataset)
    """
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y-%m%d-%H%M-%S")
        log_dir = Path(f"logs/{timestamp}")
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Start Loading metadata and csv")

    # Load metadata.yaml first
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    logging.info(f"Loaded metadata: {metadata}")

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        tags=wandb_tags,
        log_model=wandb_log_model,
        save_code=wandb_save_code,
    )

    # Create PRFDataset with the new signature
    dataset = PRFDataset(
        csv_path=csv_path,
        max_tgt_length=metadata["max_tgt_length"],
        max_src_points=metadata["max_src_points"],
        src_vocab_list=metadata["src_vocab_list"],
        tgt_vocab_list=metadata["tgt_vocab_list"],
    )
    logging.info("Finished Loading csv")
    logging.info("Example data from PRFDataset:")
    logging.info(f"First item: {dataset[0]}")
    logging.info(f"Dataset size: {len(dataset)}")
    

    # Get dataset configuration with separate tokenizers
    config = dataset.get_config()

    # Get separate tokenizers
    src_tokenizer = dataset.get_src_tokenizer()
    tgt_tokenizer = dataset.get_tgt_tokenizer()

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Use custom collate function to convert data format
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # Create vocab dictionaries from the custom tokenizers
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab

    # Calculate proper maximum sequence lengths from a sample batch
    sample_batch = next(iter(train_loader))
    src_sample, tgt_sample = sample_batch

    # Use actual vocabulary sizes instead of flattened input size
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Set reasonable maximum lengths with some buffer
    src_max_len = src_sample.shape[1] + 100  # Add buffer for dynamic lengths
    tgt_max_len = tgt_sample.shape[1] + 100  # Add buffer for dynamic lengths

    lightning_module = LitTransformer(
        src_vocab_size=src_vocab_size,  # Use actual src vocab size
        tgt_vocab_size=tgt_vocab_size,  # Use actual tgt vocab size
        src_max_len=src_max_len,  # Use calculated max length with buffer
        tgt_max_len=tgt_max_len,  # Use calculated max length with buffer
        learning_rate=learning_rate,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    # Set up step-based checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir / "checkpoints",
        filename="model-{step}-{val_loss:.2f}",
        save_top_k=save_top_k,
        mode="min",
        every_n_train_steps=checkpoint_every_n_steps,
        save_last=True,  # Always save the last checkpoint
    )

    
    # Log experiment configuration to WandB
    wandb_logger.experiment.config.update({
        "dataset": {
            "csv_path": csv_path,
            "metadata_path": metadata_path,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "src_max_len": src_max_len,
            "tgt_max_len": tgt_max_len,
        },
        "model": {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": learning_rate,
        },
        "training": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "accelerator": accelerator,
            "num_workers": num_workers,
            "checkpoint_every_n_steps": checkpoint_every_n_steps,
            "val_check_interval": val_check_interval,
            "limit_val_batches": limit_val_batches,
            "log_every_n_steps": log_every_n_steps,
        }
    })

    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        # Step-based validation and logging settings
        check_val_every_n_epoch=None,  # Disable epoch-based validation
        val_check_interval=val_check_interval,  # Validate every N training steps
        limit_val_batches=limit_val_batches,  # Limit validation batches for speed
        log_every_n_steps=log_every_n_steps,  # Log training metrics every N steps
        callbacks=[checkpoint_callback],
        logger=wandb_logger,  # Add WandB logger
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, train_loader, val_loader)

    # Display training results
    logging.info("Training completed!")
    logging.info(f"Training logs saved to: {log_dir}")

    # Display tokenizer information
    logging.info(f"Source tokenizer vocab size: {len(src_tokenizer.vocab)}")
    logging.info(f"Target tokenizer vocab size: {len(tgt_tokenizer.vocab)}")
    logging.info(f"Source padding ID: {config['src_padding_id']}")
    logging.info(f"Target padding ID: {config['tgt_padding_id']}")
    logging.info(f"Source max length: {src_max_len}")
    logging.info(f"Target max length: {tgt_max_len}")

    # Test the model
    test_results = trainer.test(lightning_module, test_loader)
    logging.info(f"Test results: {test_results}")

    return lightning_module, trainer, dataset


@hydra.main(version_base=None, config_path=".", config_name="training_config")
def main(cfg: DictConfig):
    """
    Main function to train the transformer model using configuration file.
    """
    log_dir = Path(HydraConfig.get().run.dir)

    # Extract step-based settings from config with defaults
    checkpoint_cfg = cfg.get("checkpoint", {})
    validation_cfg = cfg.get("validation", {})
    logging_cfg = cfg.get("logging", {})
    wandb_cfg = cfg.get("wandb", {})

    # Call the train function with parameters from config
    model, trainer, dataset = train(
        csv_path=cfg.csv_path,
        metadata_path=cfg.metadata_path,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        learning_rate=eval(cfg.learning_rate),
        d_model=cfg.transformer.d_model,
        nhead=cfg.transformer.nhead,
        num_encoder_layers=cfg.transformer.num_encoder_layers,
        num_decoder_layers=cfg.transformer.num_decoder_layers,
        dim_feedforward=cfg.transformer.dim_feedforward,
        dropout=cfg.transformer.dropout,
        accelerator=cfg.accelerator,
        log_dir=str(log_dir),
        num_workers=cfg.get("num_workers", 4),
        # Step-based configuration from config file
        checkpoint_every_n_steps=checkpoint_cfg.get("every_n_train_steps", 1000),
        val_check_interval=validation_cfg.get("val_check_interval", 2000),
        limit_val_batches=validation_cfg.get("limit_val_batches", 100),
        log_every_n_steps=logging_cfg.get("log_every_n_steps", 50),
        save_top_k=checkpoint_cfg.get("save_top_k", 3),
        # WandB configuration from config file
        wandb_project=wandb_cfg.get("project", "PrfSR-experiments"),
        wandb_entity=wandb_cfg.get("entity"),
        wandb_name=wandb_cfg.get("name"),
        wandb_tags=wandb_cfg.get("tags", []),
        wandb_log_model=wandb_cfg.get("log_model", "all"),
        wandb_log_gradients=wandb_cfg.get("log_gradients", False),
        wandb_save_code=wandb_cfg.get("save_code", True),
    )

    logging.info("Training completed successfully!")
    return model, trainer, dataset


if __name__ == "__main__":
    main()
