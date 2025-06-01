import logging
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

from src.models.dataset import PRFDataset
from src.models.train import LitTransformer


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
        # Convert numpy arrays to tensors and flatten src if needed
        src = torch.tensor(item["src"], dtype=torch.long)
        tgt = torch.tensor(item["tgt"], dtype=torch.long)

        # Flatten src array: (d+1, max_src_points) -> (flattened_size,)
        src_flattened = src.flatten()

        src_list.append(src_flattened)
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
):
    """
    Train the transformer model using PRFDataset with separate src and tgt tokenizers.

    Args:
        csv_path: Path to the CSV file containing training data
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

    logging.info("Start Loading csv")
    dataset = PRFDataset(csv_path)
    logging.info("Finished Loading csv")

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
        train_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
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

    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
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

    # Call the train function with parameters from config
    model, trainer, dataset = train(
        csv_path=cfg.csv_path,
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
    )

    logging.info("Training completed successfully!")
    return model, trainer, dataset


if __name__ == "__main__":
    main()
