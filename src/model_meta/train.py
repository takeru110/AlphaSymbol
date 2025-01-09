import copy
import logging
import pickle
from pathlib import Path

import hydra
import lightning as pl
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor, nn, optim, utils

from src.model_meta.data import PREDataModule
from src.model_meta.models import PositionalEncoding

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()],
)


class LitTransformer(pl.LightningModule):
    def __init__(
        self,
        src_token_num,
        tgt_token_num,
        token_embed_dim,
        max_src_dim,
        max_tgt_dim,
        src_padding_idx,
        tgt_padding_idx,
        emb_expansion_factor,
        t_config,
        learning_rate=3e-4,
    ):
        """
        Initialize the LitTransformer model.
        Args:
        - src_token_num (int): Number of tokens in the source vocabulary.
        - tgt_token_num (int): Number of tokens in the target vocabulary.
        - token_embed_dim (int): Dimension of the token embeddings.
        - max_src_dim (int): Maximum dimension of the source input.
        - max_tgt_dim (int): Maximum dimension of the target input.
        - src_padding_idx (int): Padding index for the source tokens.
        - tgt_padding_idx (int): Padding index for the target tokens.
        - emb_expansion_factor (int): Factor to expand the embedding dimension.
        - t_config (DictConfig): Configuration for the transformer model.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.src_embedding = nn.Embedding(
            src_token_num, token_embed_dim, padding_idx=src_padding_idx
        )
        self.token_embed_dim = token_embed_dim
        self.max_src_dim = max_src_dim
        size = self.token_embed_dim * self.max_src_dim
        hidden_size = emb_expansion_factor * size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tgt_embedding = nn.Embedding(
            tgt_token_num, hidden_size, padding_idx=tgt_padding_idx
        )
        self.pos_enc = PositionalEncoding(self.hidden_size, max_tgt_dim)
        self.transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=t_config.nhead,
            num_encoder_layers=t_config.num_encoder_layers,
            num_decoder_layers=t_config.num_decoder_layers,
            dim_feedforward=t_config.dim_feedforward,
            dropout=t_config.dropout,
        )
        self.fc_out = nn.Linear(self.hidden_size, tgt_token_num)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_padding_idx)

    def forward(
        self,
        src_input,
        tgt_input,
    ):
        """
        Args:
        - src_input: Tensor of shape (N, S: seq len, Emb)
        - tgt_input: Tensor of shape (N, T: seq len, Emb)
        """
        # embed input
        src_embeddings = self.src_embedding(src_input)
        batch_size, seq_len, _, _ = src_embeddings.shape
        src_embeddings = src_embeddings.view(batch_size, seq_len, -1)
        src_embeddings = F.relu(self.fc1(src_embeddings))
        src_embeddings = self.fc2(src_embeddings)
        src_embeddings = src_embeddings.permute(1, 0, 2)

        # embed target
        tgt_embeddings = self.tgt_embedding(tgt_input) * torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        tgt_embeddings = tgt_embeddings.permute(1, 0, 2)
        tgt_embeddings = self.pos_enc(tgt_embeddings)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_embeddings.size(0)
        )
        output = self.transformer(
            src_embeddings, tgt_embeddings, tgt_mask=tgt_mask
        )
        output = self.fc_out(output)  # (T, N, C)
        return output

    def training_step(self, batch, batch_idx):
        src_batch, tgt_batch = batch
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        output = self(src_batch, tgt_input)  # (T, N, C)
        output = output.permute(1, 2, 0)  # (N, C, T)
        loss = self.loss_fn(output, tgt_output)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
    log_dir = Path(HydraConfig.get().run.dir)
    data_module = PREDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.batch_size,
        max_value=cfg.max_value,
        num_workers=cfg.num_workers,
        test_ratio=cfg.test_ratio,
        val_ratio=cfg.val_ratio,
    )
    model = LitTransformer(
        src_token_num=data_module.src_token_num,
        tgt_token_num=len(data_module.tgt_vocab),
        token_embed_dim=cfg.token_embed_dim,
        max_src_dim=data_module.point_vector_size,
        max_tgt_dim=data_module.tgt_input_size,
        src_padding_idx=data_module.src_pad_idx,
        tgt_padding_idx=data_module.tgt_vocab["<pad>"],
        emb_expansion_factor=cfg.emb_expansion_factor,
        t_config=cfg.transformer,
        learning_rate=eval(cfg.learning_rate),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.max_epoch,
        default_root_dir=log_dir,
    )

    with open(f"{log_dir}/data_module.pkl", "wb") as f:
        data_module_for_save = copy.copy(data_module)
        data_module_for_save.df = None  # This attr is too large.
        data_module_for_save.train_seq = None
        data_module_for_save.val_seq = None
        data_module_for_save.test_seq = None
        pickle.dump(data_module_for_save, f)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
