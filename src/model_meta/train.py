import logging

import hydra
import lightning as pl
import torch
import torch.nn.functional as F
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
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = 3e-4
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
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@hydra.main(version_base=None, config_path=".", config_name="training_config")
def main(cfg: DictConfig):
    csv_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
    csv_path = "/home/takeru/AlphaSymbol/temp/add_in_out_extract100.csv"
    csv_path = "/home/takeru/AlphaSymbol/temp/add_in_out_extract1000.csv"
    csv_path = "/home/takeru/AlphaSymbol/temp/add_in_out.csv"
    data_module = PREDataModule(
        data_path=csv_path, batch_size=32, max_value=2000, num_workers=31
    )
    data_module.prepare_data()
    data_module.setup()
    model = LitTransformer(
        src_token_num=data_module.src_token_num,
        tgt_token_num=len(data_module.tgt_vocab),
        token_embed_dim=64,
        max_src_dim=data_module.max_input_size,
        max_tgt_dim=data_module.tgt_input_size,
        src_padding_idx=data_module.src_pad_idx,
        tgt_padding_idx=data_module.tgt_vocab["<pad>"],
        emb_expansion_factor=2,
    )
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
