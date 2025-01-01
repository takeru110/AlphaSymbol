import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim, utils

from .models import PositionalEncoding


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
        output = self.transformer(src_embeddings, tgt_embeddings, tgt_mask=tgt_mask)
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
