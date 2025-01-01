import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim, utils


class LitTransformer(pl.LightningModule):
    def __init__(
        self,
        token_num,
        token_embed_dim,
        max_input_dim,
        input_padding_idx,
        emb_expansion_factor,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = 3e-4
        self.embedding = nn.Embedding(
            token_num, token_embed_dim, padding_idx=input_padding_idx
        )
        self.token_embed_dim = token_embed_dim
        self.max_input_dim = max_input_dim
        size = self.token_embed_dim * self.max_input_dim
        hidden_size = emb_expansion_factor * size
        self.fc1 = nn.Linear(size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

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
        sequence_embeddings = self.embedding(src_input)
        batch_size, seq_len, _, _ = sequence_embeddings.shape
        sequence_embeddings = sequence_embeddings.view(batch_size, seq_len, -1)
        sequence_embeddings = F.relu(self.fc1(sequence_embeddings))
        sequence_embeddings = self.fc2(sequence_embeddings)

        # embed target

        print()

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
