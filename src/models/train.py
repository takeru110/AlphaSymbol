import lightning as L
import pandas as pd
import torch
from torch import Tensor, nn, optim, utils

from data import TransformerDataset
from models import PositionalEncoding


class LitTransformer(L.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_max_len,
        tgt_max_len,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
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
        self.src_pos_enc = PositionalEncoding(d_model, src_max_len)
        self.tgt_pos_enc = PositionalEncoding(d_model, tgt_max_len)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

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
        if self.training:
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        else:
            self.transformer.eval()
            output = self.transformer(src, tgt, src_mask=None, tgt_mask=None)
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
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


data_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
models_output_dir = "./temp/"
config_dir = "./temp/"
df = pd.read_csv(data_path)
dataset = TransformerDataset(df)
dataloadr = utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

lightning_module = LitTransformer(
    src_vocab_size=len(dataset.src_vocab),
    tgt_vocab_size=len(dataset.tgt_vocab),
    src_max_len=dataset.src_max_len,
    tgt_max_len=dataset.tgt_max_len,
    learning_rate=0.0001,
)

trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(lightning_module, dataloadr)
