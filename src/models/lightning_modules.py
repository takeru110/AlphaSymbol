import lightning as L
from torch import Tensor, nn, optim, utils

from models import PositionalEncoding, TransformerModel


class LitTransformer(L.LightningModule):
    def __init__(self, model, learning_rate=3e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        """
        Args:
        - src: Tensor of shape (N, S: seq len, Emb)
        - tgt: Tensor of shape (N, T: seq len, Emb)
        """
        ans = self.model(src, tgt)
        ans = ans.permute(1, 0, 2)
        return ans

    def training_step(self, batch, batch_idx):
        # src_batch: (N, S), tgt_batch: (N, T)
        src_batch, tgt_batch = batch
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        output = self.model(src_batch, tgt_input)  # (T, N, C)
        output = output.permute(1, 2, 0)  # (N, C, T)
        loss = self.loss_fn(output, tgt_output)
        loss.backward()
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

data_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
models_output_dir = "./temp/"
config_dir = "./temp/"
model = TransformerModel()
lightning_module = LitTransformer(