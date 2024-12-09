import lightning as L
import pandas as pd
from torch import Tensor, nn, optim, utils
from train import TransformerDataset, TransformerModel

from models import PositionalEncoding


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
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


data_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
models_output_dir = "./temp/"
config_dir = "./temp/"
df = pd.read_csv(data_path)
dataset = TransformerDataset(df)
dataloadr = utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

model = TransformerModel(
    src_vocab_size=len(dataset.src_vocab),
    tgt_vocab_size=len(dataset.tgt_vocab),
    src_max_len=dataset.src_max_len,
    tgt_max_len=dataset.tgt_max_len,
    d_model=64,
    nhead=4,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=512,
    dropout=0.1,
)
lightning_module = LitTransformer(model, learning_rate=0.0001)
trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(lightning_module, dataloadr)
