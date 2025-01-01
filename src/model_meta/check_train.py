import lightning as pl
from train import LitTransformer

from data import PREDataModule

if __name__ == "__main__":
    csv_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
    data_module = PREDataModule(data_path=csv_path, batch_size=4, max_value=100)
    model = LitTransformer(
        token_num=data_module.token_num,
        token_embed_dim=64,
        max_input_dim=5,
        input_padding_idx=data_module.pad_idx,
        emb_expansion_factor=2,
    )
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
    trainer.fit(model, data_module)
