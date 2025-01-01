import lightning as pl

from src.model_meta.data import PREDataModule
from src.model_meta.train import LitTransformer

if __name__ == "__main__":
    csv_path = "/home/takeru/AlphaSymbol/data/prfndim/d3-a2-c3-r3-status.csv"
    data_module = PREDataModule(data_path=csv_path, batch_size=4, max_value=100)
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
