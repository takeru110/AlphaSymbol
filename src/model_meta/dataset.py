import ast
from typing import Dict, Optional

import lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split


class TensorDataset(Dataset):
    """
    PyTorch DataLoaderで使用可能なDatasetクラス
    オンデマンドでtensorに変換
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            return {
                "source": torch.tensor(
                    ast.literal_eval(sample["source"]), dtype=torch.long
                ),
                "target": torch.tensor(
                    ast.literal_eval(sample["target"]), dtype=torch.long
                ),
            }
        except (ValueError, SyntaxError):
            return {
                "source": torch.tensor([[0]], dtype=torch.long),
                "target": torch.tensor([0], dtype=torch.long),
            }


class CSVDataModule(pl.LightningDataModule):
    """
    汎用的なCSVデータセット用のLightningDataModule
    source/targetカラムを持つCSVファイルに対応
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.9,
        seed: int = 42,
        collate_fn: Optional[callable] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        self.collate_fn = collate_fn

        # データセットを保存する変数
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        データセットの準備
        stage: 'fit', 'validate', 'test', 'predict'のいずれか
        """
        if self.dataset is None:
            print(f"Loading dataset from: {self.data_path}")

            # HuggingFace datasetsでCSVをロード
            raw_dataset = load_dataset(
                "csv",
                data_files={"raw": self.data_path},
                streaming=False,
                split="raw",
            )

            # TensorDatasetでラップ
            self.dataset = TensorDataset(raw_dataset)
            print(f"Dataset loaded: {len(self.dataset)} samples")

        # 学習・検証用の分割
        if stage == "fit" or stage is None:
            if self.train_dataset is None or self.val_dataset is None:
                train_size = int(self.train_val_split * len(self.dataset))
                val_size = len(self.dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

                print(f"Train dataset: {len(self.train_dataset)} samples")
                print(f"Validation dataset: {len(self.val_dataset)} samples")

        # テスト用（今回は検証セットと同じものを使用）
        if stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self.val_dataset

    def train_dataloader(self):
        """学習用DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """検証用DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """テスト用DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


def custom_collate_fn(batch):
    """
    可変長テンソルをパディングするcollate関数
    """
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    # sourceのパディング（2D -> 3D）
    max_seq_len = max(s.size(0) for s in sources)
    max_dim = max(s.size(1) for s in sources)

    padded_sources = torch.zeros(
        len(sources), max_seq_len, max_dim, dtype=torch.long
    )
    for i, src in enumerate(sources):
        seq_len, dim = src.shape
        padded_sources[i, :seq_len, :dim] = src

    # targetのパディング（1D -> 2D）
    max_target_len = max(t.size(0) for t in targets)
    padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
    for i, tgt in enumerate(targets):
        target_len = tgt.shape[0]
        padded_targets[i, :target_len] = tgt

    return {"source": padded_sources, "target": padded_targets}


# 使用例
if __name__ == "__main__":
    # データモジュールの作成
    data_module = CSVDataModule(
        data_path="data/training/mini_dataset.csv",
        batch_size=4,
        num_workers=4,
        train_val_split=0.9,
        collate_fn=custom_collate_fn,  # パディング付きcollate関数
    )

    # データの準備
    data_module.setup("fit")

    # DataLoaderの取得
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print("=== DataModule Test ===")
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")

    # 最初のバッチをテスト
    print("\n=== First Batch Test ===")
    for batch in train_loader:
        print(f"Source batch shape: {batch['source'].shape}")
        print(f"Target batch shape: {batch['target'].shape}")
        print(f"Source dtype: {batch['source'].dtype}")
        print(f"Target dtype: {batch['target'].dtype}")
        print(f"source: {batch['source']}")
        print(f"target: {batch['target']}")
        break

    print("\n=== DataModule ready for PyTorch Lightning! ===")

    # PyTorch Lightning Trainerでの使用例（コメント）
    """
    import pytorch_lightning as pl
    
    # 例: モデルとトレーナーの作成
    model = YourLightningModel()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    
    # 学習開始
    trainer.fit(model, data_module)
    """
