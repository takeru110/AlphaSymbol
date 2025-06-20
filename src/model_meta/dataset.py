import ast
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import lightning as pl
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm


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


class LengthAwareTokenBatchSampler(Sampler):
    """
    同じ長さのサンプルをグループ化し、かつトークン数制約を満たすサンプラー
    1. 同じ長さのサンプルを優先的にバッチに含める
    2. 指定したトークン数以上になるまでサンプルを追加
    3. 最大バッチサイズの制限も考慮
    """

    def __init__(
        self,
        dataset: Dataset,
        min_tokens_per_batch: int,
        max_batch_size: int = 32,
        shuffle: bool = True,
        metadata_path: Optional[str] = None,
    ):
        self.dataset = dataset
        self.min_tokens_per_batch = min_tokens_per_batch
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.metadata_path = metadata_path

        self._create_length_groups()

    def _create_length_groups(self):
        """ソース長さに基づいてサンプルをグループ化"""
        print("Creating length-based groups...")

        # metadata.yamlからpoint_num_distを読み込む
        if self.metadata_path and Path(self.metadata_path).exists():
            print(f"Loading point_num_dist from: {self.metadata_path}")
            with open(self.metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
                point_num_dist = metadata.get("point_num_dist", {})
            print(f"finished loading point_num_dist from: {self.metadata_path}")

            original_to_subset = {}
            for subset_idx, orig_idx in tqdm(
                enumerate(self.dataset.indices),
                desc="Creating orig to subset idx mapping",
            ):
                original_to_subset[orig_idx] = subset_idx

            self.tokens_groups = defaultdict(list)
            for tokens, original_indices in tqdm(
                point_num_dist.items(),
                desc="Mapping point_num_dist to subset indices",
            ):
                for orig_idx in original_indices:
                    if orig_idx not in original_to_subset:
                        continue
                    self.tokens_groups[tokens].append(
                        original_to_subset[orig_idx]
                    )

            print(f"Mapped to {len(self.tokens_groups)} groups for Subset:")
            for tokens, indices in list(self.tokens_groups.items())[:5]:
                print(
                    f"  Length {tokens}: {len(indices)} samples (subset indices: {indices[:10]}{'...' if len(indices) > 10 else ''})"
                )

        else:
            print("No metadata file provided, creating groups dynamically...")
            self._create_groups_dynamically()

    def _create_groups_dynamically(self):
        """動的にグループを作成（メタデータファイルがない場合のフォールバック）"""

        print(f"Created {len(self.tokens_groups)} groups dynamically:")
        for length, indices in list(self.tokens_groups.items())[:5]:
            print(f"  Length {length}: {len(indices)} samples")

    def __iter__(self):
        """長さベースのグループ化とトークン数制約を組み合わせたバッチ生成"""

        # 各グループ内でサンプルをシャッフル
        for _, samples in self.tokens_groups.items():
            random.shuffle(samples)
        self.token_groups = sorted(
            self.tokens_groups.items(), key=lambda x: x[0]
        )

        # batchのsourceがself.min_token_per_batch以下のトークン数になるように
        batch = []
        for tokens, indices in self.token_groups:
            for idx in indices:
                if (
                    len(batch) >= self.max_batch_size
                    or (len(batch) + 1) * tokens > self.min_tokens_per_batch
                ):
                    yield batch
                    batch = []
                batch.append(idx)
        if batch:
            yield batch

    def __len__(self):
        # 概算のバッチ数
        total_tokens = sum(
            key * len(value) for key, value in self.token_groups.items()
        )
        estimated_batches = max(1, total_tokens // self.min_tokens_per_batch)
        return estimated_batches


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
        # バッチング戦略の選択
        batching_strategy: str = "default",  # "default", "length_aware_token"
        min_tokens_per_batch: Optional[int] = None,
        max_batch_size: int = 32,
        metadata_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        self.collate_fn = collate_fn

        # バッチング戦略の設定
        self.batching_strategy = batching_strategy
        self.min_tokens_per_batch = min_tokens_per_batch
        self.max_batch_size = max_batch_size
        self.metadata_path = metadata_path

        # バリデーション
        if (
            batching_strategy == "length_aware_token"
            and min_tokens_per_batch is None
        ):
            raise ValueError(
                "min_tokens_per_batch must be specified when using length_aware_token strategy"
            )

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
        if self.batching_strategy == "length_aware_token":
            batch_sampler = LengthAwareTokenBatchSampler(
                dataset=self.train_dataset,
                min_tokens_per_batch=self.min_tokens_per_batch,
                max_batch_size=self.max_batch_size,
                shuffle=True,
                metadata_path=self.metadata_path,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
        else:
            # デフォルトの固定バッチサイズ
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


def analyze_batch_variance(dataloader, num_batches=5):
    """
    バッチ内のソース長さの分散を分析する関数
    """
    variances = []
    batch_info = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        source_lengths = [src.size(0) for src in batch["source"]]

        # 分散の計算（単一値の場合は0とする）
        if len(source_lengths) > 1:
            variance = torch.var(
                torch.tensor(source_lengths, dtype=torch.float)
            ).item()
        else:
            variance = 0.0

        variances.append(variance)

        batch_info.append(
            {
                "batch_id": i + 1,
                "batch_size": len(source_lengths),
                "lengths": source_lengths,
                "variance": variance,
                "min_length": min(source_lengths),
                "max_length": max(source_lengths),
                "mean_length": sum(source_lengths) / len(source_lengths),
            }
        )

    # NaNを除外して平均を計算
    valid_variances = [v for v in variances if not (v != v)]  # NaNでない値のみ
    avg_variance = (
        sum(valid_variances) / len(valid_variances) if valid_variances else 0
    )

    return {"average_variance": avg_variance, "batch_details": batch_info}


# 使用例
if __name__ == "__main__":
    print("=== Testing Different Batching Strategies ===")

    data_path = "data/training/superfib_r1_dataset.csv"
    metadata_path = "data/training/superfib_r1_metadata.yaml"
    batch_size = 2
    # 1. デフォルト（固定バッチサイズ）
    print("\n1. Default Fixed Batch Size")
    data_module_default = CSVDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=0,  # テスト用に0に設定
        train_val_split=0.8,
        collate_fn=custom_collate_fn,
        batching_strategy="default",
    )

    data_module_default.setup("fit")
    train_loader_default = data_module_default.train_dataloader()

    print(f"Train samples: {len(data_module_default.train_dataset)}")

    # 2. 長さを考慮したトークン数ベースのバッチング
    print("\n2. Length-Aware Token Count-based Batching")
    data_module_token = CSVDataModule(
        data_path=data_path,
        batch_size=batch_size,  # 最大バッチサイズとして機能
        num_workers=13,
        train_val_split=0.8,
        collate_fn=custom_collate_fn,
        batching_strategy="length_aware_token",
        min_tokens_per_batch=10000,  # 最小トークン数
        max_batch_size=128,
        metadata_path=metadata_path,
    )

    data_module_token.setup("fit")
    print("===Train loader===")
    train_loader_token = data_module_token.train_dataloader()

    print("Batch info for length-aware token-based batching:")
    for i, batch in enumerate(train_loader_token):
        if i >= 10:
            break
        batch_size = len(batch["source"])
        source_lengths = [src.size(0) for src in batch["source"]]
        count_dict = dict(Counter(source_lengths))
        print(
            f"  Batch {i + 1}: size={batch_size}, total_tokens={sum(source_lengths)} "
            f"Histogram of source_lengths: {count_dict}",
        )

    print("===Val loader===")
    val_loader_token = data_module_token.val_dataloader()

    print("Batch info for length-aware token-based batching:")
    for i, batch in enumerate(val_loader_token):
        if i >= 10:
            break
        batch_size = len(batch["source"])
        source_lengths = [src.size(0) for src in batch["source"]]
        count_dict = dict(Counter(source_lengths))
        print(
            f"  Batch {i + 1}: size={batch_size}, total_tokens={sum(source_lengths)} "
            f"Histogram of source_lengths: {count_dict}",
        )
