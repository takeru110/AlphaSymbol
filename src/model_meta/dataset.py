import argparse
import ast
import math
import pickle
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
        max_batch_size: int = 32,
        min_tokens_per_batch: int = 1000,
    ):
        self.dataset = dataset
        self.min_tokens_per_batch = min_tokens_per_batch
        self.max_batch_size = max_batch_size
        """長さベースのグループ化とトークン数制約を組み合わせたバッチ生成"""
        self.batch_starts = [0]
        self.batch_lengths = []
        length_counter = 1
        for idx, sample in enumerate(self.dataset):
            if idx == 0:
                continue
            if (
                length_counter * len(sample["source"])
                >= self.min_tokens_per_batch
                or length_counter >= self.max_batch_size
            ):
                self.batch_starts.append(idx)
                self.batch_lengths.append(length_counter)
                length_counter = 1
            else:
                length_counter += 1
        self.batch_lengths.append(length_counter)

        # shuffle samely
        s_r = list(zip(self.batch_lengths, self.batch_starts))
        random.shuffle(s_r)
        self.batch_lengths, self.batch_starts = map(list, zip(*s_r))

    def __iter__(self):
        for idx, length in zip(self.batch_starts, self.batch_lengths):
            yield list(range(idx, idx + length))

    def __len__(self):
        return len(self.batch_starts)


class CSVDataModule(pl.LightningDataModule):
    """
    汎用的なCSVデータセット用のLightningDataModule
    source/targetカラムを持つCSVファイルに対応
    """

    def __init__(
        self,
        dataset,  # Optional[torch.utils.data.Dataset] used when csv_path is None
        batch_size: int = 32,  # used for default batching
        num_workers: int = 4,
        train_val_split: float = 0.9,
        seed: int = 42,
        collate_fn: Optional[callable] = None,
        # バッチング戦略の選択
        batching_strategy: str = "default",  # "default", "length_aware_token"
        min_tokens_per_batch: Optional[int] = None,
        max_batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        self.collate_fn = collate_fn

        # バッチング戦略の設定
        self.batching_strategy = batching_strategy
        self.min_tokens_per_batch = min_tokens_per_batch
        self.max_batch_size = max_batch_size

        # バリデーション
        if (
            batching_strategy == "length_aware_token"
            and min_tokens_per_batch is None
        ):
            raise ValueError(
                "min_tokens_per_batch must be specified when using length_aware_token strategy"
            )

        # データセットを保存する変数
        self.dataset = dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        データセットの準備（順序保持 + ランダム抽出）
        - ランダムにサンプルを選択
        - 選択されたインデックスをソートして順序を保持
        stage: 'fit', 'validate', 'test', 'predict'のいずれか
        """
        # 学習・検証用の分割
        if stage == "fit" or stage is None:
            if self.train_dataset is None or self.val_dataset is None:
                dataset_size = len(self.dataset)
                train_size = int(self.train_val_split * dataset_size)

                # ランダムインデックス生成（再現性のためシード固定）
                torch.manual_seed(self.seed)
                all_indices = torch.randperm(dataset_size).tolist()

                # ランダムに選択されたインデックスをソートして順序保持
                train_indices = sorted(all_indices[:train_size])
                val_indices = sorted(all_indices[train_size:])

                # データセット作成
                self.train_dataset = self.dataset.select(train_indices)
                self.val_dataset = self.dataset.select(val_indices)

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
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=False,
            )
        else:
            # デフォルトの固定バッチサイズ
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
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

    def save_pickle(self, file_path: str):
        """
        Save the CSVDataModule instance to a pickle file.

        Args:
            file_path: Path to save the pickle file
        """
        pickle_path = Path(file_path)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        print(f"CSVDataModule saved to {pickle_path}")

    @classmethod
    def load_pickle(cls, file_path: str) -> "CSVDataModule":
        """
        Load a CSVDataModule instance from a pickle file.

        Args:
            file_path: Path to the pickle file

        Returns:
            Loaded CSVDataModule instance
        """
        with open(file_path, "rb") as f:
            module = pickle.load(f)

        print(f"CSVDataModule loaded from {file_path}")
        return module
        print(f"Dataset saved as pickle: {file_path}")


def custom_collate_fn(batch):
    """
    可変長テンソルをパディングするcollate関数
    """
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    # sourceのパディング（2D -> 3D）
    max_seq_len = max(len(s) for s in sources)
    max_dim = max(len(s[0]) for s in sources)

    padded_sources = torch.zeros(
        len(sources), max_seq_len, max_dim, dtype=torch.long
    )
    for i, src in enumerate(sources):
        for point_id, row in enumerate(src):
            padded_sources[i, point_id, : len(row)] = torch.tensor(row)

    # targetのパディング（1D -> 2D）
    max_target_len = max(len(t) for t in targets)
    padded_targets = torch.zeros(len(targets), max_target_len, dtype=torch.long)
    for i, tgt in enumerate(targets):
        target_len = len(tgt)
        padded_targets[i, :target_len] = torch.tensor(tgt)

    return {"source": padded_sources, "target": padded_targets}


def add_source_length(sample):
    sample["source_length"] = len(sample["source"])
    return sample


# 使用例とテスト
if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description="CSVDataModule - Load CSV data or save as pickle"
    )
    parser.add_argument("--csv-path", type=str, help="Path to the CSV file")
    parser.add_argument(
        "--metadata-path", type=str, help="Path to the metadata file"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run comprehensive test"
    )

    args = parser.parse_args()

    if args.test:
        print("🚀 Running CSVDataModule comprehensive test...")

        # CSVパスの設定
        csv_path = args.csv_path or "data/training/superfib_r1_dataset.csv"

        if not Path(csv_path).exists():
            print(f"❌ CSV file not found: {csv_path}")
            print("Please specify a valid CSV file with --csv-path")
            exit(1)

        print(f"🔍 Testing CSV loading from: {csv_path}")

        # load_datasetを使用してCSVからデータセット作成
        print(f"Loading dataset from CSV: {csv_path}")
        raw_dataset = load_dataset("csv", data_files=csv_path, split="train")
        raw_datset = raw_dataset

        # データのフォーマット（文字列からPythonオブジェクトに変換）
        def formatter(sample):
            try:
                sample["source"] = eval(sample["source"])
                sample["target"] = eval(sample["target"])
            except (ValueError, SyntaxError):
                # エラーの場合はデフォルト値を設定
                sample["source"] = [[0]]
                sample["target"] = [0]
            return sample

        # フォーマット処理を適用
        formatted_dataset = raw_dataset.map(
            formatter, batched=False, num_proc=1, load_from_cache_file=False
        )
        # 長さ情報を追加

        print(f"Dataset loaded and formatted: {len(formatted_dataset)} samples")

        def analyze_batches(dataloader, strategy_name, max_batches=10):
            """バッチの詳細分析"""
            print(f"\n=== {strategy_name} ===")

            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                print(f"Batch {batch_idx + 1}:")
                for src, tgt in zip(batch["source"], batch["target"]):
                    print("src: \n", src)
                    print("tgt: ", tgt)

                print()

            print(f"Analyzed batches of the strategy: {strategy_name}")
            print()

        # 1. デフォルト戦略のテスト

        print("🔍 Testing Default Batching Strategy...")

        datamodule_default = CSVDataModule(
            dataset=formatted_dataset,
            batch_size=2,
            num_workers=0,
            train_val_split=0.9,
            seed=42,
            collate_fn=custom_collate_fn,
            batching_strategy="default",
        )

        datamodule_default.setup("fit")
        train_loader_default = datamodule_default.train_dataloader()

        analyze_batches(train_loader_default, "Default Strategy", max_batches=5)

        # 2. Length-aware token戦略のテスト
        formatted_dataset = formatted_dataset.map(add_source_length, num_proc=1)
        formatted_dataset = formatted_dataset.sort("source_length")
        print("\n🔍 Testing Length-Aware Token Strategy...")
        datamodule_token = CSVDataModule(
            dataset=formatted_dataset,
            batch_size=16,  # max_batch_size として使用
            num_workers=0,
            train_val_split=0.8,
            seed=42,
            collate_fn=custom_collate_fn,
            batching_strategy="length_aware_token",
            min_tokens_per_batch=7,
            max_batch_size=16,
        )

        datamodule_token.setup("fit")
        train_loader_token = datamodule_token.train_dataloader()

        analyze_batches(
            train_loader_token, "Length-Aware Token Strategy", max_batches=5
        )

        # 3. 検証用DataLoaderのテスト
        print("\n🔍 Testing Validation DataLoader...")
        val_loader = datamodule_default.val_dataloader()
        analyze_batches(val_loader, "Validation DataLoader", max_batches=3)

        print("✅ Comprehensive test completed successfully!")

    else:
        print("Use --test flag to run comprehensive tests")
        print("Example: python dataset.py --test")
