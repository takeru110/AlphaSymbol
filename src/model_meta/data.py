import random
from typing import Iterable

import lightning as pl
import pandas as pd
import torch
import yaml
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CustomDataset(Dataset):
    def __init__(
        self, seq_idx, tgt_idx, src_pad_idx, tgt_pad_idx, few_pad_batches=False
    ):
        """
        Args:
            seq_idx (list): List of input sequences.
            tgt_idx (list): List of target sequences.
        """
        self.seq_idx, self.tgt_idx = (
            self.align_and_shuffle(seq_idx, tgt_idx, src_pad_idx, tgt_pad_idx)
            if few_pad_batches
            else (
                seq_idx,
                tgt_idx,
            )
        )

    def align_and_shuffle(self, seq, tgt, src_pad_idx, tgt_pad_idx):
        # Combine source and target into a single list of tuples
        data = list(zip(seq, tgt))

        # Sort by length of the source and shuffle within groups of the same length
        sorted_data = sorted(
            data, key=lambda x: len(x[0])
        )  # Sort by length of source
        grouped = {}
        for item in sorted_data:
            length = len(item[0])
            grouped.setdefault(length, []).append(item)

        # Shuffle each group randomly
        shuffled_data = []
        for group in grouped.values():
            random.shuffle(group)
            shuffled_data.extend(group)

        # Split the sorted and shuffled data back into source and target
        sorted_source, sorted_target = zip(*shuffled_data)

        # Convert back to lists
        sorted_source = list(sorted_source)
        sorted_target = list(sorted_target)
        return sorted_source, sorted_target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.seq_idx)

    def __getitem__(self, idx):
        """
        Dynamically loads and converts a single sample to tensor.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        input_data = torch.tensor(self.seq_idx[idx], dtype=torch.int64)
        target_data = torch.tensor(self.tgt_idx[idx], dtype=torch.int64)
        return input_data, target_data


class PREDataModule(pl.LightningDataModule):
    """
    This model is the data module for model of Meta AI.
    """

    def __init__(
        self,
        csv_path,
        metadata_path,
        src_vocab_list,
        tgt_vocab_list,
        min_n_tokens_in_batch,
        num_workers=0,
        test_ratio=0.2,
        val_ratio=0.25,
    ):
        """
        Args:
        - csv_path (str): path to the CSV file
        - metadata_path (str): path to the metadata YAML file
        - src_vocab_list (list): list of source vocabulary tokens
        - tgt_vocab_list (list): list of target vocabulary tokens
        - min_n_tokens_in_batch (int): the minimum number of tokens in a batch
        - num_workers (int): the number of workers for data loading (used in DataLoader)
        - test_ratio (float): the ratio of (test + val) / (test + val + train)
        - val_ratio (float): the ratio of val / (val + test)
        """
        super().__init__()
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.num_workers = num_workers
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_n_tokens_in_batch = min_n_tokens_in_batch

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = yaml.safe_load(f)

        # Set dimensions from metadata
        self.point_vector_size = self.metadata["max_src_length"] + 2  # +2 for BOS, EOS
        self.tgt_input_size = self.metadata["max_tgt_length"] + 2  # +2 for BOS, EOS

        # Create vocabularies with special tokens
        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

        # Source vocabulary
        self.src_vocab = {}
        idx = 0
        for token in special_tokens:
            self.src_vocab[token] = idx
            idx += 1
        for token in src_vocab_list:
            if token not in self.src_vocab:
                self.src_vocab[token] = idx
                idx += 1

        # Target vocabulary
        self.tgt_vocab = {}
        idx = 0
        for token in special_tokens:
            self.tgt_vocab[token] = idx
            idx += 1
        for token in tgt_vocab_list:
            if token not in self.tgt_vocab:
                self.tgt_vocab[token] = idx
                idx += 1

        # Set special token indices
        self.src_pad_idx = self.src_vocab["[PAD]"]
        self.src_sos_idx = self.src_vocab["[BOS]"]
        self.src_eos_idx = self.src_vocab["[EOS]"]
        self.src_unk_idx = self.src_vocab["[UNK]"]
        self.src_token_num = len(self.src_vocab)

        self.tgt_pad_idx = self.tgt_vocab["[PAD]"]
        self.tgt_sos_idx = self.tgt_vocab["[BOS]"]
        self.tgt_eos_idx = self.tgt_vocab["[EOS]"]
        self.tgt_unk_idx = self.tgt_vocab["[UNK]"]

        # Load dataset using datasets library
        try:
            # First try to load with datasets library
            self.dataset = HFDataset.from_csv(csv_path, cache_dir=None)
        except (NotImplementedError, Exception) as e:
            print(f"Warning: datasets library failed ({e}), falling back to pandas + datasets conversion")
            # Fallback: load with pandas first, then convert to datasets
            df = pd.read_csv(csv_path)
            self.dataset = HFDataset.from_pandas(df)

        self.setup_attrs()

    def src_add_ends(self, point: list[int]) -> list[int]:
        return [self.src_sos_idx] + point + [self.src_eos_idx]

    def src_pad_points(self, point: list[int]) -> list[int]:
        return point + [self.src_pad_idx] * (
            self.point_vector_size - len(point)
        )

    def tgt_add_ends(self, list_char: list[str]) -> list[str]:
        """add <sos> and <eos> to the target token list
        Args:
            list_char (list[str]): list of characters
        """
        return ["[BOS]"] + list_char + ["[EOS]"]

    def tgt_pad(self, list_char: list[str]) -> list[str]:
        """pad the target token list
        Args:
            list_char (list[str]): list of characters
        """
        return list_char + ["[PAD]"] * (self.tgt_input_size - len(list_char))

    def process_src_example(self, example):
        """Process source data for a single example using datasets functionality"""
        input_data, output_data = eval(example["input"]), eval(example["output"])
        point_li = []
        for x, y in zip(input_data, output_data):
            point_li.append([*x, y])

        # Apply src transformations to each point
        processed_points = []
        for p in point_li:
            p_with_ends = self.src_add_ends(p)
            p_padded = self.src_pad_points(p_with_ends)
            processed_points.append(p_padded)

        # Transform values using src_vocab
        vocab_transformed = []
        for point in processed_points:
            transformed_point = []
            for val in point:
                str_val = str(val)
                if str_val in self.src_vocab:
                    transformed_point.append(self.src_vocab[str_val])
                else:
                    transformed_point.append(self.src_unk_idx)
            vocab_transformed.append(transformed_point)

        return {"processed_src": vocab_transformed}

    def process_tgt_example(self, example):
        """Process target data for a single example using datasets functionality"""
        target_chars = list(example["expr"])

        # Apply target transformations
        chars_with_ends = self.tgt_add_ends(target_chars)
        chars_padded = self.tgt_pad(chars_with_ends)

        # Transform using tgt_vocab
        tgt_indices = []
        for char in chars_padded:
            if char in self.tgt_vocab:
                tgt_indices.append(self.tgt_vocab[char])
            else:
                tgt_indices.append(self.tgt_unk_idx)

        return {"processed_tgt": tgt_indices}

    def setup_attrs(self, stage=None):
        # Process source data using datasets map functionality
        src_processed = self.dataset.map(
            self.process_src_example, desc="Processing source data"
        )

        # Process target data using datasets map functionality
        tgt_processed = self.dataset.map(
            self.process_tgt_example, desc="Processing target data"
        )

        # Combine processed data
        seq_idx = src_processed["processed_src"]
        tgt_idx = tgt_processed["processed_tgt"]

        # Set point_num from first sequence
        self.point_num = len(seq_idx[0])

        # Shuffle data
        data_list = list(zip(seq_idx, tgt_idx))
        random.shuffle(data_list)
        shuffled_seq_idx, shuffled_tgt_idx = zip(*data_list)

        # Split the data into training, validation, and test sets
        data_len = len(shuffled_seq_idx)
        train_val_devider_idx = int(data_len - self.test_ratio * data_len)

        self.train_data = CustomDataset(
            shuffled_seq_idx[:train_val_devider_idx],
            shuffled_tgt_idx[:train_val_devider_idx],
            self.src_pad_idx,
            self.tgt_pad_idx,
            few_pad_batches=True,
        )

        val_test_seq_idx = shuffled_seq_idx[train_val_devider_idx:]
        val_test_tgt_idx = shuffled_tgt_idx[train_val_devider_idx:]
        val_test_len = len(val_test_seq_idx)
        test_val_devider_idx = int(self.val_ratio * val_test_len)
        self.val_data = CustomDataset(
            val_test_seq_idx[:test_val_devider_idx],
            val_test_tgt_idx[:test_val_devider_idx],
            self.src_pad_idx,
            self.tgt_pad_idx,
        )
        self.test_data = CustomDataset(
            val_test_seq_idx[test_val_devider_idx:],
            val_test_tgt_idx[test_val_devider_idx:],
            self.src_pad_idx,
            self.tgt_pad_idx,
        )

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train_data,
            batch_sampler=self.batch_sampler_list(self.train_data),
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_pad_idx
            ),
        )
        return data_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_sampler=self.batch_sampler_list(self.val_data),
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_pad_idx
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_sampler=self.batch_sampler_list(self.test_data),
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(
                x, self.src_pad_idx, self.tgt_pad_idx
            ),
        )

    def batch_sampler_list(self, dataset):
        min_n_token = self.min_n_tokens_in_batch
        token_num_list = [len(ps) for ps, _ in list(dataset)]
        grouped_token_num_list = []
        indices_list = []

        current_group = []
        current_indices = []
        current_sum = 0

        for i, tokens in enumerate(token_num_list):
            if (
                current_sum + tokens > min_n_token and current_group
            ):  # If adding tokens exceeds min_n_token
                grouped_token_num_list.append(current_group)
                indices_list.append(current_indices)
                current_group = []
                current_indices = []
                current_sum = 0

            current_group.append(tokens)
            current_indices.append(i)
            current_sum += tokens

        # Add the last group if not empty
        if current_group:
            grouped_token_num_list[-1].extend(current_group)
            indices_list[-1].extend(current_indices)

        random.shuffle(indices_list)
        return indices_list


def collate_fn(batch, src_pad_id, tgt_pad_id):
    """
    Custom collate function to pad sequences with <pad> token.
    """
    src_max_points = max(len(x) for x, _ in batch)
    tgt_max_length = max(len(y) for _, y in batch)

    # Pad source sequences
    x_padded = []
    for x, _ in batch:
        if len(x) < src_max_points:
            # Pad with src_pad_id
            padding = torch.full((src_max_points - len(x), x.shape[1]), src_pad_id, dtype=torch.int64)
            x_padded.append(torch.cat([x, padding], dim=0))
        else:
            x_padded.append(x)
    
    # Pad target sequences
    y_padded = []
    for _, y in batch:
        if len(y) < tgt_max_length:
            # Pad with tgt_pad_id
            padding = torch.full((tgt_max_length - len(y),), tgt_pad_id, dtype=torch.int64)
            y_padded.append(torch.cat([y, padding]))
        else:
            y_padded.append(y)

    return torch.stack(x_padded).type(torch.int64), torch.stack(y_padded).type(torch.int64)


def test_data():
    """
    モデルの挙動を確認するテスト関数
    データの読み込み、前処理、バッチ作成の各ステップを検証する
    """
    print("=" * 60)
    print("PREDataModule テスト開始")
    print("=" * 60)

    # テスト用のパラメータ設定
    csv_path = "/home/takeruito/work/PrfSR/data/training/d3-a5-c3-r5.csv"
    metadata_path = "/home/takeruito/work/PrfSR/data/training/d3-a5-c3-r5_metadata.yaml"

    # メタデータを読み込んで語彙リストを取得
    print("1. メタデータの読み込み")
    try:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        src_vocab_list = metadata["src_vocab_list"]
        tgt_vocab_list = metadata["tgt_vocab_list"]

        print(f"   ✓ メタデータ読み込み成功")
        print(f"   - max_src_length: {metadata['max_src_length']}")
        print(f"   - max_tgt_length: {metadata['max_tgt_length']}")
        print(f"   - src_vocab_size: {len(src_vocab_list)}")
        print(f"   - tgt_vocab_size: {len(tgt_vocab_list)}")
        print(f"   - src_vocab例: {src_vocab_list[:10]}")
        print(f"   - tgt_vocab例: {tgt_vocab_list[:10]}")

    except Exception as e:
        print(f"   ✗ メタデータ読み込みエラー: {e}")
        return

    print("\n" + "-" * 60)
    print("2. PREDataModuleの初期化")
    try:
        data_module = PREDataModule(
            csv_path=csv_path,
            metadata_path=metadata_path,
            src_vocab_list=src_vocab_list,
            tgt_vocab_list=tgt_vocab_list,
            min_n_tokens_in_batch=100,
            num_workers=0,
            test_ratio=0.2,
            val_ratio=0.25,
        )
        print(f"   ✓ PREDataModule初期化成功")
        print(f"   - データセット総数: {len(data_module.dataset)}")
        print(f"   - point_vector_size: {data_module.point_vector_size}")
        print(f"   - tgt_input_size: {data_module.tgt_input_size}")
        print(f"   - src_token_num: {data_module.src_token_num}")
        print(f"   - src_vocab例: {list(data_module.src_vocab.items())[:10]}")
        print(f"   - tgt_vocab例: {list(data_module.tgt_vocab.items())[:10]}")

    except Exception as e:
        print(f"   ✗ PREDataModule初期化エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "-" * 60)
    print("3. データセット分割の確認")
    try:
        print(f"   - 訓練データ数: {len(data_module.train_data)}")
        print(f"   - 検証データ数: {len(data_module.val_data)}")
        print(f"   - テストデータ数: {len(data_module.test_data)}")
        print(
            f"   - 合計: {len(data_module.train_data) + len(data_module.val_data) + len(data_module.test_data)}"
        )

    except Exception as e:
        print(f"   ✗ データセット分割確認エラー: {e}")
        return

    print("\n" + "-" * 60)
    print("4. 個別サンプルの確認")
    try:
        # 元のデータセットから最初のサンプルを確認
        original_sample = data_module.dataset[0]
        print(f"   元データ:")
        print(f"   - expr: {original_sample['expr']}")
        print(f"   - input: {original_sample['input']}")
        print(f"   - output: {original_sample['output']}")

        # 処理後のサンプルを確認
        test_id = 10
        train_sample = data_module.train_data[test_id]
        src_tensor, tgt_tensor = train_sample
        print(f"\n   処理後データ:")
        print(f"   - src shape: {src_tensor.shape}")
        print(f"   - tgt shape: {tgt_tensor.shape}")
        print(f"   - src データ例: {src_tensor[:10] if len(src_tensor) > 10 else src_tensor}")
        print(f"   - tgt データ例: {tgt_tensor[:20]}")

        # 語彙変換の確認
        print(f"\n   語彙変換確認:")
        print(f"   - src特殊トークン: PAD={data_module.src_pad_idx}, BOS={data_module.src_sos_idx}, EOS={data_module.src_eos_idx}")
        print(f"   - tgt特殊トークン: PAD={data_module.tgt_pad_idx}, BOS={data_module.tgt_sos_idx}, EOS={data_module.tgt_eos_idx}")

    except Exception as e:
        print(f"   ✗ 個別サンプル確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "-" * 60)
    print("5. DataLoaderの動作確認")
    try:
        # 訓練用DataLoaderのテスト
        train_loader = data_module.train_dataloader()
        print(f"   ✓ 訓練DataLoader作成成功")

        # 最初のバッチを取得
        batch_iter = iter(train_loader)
        batch_src, batch_tgt = next(batch_iter)

        print(f"   バッチ情報:")
        print(f"   - batch_src shape: {batch_src.shape}")
        print(f"   - batch_tgt shape: {batch_tgt.shape}")
        print(f"   - batch_src dtype: {batch_src.dtype}")
        print(f"   - batch_tgt dtype: {batch_tgt.dtype}")
        print(f"   - バッチサイズ: {batch_src.shape[0]}")

        # パディングの確認
        pad_count_src = (batch_src == data_module.src_pad_idx).sum().item()
        pad_count_tgt = (batch_tgt == data_module.tgt_pad_idx).sum().item()
        print(f"   - srcパディング数: {pad_count_src}")
        print(f"   - tgtパディング数: {pad_count_tgt}")

    except Exception as e:
        print(f"   ✗ DataLoader動作確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "-" * 60)
    print("6. 各DataLoaderの動作確認")
    try:
        # 各DataLoaderをテスト
        loaders = {
            "train": data_module.train_dataloader(),
            "val": data_module.val_dataloader(),
            "test": data_module.test_dataloader()
        }

        for name, loader in loaders.items():
            batch_src, batch_tgt = next(iter(loader))
            print(f"   {name} loader:")
            print(f"   - バッチ数: {len(loader)}")
            print(f"   - shape: src{batch_src.shape}, tgt{batch_tgt.shape}")

    except Exception as e:
        print(f"   ✗ 各DataLoader確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "-" * 60)
    print("7. 語彙復元テスト")
    try:
        # 語彙の逆引き辞書を作成
        src_id2token = {v: k for k, v in data_module.src_vocab.items()}
        tgt_id2token = {v: k for k, v in data_module.tgt_vocab.items()}

        # サンプルの復元
        sample_tgt = batch_tgt[0][:20]  # 最初の20トークン
        decoded_tgt = [tgt_id2token.get(idx.item(), "[UNK]") for idx in sample_tgt]
        print(f"   tgtトークン復元例:")
        print(f"   - 元ID: {sample_tgt.tolist()}")
        print(f"   - 復元: {decoded_tgt}")

    except Exception as e:
        print(f"   ✗ 語彙復元テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✓ 全てのテストが完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    # テスト実行
    test_data()
