import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset


class CustomTokenizer:
    """カスタムトークナイザークラス"""

    def __init__(self, vocab: List[str], padding_token: str = "[PAD]"):
        """
        Args:
            vocab: 語彙リスト
            padding_token: パディングトークン
        """
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
        }

        # 語彙を追加（特別トークンの後から）
        self.vocab = {}
        self.vocab.update(self.special_tokens)

        start_id = len(self.special_tokens)
        for i, token in enumerate(sorted(set(vocab))):
            if token not in self.vocab:
                self.vocab[token] = start_id + i

        # 逆マッピング
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        self.padding_token = padding_token
        self.padding_id = self.vocab.get(padding_token, self.vocab["[PAD]"])

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: str = None,
        max_length: int = None,
        truncation: bool = False,
        return_tensors=None,
    ) -> List[int]:
        """テキストをトークンIDに変換"""
        # 文字レベルでトークン化
        tokens = list(text.replace(" ", ""))  # スペースを除去して文字レベル分割

        # 数字の連続を一つのトークンとして扱う
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i].isdigit():
                # 数字の連続を収集
                num_str = ""
                while i < len(tokens) and tokens[i].isdigit():
                    num_str += tokens[i]
                    i += 1
                merged_tokens.append(num_str)
            else:
                merged_tokens.append(tokens[i])
                i += 1

        # トークンIDに変換
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab["[BOS]"])

        for token in merged_tokens:
            token_ids.append(self.vocab.get(token, self.vocab["[UNK]"]))

        if add_special_tokens:
            token_ids.append(self.vocab["[EOS]"])

        # パディングまたはトランケート
        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif padding == "max_length":
                while len(token_ids) < max_length:
                    token_ids.append(self.padding_id)

        return token_ids

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        """トークンIDをテキストに変換"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "[UNK]")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        """トークンをIDに変換"""
        return self.vocab.get(token, self.vocab["[UNK]"])


def extract_src_vocab_from_csv(csv_path: str) -> Set[str]:
    """CSVファイルからsrc（input/output）の語彙を抽出"""
    vocab = set()

    # CSVファイルをチャンクで読み込み
    try:
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                # inputとoutputの文字列から語彙を抽出
                if "input" in row and pd.notna(row["input"]):
                    input_str = str(row["input"])
                    # 数字、括弧、カンマ、スペースを抽出
                    tokens = re.findall(r"\d+|[\[\],\(\)\s]", input_str)
                    vocab.update(tokens)

                if "output" in row and pd.notna(row["output"]):
                    output_str = str(row["output"])
                    tokens = re.findall(r"\d+|[\[\],\(\)\s]", output_str)
                    vocab.update(tokens)
    except Exception:
        # チャンク読み込みが失敗した場合は通常の読み込み
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if "input" in row and pd.notna(row["input"]):
                input_str = str(row["input"])
                tokens = re.findall(r"\d+|[\[\],\(\)\s]", input_str)
                vocab.update(tokens)

            if "output" in row and pd.notna(row["output"]):
                output_str = str(row["output"])
                tokens = re.findall(r"\d+|[\[\],\(\)\s]", output_str)
                vocab.update(tokens)

    # 空文字列を除去
    vocab.discard("")
    return vocab


def extract_tgt_vocab_from_csv(csv_path: str) -> Set[str]:
    """CSVファイルからtgt（expr）の語彙を抽出"""
    vocab = set()

    # CSVファイルをチャンクで読み込み
    try:
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                if "expr" in row and pd.notna(row["expr"]):
                    expr_str = str(row["expr"])
                    # 関数名（Z,S,P,C,R）、数字、括弧、カンマ、スペースを抽出
                    tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str)
                    vocab.update(tokens)
    except Exception:
        # チャンク読み込みが失敗した場合は通常の読み込み
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if "expr" in row and pd.notna(row["expr"]):
                expr_str = str(row["expr"])
                tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str)
                vocab.update(tokens)

    # 空文字列を除去
    vocab.discard("")
    return vocab


class PRFDataset(Dataset):
    """
    Dataset for Primitive Recursive Function Symbolic Regression

    機能要件:
    - csvを読み込む
    - モデルへのsrcを作成：csvファイルの各サンプルにおいて"input"(d×nのarray), "output"(nのarray)を
      int array配列を組み合わせて、(d+1)×nのnp.array of type intを作成
    - モデルへのtgtを作成：csvファイルの"expr"をtokenizeしてpaddingし、
      max_tgt_length × n_token_kinds のarrayを作成。
      ただし、max_tgt_lengthに足りない長さはpadding_idで埋める
    """

    def __init__(
        self,
        csv_path: str,
        max_tgt_length: Optional[int] = None,
        max_src_points: Optional[int] = None,
        padding_token: str = "[PAD]",
    ):
        """
        Args:
            csv_path: CSVファイルのパス
            max_tgt_length: 最大ターゲット長（Noneの場合は自動計算）
            max_src_points: 最大ソース点数（Noneの場合は自動計算）
            padding_token: パディングトークン
        """
        # datasetsを使用してCSVを読み込む（大きなファイルにも対応）
        # pandasで読み込んでからdatasetsに変換する方法を使用
        from datasets import Dataset

        # まずpandasでCSVを読み込み（チャンク処理で大きなファイルにも対応）
        try:
            # 大きなファイルの場合はチャンク読み込みを試行
            df_chunks = []
            chunk_size = 10000  # 1万行ずつ処理

            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                df_chunks.append(chunk)

            # 全チャンクを結合
            df = pd.concat(df_chunks, ignore_index=True)

        except Exception:
            # チャンク読み込みが失敗した場合は通常の読み込み
            df = pd.read_csv(csv_path)

        # pandasデータフレームをdatasetsオブジェクトに変換
        self.dataset = Dataset.from_pandas(df)

        # 必要なカラムの存在確認
        required_columns = ["expr"]
        for col in required_columns:
            if col not in self.dataset.column_names:
                raise ValueError(f"CSV file must contain column: {col}")

        # CSVファイルから語彙を抽出
        print("Extracting vocabularies from CSV...")
        src_vocab_set = extract_src_vocab_from_csv(csv_path)
        tgt_vocab_set = extract_tgt_vocab_from_csv(csv_path)

        print(f"Source vocabulary size: {len(src_vocab_set)}")
        print(f"Target vocabulary size: {len(tgt_vocab_set)}")
        print(f"Source vocab sample: {sorted(list(src_vocab_set))[:10]}")
        print(f"Target vocab sample: {sorted(list(tgt_vocab_set))[:10]}")

        # 分離されたtokenizerの初期化
        self.src_tokenizer = CustomTokenizer(list(src_vocab_set), padding_token)
        self.tgt_tokenizer = CustomTokenizer(list(tgt_vocab_set), padding_token)

        self.padding_token = padding_token
        self.src_padding_id = self.src_tokenizer.padding_id
        self.tgt_padding_id = self.tgt_tokenizer.padding_id

        # 最大ターゲット長の計算
        if max_tgt_length is None:
            # 全てのexprをtokenizeして最大長を計算
            expr_lengths = []
            for example in self.dataset:
                tokens = self.tgt_tokenizer.encode(
                    str(example["expr"]),
                    add_special_tokens=True,
                    return_tensors=None,
                )
                expr_lengths.append(len(tokens))
            self.max_tgt_length = max(expr_lengths) if expr_lengths else 128
        else:
            self.max_tgt_length = max_tgt_length

        # 最大ソース点数の計算
        if max_src_points is None:
            max_points = 0
            for example in self.dataset:
                try:
                    if "input" in example and pd.notna(example["input"]):
                        input_data = (
                            eval(example["input"])
                            if isinstance(example["input"], str)
                            else example["input"]
                        )
                        max_points = max(max_points, len(input_data))
                except (ValueError, SyntaxError, TypeError):
                    continue
            self.max_src_points = max_points if max_points > 0 else 20
        else:
            self.max_src_points = max_src_points

        # 語彙サイズとエイリアスの設定
        self.src_vocab_size = len(self.src_tokenizer.vocab)
        self.tgt_vocab_size = len(self.tgt_tokenizer.vocab)

        # TransformerDatasetとの互換性のためのエイリアス
        self.src_max_len = self.max_src_points
        self.tgt_max_len = self.max_tgt_length

        # データの前処理
        self._preprocess_data()

    def _preprocess_data(self):
        """データの前処理を行う"""
        self.processed_data = []

        for idx, example in enumerate(self.dataset):
            try:
                # srcの作成：input/outputがある場合はそれを使用、ない場合はダミーデータ
                if (
                    "input" in example
                    and "output" in example
                    and pd.notna(example["input"])
                    and pd.notna(example["output"])
                ):
                    # inputとoutputの解析
                    input_data = (
                        eval(example["input"])
                        if isinstance(example["input"], str)
                        else example["input"]
                    )
                    output_data = (
                        eval(example["output"])
                        if isinstance(example["output"], str)
                        else example["output"]
                    )
                    src = self._create_src_array(input_data, output_data)
                else:
                    # ダミーのsrcデータを作成（数値の配列として）
                    src = np.zeros((2, self.max_src_points), dtype=int)

                # tgtの作成：tokenizeしてpadding
                tgt = self._create_tgt_array(example["expr"])

                self.processed_data.append(
                    {"src": src, "tgt": tgt, "original_expr": example["expr"]}
                )

            except Exception as e:
                print(f"Warning: Skipping row {idx} due to error: {e}")
                continue

    def _create_src_array(
        self, input_data: List[List[int]], output_data: List[int]
    ) -> np.ndarray:
        """
        input_data: d×nのリスト（d次元のn個の入力点）
        output_data: nのリスト（n個の出力値）

        Returns:
            (d+1)×max_src_pointsのnp.array of type int（パディング済み）
        """
        # input_dataをnumpy arrayに変換
        if len(input_data) == 0:
            raise ValueError("Input data is empty")

        # input_dataがリストのリストの場合の処理
        if isinstance(input_data[0], (list, tuple)):
            # input_data: [(x1_1, x1_2, ...), (x2_1, x2_2, ...), ...] 形式
            # これをd×n形式に変換
            input_array = np.array(input_data).T  # 転置してd×nにする
        else:
            # input_data: [x1, x2, x3, ...] 形式（1次元の場合）
            input_array = np.array(input_data).reshape(1, -1)

        # output_dataをnumpy arrayに変換
        output_array = np.array(output_data).reshape(1, -1)

        # inputとoutputを結合して(d+1)×nの配列を作成
        src_array = np.vstack([input_array, output_array]).astype(int)

        # パディングまたはトランケート
        current_points = src_array.shape[1]
        if current_points > self.max_src_points:
            # 点数が多すぎる場合はトランケート
            src_array = src_array[:, : self.max_src_points]
        elif current_points < self.max_src_points:
            # 点数が少ない場合はパディング（0で埋める）
            padding_width = ((0, 0), (0, self.max_src_points - current_points))
            src_array = np.pad(
                src_array, padding_width, mode="constant", constant_values=0
            )

        return src_array

    def _create_tgt_array(self, expr: str) -> np.ndarray:
        """
        exprをtokenizeしてpaddingし、max_tgt_length のarrayを作成

        Args:
            expr: 式の文字列

        Returns:
            tokenizeされたarray
        """
        # 式をtokenize（special tokensを追加）
        token_ids = self.tgt_tokenizer.encode(
            str(expr),
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_tgt_length,
            truncation=True,
            return_tensors=None,
        )

        return np.array(token_ids, dtype=int)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                - 'src': (d+1)×max_src_points numpy array of integers
                - 'tgt': max_tgt_length numpy array of token IDs
                - 'original_expr': original expression string
        """
        return self.processed_data[idx]

    def get_src_vocab_size(self) -> int:
        """src語彙サイズを返す"""
        return len(self.src_tokenizer.vocab)

    def get_tgt_vocab_size(self) -> int:
        """tgt語彙サイズを返す"""
        return len(self.tgt_tokenizer.vocab)

    def get_src_tokenizer(self) -> CustomTokenizer:
        """src tokenizerを返す"""
        return self.src_tokenizer

    def get_tgt_tokenizer(self) -> CustomTokenizer:
        """tgt tokenizerを返す"""
        return self.tgt_tokenizer

    def get_config(self) -> Dict[str, Any]:
        """設定情報を返す"""
        return {
            "src_vocab_size": self.get_src_vocab_size(),
            "tgt_vocab_size": self.get_tgt_vocab_size(),
            "max_tgt_length": self.max_tgt_length,
            "max_src_points": self.max_src_points,
            "src_max_len": self.src_max_len,
            "tgt_max_len": self.tgt_max_len,
            "src_padding_id": self.src_padding_id,
            "tgt_padding_id": self.tgt_padding_id,
            "pad_token": self.padding_token,
        }


def load_prf_dataset(csv_path: str, **kwargs) -> PRFDataset:
    """
    datasetsライブラリのload_datasetスタイルでデータセットを読み込む関数

    Args:
        csv_path: CSVファイルのパス
        **kwargs: PRFDatasetの追加引数

    Returns:
        PRFDataset instance
    """
    return PRFDataset(csv_path, **kwargs)


# 使用例とテスト用の関数
def test_dataset(csv_path: str):
    """データセットのテスト用関数"""
    try:
        dataset = load_prf_dataset(csv_path)
        print("Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Config: {dataset.get_config()}")

        if len(dataset) > 0:
            test_id = 3
            sample = dataset[test_id]
            print(
                f"Sample shape - src: {sample['src'].shape}, tgt: {sample['tgt'].shape}"
            )
            print(f"Sample src:\n{sample['src']}")
            print(
                f"Sample tgt: {sample['tgt'][:20]}..."
            )  # 最初の20個のトークンのみ表示
            print(f"Original expr: {sample['original_expr']}")

            # トークンのデコードもテスト
            decoded = dataset.tgt_tokenizer.decode(
                sample["tgt"], skip_special_tokens=True
            )
            print(f"Decoded tgt: {decoded}")

    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # テスト実行用
    import sys

    if len(sys.argv) > 1:
        test_dataset(sys.argv[1])
    else:
        print("Usage: python dataset.py <csv_path>")
