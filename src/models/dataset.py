import logging
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


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
        
        # デバッグ情報
        print(f"Tokenizer initialized with {len(self.vocab)} total tokens")
        print(f"Max token ID: {max(self.vocab.values())}")

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
            token_id = self.vocab.get(token, self.vocab["[UNK]"])
            # デバッグ: 範囲外チェック
            if token_id >= len(self.vocab):
                logging.warning(f"Token ID {token_id} for token '{token}' exceeds vocab size {len(self.vocab)}")
                token_id = self.vocab["[UNK]"]
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.vocab["[EOS]"])

        # パディングまたはトランケート
        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif padding == "max_length":
                while len(token_ids) < max_length:
                    token_ids.append(self.padding_id)

        # 最終チェック: すべてのトークンIDが範囲内にあることを確認
        max_valid_id = len(self.vocab) - 1
        token_ids = [min(tid, max_valid_id) for tid in token_ids]

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

    def to_one_hot(self, token_ids: List[int]) -> np.ndarray:
        """
        トークンIDのリストをone-hotエンコーディングに変換
        
        Args:
            token_ids: トークンIDのリスト
            
        Returns:
            one-hot エンコードされた配列 (len(token_ids), vocab_size)
        """
        vocab_size = len(self.vocab)
        one_hot = np.zeros((len(token_ids), vocab_size), dtype=np.float32)
        
        for i, token_id in enumerate(token_ids):
            if 0 <= token_id < vocab_size:
                one_hot[i, token_id] = 1.0
        
        return one_hot
    
    def encode_to_one_hot(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: str = None,
        max_length: int = None,
        truncation: bool = False,
    ) -> np.ndarray:
        """
        テキストを直接one-hotエンコーディングに変換
        
        Args:
            text: 入力テキスト
            add_special_tokens: 特別トークンを追加するか
            padding: パディング方法
            max_length: 最大長
            truncation: トランケートするか
            
        Returns:
            one-hot エンコードされた配列 (sequence_length, vocab_size)
        """
        # まずトークンIDに変換
        token_ids = self.encode(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation
        )
        
        # one-hotに変換
        return self.to_one_hot(token_ids)
        

def extract_src_vocab_from_csv(csv_path: str) -> Set[str]:
    """CSVファイルからsrc（input/output）の語彙を抽出"""
    vocab = set()

    # CSVファイルをチャンクで読み込み
    try:
        chunk_size = 1000

        logging.info(f"Starting src vocabulary extraction from {csv_path}")
        logging.debug(f"Using chunk size: {chunk_size}")

        # まずファイルの総行数を取得（進捗表示のため）
        total_rows = sum(1 for _ in open(csv_path)) - 1  # ヘッダーを除く
        logging.info(f"Total rows to process: {total_rows}")

        # tqdmで進捗表示付きでチャンク処理
        with tqdm(
            total=total_rows, desc="Extracting src vocab", unit="rows"
        ) as pbar:
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

                    pbar.update(1)

                    # 語彙サイズの更新（1000行ごと）
                    if pbar.n % 1000 == 0:
                        pbar.set_postfix(vocab_size=len(vocab))

    except Exception as e:
        logging.warning(f"Chunk reading failed: {e}. Trying normal reading...")
        # チャンク読み込みが失敗した場合は通常の読み込み
        df = pd.read_csv(csv_path)

        logging.info(f"Fallback: Processing {len(df)} rows in single batch")

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Extracting src vocab (fallback)",
            unit="rows",
        ):
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

    logging.info(
        f"Src vocabulary extraction completed. Final vocab size: {len(vocab)}"
    )
    logging.debug(f"Sample src tokens: {sorted(list(vocab))[:20]}")

    return vocab


def extract_tgt_vocab_from_csv(csv_path: str) -> Set[str]:
    """CSVファイルからtgt（expr）の語彙を抽出"""
    vocab = set()

    # CSVファイルをチャンクで読み込み
    try:
        chunk_size = 1000

        logging.info(f"Starting tgt vocabulary extraction from {csv_path}")
        logging.debug(f"Using chunk size: {chunk_size}")

        # まずファイルの総行数を取得（進捗表示のため）
        total_rows = sum(1 for _ in open(csv_path)) - 1  # ヘッダーを除く

        # tqdmで進捗表示付きでチャンク処理
        with tqdm(
            total=total_rows, desc="Extracting tgt vocab", unit="rows"
        ) as pbar:
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    if "expr" in row and pd.notna(row["expr"]):
                        expr_str = str(row["expr"])
                        # 関数名（Z,S,P,C,R）、数字、括弧、カンマ、スペースを抽出
                        tokens = re.findall(
                            r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str
                        )
                        vocab.update(tokens)

                    pbar.update(1)

                    # 語彙サイズの更新（1000行ごと）
                    if pbar.n % 1000 == 0:
                        pbar.set_postfix(vocab_size=len(vocab))

    except Exception as e:
        logging.warning(f"Chunk reading failed: {e}. Trying normal reading...")
        # チャンク読み込みが失敗した場合は通常の読み込み
        df = pd.read_csv(csv_path)

        logging.info(f"Fallback: Processing {len(df)} rows in single batch")

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Extracting tgt vocab (fallback)",
            unit="rows",
        ):
            if "expr" in row and pd.notna(row["expr"]):
                expr_str = str(row["expr"])
                tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str)
                vocab.update(tokens)

    # 空文字列を除去
    vocab.discard("")

    logging.info(
        f"Tgt vocabulary extraction completed. Final vocab size: {len(vocab)}"
    )
    logging.debug(f"Sample tgt tokens: {sorted(list(vocab))[:20]}")

    return vocab


class PRFDataset(Dataset):
    """
    Dataset for Primitive Recursive Function Symbolic Regression

    機能要件:
    - csvを読み込む
    - メタデータパラメータから初期化
    - モデルへのsrcを作成：csvファイルの各サンプルにおいて"input"(d×nのarray), "output"(nのarray)を
      int array配列を組み合わせて、(d+1)×nのnp.array of type intを作成
    - モデルへのtgtを作成：csvファイルの"expr"をtokenizeしてpaddingし、
      max_tgt_length × n_token_kinds のarrayを作成。
      ただし、max_tgt_lengthに足りない長さはpadding_idで埋める
    """

    def __init__(
        self,
        csv_path: str,
        max_tgt_length: int,
        max_src_points: int,
        src_vocab_list: List[str],
        tgt_vocab_list: List[str],
        padding_token: str = "[PAD]",
    ):
        """
        Args:
            csv_path: CSVファイルのパス
            max_tgt_length: 最大ターゲット長
            max_src_points: 最大ソース点数
            src_vocab_list: ソース語彙リスト
            tgt_vocab_list: ターゲット語彙リスト
            padding_token: パディングトークン
        """
        # パラメータを設定
        self.max_tgt_length = max_tgt_length
        self.max_src_points = max_src_points

        logging.info("Initializing PRFDataset with parameters:")
        logging.info(f"  - max_tgt_length: {self.max_tgt_length}")
        logging.info(f"  - max_src_points: {self.max_src_points}")
        logging.info(f"  - src_vocab_size: {len(src_vocab_list)}")
        logging.info(f"  - tgt_vocab_size: {len(tgt_vocab_list)}")

        # CSVファイルを読み込み
        try:
            # 大きなファイルの場合はチャンク読み込みを試行
            df_chunks = []
            chunk_size = 10000  # 1万行ずつ処理

            logging.debug(
                f"Loading CSV file: {csv_path} in chunks of {chunk_size} rows"
            )
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                df_chunks.append(chunk)

            logging.debug(
                f"Finished loading CSV file: {csv_path} in chunks of {chunk_size} rows"
            )
            # 全チャンクを結合
            df = pd.concat(df_chunks, ignore_index=True)

        except Exception:
            # チャンク読み込みが失敗した場合は通常の読み込み
            df = pd.read_csv(csv_path)

        from datasets import Dataset

        # pandasデータフレームをdatasetsオブジェクトに変換
        self.dataset = Dataset.from_pandas(df)
        logging.debug(f"Loaded dataset with {len(self.dataset)} samples")

        # 必要なカラムの存在確認
        required_columns = ["expr"]
        for col in required_columns:
            if col not in self.dataset.column_names:
                raise ValueError(f"CSV file must contain column: {col}")

        # 語彙でtokenizerを初期化
        print("Initializing tokenizers from vocabulary...")
        print(f"Source vocabulary size: {len(src_vocab_list)}")
        print(f"Target vocabulary size: {len(tgt_vocab_list)}")
        print(f"Source vocab sample: {sorted(src_vocab_list)[:10]}")
        print(f"Target vocab sample: {sorted(tgt_vocab_list)[:10]}")

        # 分離されたtokenizerの初期化
        self.src_tokenizer = CustomTokenizer(src_vocab_list, padding_token)
        self.tgt_tokenizer = CustomTokenizer(tgt_vocab_list, padding_token)

        self.padding_token = padding_token
        self.src_padding_id = self.src_tokenizer.padding_id
        self.tgt_padding_id = self.tgt_tokenizer.padding_id

        # 語彙サイズとエイリアスの設定
        self.src_vocab_size = len(self.src_tokenizer.vocab)
        self.tgt_vocab_size = len(self.tgt_tokenizer.vocab)

        # TransformerDatasetとの互換性のためのエイリアス
        self.src_max_len = self.max_src_points
        self.tgt_max_len = self.max_tgt_length

    def _create_src_array(
        self, input_data: List[List[int]], output_data: List[int]
    ) -> np.ndarray:
        """
        input_data: d×nのリスト（d次元のn個の入力点）
        output_data: nのリスト（n個の出力値）

        Returns:
            tokenizeされたsrcのarray
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

        # 数値配列を文字列に変換してからトークン化
        src_str = str(src_array.tolist())
        
        # srcトークナイザーでエンコード
        src_tokens = self.src_tokenizer.encode(
            src_str,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_src_points,
            truncation=True,
            return_tensors=None,
        )

        return np.array(src_tokens, dtype=int)

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
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                - 'src': (d+1)×max_src_points numpy array of integers
                - 'tgt': max_tgt_length numpy array of token IDs
                - 'original_expr': original expression string
        """
        example = self.dataset[idx]

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

            return {"src": src, "tgt": tgt, "original_expr": example["expr"]}

        except Exception as e:
            logging.warning(f"Error processing sample {idx}: {e}")
            # エラーが発生した場合はダミーデータを返す
            src = np.zeros((2, self.max_src_points), dtype=int)
            tgt = np.zeros(self.max_tgt_length, dtype=int)
            return {
                "src": src,
                "tgt": tgt,
                "original_expr": str(example.get("expr", "")),
            }

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


def load_prf_dataset(csv_path: str, metadata_path: str, **kwargs) -> PRFDataset:
    """
    datasetsライブラリのload_datasetスタイルでデータセットを読み込む関数

    Args:
        csv_path: CSVファイルのパス
        metadata_path: metadata.yamlファイルのパス
        **kwargs: PRFDatasetの追加引数

    Returns:
        PRFDataset instance
    """
    # metadata.yamlファイルを読み込み
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        # メタデータから値を取得
        max_tgt_length = metadata["max_tgt_length"]
        max_src_points = metadata["max_src_points"]
        src_vocab_list = metadata["src_vocab_list"]
        tgt_vocab_list = metadata["tgt_vocab_list"]

        return PRFDataset(
            csv_path=csv_path,
            max_tgt_length=max_tgt_length,
            max_src_points=max_src_points,
            src_vocab_list=src_vocab_list,
            tgt_vocab_list=tgt_vocab_list,
            **kwargs,
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    except KeyError as e:
        raise KeyError(f"Required key missing in metadata file: {e}")
    except Exception as e:
        raise Exception(f"Error loading metadata file: {e}")


# 使用例とテスト用の関数
def test_dataset(csv_path: str, metadata_path: str):
    """データセットのテスト用関数"""
    try:
        # metadata.yamlを読み込んでパラメータを取得
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        print(f"Finished Loading metadata from: {metadata_path}")

        # PRFDatasetを直接パラメータで初期化
        dataset = PRFDataset(
            csv_path=csv_path,
            max_tgt_length=metadata["max_tgt_length"],
            max_src_points=metadata["max_src_points"],
            src_vocab_list=metadata["src_vocab_list"],
            tgt_vocab_list=metadata["tgt_vocab_list"],
        )

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

    if len(sys.argv) > 2:
        test_dataset(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python dataset.py <csv_path> <metadata_path>")
