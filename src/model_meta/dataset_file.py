#!/usr/bin/env python3
"""
Dataset File Creator

raw.csvファイルから、モデル学習用の数値化されたdataset.csvを作成する。

機能:
- raw.csvのinputs, outputs, exprカラムからsource, targetカラムを生成
- metadata.yamlの語彙リストを使用してトークンを数値化
- sourceは2次元配列をパディングして数値化
- targetはexprをBOS/EOSトークン付きで数値化
"""

import argparse
import ast
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


def setup_logging(verbose: bool) -> None:
    """ログ設定を初期化"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_default_metadata_path(input_path: str) -> str:
    """入力ファイルパスからデフォルトのメタデータファイルパスを生成"""
    if input_path.endswith(".csv"):
        return input_path[:-4] + "_metadata.yaml"
    return input_path + "_metadata.yaml"


def create_default_output_path(input_path: str) -> str:
    """入力ファイルパスからデフォルトの出力ファイルパスを生成"""
    if input_path.endswith(".csv"):
        return input_path[:-4] + "_dataset.csv"
    return input_path + "_dataset.csv"


def print_arguments(
    input_path: str, metadata_path: str, output_path: str
) -> None:
    """コマンドライン引数の値を表示"""
    print("📋 Command line arguments:")
    print("=" * 50)
    print(f"📁 Input (-i):     {input_path}")
    print(f"📋 Metadata (-m):  {metadata_path}")
    print(f"💾 Output (-o):    {output_path}")
    print("=" * 50)


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """メタデータYAMLファイルを読み込み"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    logging.info(f"Loaded metadata from: {metadata_path}")
    logging.debug(f"Metadata keys: {list(metadata.keys())}")

    return metadata


def process_input_to_source(
    input_str: str,
    src_vocab_list: List[str],
    max_point_dim: int,
    pad_index: int,
) -> str:
    """inputsカラムの値をsourceカラムの値に変換"""
    try:
        # 文字列をPythonオブジェクトに変換
        input_data = ast.literal_eval(input_str)

        # 各点を語彙インデックスに変換
        indexed_points = []
        for point in input_data:
            indexed_point = []
            for value in point:
                str_value = str(value)
                if str_value in src_vocab_list:
                    indexed_point.append(src_vocab_list.index(str_value))
                else:
                    logging.warning(
                        f"Unknown token '{str_value}' not in src_vocab_list"
                    )
                    indexed_point.append(
                        pad_index
                    )  # 未知のトークンはPADで埋める

            # max_point_dimまでパディング
            while len(indexed_point) < max_point_dim:
                indexed_point.append(pad_index)

            # max_point_dimを超える場合は切り捨て
            indexed_point = indexed_point[:max_point_dim]

            indexed_points.append(indexed_point)

        # 結果を文字列として返す
        return str(indexed_points)

    except (ValueError, SyntaxError) as e:
        logging.error(f"Error parsing input: {input_str}, error: {e}")
        return str(
            [[pad_index] * max_point_dim]
        )  # エラー時はPADで埋めた単一点を返す


def process_expr_to_target(expr_str: str, tgt_vocab_list: List[str]) -> str:
    """exprカラムの値をtargetカラムの値に変換"""
    try:
        # 空白をstripする
        cleaned_expr = expr_str.strip()

        # 各文字をトークン化（関数名、数字、括弧、カンマを抽出）
        tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),]", cleaned_expr)

        # トークンを語彙インデックスに変換
        indexed_tokens = []

        # [BOS]を先頭に追加
        bos_index = tgt_vocab_list.index("[BOS]")
        indexed_tokens.append(bos_index)

        # 各トークンを変換
        for token in tokens:
            if token in tgt_vocab_list:
                indexed_tokens.append(tgt_vocab_list.index(token))
            else:
                logging.warning(
                    f"Unknown token '{token}' not in tgt_vocab_list"
                )
                # 未知のトークンは[PAD]で代替（エラー処理）
                pad_index = tgt_vocab_list.index("[PAD]")
                indexed_tokens.append(pad_index)

        # [EOS]を末尾に追加
        eos_index = tgt_vocab_list.index("[EOS]")
        indexed_tokens.append(eos_index)

        # 結果を文字列として返す（パディングは行わない）
        return str(indexed_tokens)

    except (ValueError, KeyError) as e:
        logging.error(f"Error processing expr: {expr_str}, error: {e}")
        # エラー時は[BOS][PAD][EOS]のみ返す
        try:
            bos_index = tgt_vocab_list.index("[BOS]")
            eos_index = tgt_vocab_list.index("[EOS]")
            pad_index = tgt_vocab_list.index("[PAD]")
            return str([bos_index, pad_index, eos_index])
        except ValueError:
            return str([0, 0, 0])  # 最後の手段


def process_csv_chunk(
    chunk_df: pd.DataFrame, metadata: Dict[str, Any]
) -> pd.DataFrame:
    """CSVのチャンクを処理してsourceとtargetカラムを作成"""
    src_vocab_list = metadata["src_vocab_list"]
    tgt_vocab_list = metadata["tgt_vocab_list"]
    max_point_dim = metadata["max_point_dim"]
    pad_index = src_vocab_list.index("[PAD]")

    # inputsカラムが存在するかチェック（input vs inputs）
    input_col = None
    if "inputs" in chunk_df.columns:
        input_col = "inputs"
    elif "input" in chunk_df.columns:
        input_col = "input"
    else:
        raise ValueError("Neither 'inputs' nor 'input' column found in CSV")

    # exprカラムの存在確認
    if "expr" not in chunk_df.columns:
        raise ValueError("'expr' column not found in CSV")

    logging.debug(f"Using input column: {input_col}")

    # sourceとtargetカラムを作成
    source_data = []
    target_data = []

    for _, row in chunk_df.iterrows():
        # sourceカラムの処理
        input_value = row[input_col]
        if pd.notna(input_value):
            source_value = process_input_to_source(
                str(input_value), src_vocab_list, max_point_dim, pad_index
            )
            source_data.append(source_value)
        else:
            # NaNの場合はPADで埋めた単一点
            pad_point = str([[pad_index] * max_point_dim])
            source_data.append(pad_point)

        # targetカラムの処理
        expr_value = row["expr"]
        if pd.notna(expr_value):
            target_value = process_expr_to_target(
                str(expr_value), tgt_vocab_list
            )
            target_data.append(target_value)
        else:
            # NaNの場合は[BOS][PAD][EOS]
            try:
                bos_index = tgt_vocab_list.index("[BOS]")
                eos_index = tgt_vocab_list.index("[EOS]")
                pad_index_tgt = tgt_vocab_list.index("[PAD]")
                fallback_target = str([bos_index, pad_index_tgt, eos_index])
                target_data.append(fallback_target)
            except ValueError:
                target_data.append(str([0, 0, 0]))

    # 結果のDataFrameを作成
    result_df = pd.DataFrame({"source": source_data, "target": target_data})

    return result_df


def process_chunk_parallel(
    chunk_data: Tuple[int, pd.DataFrame, Dict[str, Any]],
) -> pd.DataFrame:
    """並列処理用のチャンク処理関数"""
    chunk_id, chunk_df, metadata = chunk_data
    return process_csv_chunk(chunk_df, metadata)


def process_csv_parallel(
    input_path: str,
    metadata: Dict[str, Any],
    n_workers: int = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """並列処理でCSVファイルを処理"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # 最大8プロセス

    logging.info(f"Starting parallel processing with {n_workers} workers")

    # チャンクに分割してDataFrameのリストを作成
    chunks = []
    chunk_id = 0

    print("📚 Loading and splitting CSV into chunks...")
    chunk_reader = pd.read_csv(input_path, chunksize=chunk_size)

    for chunk_df in chunk_reader:
        chunks.append((chunk_id, chunk_df, metadata))
        chunk_id += 1

    logging.info(f"Created {len(chunks)} chunks for processing")

    # 並列処理で各チャンクを処理
    output_chunks = []

    print(f"⚡ Processing {len(chunks)} chunks with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 全てのチャンクをサブミット
        futures = {
            executor.submit(process_chunk_parallel, chunk): chunk[0]
            for chunk in chunks
        }

        # 進捗バー付きで結果を取得
        completed = 0
        total_chunks = len(chunks)

        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                processed_chunk = future.result()
                output_chunks.append(processed_chunk)

                completed += 1
                if completed % 10 == 0 or completed == total_chunks:  # 進捗表示
                    print(f"  Processed {completed}/{total_chunks} chunks")

            except Exception as e:
                logging.error(f"Error processing chunk {chunk_id}: {e}")

    # 結果をまとめて返す
    if output_chunks:
        return pd.concat(output_chunks, ignore_index=True)
    else:
        return pd.DataFrame()


def process_csv_sequential(
    input_path: str, metadata: Dict[str, Any], chunk_size: int = 1000
) -> pd.DataFrame:
    """順次処理でCSVファイルを処理"""
    output_chunks = []
    total_rows = 0
    chunk_count = 0

    for chunk_df in pd.read_csv(input_path, chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk_df)

        # チャンクを処理
        processed_chunk = process_csv_chunk(chunk_df, metadata)
        output_chunks.append(processed_chunk)

        if chunk_count % 10 == 0:  # 10チャンクごとに進捗表示
            print(f"  Processed {chunk_count} chunks ({total_rows:,} rows)")

    # 結果をまとめて返す
    if output_chunks:
        return pd.concat(output_chunks, ignore_index=True)
    else:
        return pd.DataFrame()


def main() -> int:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Create dataset.csv from raw.csv and metadata.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本的な使用法
  python dataset_file.py -i data/raw.csv
  
  # 全オプション指定
  python dataset_file.py -i data/raw.csv -m data/metadata.yaml -o data/dataset.csv
  
  # 詳細ログ付き
  python dataset_file.py -i data/raw.csv -v
  
  # 並列処理（4ワーカー、チャンクサイズ500）
  python dataset_file.py -i data/raw.csv -p -w 4 -c 500
        """,
    )

    # 必須引数
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input raw CSV file path (required)",
    )

    # オプション引数
    parser.add_argument(
        "-m",
        "--metadata",
        help="Metadata YAML file path (default: {input}_metadata.yaml)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output dataset CSV file path (default: {input}_dataset.csv)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Enable parallel processing",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count, max 8)",
    )

    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for processing (default: 1000)",
    )

    args = parser.parse_args()

    # ログ設定
    setup_logging(args.verbose)

    # デフォルト値の設定
    input_path = args.input
    metadata_path = args.metadata or create_default_metadata_path(input_path)
    output_path = args.output or create_default_output_path(input_path)

    # 引数の値を表示
    print_arguments(input_path, metadata_path, output_path)

    try:
        # メタデータ読み込み
        print("\n📋 Loading metadata...")
        metadata = load_metadata(metadata_path)

        # ファイルサイズとレコード数の事前確認
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        total_rows = sum(1 for _ in open(input_path)) - 1  # ヘッダーを除く

        print(f"📊 File info: {file_size_mb:.2f} MB, {total_rows:,} rows")

        # 並列処理か順次処理かを選択
        if args.parallel:
            print("\n⚡ Starting parallel processing...")
            print(
                f"🔧 Workers: {args.workers or 'auto'}, Chunk size: {args.chunk_size}"
            )

            final_df = process_csv_parallel(
                input_path,
                metadata,
                n_workers=args.workers,
                chunk_size=args.chunk_size,
            )
        else:
            print("\n🚀 Starting sequential processing...")
            final_df = process_csv_sequential(
                input_path, metadata, chunk_size=args.chunk_size
            )

        # 結果を保存
        print("💾 Saving results...")
        final_df.to_csv(output_path, index=False, encoding="utf-8")

        print("\n✅ Processing completed!")
        print(f"📊 Total processed: {len(final_df):,} samples")
        print(f"💾 Output saved to: {output_path}")

        return 0

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
