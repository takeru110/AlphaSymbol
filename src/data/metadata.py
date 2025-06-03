#!/usr/bin/env python3
"""
メタデータ計算ツール

PRFDatasetの初期化処理を参考にして、以下の値を計算しYAMLファイルに出力する：
- src_vocab_list: ソース語彙リスト
- tgt_vocab_list: ターゲット語彙リスト
- max_tgt_length: 最大ターゲット長
- max_src_points: 最大ソース点数
"""

import argparse
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


def encode_expr(expr_str: str) -> List[str]:
    """exprをトークン化（カスタムトークナイザーの処理を模擬）"""
    # 関数名（Z,S,P,C,R）、数字、括弧、カンマ、スペースを抽出
    tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str)
    # 空文字列を除去
    tokens = [token for token in tokens if token.strip()]
    # BOS, EOSトークンを追加
    return ["[BOS]"] + tokens + ["[EOS]"]


def process_chunk(chunk_data: Tuple[int, pd.DataFrame]) -> Tuple[Set[str], Set[str], int, int, int]:
    """
    チャンクを処理してメタデータを抽出する関数（並列処理用）
    
    Args:
        chunk_data: (chunk_id, DataFrame) のタプル
        
    Returns:
        Tuple[src_vocab_set, tgt_vocab_set, max_tgt_length, max_src_points, skipped_count]
    """
    chunk_id, chunk_df = chunk_data
    
    src_vocab = set()
    tgt_vocab = set()
    max_tgt_length = 0
    max_src_points = 0
    skipped_count = 0
    
    for _, row in chunk_df.iterrows():
        # 1. src語彙の抽出（input/output）
        if "input" in row and pd.notna(row["input"]):
            input_str = str(row["input"])
            # 数字、括弧、カンマ、スペースを抽出
            tokens = re.findall(r"\d+|[\[\],\(\)\s]", input_str)
            src_vocab.update(tokens)

            # 3. max_src_pointsの計算
            try:
                input_data = (
                    eval(input_str)
                    if isinstance(input_str, str)
                    else input_str
                )
                points = len(input_data)
                max_src_points = max(max_src_points, points)
            except (ValueError, SyntaxError, TypeError):
                skipped_count += 1

        if "output" in row and pd.notna(row["output"]):
            output_str = str(row["output"])
            tokens = re.findall(r"\d+|[\[\],\(\)\s]", output_str)
            src_vocab.update(tokens)

        # 2. tgt語彙の抽出とmax_tgt_lengthの計算（expr）
        if "expr" in row and pd.notna(row["expr"]):
            expr_str = str(row["expr"])

            # tgt語彙の抽出
            tokens = re.findall(
                r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str
            )
            tgt_vocab.update(tokens)

            # max_tgt_lengthの計算
            encoded_tokens = encode_expr(expr_str)
            max_tgt_length = max(
                max_tgt_length, len(encoded_tokens)
            )
    
    # 空文字列を除去
    src_vocab.discard("")
    tgt_vocab.discard("")
    
    return src_vocab, tgt_vocab, max_tgt_length, max_src_points, skipped_count


def calculate_metadata_parallel(
    csv_path: str,
    n_workers: int = None,
    chunk_size: int = 1000
) -> Tuple[Set[str], Set[str], int, int]:
    """
    並列処理でCSVファイルからメタデータを計算
    
    Args:
        csv_path: CSVファイルのパス
        n_workers: ワーカープロセス数（Noneの場合はCPU数）
        chunk_size: チャンクサイズ
        
    Returns:
        Tuple[src_vocab_set, tgt_vocab_set, max_tgt_length, max_src_points]
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # 最大8プロセス
    
    logging.info(f"Starting parallel processing with {n_workers} workers")
    
    # ファイルの総行数を取得
    total_rows = sum(1 for _ in open(csv_path)) - 1  # ヘッダーを除く
    logging.info(f"Total rows to process: {total_rows}")
    
    # チャンクに分割してDataFrameのリストを作成
    chunks = []
    chunk_id = 0
    
    print("📚 Loading and splitting CSV into chunks...")
    chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size)
    
    for chunk_df in tqdm(chunk_reader, desc="Loading chunks"):
        chunks.append((chunk_id, chunk_df))
        chunk_id += 1
    
    logging.info(f"Created {len(chunks)} chunks for processing")
    
    # 並列処理で各チャンクを処理
    src_vocab_global = set()
    tgt_vocab_global = set()
    max_tgt_length_global = 0
    max_src_points_global = 0
    total_skipped = 0
    
    print(f"⚡ Processing {len(chunks)} chunks with {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 全てのチャンクをサブミット
        futures = {executor.submit(process_chunk, chunk): chunk[0] 
                  for chunk in chunks}
        
        # 進捗バー付きで結果を取得
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunk") as pbar:
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    src_vocab, tgt_vocab, max_tgt_length, max_src_points, skipped_count = future.result()
                    
                    # 結果をマージ
                    src_vocab_global.update(src_vocab)
                    tgt_vocab_global.update(tgt_vocab)
                    max_tgt_length_global = max(max_tgt_length_global, max_tgt_length)
                    max_src_points_global = max(max_src_points_global, max_src_points)
                    total_skipped += skipped_count
                    
                    pbar.set_postfix(
                        src_vocab=len(src_vocab_global),
                        tgt_vocab=len(tgt_vocab_global),
                        max_tgt_len=max_tgt_length_global,
                        max_src_pts=max_src_points_global,
                        skipped=total_skipped,
                        refresh=False
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_id}: {e}")
                    
                pbar.update(1)
    
    # デフォルト値の設定
    if max_tgt_length_global == 0:
        max_tgt_length_global = 128
    if max_src_points_global == 0:
        max_src_points_global = 20

    if total_skipped > 0:
        logging.warning(f"Skipped {total_skipped} samples during processing")

    logging.info("Parallel processing completed:")
    logging.info(f"  - Src vocabulary size: {len(src_vocab_global)}")
    logging.info(f"  - Tgt vocabulary size: {len(tgt_vocab_global)}")
    logging.info(f"  - Max target length: {max_tgt_length_global}")
    logging.info(f"  - Max source points: {max_src_points_global}")

    return src_vocab_global, tgt_vocab_global, max_tgt_length_global, max_src_points_global


def calculate_all_metadata(
    csv_path: str,
) -> Tuple[Set[str], Set[str], int, int]:
    """
    CSVファイルを1回だけ走査して全てのメタデータを計算（シーケンシャル版）

    Returns:
        Tuple[src_vocab_set, tgt_vocab_set, max_tgt_length, max_src_points]
    """
    src_vocab = set()
    tgt_vocab = set()
    max_tgt_length = 0
    max_src_points = 0
    skipped_count = 0

    try:
        chunk_size = 1000

        # ファイルの総行数を取得
        total_rows = sum(1 for _ in open(csv_path)) - 1  # ヘッダーを除く
        logging.info(f"Total rows to process: {total_rows}")

        with tqdm(
            total=total_rows,
            desc="🔍 Processing CSV data",
            unit="rows",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    # 1. src語彙の抽出（input/output）
                    if "input" in row and pd.notna(row["input"]):
                        input_str = str(row["input"])
                        # 数字、括弧、カンマ、スペースを抽出
                        tokens = re.findall(r"\d+|[\[\],\(\)\s]", input_str)
                        src_vocab.update(tokens)

                        # 3. max_src_pointsの計算
                        try:
                            input_data = (
                                eval(input_str)
                                if isinstance(input_str, str)
                                else input_str
                            )
                            points = len(input_data)
                            max_src_points = max(max_src_points, points)
                        except (ValueError, SyntaxError, TypeError) as e:
                            skipped_count += 1
                            logging.debug(
                                f"Skipping input data due to error: {e}"
                            )

                    if "output" in row and pd.notna(row["output"]):
                        output_str = str(row["output"])
                        tokens = re.findall(r"\d+|[\[\],\(\)\s]", output_str)
                        src_vocab.update(tokens)

                    # 2. tgt語彙の抽出とmax_tgt_lengthの計算（expr）
                    if "expr" in row and pd.notna(row["expr"]):
                        expr_str = str(row["expr"])

                        # tgt語彙の抽出
                        tokens = re.findall(
                            r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str
                        )
                        tgt_vocab.update(tokens)

                        # max_tgt_lengthの計算
                        encoded_tokens = encode_expr(expr_str)
                        max_tgt_length = max(
                            max_tgt_length, len(encoded_tokens)
                        )

                    pbar.update(1)

                    # 統計情報を100行ごとに更新
                    if pbar.n % 100 == 0:
                        pbar.set_postfix(
                            src_vocab=len(src_vocab),
                            tgt_vocab=len(tgt_vocab),
                            max_tgt_len=max_tgt_length,
                            max_src_pts=max_src_points,
                            skipped=skipped_count,
                            refresh=False,
                        )

    except Exception as e:
        logging.warning(f"Chunk reading failed: {e}. Trying normal reading...")
        # チャンク読み込みが失敗した場合は通常の読み込み
        df = pd.read_csv(csv_path)

        logging.info(f"Fallback: Processing {len(df)} rows in single batch")

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="🔍 Processing CSV data (fallback)",
            unit="rows",
        ):
            # 同じ処理をフォールバック用に実行
            if "input" in row and pd.notna(row["input"]):
                input_str = str(row["input"])
                tokens = re.findall(r"\d+|[\[\],\(\)\s]", input_str)
                src_vocab.update(tokens)

                try:
                    input_data = (
                        eval(input_str)
                        if isinstance(input_str, str)
                        else input_str
                    )
                    points = len(input_data)
                    max_src_points = max(max_src_points, points)
                except (ValueError, SyntaxError, TypeError) as e:
                    skipped_count += 1
                    logging.debug(f"Skipping input data due to error: {e}")

            if "output" in row and pd.notna(row["output"]):
                output_str = str(row["output"])
                tokens = re.findall(r"\d+|[\[\],\(\)\s]", output_str)
                src_vocab.update(tokens)

            if "expr" in row and pd.notna(row["expr"]):
                expr_str = str(row["expr"])
                tokens = re.findall(r"[ZSPCRPRF]+|\d+|[\(\),\s]", expr_str)
                tgt_vocab.update(tokens)

                encoded_tokens = encode_expr(expr_str)
                max_tgt_length = max(max_tgt_length, len(encoded_tokens))

    # 空文字列を除去
    src_vocab.discard("")
    tgt_vocab.discard("")

    # デフォルト値の設定
    if max_tgt_length == 0:
        max_tgt_length = 128
    if max_src_points == 0:
        max_src_points = 20

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} samples during processing")

    logging.info("Processing completed:")
    logging.info(f"  - Src vocabulary size: {len(src_vocab)}")
    logging.info(f"  - Tgt vocabulary size: {len(tgt_vocab)}")
    logging.info(f"  - Max target length: {max_tgt_length}")
    logging.info(f"  - Max source points: {max_src_points}")

    return src_vocab, tgt_vocab, max_tgt_length, max_src_points


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Calculate metadata from CSV file and output to YAML"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output YAML file path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true", 
        help="Enable parallel processing"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help="Number of worker processes (default: CPU count, max 8)"
    )
    parser.add_argument(
        "-c", "--chunk-size", type=int, default=1000,
        help="Chunk size for processing (default: 1000)"
    )

    args = parser.parse_args()

    # ログレベルの設定
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        print(f"🔍 Processing CSV file: {args.input}")

        # ファイルサイズとレコード数の事前確認
        file_size_mb = os.path.getsize(args.input) / (1024 * 1024)
        total_rows = sum(1 for _ in open(args.input)) - 1  # ヘッダーを除く

        print(f"📊 File info: {file_size_mb:.2f} MB, {total_rows:,} rows")

        # 並列処理か順次処理かを選択
        if args.parallel:
            print(f"\n⚡ Starting parallel metadata calculation...")
            print(f"🔧 Workers: {args.workers or 'auto'}, Chunk size: {args.chunk_size}")
            
            src_vocab_set, tgt_vocab_set, max_tgt_length, max_src_points = (
                calculate_metadata_parallel(
                    args.input, 
                    n_workers=args.workers,
                    chunk_size=args.chunk_size
                )
            )
        else:
            print("\n🚀 Starting single-pass metadata calculation...")
            
            src_vocab_set, tgt_vocab_set, max_tgt_length, max_src_points = (
                calculate_all_metadata(args.input)
            )

        print("\n🔄 Converting vocabularies to sorted lists...")
        # リストに変換してソート
        src_vocab_list = sorted(list(src_vocab_set))
        tgt_vocab_list = sorted(list(tgt_vocab_set))

        # YAMLに出力するデータを作成（2つのlistが最後になる並び）
        metadata = {
            "max_tgt_length": max_tgt_length,
            "max_src_points": max_src_points,
            "src_vocab_list": src_vocab_list,
            "tgt_vocab_list": tgt_vocab_list,
        }

        # YAMLファイルに書き出し
        print(f"💾 Writing metadata to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

        # 結果のサマリーを表示
        print("\n✅ Metadata calculation completed!")
        print("=" * 50)
        print(f"📏 Max target length:     {max_tgt_length:,}")
        print(f"📊 Max source points:     {max_src_points:,}")
        print(f"📚 Source vocabulary size: {len(src_vocab_list):,}")
        print(f"🎯 Target vocabulary size: {len(tgt_vocab_list):,}")
        print("=" * 50)
        print(f"💾 Results saved to: {args.output}")

    except Exception as e:
        logging.error(f"❌ Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
