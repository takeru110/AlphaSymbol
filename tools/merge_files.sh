#!/bin/bash

input_directory=$1
output_file=$2

# 結合結果を保存するファイル名
output_file="merged_output.txt"

if [ -z "$input_directory" ] || [ -z "$output_file" ]; then
    echo "Usage: ./merge_files.sh <Dir Name> <Output File>"
    exit 1
fi

if [ ! -d "$input_directory" ]; then
    echo "Error: The specified directory is not exist"
    exit 1
fi

# 結合結果をクリア（すでに存在している場合）
> "$output_file"

# 指定したディレクトリ内の全てのファイルを対象
find "$input_directory" -type f | while read file; do
  # ディレクトリは無視
  if [ -f "$file" ]; then
    # ファイル名と拡張子を出力
    echo "$file" >> "$output_file"
    
    # ファイルの内容を囲む
    echo '```' >> "$output_file"
    cat "$file" >> "$output_file"
    echo '```' >> "$output_file"
    
    # 次のファイルと区別するために改行
    echo "" >> "$output_file"
  fi
done

echo "全ファイルを$OUTPUT_FILEに結合しました。"
