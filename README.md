# PrfSR

## Overview
PrfSR（Primitive Recursive Function Symbolic Regression）は、原始帰納関数を元にした記号体系PRFを用いたSymbolic Regressionを行うソフトウェアです。記号生成とディープラーニングのアプローチを組み合わせて、アルゴリズミックな自然数関数に対するPRF式の生成と学習に焦点を当てています。

## Demo

main.pyを実行して、予測したい数列の先頭部分を入力するとその数列のパターンを表すPRF式と具体的な数列を出力します。

```
> python main.py
Please input numbers: 1, 2, 3, 4, 5

Now inferencing...

Prf expression: C(S(), P(1, 1))
Values: 1, 2, 3, 4, 5, 6, 7, 8, 9...
```


## Description
このプロジェクトは数値として入力した自然数関数に対して、そのアルゴリズムを予測するものです。

このプロジェクトは、記号回帰に対して2段階のアプローチを実装しています：

1. **記号表現の生成**:
   - 指定された深さまでの原始帰納関数を生成
   - 以下の関数パターンをサポート：
     - 基本関数（Z, S, P）
     - 合成（C）
     - 原始帰納（R）

2. **ニューラルネットワークによる学習**:
   - Transformerベースのアーキテクチャを使用した記号表現の学習
   - 検証とテスト分割を含む学習パイプライン


## VS. Traditional Methods
- **利点**:
  - 原始帰納関数の体系的生成
  - ディープラーニングによる複雑な数学的表現の処理
  - 異なる深さやアリティへのスケーラビリティ
  - バイトレベル比較による効率的な重複検出
- **制限事項**:
  - 深さに応じて計算複雑性が増加
  - 生成された式の保存に必要なメモリ要件
  - ニューラルネットワークコンポーネントの学習時間

## Requirements
- Python >= 3.12
- PyTorch >= 2.7.0
- NumPy >= 2.2.6
- Pandas >= 2.2.3
- Lightning >= 1.8.6
- Hydra >= 1.3.2
- CUDAサポート（オプション、requirements-cuda.txtを参照）

## Usage
1. **記号表現の生成**:
   ```python
   from src.data.generate_by_depth import generate_by_depth
   
   # 深さ3、最大アリティ2までの式を生成
   expressions = generate_by_depth(
       depth=3,
       max_arity=2,
       max_c=3,
       max_r=3,
       eq_domain=[...],  # 評価用の定義域
       output_file="expressions.csv"
   )
   ```

2. **モデルの学習**:
   ```bash
   # デフォルト設定を使用
   python src/models/train.py
   
   # カスタム設定を使用
   python src/models/train.py --config-name custom_config
   ```

## Installation
1. リポジトリのクローン
2. 仮想環境の作成:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Unix/macOSの場合
   ```
3. 依存パッケージのインストール:
   ```bash
   # CPUのみの場合
   pip install -r requirements.txt
   
   # CUDAサポートを含む場合
   pip install -r requirements-cuda.txt
   ```

## File Structure
```
.
├── src/
│   ├── data/                    # データ生成と処理
│   │   ├── generate_by_depth.py # 深さを制限して式を全探索する
│   │   ├── generate_random.py   # ランダムに式を生成する
│   │   ├── pipeline.py         # データ処理パイプライン
│   │   └── prfndim_utils.py    # ユーティリティ関数
│   ├── models/                  # Neural Network
│   │   ├── train.py            # 学習パイプライン
│   │   └── ...                 # モデル実装
│   └── visualization/          # 可視化ツール
├── tests/                      # テストファイル
├── notebooks/                  # Jupyterノートブック
├── data/                       # 生成されたデータ
└── logs/                       # 学習ログ
```

## Contribution
(追加予定)

## License
このプロジェクトは、LICENSEファイルに含まれる条項に基づいてライセンスされています。

## Author
(追加予定 - 作者情報の追加が必要)

## Notes
(追加予定)