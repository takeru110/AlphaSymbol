import logging
from pathlib import Path
from datetime import datetime
import numpy as np

import pytest


from src.data.generate_random import (
    generate_random,
)


def test_generate_random():


    # 現在の日時を取得
    now = datetime.now()

    # ファイル名を生成
    timestamp = f"{now.strftime('%Y%m%d_%H%M%S')}.py"
    output_file = Path(__file__).parent.parent.parent / f"output/generate_random_{timestamp}.csv"
    init_csv = Path(__file__).parent.parent.parent / "data/d2-a3-c3-r3.csv"

    sample_num = 10
    sample_max = 10
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, sample_max + 1)
    ]
    print(output_file)
    generate_random(
        sample_num = 20,
        max_arity = 3,
        max_c_args = 3,
        max_r_args = 3,
        eq_domain = eq_domain,
        output_path = output_file,
        init_csv=init_csv
    )