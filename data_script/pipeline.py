from pathlib import Path

import numpy as np
import pandas as pd
from generate_by_depth import generate_by_depth
from generate_random import generate_random
from unify_output_csv import unify_output_csv

from data_script.input_output_columns import input_output_columns
from data_script.status_columns import status_columns

max_value = 10
sample_num = 10
depth = 3
max_arity = 3
max_c = 3
max_r = 3


eq_domain = [np.zeros((sample_num, 1))] + [
    np.random.randint(1, max_value + 1, size=(sample_num, dim))
    for dim in range(1, max_arity + 1)
]
eq_domain[1] = np.arange(10).reshape(10, 1)

temp_dir = Path("/home/takeru/AlphaSymbol/temp/")
after_gen_depth = temp_dir / "generate_by_depth.csv"
after_gen_random = temp_dir / "generate_random.csv"
after_status = temp_dir / "add_status.csv"
after_unify = temp_dir / "unify_output.csv"
after_in_out = temp_dir / "add_in_out.csv"


generate_by_depth(
    depth=depth,
    max_arity=max_arity,
    max_c=max_c,
    max_r=max_r,
    eq_domain=eq_domain,
    output_file=after_gen_depth,
)

generate_random(
    sample_num=sample_num,
    max_p_arity=max_arity,
    max_c_args=max_c,
    max_r_args=max_r,
    eq_domain=eq_domain,
    output_path=after_gen_random,
    init_csv=after_gen_depth,
)

df_random = pd.read_csv(after_gen_random)
df_depth = pd.read_csv(after_gen_depth)
df_cat = pd.concat([df_random, df_depth], axis=0)

df_status = status_columns(df_cat)
df_sorted = df_status.sort_values(by=["arity", "len", "depth"])
df_sorted.to_csv(after_status, index=False)

unify_output_csv(
    input_file=after_status,
    output_file=after_unify,
    max_arity=max_arity,
    eq_domain=eq_domain,
    buffer_size=10000,
)

df_unify = pd.read_csv(after_unify)
df_in_out = input_output_columns(df_unify, eq_domain)
df_in_out.to_csv(after_in_out, index=False)
