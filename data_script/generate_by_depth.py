import argparse
import logging
import time
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z, expr_list_to_str

BATCH_SIZE = 100
OUTPUT_FILE: Optional[Path] = None
buffer_counter = 0
saved_expr_counter = 0
data_buffer: list[Expr] = []


def add_data(new_data: Expr):
    if OUTPUT_FILE is None:
        return
    global data_buffer, buffer_counter

    # データをバッファに追加
    data_buffer.append(new_data)
    buffer_counter = len(data_buffer)

    # バッチサイズを超えた場合、CSVファイルに追記してバッファをクリア
    if buffer_counter >= BATCH_SIZE:
        save_to_csv(data_buffer)
        data_buffer.clear()


def save_to_csv(data):
    global saved_expr_counter
    # データをDataFrameに変換
    df = pd.DataFrame()
    df["expr"] = data
    saved_expr_counter += len(data)

    # ファイルが存在しない場合は新規作成、存在する場合は追記
    assert OUTPUT_FILE.exists(), f"Output file is not created: {OUTPUT_FILE}"
    df.to_csv(OUTPUT_FILE, mode="a", index=False)
    logging.info(
        "Output file is updated. There are %d exprs", saved_expr_counter
    )


def output_bytes_not_const(
    expr: Expr, eq_domain: list[npt.NDArray]
) -> tuple[bytes, bool]:
    """
    Returns:
    - bytes: output of Expr.eval on eq_domain
    - bool: True if no OverflowError, False otherwise
    """
    assert expr.arity is not None
    try:
        ans: npt.NDArray = np.array(
            tuple(expr.eval(*x) for x in eq_domain[expr.arity])
        ).flatten()
        return ans.tobytes(), True
    except OverflowError:
        ans = np.empty((eq_domain[expr.arity]).shape[0])
        ans.fill(None)
        logging.debug("output oveflowed")
        return ans.tobytes(), False


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[npt.NDArray]
) -> tuple[bytes, bool]:
    """
    Returns:
    - bytes: output of Expr.eval on eq_domain
    - bool: True if no OverflowError, False otherwise
    """
    assert input_size >= 1
    try:
        ret_arr = np.array(
            tuple(expr.eval(0) for _ in range(len(eq_domain[input_size])))
        )
        return ret_arr.tobytes(), True
    except OverflowError:
        ret_arr = np.empty(eq_domain[input_size].shape[0], dtype=object)
        ret_arr.fill(None)
        return ret_arr.tobytes(), False


def one_depth_exprs(
    max_arity: int, eq_domain: list[npt.NDArray]
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    exprs: list[list[list[Expr]]] = [[], [[] for _ in range(max_arity + 1)]]
    visited: list[set[bytes]] = [set() for _ in range(max_arity + 1)]

    logging.debug("Z() added")
    exprs[1][0].append(Z())
    add_data(Z())
    for input_size in range(1, max_arity + 1):
        output_bytes, _ = output_bytes_const(Z(), input_size, eq_domain)
        visited[input_size].add(output_bytes)

    b_output, is_success = output_bytes_not_const(S(), eq_domain)
    if is_success:
        visited[1].add(b_output)
        logging.debug("S() added")
        exprs[1][1].append(S())
        add_data(S())

    for i in range(1, max_arity + 1):
        for j in range(1, i + 1):
            b_output, is_success = output_bytes_not_const(P(i, j), eq_domain)
            if is_success:
                exprs[1][i].append(P(i, j))
                add_data(P(i, j))
                visited[i].add(b_output)
    return exprs, visited


def gen_exprs_by_arity(exprs):
    ret_li = [[] for _ in range(len(exprs[1]))]
    for exprs_by_arity in exprs:
        for arity in range(len(exprs_by_arity)):
            ret_li[arity].extend(exprs_by_arity[arity])
    return ret_li


def c_pattern(depth, exprs_by_arity, max_arity, max_c):
    c_exprs = [[] for _ in range(max_arity + 1)]
    max_base_arity = min(max_arity, max_c - 1)
    outer_arities = tuple(range(1, max_base_arity + 1))
    inner_arities = tuple(range(max_arity + 1))
    args_arities_list = tuple(
        (outer_arity, *inner_airty_and_zero)
        for outer_arity in outer_arities
        for inner_arity in inner_arities
        for inner_airty_and_zero in product(
            (0, inner_arity), repeat=outer_arity
        )
    )
    for args in args_arities_list:
        # args is like (5, 2, 2, 0, 2, 2)
        new_args_list = list(product(*((exprs_by_arity[arg]) for arg in args)))
        new_args_list_filterd_depth = [
            tup
            for tup in new_args_list
            if any(expr.depth == depth - 1 for expr in tup)
        ]
        for new_args in new_args_list_filterd_depth:
            new_arity = max(args[1:])
            c_exprs[new_arity].extend([C(*new_args)])
    return c_exprs


def get_r_arity(args: tuple):
    assert len(args) >= 3
    n = (len(args) - 1) // 2
    t = args[0]
    a_s = args[1 : n + 1]
    b_s = args[n + 1 :]
    a_arity = max(a_s)
    b_arity = max(b_s)
    if b_arity == 0:
        if a_arity == 0:
            return 0
        else:
            return a_arity - t
    return b_arity


def r_pattern(depth, exprs_by_arity, max_arity, max_r):
    t_a_b_list = [
        (n, d + n + 1, d)
        for d in range(0, max_arity - 1)
        for n in range(1, min(max_arity - d - 1, (max_r - 1) // 2) + 1)
    ]

    args_list = [
        (t, *a_s, *b_s)
        for t, a, b in t_a_b_list
        for a_s in product((0, a), repeat=t)
        for b_s in product((b,), repeat=t)
    ]

    r_exprs: list[list[Expr]] = [[] for _ in range(max_arity + 1)]
    for args in args_list:
        expr_list = tuple(product(*(exprs_by_arity[arg] for arg in args)))
        expr_list_filterd_depth = [
            tup
            for tup in expr_list
            if any(expr.depth == depth - 1 for expr in tup)
        ]
        arity = get_r_arity(args)
        r_exprs[arity].extend([R(*args) for args in expr_list_filterd_depth])
    return r_exprs


def if_not_visited_then_update_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    max_dim: int,
    eq_domain: list[npt.NDArray],
) -> tuple[list[list[Expr]], list[set[bytes]], bool]:
    assert expr.arity is None
    is_visited = True
    for dim in range(1, max_dim + 1):
        out_bytes, is_success = output_bytes_const(expr, dim, eq_domain)
        if not is_success:
            return exprs, outputs, False
        if out_bytes not in outputs[dim]:
            outputs[dim].add(out_bytes)
            is_visited = False

    if not is_visited:
        exprs[0].append(expr)
        add_data(expr)
        logging.debug(f"{expr} added")
        is_updated = True
    else:
        is_updated = False
    return exprs, outputs, is_updated


def if_not_visited_then_update_not_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    eq_domain: list[npt.NDArray],
) -> tuple[list[list[Expr]], list[set[bytes]], bool]:
    assert expr.arity is not None
    out_bytes, is_success = output_bytes_not_const(expr, eq_domain)
    if not is_success:
        return exprs, outputs, False
    if out_bytes in outputs[expr.arity]:
        return exprs, outputs, False
    logging.debug(f"{expr} added")
    exprs[expr.arity].append(expr)
    add_data(expr)
    outputs[expr.arity].add(out_bytes)
    return exprs, outputs, True


def _generate_prfndim_by_depth(
    depth: int,
    max_arity: int,
    max_c: int,
    max_r: int,
    eq_domain: list[npt.NDArray],
    visited: list[set[bytes]],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    Args:
    - exprs: list[list[list[Expr]]]: list of list of Exprs
        - exprs[depth][arity]
    - visited: list[set[bytes]]
        - visited[input_size]
    """
    assert depth > 0, "Expr size must be positive"
    if depth == 1:
        return one_depth_exprs(max_arity, eq_domain)

    exprs, visited = _generate_prfndim_by_depth(
        depth - 1, max_arity, max_c, max_r, eq_domain, visited
    )
    exprs_by_arity = gen_exprs_by_arity(exprs)
    exprs.append([[] for _ in range(max_arity + 1)])

    c_exprs = c_pattern(depth, exprs_by_arity, max_arity, max_c)

    for expr in c_exprs[0]:
        exprs[depth], visited, _ = if_not_visited_then_update_const(
            exprs[depth], visited, expr, max_arity, eq_domain
        )

    for _exprs in c_exprs[1:]:
        for expr in _exprs:
            exprs[depth], visited, _ = if_not_visited_then_update_not_const(
                exprs[depth], visited, expr, eq_domain
            )

    r_exprs = r_pattern(depth, exprs_by_arity, max_arity, max_r)

    for expr in r_exprs[0]:
        exprs[depth], visited, _ = if_not_visited_then_update_const(
            exprs[depth], visited, expr, max_arity, eq_domain
        )

    for _exprs in r_exprs[1:]:
        for expr in _exprs:
            exprs[depth], visited, _ = if_not_visited_then_update_not_const(
                exprs[depth], visited, expr, eq_domain
            )

    return exprs, visited


def generate_by_depth(
    depth: int,
    max_arity: int,
    max_c: int,
    max_r: int,
    eq_domain: list[npt.NDArray],
    output_file: Optional[Path] = None,
) -> list[Expr]:
    if output_file is not None:
        global OUTPUT_FILE
        if output_file.exists():
            output_file.unlink()
        output_file.touch()

        OUTPUT_FILE = output_file

    visited: list[set[bytes]] = [set() for _ in range(max_arity + 1)]
    exprs, _ = _generate_prfndim_by_depth(
        depth,
        max_arity,
        max_c,
        max_r,
        eq_domain,
        visited,
    )
    ret: list[Expr] = [
        expr
        for fix_depth in exprs
        for fix_depth_arity in fix_depth
        for expr in fix_depth_arity
    ]
    if OUTPUT_FILE is not None:
        save_to_csv(data_buffer)
        data_buffer.clear()
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all valid prfndim expressions by depth"
    )
    start_time = time.time()
    parser.add_argument(
        "--depth", type=int, help="Max depth of expressions for searching"
    )
    parser.add_argument(
        "--max_arity", type=int, help="Max arity of expressions"
    )
    parser.add_argument("--max_c", type=int, help="Max arity of C expressions")
    parser.add_argument("--max_r", type=int, help="Max arity of R expressions")
    parser.add_argument(
        "--output", type=str, help="Output CSV file path", default=None
    )

    parser.add_argument(
        "--sample_num",
        default=30,
        type=int,
        help="Number of samples to decide expressions are equal",
    )

    parser.add_argument(
        "--sample_max",
        default=10,
        type=int,
        help="Max value of sample to decide expressions are equal",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s-%(levelname)s-%(message)s",
        force=True,
    )
    output_path = Path(args.output)
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, args.sample_max + 1, size=(args.sample_num, dim))
        for dim in range(1, args.max_arity + 1)
    ]
    eq_domain[1] = np.arange(10).reshape(10, 1)

    ret = generate_by_depth(
        args.depth,
        args.max_arity,
        args.max_c,
        args.max_r,
        eq_domain,
        output_file=output_path,
    )
    end_time = time.time()
    logging.info(f"Time: {end_time - start_time}")
