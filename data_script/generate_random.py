import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z

BATCH_SIZE = 20

OUTPUT_FILE = None
data_buffer: list[Expr] = []
saved_expr_counter = 0


def add_data(new_data: Expr):
    if OUTPUT_FILE is None:
        return
    global data_buffer

    # データをバッファに追加
    data_buffer.append(new_data)

    # バッチサイズを超えた場合、CSVファイルに追記してバッファをクリア
    if len(data_buffer) >= BATCH_SIZE:
        save_to_csv(data_buffer)
        data_buffer.clear()


def save_to_csv(data):
    global saved_expr_counter
    if OUTPUT_FILE is None:
        return
    # データをDataFrameに変換
    df = pd.DataFrame()
    df["expr"] = data

    saved_expr_counter += len(data)
    logging.info(
        "Output file is updated. There are %d exprs", saved_expr_counter
    )

    # ファイルが存在しない場合は新規作成、存在する場合は追記
    assert OUTPUT_FILE.exists(), "Output file does not exist"
    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)


def output_bytes_not_const(expr: Expr, eq_domain: list[npt.NDArray]) -> bytes:
    assert expr.arity is not None
    ans: npt.NDArray = np.array([expr.eval(*x) for x in eq_domain[expr.arity]])
    return ans.tobytes()


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[npt.NDArray]
) -> bytes:
    assert input_size >= 1
    ans = np.array([expr.eval(0) for _ in range(len(eq_domain[input_size]))])
    return ans.tobytes()


def init_exprs(
    max_p_arity: int, eq_domain: list[npt.NDArray]
) -> tuple[list[list[Expr]], list[set[bytes]]]:
    exprs: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
    outputs: list[set[bytes]] = [set() for _ in range(max_p_arity + 1)]
    exprs[0].append(Z())
    for input_size in range(1, max_p_arity + 1):
        outputs[input_size].add(output_bytes_const(Z(), input_size, eq_domain))
    try:
        outputs[1].add(output_bytes_not_const(S(), eq_domain))
        exprs[1].append(S())
    except OverflowError:
        pass
    for i in range(1, max_p_arity + 1):
        for j in range(1, i + 1):
            try:
                exprs[i].append(P(i, j))
                outputs[i].add(output_bytes_not_const(P(i, j), eq_domain))
            except OverflowError:
                pass
    return exprs, outputs


def init_exprs_from_csv(
    init_file: Path, max_arity: int, eq_domain: list[npt.NDArray]
) -> tuple[list[list[Expr]], list[set[bytes]]]:
    exprs: list[list[Expr]] = [[] for _ in range(max_arity + 1)]
    outputs: list[set[bytes]] = [set() for _ in range(max_arity + 1)]
    df = pd.read_csv(init_file)
    for index, row in df.iterrows():
        expr: Expr = eval(row["expr"])
        if expr.arity is None:
            try:
                for input_size in range(1, max_arity + 1):
                    outputs[input_size].add(
                        output_bytes_const(Z(), input_size, eq_domain)
                    )
            except OverflowError:
                continue
            exprs[0].append(expr)
        else:
            try:
                outputs[expr.arity].add(output_bytes_not_const(expr, eq_domain))
            except OverflowError:
                continue
            exprs[expr.arity].append(expr)
    return exprs, outputs


def if_not_visited_then_update_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    max_dim: int,
    eq_domain: list[npt.NDArray],
) -> tuple[list[set[bytes]], bool]:
    assert expr.arity is None
    is_visited = True
    for dim in range(1, max_dim + 1):
        try:
            # when OverflowError occurs, it occurs in all iteration
            out_bytes = output_bytes_const(expr, dim, eq_domain)
        except OverflowError:
            return outputs, False
        if out_bytes in outputs[dim]:
            continue
        outputs[dim].add(out_bytes)
        is_visited = False
    is_new = not is_visited
    return outputs, is_new


def if_not_visited_then_update_not_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    eq_domain: list[npt.NDArray],
) -> tuple[list[set[bytes]], bool]:
    assert expr.arity is not None
    try:
        out_bytes = output_bytes_not_const(expr, eq_domain)
    except OverflowError:
        return outputs, False
    if out_bytes in outputs[expr.arity]:
        return outputs, False
    outputs[expr.arity].add(out_bytes)
    return outputs, True


def generate_random(
    sample_num,
    max_p_arity,
    max_c_args,
    max_r_args,
    eq_domain,
    output_path: Optional[Path],
    init_csv: Optional[Path],
):
    if output_path is not None:
        global OUTPUT_FILE
        if output_path.exists():
            output_path.unlink()
        output_path.touch()
        OUTPUT_FILE = output_path
        with open(OUTPUT_FILE, "w") as f:
            f.write("expr\n")

    # gen_exprs[arity]: list of Expr with arity
    # new exprs is generated combining exprs in gen_exprs

    if init_csv is None:
        gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
    else:
        gen_exprs, outputs = init_exprs_from_csv(
            init_csv, max_p_arity, eq_domain
        )
    new_exprs: list[Expr] = []
    # visited[arity] set of output of Expr with arity
    while len(new_exprs) < sample_num:
        if random.random() < 0.5:
            base_arity: int = random.randint(
                1, min(max_p_arity, max_c_args - 1)
            )
            arg_arity: int = random.randint(1, max_p_arity)
            base: Expr = random.choice(gen_exprs[base_arity])
            args = tuple(
                random.choice(gen_exprs[arg_arity] + gen_exprs[0])
                for _ in range(base_arity)
            )
            new_expr_c = C(base, *args)
            if new_expr_c.arity is None:
                outputs, is_new = if_not_visited_then_update_const(
                    gen_exprs,
                    outputs,
                    new_expr_c,
                    max_p_arity,
                    eq_domain,
                )
                if is_new:
                    new_exprs.append(new_expr_c)

            else:
                outputs, is_new = if_not_visited_then_update_not_const(
                    gen_exprs, outputs, new_expr_c, eq_domain
                )
                if is_new:
                    new_exprs.append(new_expr_c)
        else:
            while True:
                try:
                    term_arity = random.randint(1, (max_r_args - 1) // 2)
                    step_arity = random.randint(
                        term_arity + 1, min(term_arity + 1, max_p_arity)
                    )
                    base_arity = random.randint(0, step_arity - term_arity - 1)
                    term: Expr = random.choice(gen_exprs[term_arity])
                    steps: tuple[Expr, ...] = tuple(
                        random.choice(gen_exprs[step_arity])
                        for _ in range(term_arity)
                    )
                    bases: tuple[Expr, ...] = tuple(
                        random.choice(gen_exprs[base_arity] + gen_exprs[0])
                        for _ in range(term_arity)
                    )
                    break
                except ValueError:
                    continue
            new_expr_r = R(term, *steps, *bases)
            if new_expr_r.arity is None:
                outputs, is_updated = if_not_visited_then_update_const(
                    gen_exprs,
                    outputs,
                    new_expr_r,
                    max_p_arity,
                    eq_domain,
                )
                if is_updated:
                    new_exprs.append(new_expr_r)
            else:
                outputs, is_updated = if_not_visited_then_update_not_const(
                    gen_exprs, outputs, new_expr_r, eq_domain
                )
                if is_updated:
                    new_exprs.append(new_expr_r)

    for item in new_exprs:
        add_data(item)
    logging.debug(f"Iter {iter}: {len(new_exprs)} is added")

    save_to_csv(data_buffer)
    data_buffer.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate expressions randomly"
    )

    parser.add_argument("--sample", type=int)
    parser.add_argument("-p", "--max_p_arity", type=int)
    parser.add_argument("-c", "--max_c_args", type=int)
    parser.add_argument("-r", "--max_r_args", type=int)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--sample_max", type=int, default=10)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--init_csv", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s-%(levelname)s-%(message)s",
        force=True,
    )

    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, args.sample_max + 1, size=(args.sample_num, dim))
        for dim in range(1, args.sample_max + 1)
    ]
    eq_domain[1] = np.array(range(10)).reshape(10, 1)

    output_path = None if args.output is None else Path(args.output)
    init_csv_path = None if args.init_csv is None else Path(args.init_csv)

    exprs = generate_random(
        args.sample,
        args.max_p_arity,
        args.max_c_args,
        args.max_r_args,
        eq_domain,
        output_path,
        init_csv_path,
    )
