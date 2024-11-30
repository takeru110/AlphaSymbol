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
counter = 0


def add_data(new_data: Expr):
    if OUTPUT_FILE is None:
        return
    global data_buffer, counter

    # データをバッファに追加
    data_buffer.append(new_data)
    counter = len(data_buffer)

    # バッチサイズを超えた場合、CSVファイルに追記してバッファをクリア
    if counter >= BATCH_SIZE:
        save_to_csv(data_buffer)
        data_buffer.clear()


def save_to_csv(data):
    global saved_expr_counter
    if OUTPUT_FILE is None:
        return
    # データをDataFrameに変換
    df = pd.DataFrame(data)
    saved_expr_counter += len(data)
    logging.info(
        "Output file is updated. There are %d exprs", saved_expr_counter
    )

    # ファイルが存在しない場合は新規作成、存在する場合は追記
    assert OUTPUT_FILE.exists(), "Output file does not exist"
    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)


def output_bytes_not_const(expr: Expr, eq_domain: list[npt.NDArray]) -> bytes:
    assert expr.arity is not None
    ans: npt.NDArray = np.array(
        tuple(expr.eval(*x) for x in eq_domain[expr.arity])
    )
    return ans.tobytes()


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[npt.NDArray]
) -> bytes:
    assert input_size >= 1
    return np.array(
        expr.eval(0) for _ in range(len(eq_domain[input_size]))
    ).tobytes()


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
) -> tuple[list[list[Expr]], list[set[bytes]], bool]:
    assert expr.arity is None
    is_visited = False
    for dim in range(1, max_dim + 1):
        try:
            # when OverflowError occurs, it occurs in all iteration
            out_bytes = output_bytes_const(expr, dim, eq_domain)
        except OverflowError:
            return exprs, outputs, False
        if out_bytes in outputs[dim]:
            continue
        outputs[dim].add(out_bytes)
        is_visited = True

    if not is_visited:
        exprs[0].append(expr)
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
    try:
        out_bytes = output_bytes_not_const(expr, eq_domain)
    except OverflowError:
        return exprs, outputs, False
    if out_bytes in outputs[expr.arity]:
        return exprs, outputs, False
    exprs[expr.arity].append(expr)
    outputs[expr.arity].add(out_bytes)
    return exprs, outputs, True


def generate_random_prfndim(
    max_count,
    max_generate_count,
    max_p_arity,
    max_c_args,
    max_r_args,
    eq_domain,
    output_path: Optional[Path],
    from_csv,
) -> list[Expr]:
    if output_path is not None:
        global OUTPUT_FILE
        if output_path.exists():
            output_path.unlink()
        output_path.touch()
        OUTPUT_FILE = output_path

    ret_exprs: list[Expr] = []
    gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)

    if from_csv:
        init_file = Path(
            "/home/takeru/AlphaSymbol/data/prfndim/d4-a3-c2-r3.csv"
        )
        gen_exprs_from_file, outputs_from_file = init_exprs_from_csv(
            init_file, max_p_arity, eq_domain
        )

        for i in range(0, max_p_arity + 1):
            gen_exprs[i].extend(gen_exprs_from_file[i])
            outputs[i].update(outputs_from_file[i])

    ret_exprs.extend([expr for exprs in gen_exprs for expr in exprs])

    for exprs in gen_exprs:
        for expr in exprs:
            add_data(expr)

    init_list: list[Expr] = []
    for i in range(0, max_p_arity + 1):
        init_list.extend(gen_exprs[i])

    for iter in range(max_count):
        # gen_exprs[arity]: list of Expr with arity
        gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
        if from_csv:
            gen_exprs_from_file, outputs_from_file = init_exprs_from_csv(
                init_file, max_p_arity, eq_domain
            )
            for i in range(0, max_p_arity + 1):
                gen_exprs[i].extend(gen_exprs_from_file[i])
                outputs[i].update(outputs_from_file[i])

        # visited[arity] set of output of Expr with arity
        counter = 0
        while counter < max_generate_count:
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
                    gen_exprs, outputs, is_updated = (
                        if_not_visited_then_update_const(
                            gen_exprs,
                            outputs,
                            new_expr_c,
                            max_p_arity,
                            eq_domain,
                        )
                    )
                    counter += 1 if is_updated else 0
                else:
                    gen_exprs, outputs, is_updated = (
                        if_not_visited_then_update_not_const(
                            gen_exprs, outputs, new_expr_c, eq_domain
                        )
                    )
                    counter += 1 if is_updated else 0
            else:
                while True:
                    try:
                        term_arity = random.randint(1, (max_r_args - 1) // 2)
                        step_arity = random.randint(
                            term_arity + 1, min(term_arity + 1, max_p_arity)
                        )
                        base_arity = random.randint(
                            0, step_arity - term_arity - 1
                        )
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
                    gen_exprs, outputs, is_updated = (
                        if_not_visited_then_update_const(
                            gen_exprs,
                            outputs,
                            new_expr_r,
                            max_p_arity,
                            eq_domain,
                        )
                    )
                    counter += 1 if is_updated else 0
                else:
                    gen_exprs, outputs, is_updated = (
                        if_not_visited_then_update_not_const(
                            gen_exprs, outputs, new_expr_r, eq_domain
                        )
                    )
                    counter += 1 if is_updated else 0

        new_list = []

        for i in range(0, max_p_arity + 1):
            new_list.extend(gen_exprs[i])

        for item in set(new_list) - set(init_list):
            add_data(item)

        ret_exprs.extend(list(set(new_list) - set(init_list)))
        logging.debug(f"Iter {iter}: {len(new_list) - len(init_list)} is added")

    save_to_csv(data_buffer)
    data_buffer.clear()

    ret_exprs.extend(init_list)
    return ret_exprs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate expressions randomly"
    )

    parser.add_argument("--iter", type=int)
    parser.add_argument("--sampling", type=int)
    parser.add_argument("-p", "--max_p_arity", type=int)
    parser.add_argument("-c", "--max_c_args", type=int)
    parser.add_argument("-r", "--max_r_args", type=int)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--sample_max", type=int, default=10)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--init_csv", action="store_true")

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

    exprs = generate_random_prfndim(
        args.iter,
        args.sampling,
        args.max_p_arity,
        args.max_c_args,
        args.max_r_args,
        eq_domain,
        output_path,
        args.init_csv,
    )
