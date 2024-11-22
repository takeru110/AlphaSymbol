import logging
import random
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z

logging.basicConfig(level=logging.DEBUG)


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
    from_csv,
) -> list[Expr]:
    ret_exprs: list[Expr] = []
    init_file = Path("/home/takeru/AlphaSymbol/data/prfndim/d4-a3-c2-r3.csv")

    gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
    if from_csv:
        gen_exprs_from_file, outputs_from_file = init_exprs_from_csv(
            init_file, max_p_arity, eq_domain
        )

        for i in range(0, max_p_arity + 1):
            gen_exprs[i].extend(gen_exprs_from_file[i])
            outputs[i].update(outputs_from_file[i])

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

        ret_exprs.extend(list(set(new_list) - set(init_list)))
        logging.info(f"Iter {iter}: {len(new_list) - len(init_list)} is added")
    ret_exprs.extend(init_list)
    return ret_exprs


if __name__ == "__main__":
    sample_num = 5
    sample_max = 10

    max_count = 10000
    max_generate_count = 5
    max_p_arity = 4
    max_c_args = 4
    max_r_args = 9

    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, sample_max + 1)
    ]
    eq_domain[1] = np.array(range(10)).reshape(10, 1)

    exprs = generate_random_prfndim(
        max_count,
        max_generate_count,
        max_p_arity,
        max_c_args,
        max_r_args,
        eq_domain,
        False,
    )

    df = pd.DataFrame(exprs)
    df.columns = ["expr"]

    df_unique = df.drop_duplicates(subset=["expr"])

    outputs_li = []
    inputs_li = []
    len_li = []
    arity_li = []
    for index, row in df_unique.iterrows():
        expr = row["expr"]
        if expr.arity is None:
            inputs = np.full((sample_num, 1), 0)
        else:
            inputs = eq_domain[expr.arity]
        outputs = tuple(expr.eval(*args) for args in inputs)
        inputs_li.append(
            tuple(tuple(int(arg) for arg in args) for args in inputs)
        )
        outputs_li.append(tuple(int(output) for output in outputs))
        len_li.append(len(str(expr)))
        arity_li.append(expr.arity)
    df_unique.loc[:, "inputs"] = inputs_li
    df_unique.loc[:, "outputs"] = outputs_li
    df_unique.loc[:, "len"] = len_li
    df_unique.loc[:, "arity"] = arity_li

    df_unique_output = df_unique.drop_duplicates(subset=["arity", "outputs"])
    df_unique_output = df_unique_output.sort_values(
        by=["arity", "len"]
    ).reset_index(drop=True)
    print(df_unique_output)

    df_unique_output.to_csv("./data/prfndim/unique_output.csv", index=True)
