import logging
import random

import numpy as np
import numpy.typing as npt
import pandas as pd

from prfndim.prfndim import C, Expr, P, R, S, Z

logging.basicConfig(level=logging.DEBUG)


def output_bytes_not_const(expr: Expr, eq_domain: list[npt.NDArray]) -> bytes:
    assert expr.arity is not None
    return np.array(expr.eval(x) for x in eq_domain[expr.arity]).tobytes()


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
    exprs[1].append(S())
    outputs[1].add(output_bytes_not_const(S(), eq_domain))
    for i in range(1, max_p_arity + 1):
        for j in range(1, i + 1):
            exprs[i].append(P(i, j))
            outputs[i].add(output_bytes_not_const(P(i, j), eq_domain))
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
        out_bytes = output_bytes_const(expr, dim, eq_domain)
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
    out_bytes = output_bytes_not_const(expr, eq_domain)
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
) -> list[Expr]:
    ret_exprs: list[Expr] = []
    gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
    for i in range(0, max_p_arity + 1):
        ret_exprs.extend(gen_exprs[i])

    for iter in range(max_count):
        # gen_exprs[arity]: list of Expr with arity
        gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
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

        gen_exprs, outputs = init_exprs(max_p_arity, eq_domain)
        init_list: list[Expr] = []
        for i in range(0, max_p_arity + 1):
            init_list.extend(gen_exprs[i])
        ret_exprs.extend(list(set(new_list) - set(init_list)))
        logging.info(f"Iter {iter}: {len(new_list) - len(init_list)} is added")
    return ret_exprs


if __name__ == "__main__":
    sample_num = 5
    sample_max = 10

    max_count = 1000
    max_generate_count = 10
    max_p_arity = 4
    max_c_args = 4
    max_r_args = 7
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(dim, sample_num))
        for dim in range(1, sample_max + 1)
    ]

    exprs = generate_random_prfndim(
        max_count,
        max_generate_count,
        max_p_arity,
        max_c_args,
        max_r_args,
        eq_domain,
    )

    df = pd.DataFrame(exprs)
    df.columns = ["expr"]
    df.to_csv("./data/prfndim/raw_prfndim.csv", index=False)
