import logging
import random

import numpy as np
import pandas as pd

from prfndim.prfndim import C, Expr, P, R, S, Z

logging.basicConfig(level=logging.DEBUG)


def init_exprs(max_p_arity):
    exprs: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
    exprs[0].append(Z())
    exprs[1].append(S())
    for i in range(1, max_p_arity + 1):
        for j in range(1, i + 1):
            exprs[i].append(P(i, j))
    return exprs


def generate_random_prfndim(
    max_count,
    max_generate_count,
    max_p_arity,
    max_c_args,
    max_r_args,
    eq_domain,
    visited,
) -> list[Expr]:
    ret_exprs: list[Expr] = []
    for _ in range(max_count):
        gen_exprs: list[list[Expr]] = init_exprs(max_p_arity)
        for _ in range(max_generate_count):
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
                    gen_exprs[0].append(new_expr_c)
                else:
                    gen_exprs[new_expr_c.arity].append(new_expr_c)
                ret_exprs.append(new_expr_c)
                logging.debug(f"append {new_expr_c}")
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
                    gen_exprs[0].append(new_expr_r)
                else:
                    gen_exprs[new_expr_r.arity].append(new_expr_r)
                ret_exprs.append(new_expr_r)
                logging.debug(f"append {new_expr_r}")
    init_exprs_li: list[list[Expr]] = init_exprs(max_p_arity)
    ret_exprs.extend([item for lists in init_exprs_li for item in lists])
    return ret_exprs


if __name__ == "__main__":
    sample_num = 5
    sample_max = 10

    max_count = 10000
    max_generate_count = 20
    max_p_arity = 4
    max_c_args = 4
    max_r_args = 5
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(dim, sample_num))
        for dim in range(1, sample_max + 1)
    ]
    visited = [set() for _ in range(max_p_arity + 1)]

    exprs = generate_random_prfndim(
        max_count,
        max_generate_count,
        max_p_arity,
        max_c_args,
        max_r_args,
        eq_domain,
        visited,
    )

    df = pd.DataFrame(exprs)
    df.to_csv("./data/prfndim/raw_prfndim.csv", index=False)
