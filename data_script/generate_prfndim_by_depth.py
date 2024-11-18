import logging
from itertools import product

import numpy as np
import numpy.typing as npt

from prfndim.prfndim import C, Expr, P, R, S, Z, expr_list_to_str


def output_bytes_not_const(expr: Expr, eq_domain: list[npt.NDArray]) -> bytes:
    assert expr.arity is not None
    ans: npt.NDArray = np.array(
        tuple(expr.eval(x) for x in eq_domain[expr.arity])
    ).flatten()
    logging.debug("output_bytes_not_const: %s", ans)
    return ans.tobytes()


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[npt.NDArray]
) -> bytes:
    assert input_size >= 1
    return np.array(
        expr.eval(0) for _ in range(len(eq_domain[input_size]))
    ).tobytes()


def one_depth_exprs(
    max_arity: int, eq_domain: list[npt.NDArray]
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    exprs: list[list[list[Expr]]] = [[], [[] for _ in range(max_arity + 1)]]
    visited: list[set[bytes]] = [set() for _ in range(max_arity + 1)]
    exprs[1][0].append(Z())
    for input_size in range(1, max_arity + 1):
        visited[input_size].add(output_bytes_const(Z(), input_size, eq_domain))
    exprs[1][1].append(S())
    visited[1].add(output_bytes_not_const(S(), eq_domain))
    for i in range(1, max_arity + 1):
        for j in range(1, i + 1):
            exprs[1][i].append(P(i, j))
            visited[i].add(output_bytes_not_const(P(i, j), eq_domain))
    return exprs, visited


def gen_exprs_by_arity(exprs):
    ret_li = [[] for _ in range(len(exprs[1]))]
    for exprs_by_arity in exprs:
        for arity in range(len(exprs_by_arity)):
            ret_li[arity].extend(exprs_by_arity[arity])
    return ret_li


def c_pattern(exprs_by_arity, max_arity, max_c):
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
        new_args_list = tuple(product(*((exprs_by_arity[arg]) for arg in args)))
        for new_args in zip(new_args_list):
            new_arity = max(args[1:])
            c_exprs[new_arity].extend([C(*args) for args in new_args])
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


def r_pattern(exprs_by_arity, max_arity, max_r):
    t_a_b_list = [
        (n, d + n + 1, d)
        for d in range(0, max_arity - 1)
        for n in range(1, min(max_arity - d - 1, (max_r - 1) // 2) + 1)
    ]
    logging.debug(f"t_a_b: {t_a_b_list}")

    args_list = [
        (t, *a_s, *b_s)
        for t, a, b in t_a_b_list
        for a_s in product((0, a), repeat=t)
        for b_s in product((b,), repeat=t)
    ]

    logging.debug(f"args_list: {args_list}")

    r_exprs: list[list[Expr]] = [[] for _ in range(max_arity + 1)]
    for args in args_list:
        expr_list = tuple(product(*(exprs_by_arity[arg] for arg in args)))
        arity = get_r_arity(args)
        r_exprs[arity].extend([R(*args) for args in expr_list])
    return r_exprs


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

    c_exprs = c_pattern(exprs_by_arity, max_arity, max_c)

    for expr in c_exprs[0]:
        exprs[depth], visited, _ = if_not_visited_then_update_const(
            exprs[depth], visited, expr, max_arity, eq_domain
        )

    for _exprs in c_exprs[1:]:
        for expr in _exprs:
            exprs[depth], visited, _ = if_not_visited_then_update_not_const(
                exprs[depth], visited, expr, eq_domain
            )

    r_exprs = r_pattern(exprs_by_arity, max_arity, max_r)

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
