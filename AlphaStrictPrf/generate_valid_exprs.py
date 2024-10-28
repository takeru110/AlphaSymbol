from itertools import product
from typing import List

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z


def expr_list_to_str(exprs):
    if isinstance(exprs, Expr):
        return str(exprs)
    return [expr_list_to_str(expr) for expr in exprs]


def generate_valid_exprs(
    max_depth: int,
    max_p_arity: int,
    max_c_args: int,
) -> List[List[List[Expr]]]:
    """
    Generate all possible PRF expressions up to a given depth.

    Args:
    - depth (int): The depth of recursion for generating expressions.
    - max_c_args (int): Maximum number of arguments allowed for C.

    Returns:
    - Dict[int, List[Expr]]: Dictionary of generated PRF expressions grouped by their arity.
    """
    if max_depth == 0:
        return [[]]

    if max_depth == 1:
        depth1_exprs = [[Z()], [S(), P(1, 1)]]

        for arity in range(2, max_p_arity + 1):
            li_same_arity = [P(arity, i) for i in range(1, arity + 1)]
            depth1_exprs.append(li_same_arity)
        return [[], depth1_exprs]

    pre_exprs = generate_valid_exprs(max_depth - 1, max_p_arity, max_c_args)
    pre_max_d_exprs = pre_exprs[max_depth - 1]

    max_d_exprs: List[List[Expr]] = [[] for _ in range(max_p_arity + 1)]

    pre_max_by_arity = [[] for _ in range(max_p_arity + 1)]
    for depth in range(len(pre_exprs)):
        for arity in range(len(pre_exprs[depth])):
            pre_max_by_arity[arity].extend(pre_exprs[depth][arity])

    # pattern of C(depth is n-1, *)
    for base_arity, base_funcs in enumerate(pre_max_d_exprs[:max_c_args]):
        if base_arity == 0:
            continue

        # scanning by args arity (== output arity)
        for args_arity, same_arity_exprs in enumerate(pre_max_by_arity):
            same_arity_depth = [
                C(base, *args)
                for base in base_funcs
                for args in product(
                    same_arity_exprs + pre_max_by_arity[0], repeat=base_arity
                )
                if args_arity == 0 or not all((arg.arity() is None) for arg in args)
            ]
            max_d_exprs[args_arity].extend(same_arity_depth)

    # pattern of C(*, depth is n-1)
    base_arity_list = [[] for _ in range(max_p_arity + 1)]
    for depth in range(len(pre_exprs)):
        for arity in range(len(pre_exprs[depth])):
            base_arity_list[arity].extend(pre_exprs[depth][arity])

    none_arity_exprs: List[Expr] = []
    for same_arity in pre_exprs[1:]:
        none_arity_exprs.extend(same_arity[0])

    for base_arity, base_funcs in enumerate(base_arity_list[:max_c_args]):
        if base_arity == 0:
            continue

        for args_arity, same_arity_exprs in enumerate(pre_max_d_exprs):
            same_arity_depth = [
                C(base, *args)
                for base in base_funcs
                for args in product(
                    same_arity_exprs + none_arity_exprs, repeat=base_arity
                )
                if args_arity == 0 or not all((arg.arity() is None) for arg in args)
            ]
            max_d_exprs[args_arity].extend(same_arity_depth)

    # Pattern R(None max, >=2)

    # Expr lists which are arity none and depth is max_depth - 1
    none_max_d_exprs = pre_exprs[max_depth - 1][0]

    for steps_by_depth in pre_exprs:
        for step_arity, steps in enumerate(steps_by_depth):
            if step_arity < 2:
                continue
            max_d_exprs[step_arity - 1].extend(
                [R(base, step) for base in none_max_d_exprs for step in steps]
            )

    # make none arity exprs
    none_arity_exprs = []
    for same_arity in pre_exprs[1:]:
        none_arity_exprs.extend(same_arity[0])

    # R(None, >=2 max)
    for step_arity, same_arity_steps in enumerate(pre_exprs[max_depth - 1]):
        if step_arity < 2:
            continue
        max_d_exprs[step_arity - 1].extend(
            [R(base, _step) for base in none_arity_exprs for _step in same_arity_steps]
        )

    # Pattern R(i, i+2) (i = 1, 2, ...)
    for i in range(1, max_p_arity - 1):
        new_exprs1 = [
            R(base, step)
            for base in pre_exprs[max_depth - 1][i]
            for step in pre_max_by_arity[i + 2]
        ]
        max_d_exprs[i + 1].extend(new_exprs1)
        new_exprs2 = [
            R(base, step)
            for base in pre_max_by_arity[i]
            for step in pre_exprs[max_depth - 1][i + 2]
        ]
        max_d_exprs[i + 1].extend(new_exprs2)

    # delete duplicated elements
    for i in range(max_p_arity + 1):
        max_d_exprs[i] = list(set(max_d_exprs[i]))

    ret: List[List[Expr]] = pre_exprs.copy()
    ret.append(max_d_exprs)
    return ret
