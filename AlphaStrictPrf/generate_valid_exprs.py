from itertools import product
from typing import List

from AlphaStrictPrf.strict_prf import C, Expr, P, S, Z


def expr_list_to_str(exprs):
    if isinstance(exprs, Expr):
        return str(exprs)
    return [expr_list_to_str(expr) for expr in exprs]


def generate_valid_exprs(
    max_depth: int,
    max_p_arity: int,
    max_c_args: int,
) -> List[List[Expr]]:
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

    ret = []
    ret.extend(pre_max_d_exprs)
    ret.append([])

    max_d_exprs = []
    # pattern of C(depth is n-1, *)
    for base_arity, base_funcs in enumerate(pre_max_d_exprs):
        if base_arity == 0:
            continue

        args_candidates_by_arity = [[] for _ in range(max_p_arity + 1)]
        for depth in range(len(pre_exprs)):
            for arity in range(len(pre_exprs[depth])):
                args_candidates_by_arity[arity].extend(pre_exprs[depth][arity])

        # scanning by args arity (== output arity)
        for args_arity, same_arity_exprs in enumerate(args_candidates_by_arity):
            for args in product(same_arity_exprs, repeat=base_arity):
                for base in base_funcs:
                    continue

            same_arity_depth = [
                C(base, *args)
                for base in base_funcs
                for args in product(same_arity_exprs, repeat=base_arity)
            ]
            ret[max_depth].append(same_arity_depth)

    return [[], depth1_exprs]
