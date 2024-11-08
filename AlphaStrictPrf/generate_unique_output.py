from itertools import product
from typing import Any, List

import numpy as np
import pandas as pd

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z


def list_to_tuple(lst):
    if isinstance(lst, list):
        # 各要素を再帰的にタプルに変換
        return tuple(list_to_tuple(sub) for sub in lst)
    else:
        return lst


def create_ndim_array(base_array, d, func) -> tuple:
    """
    任意の次元の多次元配列を生成し、各要素はd個の引数をもつ関数の結果とする。

    Args:
    - base_array (list): 1次元の配列 (長さn)
    - d (int): 次元数
    - func (callable): d個の引数を持つ関数

    Returns:
    - list: n^dのd次元配列
    """
    assert d >= 1, "d must be greater than or equal to 1"
    n = len(base_array)
    # インデックスの組み合わせを生成
    indices_combinations = product(range(n), repeat=d)

    # 各組み合わせに対して関数を適用し、多次元リストに格納
    result = [func(*indices) for indices in indices_combinations]
    ret_li = np.array(result).reshape((n,) * d).tolist()
    return list_to_tuple(ret_li)


def append_if_unique(
    expr: Expr,
    arity: int,
    eq_domain: List[int],
    exprs: List[Expr],
    visited: list[set[tuple[Any]]],
) -> tuple[List[Expr], list[set[tuple[Any]]]]:
    """
    if the output of expr in eq_domain is not in visited array, append expr to exprs.
    Args:
    - expr (Expr): PRF expression to be appended.
    - arity (int): Arity of "expr".
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    - exprs (List[Expr]): List of PRF expressions to be updated.
    - visited (list[set[Any]]): List of visited outputs.
    Return:
    - None: But, visited and exprs are updated.
    """
    assert arity >= 1, "arity must be greater than or equal to 1"
    output = create_ndim_array(eq_domain, arity, expr.evaluate)
    if output not in visited[arity]:
        exprs.append(expr)
        visited[arity].add(output)
    return exprs, visited


def append_c_pattern(
    new_exprs: list[list[Expr]],
    visited: list[set[tuple[Any]]],
    bases: list[Expr],
    bases_arity: int,
    args: list[Expr],
    args_arity: int,
    const_args: list[Expr],
    eq_domain: list[int],
) -> tuple[list[list[Expr]], list[set[tuple[Any]]]]:
    """append C(base, *args)
    Append PRF expressions of the form C(base, *arg) to new_exprs if the output is not in visited.
    This function is used when args_arity > 0.

    Args
    - new_exprs (list[list[Expr]]): List of PRF expressions max_depth to be updated.
        - new_exprs[Arity]
    - visited (list[set[tuple[Any]]]): List of visited outputs.
    - bases (list[Expr]): List of base functions.
    - bases_arity (int): Arity of base functions.
    - args (list[Expr]): List of arguments.
    - args_arity (int): Arity of arguments.
    - const_args (list[Expr]): List of constant arguments.
    - eq_domain (list[int]): List of inputs to defining semantic equivalence of Exprs.

    Return
    - new_exprs (list[list[Expr]]): Updated list of PRF expressions.
    - visited (list[set[tuple[Any]]]): Updated list of visited outputs.
    """
    assert args_arity >= 1, "args_arity must be greater than or equal to 1"
    for base, args in product(
        bases, product(args + const_args, repeat=bases_arity)
    ):
        if not all((arg.arity() is None) for arg in args):
            new_exprs[args_arity], visited = append_if_unique(
                C(base, *args),
                args_arity,
                eq_domain,
                new_exprs[args_arity],
                visited,
            )
    return new_exprs, visited


def append_c_pattern_const_args(
    new_exprs: list[list[Expr]],
    visited: list[set[tuple[Any]]],
    bases: list[Expr],
    bases_arity: int,
    args: list[Expr],
    const_args: list[Expr],
    eq_domain: list[int],
) -> tuple[list[list[Expr]], list[set[tuple[Any]]]]:
    """append C(base, *args)
    Append PRF expressions of the form C(base, *arg) to new_exprs if the output is not in visited.
    this function is for the case of args' arity == 0.

    Args
    - new_exprs (list[list[Expr]]): List of PRF expressions of max depth to be updated.
        - new_exprs[Arity]
    - visited (list[set[tuple[Any]]]): List of visited outputs.
    - bases (list[Expr]): List of base functions.
    - bases_arity (int): Arity of base functions.
    - args (list[Expr]): List of arguments.
    - const_args (list[int]): List of constant arguments.
    - eq_domain (list[int]): List of inputs to defining semantic equivalence of Exprs.

    Return
    - new_exprs (list[list[Expr]]): Updated list of PRF expressions.
    - visited (list[set[tuple[Any]]]): Updated list of visited outputs.
    """
    for base, args in product(
        bases, product(args + const_args, repeat=bases_arity)
    ):
        for value_arity in range(1, max_p_arity + 1):
            new_exprs[0], visited = append_if_unique(
                C(base, *args),
                value_arity,
                eq_domain,
                new_exprs[0],
                visited,
            )
    return new_exprs, visited


def generate_exprs_unique_output(
    max_depth: int,
    max_p_arity: int,
    max_c_args: int,
    eq_domain: list[int],
    visited: list[set[Any]],
) -> tuple[list[list[list[Expr]]], list[set[Any]]]:
    """
    Generate all possible PRF expressions up to a given depth.

    Args:
    - max_depth (int): The depth of recursion for generating expressions.
    - max_p_arity: (int): Maximum number of arity of P.
    - max_c_args (int): Maximum number of arguments allowed for C.
    - eq_domains: List of inputs to defining semantic equivalence of Exprs.
    - visited: Set of visited outputs.

    Returns:
    - List[List[List[Expr]]]: Dictionary of generated PRF expressions grouped by their arity and depth.
        ret[depth][arity] is a list of PRF expressions of depth `depth` and arity `arity`.
    """

    assert max_depth >= 1, "max_depth must be greater than or equal to 1"

    if max_depth == 1:
        depth1_exprs: list[list[Any]] = [[] for _ in range(max_p_arity + 1)]
        depth1_exprs[0] = [Z()]
        depth1_exprs[1] = [S(), P(1, 1)]
        visited[1].update(
            [
                tuple([S().evaluate(x) for x in eq_domain]),
                tuple([P(1, 1).evaluate(x) for x in eq_domain]),
            ]
        )

        for arity in range(1, max_p_arity + 1):
            # arity==0 append "visited" abnormally
            zero_super_cube: tuple[Any] = list_to_tuple(
                np.zeros((len(eq_domain),) * arity).tolist()
            )
            visited[arity].add(zero_super_cube)

        for arity in range(2, max_p_arity + 1):
            for i in range(1, arity + 1):
                depth1_exprs[arity], visited = append_if_unique(
                    P(arity, i), arity, eq_domain, depth1_exprs[arity], visited
                )
        return [[], depth1_exprs], visited

    # pre_exprs[depth][arity] is a list of PRF expressions of depth `depth` and arity `arity` for return.
    pre_exprs, visited = generate_exprs_unique_output(
        max_depth - 1, max_p_arity, max_c_args, eq_domain, visited
    )

    ret_exprs: list[list[list[Expr]]] = pre_exprs.copy()
    ret_exprs.append([[] for _ in range(max_p_arity + 1)])
    # max_d_exprs[arity] is a list of PRF expressions of depth `max_depth` and arity `arity` for return.

    # rearrange pre_exprs by arity
    pre_max_by_arity: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
    for depth in range(len(pre_exprs)):
        for arity in range(len(pre_exprs[depth])):
            pre_max_by_arity[arity].extend(pre_exprs[depth][arity])

    # pattern of C(depth is n-1, *)
    for base_arity, base_funcs in enumerate(
        pre_exprs[-1][1:max_c_args], start=1
    ):
        for args_arity, same_arity_exprs in enumerate(pre_max_by_arity):
            if args_arity == 0:
                ret_exprs[-1], visited = append_c_pattern_const_args(
                    new_exprs=ret_exprs[-1],
                    visited=visited,
                    bases=base_funcs,
                    bases_arity=base_arity,
                    args=same_arity_exprs,
                    const_args=pre_max_by_arity[0],
                    eq_domain=eq_domain,
                )
            else:
                ret_exprs[-1], visited = append_c_pattern(
                    new_exprs=ret_exprs[-1],
                    visited=visited,
                    bases=base_funcs,
                    bases_arity=base_arity,
                    args=same_arity_exprs,
                    args_arity=args_arity,
                    const_args=pre_max_by_arity[0],
                    eq_domain=eq_domain,
                )

    # pattern of C(*, depth is n-1)
    base_arity_list: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
    for depth in range(len(pre_exprs)):
        for arity in range(len(pre_exprs[depth])):
            base_arity_list[arity].extend(pre_exprs[depth][arity])

    none_arity_exprs: List[Expr] = []
    for same_arity in pre_exprs[1:]:
        none_arity_exprs.extend(same_arity[0])

    for base_arity, base_funcs in enumerate(
        base_arity_list[1:max_c_args], start=1
    ):
        for args_arity, same_arity_exprs in enumerate(pre_exprs[-1]):
            if args_arity == 0:
                ret_exprs[-1], visited = append_c_pattern_const_args(
                    new_exprs=ret_exprs[-1],
                    visited=visited,
                    bases=base_funcs,
                    bases_arity=base_arity,
                    args=same_arity_exprs,
                    const_args=none_arity_exprs,
                    eq_domain=eq_domain,
                )
            else:
                ret_exprs[-1], visited = append_c_pattern(
                    new_exprs=ret_exprs[-1],
                    visited=visited,
                    bases=base_funcs,
                    bases_arity=base_arity,
                    args=same_arity_exprs,
                    args_arity=args_arity,
                    const_args=none_arity_exprs,
                    eq_domain=eq_domain,
                )
    # Pattern R(None max, >=2)

    for steps_by_depth in pre_exprs:
        for step_arity, steps in enumerate(steps_by_depth):
            if step_arity < 2:
                continue
            for base in pre_exprs[-1][0]:
                for step in steps:
                    ret_exprs[-1][step_arity - 1], visited = append_if_unique(
                        R(base, step),
                        step_arity - 1,
                        eq_domain,
                        ret_exprs[-1][step_arity - 1],
                        visited,
                    )

    # make none arity exprs
    none_arity_exprs = []
    for same_arity in pre_exprs[1:]:
        none_arity_exprs.extend(same_arity[0])

    # R(None, >=2 max)
    for step_arity, same_arity_steps in enumerate(pre_exprs[max_depth - 1]):
        if step_arity < 2:
            continue
        for base in pre_exprs[-1][0]:
            for _step in same_arity_steps:
                ret_exprs[-1][step_arity - 1], visited = append_if_unique(
                    R(base, _step),
                    step_arity - 1,
                    eq_domain,
                    ret_exprs[-1][step_arity - 1],
                    visited,
                )

    # Pattern R(i, i+2) (i = 1, 2, ...)
    for i in range(1, max_p_arity - 1):
        for base in pre_exprs[-1][i]:
            for step in pre_max_by_arity[i + 2]:
                ret_exprs[-1][i + 1], visited = append_if_unique(
                    R(base, step),
                    i + 1,
                    eq_domain,
                    ret_exprs[-1][i + 1],
                    visited,
                )
        for base in pre_max_by_arity[i]:
            for step in pre_exprs[-1][i + 2]:
                ret_exprs[-1][i + 1], visited = append_if_unique(
                    R(base, step),
                    i + 1,
                    eq_domain,
                    ret_exprs[-1][i + 1],
                    visited,
                )

    # delete duplicated elements
    for i in range(max_p_arity + 1):
        ret_exprs[-1][i] = list(set(ret_exprs[-1][i]))

    return ret_exprs, visited


def generate_expression_table(
    max_depth: int,
    max_p_arity: int,
    max_c_args: int,
    arity: int,
    inputs: List[Any],
) -> pd.DataFrame:
    """
    This function generats DataFrame of semantically-valid SPF expressions and their status.
    Args:
    - max_depth (int): The depth of recursion for generating expressions.
    - max_p_arity: (int): Maximum number of arity of P.
    - max_c_args (int): Maximum number of arguments allowed for C.
    - arity (int): Arity of the target expressions. This is needed for generating outputs from inputs
    - inputs (List[Any]): List of inputs for generating outputs.
    """
    eq_domain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    visited: list[set[tuple[Any]]] = [set() for _ in range(max_p_arity + 1)]
    exprs_by_depth_and_arity, visited = generate_exprs_unique_output(
        max_depth, max_p_arity, max_c_args, eq_domain, visited
    )

    data: list[dict[str, Any]] = []

    for depth in range(len(exprs_by_depth_and_arity)):
        if depth == 0:
            continue
        exprs: set[Expr] = set(
            exprs_by_depth_and_arity[depth][0]
            + exprs_by_depth_and_arity[depth][arity]
        )
        for expr in exprs:
            # 各式について必要な情報を取得
            is_valid = expr.is_valid()
            outputs = [expr.evaluate(x) for x in inputs]  # 出力を計算

            data.append(
                {
                    "SPF": str(expr),
                    "valid": is_valid,
                    "depth": depth,
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )

    # DataFrameを生成
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # 使用例
    max_depth = 4
    max_p_arity = 2
    max_c_args = 2
    df_expr_table = generate_expression_table(
        max_depth=max_depth,
        max_p_arity=max_p_arity,
        max_c_args=max_c_args,
        arity=1,
        inputs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    df_expr_table.to_csv(
        f"./data/d{max_depth}-p{max_p_arity}-c{max_c_args}-unique.csv",
        index=False,
        encoding="utf-8-sig",
    )  # UTF-8でエンコードして保存
    print(
        f"CSV file is generated: d{max_depth}-p{max_p_arity}-c{max_c_args}-unique.csv"
    )
