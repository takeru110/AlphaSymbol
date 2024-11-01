from itertools import product
from typing import Any, List

import numpy as np
import pandas as pd

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
    - max_depth (int): The depth of recursion for generating expressions.
    - max_p_arity: (int): Maximum number of arity of P.
    - max_c_args (int): Maximum number of arguments allowed for C.

    Returns:
    - List[List[List[Expr]]]: Dictionary of generated PRF expressions grouped by their arity and depth.
        ret[depth][arity] is a list of PRF expressions of depth `depth` and arity `arity`.
    """
    if max_depth == 0:
        return [[]]

    if max_depth == 1:
        depth1_exprs: list[list[Expr]] = [[Z()], [S(), P(1, 1)]]

        for arity in range(2, max_p_arity + 1):
            li_same_arity: list[Expr] = [P(arity, i) for i in range(1, arity + 1)]
            depth1_exprs.append(li_same_arity)
        return [[], depth1_exprs]

    pre_exprs = generate_valid_exprs(max_depth - 1, max_p_arity, max_c_args)
    pre_max_d_exprs = pre_exprs[max_depth - 1]

    max_d_exprs: List[List[Expr]] = [[] for _ in range(max_p_arity + 1)]

    pre_max_by_arity: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
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
    base_arity_list: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
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

    ret: List[List[List[Expr]]] = pre_exprs.copy()
    ret.append(max_d_exprs)
    return ret


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
):
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
    return


def generate_exprs_unique_output(
    max_depth: int,
    max_p_arity: int,
    max_c_args: int,
    eq_domain: list[int],
    visited: List[set[Any]],
) -> tuple[List[List[List[Expr]]], List[set[Any]]]:
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
                tuple([Z().evaluate(x) for x in eq_domain]),
                tuple([S().evaluate(x) for x in eq_domain]),
                tuple([P(1, 1).evaluate(x) for x in eq_domain]),
            ]
        )

        for arity in range(2, max_p_arity + 1):
            for i in range(1, arity + 1):
                append_if_unique(
                    P(arity, i), arity, eq_domain, depth1_exprs[arity], visited
                )
        return [[], depth1_exprs], visited

    pre_exprs, visited = generate_exprs_unique_output(
        max_depth - 1, max_p_arity, max_c_args, eq_domain, visited
    )
    pre_max_d_exprs = pre_exprs[max_depth - 1]

    max_d_exprs: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]

    pre_max_by_arity: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]

    for depth in range(len(pre_exprs)):
        for arity in range(len(pre_exprs[depth])):
            pre_max_by_arity[arity].extend(pre_exprs[depth][arity])

    # pattern of C(depth is n-1, *)
    for base_arity, base_funcs in enumerate(pre_max_d_exprs[:max_c_args]):
        if base_arity == 0:
            continue

        # scanning by args arity (== output arity)
        for args_arity, same_arity_exprs in enumerate(pre_max_by_arity):
            for base in base_funcs:
                for args in product(
                    same_arity_exprs + pre_max_by_arity[0], repeat=base_arity
                ):
                    if args_arity == 0:
                        for value_arity in range(1, max_p_arity + 1):
                            # append if C(base, *args) is not duplicated as arity=="arity" function.
                            append_if_unique(
                                C(base, *args),
                                value_arity,
                                eq_domain,
                                max_d_exprs[args_arity],
                                visited,
                            )
                    elif not all((arg.arity() is None) for arg in args):
                        append_if_unique(
                            C(base, *args),
                            args_arity,
                            eq_domain,
                            max_d_exprs[args_arity],
                            visited,
                        )

    # pattern of C(*, depth is n-1)
    base_arity_list: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
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
            for base in base_funcs:
                for args in product(
                    same_arity_exprs + none_arity_exprs, repeat=base_arity
                ):
                    if args_arity == 0:
                        for val_arity in range(1, max_p_arity + 1):
                            append_if_unique(
                                C(base, *args),
                                val_arity,
                                eq_domain,
                                max_d_exprs[args_arity],
                                visited,
                            )
                    elif not all((arg.arity() is None) for arg in args):
                        append_if_unique(
                            C(base, *args),
                            args_arity,
                            eq_domain,
                            max_d_exprs[args_arity],
                            visited,
                        )

    # Pattern R(None max, >=2)

    # Expr lists which are arity none and depth is max_depth - 1
    none_max_d_exprs = pre_exprs[max_depth - 1][0]

    for steps_by_depth in pre_exprs:
        for step_arity, steps in enumerate(steps_by_depth):
            if step_arity < 2:
                continue
            for base in none_max_d_exprs:
                for step in steps:
                    append_if_unique(
                        R(base, step),
                        step_arity - 1,
                        eq_domain,
                        max_d_exprs[step_arity - 1],
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
        for base in none_max_d_exprs:
            for _step in same_arity_steps:
                append_if_unique(
                    R(base, _step),
                    step_arity - 1,
                    eq_domain,
                    max_d_exprs[step_arity - 1],
                    visited,
                )

    # Pattern R(i, i+2) (i = 1, 2, ...)
    for i in range(1, max_p_arity - 1):
        for base in pre_exprs[max_depth - 1][i]:
            for step in pre_max_by_arity[i + 2]:
                append_if_unique(
                    R(base, step),
                    i + 1,
                    eq_domain,
                    max_d_exprs[i + 1],
                    visited,
                )
        for base in pre_max_by_arity[i]:
            for step in pre_exprs[max_depth - 1][i + 2]:
                append_if_unique(
                    R(base, step),
                    i + 1,
                    eq_domain,
                    max_d_exprs[i + 1],
                    visited,
                )

    # delete duplicated elements
    for i in range(max_p_arity + 1):
        max_d_exprs[i] = list(set(max_d_exprs[i]))

    ret: List[List[List[Expr]]] = pre_exprs.copy()
    ret.append(max_d_exprs)
    return ret, visited


def generate_expression_table(
    max_depth: int, max_p_arity: int, max_c_args: int, arity: int, inputs: List[Any]
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
    exprs_by_depth_and_arity = generate_valid_exprs(max_depth, max_p_arity, max_c_args)

    data = []

    for depth in range(len(exprs_by_depth_and_arity)):
        if depth == 0:
            continue
        exprs = set(
            exprs_by_depth_and_arity[depth][0] + exprs_by_depth_and_arity[depth][arity]
        )
        for expr in exprs:
            # 各式について必要な情報を取得
            is_valid = expr.validate_semantic()
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
    df_expr_table = generate_expression_table(
        max_depth=2, max_p_arity=2, max_c_args=2, arity=1, inputs=[1, 2, 3]
    )
    df_expr_table.to_csv(
        "./data/d2-p2-c2.csv",
        index=False,
        encoding="utf-8-sig",
    )  # UTF-8でエンコードして保存
    print("CSV file is generated: expression_table.csv")
