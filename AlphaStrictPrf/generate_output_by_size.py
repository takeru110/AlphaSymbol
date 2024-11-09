from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z, expr_to_str_rec


def list_to_tuple(lst):
    if isinstance(lst, list):
        # 各要素を再帰的にタプルに変換
        return tuple(list_to_tuple(sub) for sub in lst)
    else:
        return lst


def create_ndim_array(base_array, d, func) -> NDArray:
    """
    任意の次元の多次元配列を生成し、各要素はd個の引数をもつ関数の結果とする。

    Args:
    - base_array (list): 1次元の配列 (長さn)
    - d (int): 次元数
    - func (callable): d個の引数を持つ関数

    Returns:
    - tuple: n^dのd次元配列
    """
    assert d >= 1, "d must be greater than or equal to 1"
    n = len(base_array)
    # インデックスの組み合わせを生成
    indices_combinations = product(base_array, repeat=d)

    # 各組み合わせに対して関数を適用し、多次元リストに格納
    result = [func(*indices) for indices in indices_combinations]
    return np.array(result)


def ndim_output_map(base_array: list[int], d: int, f: Callable) -> NDArray:
    meshgrid = np.meshgrid(*([base_array] * d), indexing="ij")
    array = np.array(meshgrid)
    transposed = array.transpose(*range(1, d + 1), 0)
    flatten = transposed.reshape(-1, d)
    results = np.array([f(*t) for t in flatten])
    results_reshaped = results.reshape((len(base_array),) * d)
    return results_reshaped


def if_visited_then_append_for_const(
    exprs: list[Expr],
    visited: list[set[bytes]],
    expr: Expr,
    max_p_arity: int,
    eq_domain: list[int],
) -> tuple[list[Expr], list[set[bytes]]]:
    """
    check if the output of constant function is already visited
    and update "exprs" and "visited".
    """
    is_new = False
    for num_inputs in range(1, max_p_arity + 1):
        output_array = ndim_output_map(eq_domain, num_inputs, expr.evaluate)
        if output_array.tobytes() not in visited[num_inputs]:
            visited[num_inputs].add(output_array.tobytes())
            is_new = True
    if is_new:
        exprs.append(expr)
        return exprs, visited
    return exprs, visited


def if_visited_then_append(
    exprs: list[Expr],
    visited: list[set[bytes]],
    expr: Expr,
    expr_arity,
    eq_domain: list[int],
) -> tuple[list[Expr], list[set[bytes]]]:
    output = ndim_output_map(eq_domain, expr_arity, expr.evaluate)
    if output.tobytes() not in visited[expr_arity]:
        visited[expr_arity].add(output.tobytes())
        exprs.append(expr)
    return exprs, visited


def update_c_pattern_const(
    exprs: list[list[list[Expr]]],
    visited: list[set[bytes]],
    base_arity: int,
    base_size: int,
    output_size: int,
    max_p_arity: int,
    eq_domain: list[int],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    Append PRF expressions of C of output-arity==0 to "exprs" and "visited".
    Args:
    - exprs (list[list[list[Expr]]]): List of PRF expressions to be updated.
    - visited (list[set[bytes]]): List of visited outputs.
    - base_arity (int): Arity of base PRF expression.
    - base_size (int): Size of base PRF expression.
    - max_c_args (int): Maximum arity of C.
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    Return:
    - tuple[list[list[list[Expr]]], list[set[Any]]]: Updated "exprs" and "visited".
    """
    base_exprs = exprs[base_size][base_arity]
    arg_exprs = exprs[output_size - base_size - 1][0]

    for base, args in product(
        base_exprs, product(arg_exprs, repeat=base_arity)
    ):
        candidate = C(base, *args)
        exprs[output_size][0], visited = if_visited_then_append_for_const(
            exprs[output_size][0], visited, candidate, max_p_arity, eq_domain
        )
    return exprs, visited


def update_c_pattern(
    pre_exprs: list[list[list[Expr]]],
    visited: list[set[bytes]],
    base_arity: int,
    args_arity: int,
    base_size: int,
    output_size: int,
    eq_domain: list[int],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    Append PRF expressions of C to "exprs" and "visited".
    Args:
    - pre_exprs (list[list[list[Expr]]]): List of PRF expressions to be updated.
    - visited (list[set[bytes]]): List of visited outputs.
    - base_arity (int): Arity of base PRF expression.
    - args_arity (int): Arity of argument PRF expression.
    - base_size (int): Size of base PRF expression.
    - output_size (int): Size of output PRF expression.
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    Return:
    - tuple[list[list[list[Expr]]], list[set[bytes]]]: Updated "exprs" and "visited".
    """
    assert args_arity >= 1, "arg_arity must not be 0"
    assert base_arity >= 1, "base_arity must not be 0"

    base_exprs = pre_exprs[base_size][base_arity]

    arg_exprs = (
        pre_exprs[output_size - base_size - 1][args_arity]
        + pre_exprs[output_size - base_size - 1][0]
    )
    # remove args all which elements are arity 0
    args_set = set(product(arg_exprs, repeat=base_arity)) - set(
        product(pre_exprs[output_size - base_size - 1][0], repeat=base_arity)
    )

    for base, args in product(base_exprs, args_set):
        pre_exprs[output_size][args_arity], visited = if_visited_then_append(
            pre_exprs[output_size][args_arity],
            visited,
            C(base, *args),
            args_arity,
            eq_domain,
        )
    return pre_exprs, visited


def update_r_pattern_const(
    exprs: list[list[list[Expr]]],
    visited: list[set[bytes]],
    base_size: int,
    step_arity: int,
    output_size: int,
    eq_domain: list[int],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    Append PRF expressions of R of output-arity==0 to "exprs" and "visited".
    Args:
    - exprs (list[list[list[Expr]]]): List of PRF expressions to be updated.
    - visited (list[set[bytes]]): List of visited outputs.
    - base_arity (int): Arity of base PRF expression.
    - base_size (int): Size of base PRF expression.
    - step_arity (int),
    - output_size: int,
    - max_p_arity: int,
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    Return:
    - tuple[list[list[list[Expr]]], list[set[Any]]]: Updated "exprs" and "visited".
    """
    for base in exprs[base_size][0]:
        for step in exprs[output_size - base_size - 1][step_arity]:
            exprs[output_size][step_arity - 1], visited = (
                if_visited_then_append(
                    exprs[output_size][step_arity - 1],
                    visited,
                    R(base, step),
                    step_arity - 1,
                    eq_domain,
                )
            )
    return exprs, visited


def update_r_pattern(
    exprs: list[list[list[Expr]]],
    visited: list[set[bytes]],
    base_arity: int,
    base_size: int,
    output_size: int,
    eq_domain: list[int],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    add PRF expressions of R to "exprs" and "visited".
    this is arity R(n, n+2) (n>=1) pattern.

    """
    assert base_arity >= 1, "base_arity must not be 0"
    for base in exprs[base_size][base_arity]:
        for arg in exprs[output_size - base_size - 1][base_arity + 2]:
            if_visited_then_append(
                exprs[output_size][base_arity + 2],
                visited,
                R(base, arg),
                base_arity + 1,
                eq_domain,
            )
    return exprs, visited


def _generate_output_by_size(
    n: int,
    max_p_arity: int,
    max_c_args: int,
    eq_domain: list[int],
    visited: list[set[bytes]],
) -> tuple[list[list[list[Expr]]], list[set[bytes]]]:
    """
    Generate PRF expressions of size <= n.
    size means the number of PRF symbol in the expression.
    Args:
    - n (int): Size of PRF expressions.
    - max_p_arity (int): Maximum arity of P.
    - max_c_args (int): Maximum the number of arguments of C.
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    - visited (list[set[bytes]]): List of visited outputs.
        - bytes are generated by np.array.tobytes()
        - visited[num of input of a expression] is a set of bytes of output array.
    """
    assert n > 0, "Expr size must be positive"
    if n == 1:
        exprs: list[list[list[Expr]]] = [
            [[] for _ in range(max_p_arity + 1)],
            [[] for _ in range(max_p_arity + 1)],
        ]
        exprs[1][0].append(Z())
        exprs[1][1].append(S())  # S の式をアリティ 1 のリストに追加
        # arity=0 needs spetial treatment for input dimension
        for arity in range(1, max_p_arity + 1):
            visited[arity].add(np.zeros((len(eq_domain),) * arity).tobytes())
        visited[1].add((np.array(eq_domain) + 1).tobytes())

        for arity in range(1, max_p_arity + 1):
            mesh = np.meshgrid(*((eq_domain,) * arity), indexing="ij")
            for i in range(1, arity + 1):
                exprs[1][arity].append(P(arity, i))
                visited[arity].add(mesh[i - 1].tobytes())
        return exprs, visited

    pre_exprs, visited = _generate_output_by_size(
        n - 1, max_p_arity, max_c_args, eq_domain, visited
    )
    pre_exprs.append([[] for _ in range(max_p_arity + 1)])

    # update C pattern
    for k in range(1, n - 1):  # size of base
        for b_arity in range(1, min(max_p_arity + 1, max_c_args)):
            pre_exprs, visited = update_c_pattern_const(
                pre_exprs,
                visited,
                base_arity=b_arity,
                base_size=k,
                output_size=n,
                max_p_arity=max_p_arity,
                eq_domain=eq_domain,
            )
            for arity in range(1, max_p_arity + 1):
                pre_exprs, visited = update_c_pattern(
                    pre_exprs,
                    visited,
                    base_arity=b_arity,
                    args_arity=arity,
                    base_size=k,
                    output_size=n,
                    eq_domain=eq_domain,
                )

    for k in range(1, n - 1):
        for s_arity in range(2, max_p_arity + 1):
            pre_exprs, visited = update_r_pattern_const(
                pre_exprs,
                visited,
                base_size=k,
                step_arity=s_arity,
                output_size=n,
                eq_domain=eq_domain,
            )
    for k in range(1, n - 1):
        for b_arity in range(1, max_p_arity - 1):
            pre_exprs, visited = update_r_pattern(
                pre_exprs,
                visited,
                base_arity=b_arity,
                base_size=k,
                output_size=n,
                eq_domain=eq_domain,
            )

    return pre_exprs, visited


def generate_output_by_size(
    size: int,
    max_p_arity: int,
    max_c_args: int,
    eq_domain: list[int],
):
    """
    Generate PRF expressions of size <= n.
    size means the number of PRF symbol in the expression.
    Args:
    - n (int): Size of PRF expressions.
    - max_p_arity (int): Maximum arity of P = maximum number of arguments all expressions.
    - max_c_args (int): Maximum the number of arguments of C.
    - eq_domain (List[int]): List of inputs to defining semantic equivalence of Exprs.
    """
    visited: list[set[bytes]] = [set() for _ in range(size + 1)]
    return _generate_output_by_size(
        size,
        max_p_arity,
        max_c_args,
        eq_domain,
        visited,
    )


def generate_expression_table(
    max_size: int,
    max_p_arity: int,
    max_c_args: int,
    arity: int,
    inputs: list[Any],
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
    exprs_by_depth_and_arity, visited = generate_output_by_size(
        max_size, max_p_arity, max_c_args, eq_domain
    )

    data: list[dict[str, Any]] = []

    for size in range(1, len(exprs_by_depth_and_arity)):
        exprs: set[Expr] = set(
            exprs_by_depth_and_arity[size][0]
            + exprs_by_depth_and_arity[size][arity]
        )
        for expr in exprs:
            # 各式について必要な情報を取得
            is_valid = expr.is_valid()
            outputs = [expr.evaluate(x) for x in inputs]  # 出力を計算

            data.append(
                {
                    "SPF": str(expr),
                    "valid": is_valid,
                    "size": size,
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )

    # DataFrameを生成
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # 使用例
    max_size = 5
    max_p_arity = 3
    max_c_args = 2
    df_expr_table = generate_expression_table(
        max_size=max_size,
        max_p_arity=max_p_arity,
        max_c_args=max_c_args,
        arity=1,
        inputs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    filename = f"size{max_size}-p{max_p_arity}-c{max_c_args}-unique.csv"
    df_expr_table.to_csv(
        f"./data/{filename}",
        index=False,
        encoding="utf-8-sig",
    )  # UTF-8でエンコードして保存
    print(f"CSV file is generated: {filename}")
