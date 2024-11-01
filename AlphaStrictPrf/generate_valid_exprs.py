from itertools import product
from typing import Any, List

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

    ret: List[List[List[Expr]]] = pre_exprs.copy()
    ret.append(max_d_exprs)
    return ret


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
        "expression_table.csv", index=False, encoding="utf-8-sig"
    )  # UTF-8でエンコードして保存
    print("CSV file is generated: expression_table.csv")
