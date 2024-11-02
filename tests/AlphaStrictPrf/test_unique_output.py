from itertools import product
from AlphaStrictPrf.generate_unique_output import (
    append_if_unique,
    create_ndim_array,
    generate_exprs_unique_output,
)
from typing import Any, List

import numpy as np

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z

def expr_list_to_str(exprs):
    if isinstance(exprs, Expr):
        return str(exprs)
    return [expr_list_to_str(expr) for expr in exprs]

# テスト関数の作成
def test_create_ndim_array():
    # テスト用の関数: 引数を足し合わせたものを返す
    def test_func(*args):
        return sum(args)

    # パラメータ
    base_array = [0, 1, 2]  # len=3の配列
    d = 2  # 2次元
    expected_output = ((0, 1, 2), (1, 2, 3), (2, 3, 4))

    # 関数を実行
    result = create_ndim_array(base_array, d, test_func)

    # 結果を検証
    assert result == expected_output, f"Expected {expected_output} but got {result}"


def test_create_ndim_array_3d():
    # 別のテストケース: 3次元配列
    def test_func(*args):
        return sum(args)

    base_array = [0, 1]  # len=2の配列
    d = 3  # 3次元
    expected_output = (((0, 1), (1, 2)), ((1, 2), (2, 3)))

    result = create_ndim_array(base_array, d, test_func)
    assert result == expected_output, f"Expected {expected_output} but got {result}"


def test_append_if_unique():
    exprs = []
    visited = [set() for _ in range(3)]
    eq_domain = [0, 1, 2]

    # Z() should be appended in arity = 1 in this time because input dim == 1
    expr1 = Z()
    append_if_unique(expr1, 1, eq_domain, exprs, visited)
    assert visited == [set(), {(0, 0, 0)}, set()], "Error in append_if_unique"

    expr2 = S()
    append_if_unique(expr2, 1, eq_domain, exprs, visited)
    assert visited == [
        set(),
        {(1, 2, 3), (0, 0, 0)},
        set(),
    ], "Error in append_if_unique"

    # Test for duplicate
    expr3 = C(P(1, 1), S())
    append_if_unique(expr3, 1, eq_domain, exprs, visited)
    assert visited == [
        set(),
        {(1, 2, 3), (0, 0, 0)},
        set(),
    ], "Error in append_if_unique"

    # Test for dim=2
    expr4 = P(2, 1)
    append_if_unique(expr4, 2, eq_domain, exprs, visited)
    assert visited == [
        set(),
        {(1, 2, 3), (0, 0, 0)},
        {((0, 0, 0), (1, 1, 1), (2, 2, 2))},
    ], "Error in append_if_unique"


def test_args_1():
    max_p_arity = 2
    visited = [set() for _ in range(max_p_arity + 1)]
    exprs, visited = generate_exprs_unique_output(1, max_p_arity, 2, [0, 1, 2], visited)
    exp_exprs = [[], [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]]]

    assert exprs == exp_exprs, f"Error: {exprs} is ont {exp_exprs}"
    exp_visited = [
            set(),
            {(0, 0, 0), (1, 2, 3), (0, 1, 2)},
            {((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 1, 1), (2, 2, 2)), ((0, 1, 2), (0, 1, 2), (0, 1, 2))}
    ]
    assert visited == exp_visited, f"Error: {visited} is not {exp_visited}"


def test_args_2():
    max_p_arity = 2
    visited = [set() for _ in range(max_p_arity + 1)]
    exprs, visited = generate_exprs_unique_output(2, max_p_arity, 2, [0, 1, 2], visited)
    exp_exprs = [
        [],
        [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]],
        [
            list(
                set(
                    [
                        C(S(), Z()),
                    ]
                )
            ),
            list(
                set(
                    [
                        C(S(), S()),
                        R(Z(), P(2, 1)),
                    ]
                )
            ),
            list(
                set(
                    [
                        C(S(), P(2, 1)),
                        C(S(), P(2, 2)),
                    ]
                )
            ),
        ],
    ]
    assert (
        exprs == exp_exprs
    ), f"Error: {expr_list_to_str(exprs)} is not {expr_list_to_str(exp_exprs)}"

