import numpy as np

from AlphaStrictPrf.generate_output_by_size import (
    _generate_output_by_size,
    ndim_output_map,
)
from AlphaStrictPrf.strict_prf import C, P, R, S, Z, expr_to_str_rec

expr_to_str_rec([Z(), S(), P(1, 1), P(2, 1), P(2, 2)])


def test_ndim_output_map_add():
    base_array = [3, 4, 5]
    d = 2

    def f(x, y):
        return x + y

    expected = np.array([[6, 7, 8], [7, 8, 9], [8, 9, 10]])
    assert np.array_equal(ndim_output_map(base_array, d, f), expected)


def test_ndim_output_map_sub():
    base_array = [3, 4, 5]
    d = 2

    def f(x, y):
        return x - y

    expected = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    output = ndim_output_map(base_array, d, f)
    assert np.array_equal(expected, output), "Error: output is not as expected."


def test_ndim_output_map():
    base_array = [0, 1]
    d = 3

    def f(x, y, z):
        return max(x, y, z)

    expected = np.array([[[0, 1], [1, 1]], [[1, 1], [1, 1]]])
    output = ndim_output_map(base_array, d, f)
    assert np.array_equal(expected, output), "Error: output is not as expected."


def test_size_1():
    visited = [set() for _ in range(3)]
    output, visietd = _generate_output_by_size(
        1,
        2,
        2,
        [0, 1, 2],
        visited,
    )
    expected = [[], [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]]]
    assert output == expected, "Error: output is not as expected."


def test_size_3():
    size = 3
    max_p_arity = 2
    visited = [set() for _ in range(size + 1)]
    output, visietd = _generate_output_by_size(
        size,
        max_p_arity=max_p_arity,
        max_c_args=2,
        eq_domain=[0, 1, 2, 3],
        visited=visited,
    )
    expected = [
        [[], [], []],
        [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]],
        [[], [], []],
        [
            [C(S(), Z())],
            [
                C(S(), S()),
                R(Z(), P(2, 1)),
            ],
            [
                C(S(), P(2, 1)),
                C(S(), P(2, 2)),
            ],
        ],
    ]
    for i in range(1, size + 1):
        for j in range(max_p_arity + 1):
            assert set(output[i][j]) == set(
                expected[i][j]
            ), "Error: output is not as expected."


def test_size_4():
    size = 5
    max_p_arity = 2
    visited = [set() for _ in range(size + 1)]
    output, visietd = _generate_output_by_size(
        size,
        max_p_arity=max_p_arity,
        max_c_args=2,
        eq_domain=[0, 1, 2, 3],
        visited=visited,
    )
    expected = [
        [[], [], []],
        [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]],
        [[], [], []],
        [
            [C(S(), Z())],
            [
                C(S(), S()),
                R(Z(), P(2, 1)),
            ],
            [
                C(S(), P(2, 1)),
                C(S(), P(2, 2)),
            ],
        ],
        [[], [], [], []],
        [
            [C(S(), C(S(), Z()))],
            [
                C(S(), R(Z(), P(2, 1))),
                C(S(), C(S(), S())),
                R(C(S(), Z()), P(2, 1)),
            ],
            [
                C(S(), C(S(), P(2, 1))),  # [2, 2, 2], [3, 3, 3], [4, 4, 4]
                C(S(), C(S(), P(2, 2))),  # [2, 3, 4], [2, 3, 4], [2, 3, 4]
                C(
                    R(Z(), P(2, 1)), P(2, 1)
                ),  # [0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]
                C(
                    R(Z(), P(2, 1)), P(2, 2)
                ),  # [0, 0, 1, 2, 3], [0, 0, 1, 2, 3], [0, 0, 1, 2, 3]
            ],
        ],
    ]
    for i in range(1, size + 1):
        for j in range(max_p_arity + 1):
            assert set(output[i][j]) == set(
                expected[i][j]
            ), "Error: output is not as expected."
