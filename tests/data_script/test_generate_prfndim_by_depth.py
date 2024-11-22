import logging

import numpy as np

from data_script.generate_prfndim_by_depth import (
    _generate_prfndim_by_depth,
    c_pattern,
    get_r_arity,
    one_depth_exprs,
    output_bytes_const,
    output_bytes_not_const,
    r_pattern,
)
from prfndim.prfndim import C, P, R, S, Z, expr_list_to_str

logging.basicConfig(level=logging.DEBUG)


def test_c_pattern_base_arity_1():
    max_c = 2
    max_arity = 3
    exprs_by_arity = [
        [Z()],
        [S(), P(1, 1)],
        [P(2, 1), P(2, 2)],
        [P(3, 1), P(3, 2), P(3, 3)],
    ]

    c_exprs = c_pattern(exprs_by_arity, max_arity, max_c)
    assert len(c_exprs) > 0


def test_c_pattern_arity_2():
    max_c = 3
    max_arity = 3
    exprs_by_arity = [
        [Z()],
        [S(), P(1, 1)],
        [P(2, 1), P(2, 2)],
        [P(3, 1), P(3, 2), P(3, 3)],
    ]

    c_exprs = c_pattern(exprs_by_arity, max_arity, max_c)
    assert len(c_exprs) > 0


def test_r_pattern_r_3():
    max_arity = 3
    max_r = 3
    exprs_by_arity = [
        [Z()],
        [S(), P(1, 1)],
        [P(2, 1), P(2, 2)],
        [P(3, 1), P(3, 2), P(3, 3)],
    ]

    r_exprs = r_pattern(exprs_by_arity, max_arity, max_r)
    assert len(r_exprs) > 0


def test_r_pattern():
    max_arity = 5
    max_r = 5
    exprs_by_arity = [
        [Z()],
        [S(), P(1, 1)],
        [P(2, 1), P(2, 2)],
        [P(3, 1), P(3, 2), P(3, 3)],
        [P(4, 1), P(4, 2), P(4, 3), P(4, 4)],
        [P(5, 1), P(5, 2), P(5, 3), P(5, 4), P(5, 5)],
    ]

    r_exprs = r_pattern(exprs_by_arity, max_arity, max_r)
    assert len(r_exprs) > 0


def test__generate_prfndim_by_depth():
    sample_num = 5
    sample_max = 10

    max_arity = 5
    max_c = 3
    max_r = 5
    max_depth = 2
    visited = [set() for _ in range(max_arity + 1)]
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]

    exprs, output = _generate_prfndim_by_depth(
        max_depth,
        max_arity,
        max_c,
        max_r,
        eq_domain,
        visited,
    )
    assert len(exprs) > 0
    assert len(output) > 0


def test_get_r_arity():
    args = (1, 2, 0)
    assert get_r_arity(args) == 1


def test_one_depth_exprs():
    sample_num = 5
    sample_max = 10

    max_arity = 5
    visited = [set() for _ in range(max_arity + 1)]
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]
    exprs, visited = one_depth_exprs(max_arity, eq_domain)
    expr = C(S(), P(1, 1))
    output_bytes, _= output_bytes_not_const(expr, eq_domain)
    logging.debug("output_bytes: %s", output_bytes)
    assert output_bytes in visited[1]


def test_one_depth_exprs_const():
    sample_num = 5
    sample_max = 10

    max_arity = 5
    visited = [set() for _ in range(max_arity + 1)]
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]
    exprs, visited = one_depth_exprs(max_arity, eq_domain)
    expr = C(P(1, 1), Z())
    for input_size in range(1, max_arity + 1):
        output_bytes, _ = output_bytes_const(expr, input_size, eq_domain)
        assert output_bytes in visited[input_size]
    logging.debug("output_bytes: %s", output_bytes)


def test_output_bytes_const():
    e1 = C(P(1, 1), Z())
    e2 = Z()
    max_arity = 4
    sample_max = 10
    sample_num = 5
    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, sample_max + 1, size=(sample_num, dim))
        for dim in range(1, max_arity + 1)
    ]
    input_size = 1
    e1_out, _ = output_bytes_const(e1, 1, eq_domain[input_size])
    e2_out, _ = output_bytes_const(e2, 1, eq_domain[input_size])
    assert e1_out == e2_out
