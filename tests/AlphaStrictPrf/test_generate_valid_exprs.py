from AlphaStrictPrf.generate_valid_exprs import generate_valid_exprs
from AlphaStrictPrf.strict_prf import C, P, S, Z


def test_args_0():
    exprs = generate_valid_exprs(0, 2, 2)
    exp_exprs = [[]]
    assert exprs == exp_exprs


def test_args_1():
    exprs = generate_valid_exprs(1, 2, 2)
    exp_exprs = [[], [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]]]

    assert exprs == exp_exprs, "Error: generate_valid_exprs(1, 2, 2)"


def test_args_2():
    exprs = generate_valid_exprs(2, 2, 2)
    exp_exprs = [
        [],
        [[Z()], [S(), P(1, 1)], [P(2, 1), P(2, 2)]],
        [
            [C(S(), Z()), C(P(1, 1), Z())],
            [
                C(S(), S()),
                C(S(), P(1, 1)),
                C(P(1, 1), S()),
                C(P(1, 1), P(1, 1)),
                C(Z(), P(2, 1)),
                C(Z(), P(2, 2)),
            ],
            [C(S(), P(2, 1)), C(P(1, 1), P(2, 1))],
        ],
    ]
