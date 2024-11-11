import logging
from collections import deque

import pytest

from AlphaStrictPrf.strict_prf import (
    C,
    InputSizeError,
    P,
    R,
    S,
    Z,
    expr_to_str_rec,
)

logging.basicConfig(level=logging.DEBUG)

# ---- arity -----


def test_arity_validation():
    z = Z()
    assert z.arity() is None, "Z should return None for arity"
    s = S()
    assert s.arity() == 1, "S should return 1 for arity"


def test_R_arity():
    add = R(P(1, 1), C(S(), P(3, 2)))
    logging.debug("Test Sample: R(P(1, 1), C(S(), P(3, 2)))")
    assert (
        add.arity() == 2
    ), "Error: arity is wrong for R(P(1, 1), C(S(), P(3, 2)))"


def test_C_arity():
    when0_then1_else0 = C(R(S(), P(3, 3)), P(1, 1), Z())
    logging.debug("Test Sample: C(R(S(), P(3, 3)), P(1, 1), Z())")
    assert (
        when0_then1_else0.arity() == 1
    ), "Error: arity is wrong for C(R(S(), P(3, 3)), P(1, 1), Z())"


def test_invalid_R_arity():
    rss_invalid_arity = R(S(), S())
    logging.debug("Test Sample: R(S(), S())")
    assert (
        rss_invalid_arity.is_valid() is False
    ), "Error: is_valid() works incorrectly for R(S(), S())"


def test_invalid_inner_arity():
    inner_invalid = R(Z(), C(S(), P(2, 2), P(2, 2)))
    logging.debug("R(Z(), C(S(), P(2, 2), P(2, 2)))")
    assert (
        inner_invalid.is_valid() is False
    ), "Error: is_valid() works incorrectly for R(Z(), C(S(), P(2, 2), P(2, 2)))"


def test_invalid_R_arity_2():
    inner_invalid2 = R(S(), Z())
    logging.debug("R(S(), Z())")
    assert (
        inner_invalid2.is_valid() is False
    ), "Error: is_valid() works incorrectly for R(S(), Z())"


# ---- change -----


def test_change_complex():
    expr = C(R(Z(), P(2, 1)), C(P(1, 1), S()))
    pos = deque([2, 1])
    new_expr = expr.change(pos, R(Z(), P(2, 1)))
    expected_expr = C(R(Z(), P(2, 1)), C(R(Z(), P(2, 1)), S()))
    assert new_expr is expected_expr, "Error: Expr.change()"
    assert new_expr is not expr, "Error: Expr.change()"


# ---- copy -----
def test_Z_copy():
    z1 = Z()
    z2 = z1.copy()
    assert z1 == z2, "Copy of Z should be equal to the original"

    assert (
        z1 is not z2
    ), "Copy of Z should not be the same object as the original"


def test_S_copy():
    s1 = S()
    s2 = s1.copy()
    assert s1 == s2, "Copy of S should be equal to the original"
    assert (
        s1 is not s2
    ), "Copy of S should not be the same object as the original"


def test_P_copy():
    p1 = P(3, 2)
    p2 = p1.copy()
    assert p1 == p2, "Copy of P should be equal to the original"
    assert (
        p1 is not p2
    ), "Copy of P should not be the same object as the original"


def test_C_copy():
    c1 = C(S(), Z())
    c2 = c1.copy()
    assert c1 == c2, "Copy of C should be equal to the original"
    assert (
        c1 is not c2
    ), "Copy of C should not be the same object as the original"


def test_R_copy():
    r1 = R(Z(), S())
    r2 = r1.copy()
    assert r1 == r2, "Copy of R should be equal to the original"
    assert (
        r1 is not r2
    ), "Copy of R should not be the same object as the original"


def test_copy_complex():
    assert Z().copy() == Z(), "Error: Z().copy"
    assert S().copy() == S(), "Error: S().copy"
    assert P(3, 1).copy() == P(3, 1), "Error: P(1, 2).copy"
    assert C(S(), Z()).copy() == C(S(), Z()), "Error: C(S(), Z()).copy"
    assert R(P(1, 1), P(3, 1)).copy() == R(
        P(1, 1), P(3, 1)
    ), "Error: R(C(1, 1), P(3, 1).copy"


# ---- equality -----
def test_Z_equality():
    z1 = Z()
    z2 = Z()
    assert z1 == z2, "Two instances of Z should be equal"


def test_S_equality():
    s1 = S()
    s2 = S()
    assert s1 == s2, "Two instances of S should be equal"


def test_C_equality():
    c1 = C(S(), Z())
    c2 = C(S(), Z())
    assert (
        c1 == c2
    ), "Two instances of C with same function and args should be equal"


def test_R_equality():
    r1 = R(Z(), S())
    r2 = R(Z(), S())
    assert (
        r1 == r2
    ), "Two instances of R with same base and step should be equal"


def test_complex_equality():
    expr1 = C(S(), Z())
    expr2 = C(S(), Z())
    assert expr1 == expr2, "Error: Expr.__eq__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), S(), S()))
    assert expr1 == expr2, "Error: Expr.__eq__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), Z(), Z()))
    assert expr1 != expr2, "Error: Expr.__eq__()"


def test_set_equality():
    expr_set1 = {C(S(), Z()), R(S(), C(P(3, 1), S(), S())), Z()}
    expr_set2 = {R(S(), C(P(3, 1), S(), S())), Z(), C(S(), Z())}
    assert expr_set1 == expr_set2, "Error: Expr set equality"


# ---- evaluate -----
def test_Z_evaluate():
    z = Z()
    assert z.evaluate() == 0, "Z should always evaluate to 0"
    assert (
        z.evaluate(1, 2, 3) == 0
    ), "Z should always evaluate to 0 regardless of arguments"


def test_S_evaluate():
    s = S()
    assert s.evaluate(0) == 1, "S(0) should evaluate to 1"
    assert s.evaluate(5) == 6, "S(5) should evaluate to 6"


def test_P_evaluate():
    p = P(3, 2)
    assert p.evaluate(1, 2, 3) == 2, "P(3,2) should return the second argument"


def test_invalid_P_evaluate():
    p_func = P(3, 2)
    logging.debug("P(3, 2)")
    with pytest.raises(InputSizeError):
        p_func.evaluate(10, 20)


def test_C_evaluate():
    c = C(S(), Z())
    assert (
        c.evaluate(1) == 1
    ), "C(S(), Z()) should evaluate to 1 when input is 1"
    assert (
        c.evaluate(5) == 1
    ), "C(S(), Z()) should evaluate to 1 when input is 5"


def test_R_evaluate():
    r = R(C(S(), Z()), P(2, 1))
    assert (
        r.evaluate(0) == 1
    ), "R(C(S(), Z()), P(2, 1)).evaluate(0) should return base value 0"
    assert (
        r.evaluate(1) == 0
    ), "R(C(S(), Z()), P(2, 1)) should return step value 1"
    assert (
        r.evaluate(2) == 1
    ), "R(C(S(), Z()), P(2, 1)) should return step value 2"


def test_R_evaluate_complex_1():
    add = R(P(1, 1), C(S(), P(3, 2)))
    logging.debug("Test Sample: R(P(1, 1), C(S(), P(3, 2)))")
    assert (
        add.evaluate(2, 3) == 5
    ), "Error: R(P(1, 1), C(S(), P(3, 2))) evaluation is wrong"


def test_C_evaluate_complex_2():
    when0_then1_else0 = C(R(S(), P(3, 3)), P(1, 1), Z())
    logging.debug("Test Sample: C(R(S(), P(3, 3)), P(1, 1), Z())")
    ans_sequence = [when0_then1_else0.evaluate(i) for i in range(5)]
    assert ans_sequence == [
        1,
        0,
        0,
        0,
        0,
    ], "Error: C(R(S(), P(3, 3)), P(1, 1), Z()) evaluation is wrong"


def test_invalid_R_evaluate():
    rss_invalid_arity = R(S(), S())
    logging.debug("Test Sample: R(S(), S())")
    with pytest.raises(InputSizeError):
        rss_invalid_arity.evaluate(1)


def test_invalid_inner_evaluate():
    inner_invalid = R(Z(), C(S(), P(2, 2), P(2, 2)))
    logging.debug("R(Z(), C(S(), P(2, 2), P(2, 2)))")
    with pytest.raises(InputSizeError):
        inner_invalid.evaluate(1)


def test_invalid_R_evaluate_2():
    inner_invalid2 = R(S(), Z())
    logging.debug("R(S(), Z())")
    with pytest.raises(InputSizeError):
        inner_invalid2.evaluate(1)


# ---- hash -----
def test_complex_hash():
    expr1 = C(S(), Z())
    expr2 = C(S(), Z())
    assert hash(expr1) == hash(expr2), "Error: Expr.__hash__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), S(), S()))
    assert hash(expr1) == hash(expr2), "Error: Expr.__hash__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), Z(), Z()))
    assert expr1 != expr2, "Error: Expr.__eq__()"


# ---- str -----


def test_Z_str():
    z = Z()
    assert str(z) == "Z()", "Error: Z's __str__() should return 'Z()'"


def test_S_str():
    s = S()
    assert str(s) == "S()", "Error: S's __str__() should return 'S()'"


def test_P_str():
    p = P(2, 1)
    assert str(p) == "P(2, 1)", "Error: P's __str__() should return 'P(2, 1)'"


def test_C_str():
    c = C(S(), Z())
    expected_str = "C(S(), Z())"
    assert (
        str(c) == expected_str
    ), "Error: C(S(), Z())'s __str__() should return 'C(S(), Z())'"


def test_R_str():
    r = R(Z(), S())
    expected_str = "R(Z(), S())"
    assert (
        str(r) == expected_str
    ), "Error: R(Z(), S())'s __str__() should return 'R(Z(), S())'"


def test_complex_str():
    expr1 = C(S(), Z())
    expr2 = R(S(), C(P(3, 1), S(), S()))
    expr3 = R(P(1, 1), C(S(), P(3, 2)))

    str_expr1 = "C(S(), Z())"
    str_expr2 = "R(S(), C(P(3, 1), S(), S()))"
    str_expr3 = "R(P(1, 1), C(S(), P(3, 2)))"

    assert str(expr1) == str_expr1, "Error: Expr.__str__()"
    assert str(expr2) == str_expr2, "Error: Expr.__str__()"
    assert str(expr3) == str_expr3, "Error: Expr.__str__()"


# ---- positions -----
def test_Expr_positions():
    func = C(P(1, 1), Z(), C(P(1, 1), S()))
    positions = func.positions()
    expected_positions = [
        [],
        [1],
        [2],
        [3],
        [3, 1],
        [3, 2],
    ]
    pos_comp = set(tuple(sublist) for sublist in positions)
    exp_pos_comp = set(tuple(sublist) for sublist in expected_positions)
    assert set(pos_comp) == set(exp_pos_comp), "Error: generate_positions()"


def test_positions():
    expr = C(Z(), S())
    positions = expr.positions()
    assert positions == [
        deque([]),
        deque([1]),
        deque([2]),
    ], "Position list mismatch"


# ---- tree_string -----
def test_Z_tree_string():
    z = Z()
    assert z.tree_string() == "Z", "Error: Z's tree_string() should return 'Z'"


def test_S_tree_string():
    s = S()
    assert s.tree_string() == "S", "Error: S's tree_string() should return 'S'"


def test_P_tree_string():
    p = P(2, 1)
    assert (
        p.tree_string() == "P^2_1"
    ), "Error: P's tree_string() should return 'P^2_1'"
    assert (
        p.tree_string(indent=4) == "    P^2_1"
    ), "Error: P's tree_string() with indent should return '    P^2_1'"


def test_C_tree_string():
    c = C(S(), Z())
    expected_tree_str = "C^2\n  S\n  Z"
    assert (
        c.tree_string() == expected_tree_str
    ), "Error: C(S(), Z())'s tree_string() should return 'C^2\n  S\n  Z'"


def test_R_tree_string():
    r = R(Z(), S())
    expected_tree_str = "R\n  Z\n  S"
    assert (
        r.tree_string() == expected_tree_str
    ), "Error: R(Z(), S())'s tree_string() should return 'R\n  Z\n  S'"


# ---- is_valid -----
def test_Z_is_valid():
    z = Z()
    assert z.is_valid(), "Z should always be semantically valid"


def test_S_is_valid():
    s = S()
    assert s.is_valid(), "S should always be semantically valid"


def test_P_is_valid():
    p = P(3, 2)
    assert p.is_valid(), "P(3, 2) should always be semantically valid"


def test_C_is_valid():
    c = C(S(), Z())
    assert c.is_valid(), "C(S(), Z()) should always be semantically valid"
    assert not C(
        S(), S(), S()
    ).is_valid(), "C(S(), S(), S()) should not be semantically valid"


def test_C_is_valid_complex():
    inner_invalid = C(P(2, 1), Z(), C(Z(), Z()))
    logging.debug("C(P(2, 1), Z(), P(2, 1))")
    assert (
        inner_invalid.is_valid() is False
    ), "Error: is_valid() works incorrectly"


def test_R_is_valid():
    r = R(Z(), S())
    assert not r.is_valid(), "R(Z(), S()) should always be semantically valid"

    r = R(S(), P(3, 1))
    assert r.is_valid(), "R(S(), P(3, 1)) should always be semantically valid"

    r = R(S(), S())
    assert not r.is_valid(), "R(S(), S()) should not be semantically valid"

    r = R(Z(), P(3, 1))
    assert r.is_valid(), "R(Z(), P(3, 1)) should always be semantically valid"

    r = R(Z(), P(1, 1))
    assert (
        not r.is_valid()
    ), "R(Z(), P(1, 1)) should always be semantically valid"


def test_expr_to_str_rec():
    expr = [Z(), S(), [P(1, 1), P(2, 1)], [C(S(), Z()), R(Z(), S())]]
    str_expr = [
        "Z()",
        "S()",
        ["P(1, 1)", "P(2, 1)"],
        ["C(S(), Z())", "R(Z(), S())"],
    ]
    assert str_expr == expr_to_str_rec(expr), "Error: {expr} != {str_expr}"


# ---- id ----
def test_id():
    assert id(Z()) == id(Z()), "id is not equal."
    assert id(S()) == id(S()), "id is not equal."
    assert id(P(1, 1)) == id(P(1, 1)), "id is not equal."
    assert id(C(S(), Z())) == id(C(S(), Z())), "id is not equal."
    assert id(R(S(), P(2, 1))) == id(R(S(), P(2, 1))), "id is not equal."
