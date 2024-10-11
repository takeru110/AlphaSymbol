import logging
from collections import deque

from AlphaStrictPrf.strict_prf import C, P, R, S, Z

logging.basicConfig(level=logging.DEBUG)


def test_Expr_positions():
    func = C(P(2, 1), Z(), C(P(1, 1), S()))
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


def test_Z():
    z_func = Z()
    logging.debug("Test Sample: Z()")
    assert (
        z_func.validate_semantic() is True
    ), "There are something wrong in validate_semantic()."
    assert z_func.evaluate(23) == 0, "Error: Z evaluation is wrong."
    assert (
        z_func.arity() is None
    ), f"Error: Z arity should be 0 but now {z_func.arity()}."
    assert str(z_func) == "Z()", "Error: Z.__str__() wrongly."
    assert z_func.tree_string() == "Z", "Error: Z tree is wrong."
    assert z_func.complexity() == 1.0, "Error: Complesity of Z is wrong"

    zss_func = Z(S, S)
    logging.debug("Test Sample: Z(S, S)")
    assert (
        zss_func.validate_semantic() is False
    ), "There are something wrong in validate_semantic()."


def test_S():
    s_func = S()
    logging.debug("Test Sample: S()")
    assert s_func.evaluate(31) == 32, "Error: S evaluation is wrong."
    assert (
        s_func.arity() == 1
    ), f"Error: S arity should be 1 but {s_func.arity()}."
    assert str(s_func) == "S()", "Error: S.__str__() wrongly."
    assert s_func.tree_string() == "S", "Error: S tree is wrong."
    assert s_func.complexity() == 1.0, "Error: Complesity of S is wrong"


def test_P():
    # Test for the P class with i = 2
    p_func = P(3, 2)
    logging.debug("P(3, 2)")
    assert p_func.evaluate(10, 20, 30) == 20, "Error: P evaluate is wrong."
    assert p_func.arity() == 3, "Error: P arity is wrong."
    assert p_func.tree_string() == "P^3_2", "Error: P print_tree is wrong."
    assert (
        p_func.tree_string(indent=4) == "    P^3_2"
    ), "Error: P print_tree with indent is wrong."
    assert (
        p_func.tree_string() == "P^3_2"
    ), "Error: P print tree_string is wrong."
    assert p_func.complexity() == 1.0, "Error: Complexity of P is wrong"


def test_C():
    c_func_always_one = C(S(), Z())
    logging.debug("Test Sample: C(S(), Z())")
    assert (
        c_func_always_one.validate_semantic() is True
    ), "Error: validate_semantic() works incorrectly."
    assert c_func_always_one.evaluate(34) == 1, "Error: C evaluation is wrong"
    assert c_func_always_one.arity() is None, "Error: C arity is wrong."
    assert (
        c_func_always_one.tree_string() == "C^2\n  S\n  Z"
    ), "C tree is wrong."
    assert (
        str(c_func_always_one) == "C(S(), Z())"
    ), "Error: C.__str__() is wrongly"
    assert (
        c_func_always_one.complexity() == 1.0
    ), "Error: C Complexity is wrong."

    add_two = C(S(), S())
    logging.debug("Test Sample: C(S(), S())")
    assert (
        add_two.validate_semantic() is True
    ), "Error: validate_semantic() works incorrectly."
    assert add_two.evaluate(17) == 19, "Error: C evaluation is wrong"
    assert add_two.arity() == 1, "Error: C arity is wrong."
    assert add_two.tree_string() == "C^2\n  S\n  S", "C tree is wrong."
    assert str(add_two) == "C(S(), S())", "Error: C.__str__() is wrongly"
    assert add_two.complexity() == 1.0, "Error: C Complexity is wrong."

    csss_semantic_invalid = C(S(), S(), S())
    logging.debug("Test Sample: C(S(), S(), S())")
    assert (
        csss_semantic_invalid.validate_semantic() is False
    ), "Error: validate_semantic() works incorrectly."

    inner_invalid = C(P(2, 1), Z(), Z(S()))
    logging.debug("C(P(2, 1), Z(), Z(S()))")
    assert (
        inner_invalid.validate_semantic() is False
    ), "Error: validate_semantic() works incorrectly"


def test_R():
    add = R(P(1, 1), C(S(), P(3, 2)))
    logging.debug("Test Sample: R(P(1, 1), C(S(), P(3, 2)))")
    assert (
        add.validate_semantic() is True
    ), "Error: validate_semantic() works incorrectly."
    assert add.arity() == 2, "Error: arity is wrong"
    assert add.evaluate(2, 3) == 5, "Error: R evaluation is wrong."
    assert (
        add.tree_string() == "R\n  P^1_1\n  C^2\n    S\n    P^3_2"
    ), "Error: R tree is wrong"
    assert (
        str(add) == "R(P(1, 1), C(S(), P(3, 2)))"
    ), "Error: R.__str__() is wrong"
    assert add.complexity() == 1.0, "Error: add complexity is wrong."

    when0_then1_else0 = C(R(S(), P(3, 3)), P(1, 1), Z())
    logging.debug("Test Sample: C(R(S(), P(3, 3)), P(1, 1), Z())")
    assert (
        when0_then1_else0.validate_semantic() is True
    ), "Error: validate_semantic() works incorrectly."
    ans_sequence = [when0_then1_else0.evaluate(i) for i in range(5)]
    assert ans_sequence == [1, 0, 0, 0, 0], "Error: prf is wrong"
    assert when0_then1_else0.arity() == 1, "Error: arity is wrong"

    rss_invalid_arity = R(S(), S())
    logging.debug("Test Sample: R(S(), S())")
    assert (
        rss_invalid_arity.validate_semantic() is False
    ), "Error: validate_semantic() works incorrectly."

    inner_invalid = R(Z(), C(S(), P(2, 2), P(2, 2)))
    logging.debug("R(Z(), C(S(), P(2, 2), P(2, 2)))")
    assert (
        inner_invalid.validate_semantic() is False
    ), "Error: validate_semantic() works incorrectly"

    inner_invalid2 = R(S(), Z())
    logging.debug("R(S(), Z())")
    assert (
        inner_invalid2.validate_semantic() is False
    ), "Error: validate_semantic() works incorrectly"


def test_eq():
    expr1 = C(S(), Z())
    expr2 = C(S(), Z())
    assert expr1 == expr2, "Error: Expr.__eq__()"
    assert hash(expr1) == hash(expr2), "Error: Expr.__hash__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), S(), S()))
    assert expr1 == expr2, "Error: Expr.__eq__()"
    assert hash(expr1) == hash(expr2), "Error: Expr.__hash__()"

    expr1 = R(S(), C(P(3, 1), S(), S()))
    expr2 = R(S(), C(P(3, 1), Z(), Z()))
    assert expr1 is not expr2, "Error: Expr.__eq__()"

    expr_set1 = {C(S(), Z()), R(S(), C(P(3, 1), S(), S())), Z()}
    expr_set2 = {R(S(), C(P(3, 1), S(), S())), Z(), C(S(), Z())}
    assert expr_set1 == expr_set2, "Error: Expr set equality"


def test_str():
    expr1 = C(S(), Z())
    expr2 = R(S(), C(P(3, 1), S(), S()))
    expr3 = R(P(1, 1), C(S(), P(3, 2)))

    str_expr1 = "C(S(), Z())"
    str_expr2 = "R(S(), C(P(3, 1), S(), S()))"
    str_expr3 = "R(P(1, 1), C(S(), P(3, 2)))"

    assert str(expr1) == str_expr1, "Error: Expr.__str__()"
    assert str(expr2) == str_expr2, "Error: Expr.__str__()"
    assert str(expr3) == str_expr3, "Error: Expr.__str__()"


def test_change():
    expr = C(R(Z(), P(2, 1)), C(P(1, 1), S()))
    pos = deque([2, 1])
    new_expr = expr.change(pos, R(Z(), P(2, 1)))
    expected_expr = C(R(Z(), P(2, 1)), C(R(Z(), P(2, 1)), S()))
    assert new_expr == expected_expr, "Error: Expr.change()"
    # check non-destructive
    expr_copy = expr.copy()
    for pos in expr.positions():
        expr_copy.change(pos, Z())
        assert expr_copy == expr, "Error: Expr.change()"


def test_copy():
    assert Z().copy() == Z(), "Error: Z().copy"
    assert S().copy() == S(), "Error: S().copy"
    assert P(3, 1).copy() == P(3, 1), "Error: P(1, 2).copy"
    assert C(S(), Z()).copy() == C(S(), Z()), "Error: C(S(), Z()).copy"
    assert R(P(1, 1), P(3, 1)).copy() == R(
        P(1, 1), P(3, 1)
    ), "Error: R(C(1, 1), P(3, 1).copy"
