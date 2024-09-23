import logging

from AlphaStrictPrf.strict_prf import C, P, R, S, Z

logging.basicConfig(level=logging.DEBUG)


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
    assert (
        z_func.parenthesized_string() == "Z"
    ), "Error: Z parenthesized wrongly."
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
    assert (
        s_func.parenthesized_string() == "S"
    ), "Error: S parenthesized wrongly."
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
    ), "Error: P print_parenthesized is wrong."
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
        c_func_always_one.parenthesized_string() == "C^2(S, Z)"
    ), "Error: C is parenthesized wrongly"
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
    assert (
        add_two.parenthesized_string() == "C^2(S, S)"
    ), "Error: C is parenthesized wrongly"
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
        add.parenthesized_string() == "R(P^1_1, C^2(S, P^3_2))"
    ), "Error: R parenthesis is wrong"
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
