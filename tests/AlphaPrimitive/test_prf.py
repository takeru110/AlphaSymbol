from AlphaPrimitive.prf import C, P, R, S, Z


def test_Z():
    z_func = Z()
    assert z_func.evaluate(23) == 0, "Error: Z evaluation is wrong."
    # assert z_func.arity() == 0, "Error: Z arity is wrong."
    assert str(z_func) == "Z", "Error: Z parenthesized wrongly."
    assert z_func.tree_string() == "Z", "Error: Z tree is wrong."


def test_S():
    s_func = S()
    assert s_func.evaluate(31) == 32, "Error: S evaluation is wrong."
    # assert s_func.arity() == 1, "Error: S arity is wrong."
    assert str(s_func) == "S", "Error: S parenthesized wrongly."
    assert s_func.tree_string() == "S", "Error: S tree is wrong."


def test_P():
    # Test for the P class with i = 2
    p_func = P(2)

    # Test evaluate method
    assert p_func.evaluate(10, 20, 30) == 20, "Error: P evaluate is wrong."
    assert (
        p_func.evaluate(1, 2, 3, 4) == 2
    ), "Error: P evaluate is wrong with different inputs."

    # Test arity method
    # assert p_func.arity() == 4, "Error: P arity is wrong."

    # Test print_tree method
    assert p_func.tree_string() == "P2", "Error: P print_tree is wrong."
    assert (
        p_func.tree_string(indent=4) == "    P2"
    ), "Error: P print_tree with indent is wrong."

    # Test print_parenthesized method
    assert (
        p_func.tree_string() == "P2"
    ), "Error: P print_parenthesized is wrong."


def test_C():
    c_func_always_one = C(S(), Z())
    assert c_func_always_one.evaluate(34) == 1, "Error: C evaluation is wrong"
    # assert c_func_always_one.arity() == 0, "Error: C arity is wrong."
    assert c_func_always_one.tree_string() == "C2\n  S\n  Z", "C tree is wrong."
    assert (
        str(c_func_always_one) == "C2(S, Z)"
    ), "Error: C is parenthesized wrongly"

    add_two = C(S(), S())
    assert add_two.evaluate(17) == 19, "Error: C evaluation is wrong"
    # assert add_two.arity() == 1, "Error: C arity is wrong."
    assert add_two.tree_string() == "C2\n  S\n  S", "C tree is wrong."
    assert str(add_two) == "C2(S, S)", "Error: C is parenthesized wrongly"


def test_R():
    add = R(P(1), C(S(), P(2)))
    add.max_arity = 10
    assert add.evaluate(2, 3) == 5, "Error: R evaluation is wrong."
    # assert add.arity() == 10, "Error: R arity is wrong"
    assert (
        add.tree_string() == "R\n  P1\n  C2\n    S\n    P2"
    ), "Error: R tree is wrong"
    assert str(add) == "R(P1, C2(S, P2))", "Error: R parenthesis is wrong"

    # when input is 0 then returns 1 else returns 0
    when0_then1_else0 = C(R(S(), P(3)), P(1), Z())
    ans_sequence = [when0_then1_else0.evaluate(i) for i in range(5)]
    assert ans_sequence == [1, 0, 0, 0, 0], "Error: prf is wrong"
