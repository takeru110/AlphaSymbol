import logging

import pytest

from prfndim.prfndim import (
    C,
    Expr,
    InputSizeError,
    OverflowError,
    P,
    PrfSyntaxError,
    R,
    S,
    SemanticsError,
    Z,
)


def test_Z():
    z1 = Z()
    z2 = Z()
    assert z1 is z2
    assert z1.eval(1, 2, 3) == 0
    assert str(z1) == "Z()"
    assert z1.arity is None


def test_S():
    s1 = S()
    s2 = S()
    assert s1 is s2
    assert str(s1) == "S()"
    assert s1.arity == 1


def test_P():
    p1 = P(2, 1)
    p2 = P(2, 1)
    assert p1 is p2
    assert str(p1) == "P(2, 1)"


def test_C():
    c1 = C(S(), Z())
    c2 = C(S(), Z())
    assert c1 is c2
    assert str(c1) == "C(S(), Z())"


def test_R():
    r1 = R(S(), Z(), P(1, 1))
    r2 = R(S(), Z(), P(1, 1))
    assert r1 is r2
    assert str(r1) == "R(S(), Z(), P(1, 1))"


# ---- PrfSyntaxError ----


def test_p_syntax_error():
    with pytest.raises(PrfSyntaxError):
        P(1, 2)


def test_c_syntax_error():
    with pytest.raises(PrfSyntaxError):
        C(S())


def test_r_syntax_error():
    with pytest.raises(PrfSyntaxError):
        R(S(), Z())
    with pytest.raises(PrfSyntaxError):
        R(S())


# ---- arity ----


def test_z_arity():
    assert Z().arity is None


def test_s_arity():
    assert S().arity == 1


def test_p_arity():
    assert P(2, 1).arity == 2


def test_c_arity():
    assert C(S(), Z()).arity is None
    assert C(P(2, 1), Z(), S()).arity == 1
    assert C(P(2, 2), P(2, 1), C(S(), P(2, 2))).arity == 2
    with pytest.raises(SemanticsError):
        C(P(2, 1), S()).arity


def test_r_arity():
    assert R(S(), Z(), P(1, 1)).arity == 2
    assert R(S(), P(2, 1), Z()).arity == 1
    assert R(P(2, 1), P(3, 2), P(3, 1), Z(), Z()).arity == 1
    assert R(P(2, 1), P(4, 2), P(4, 1), S(), P(1, 1)).arity == 2
    assert R(P(2, 1), Z(), Z(), Z(), Z()).arity is None
    with pytest.raises(SemanticsError):
        R(S(), S(), Z()).arity


# ---- eval ----


def test_Z_eval():
    z = Z()
    assert z.eval() == 0, "Z should always evaluate to 0"
    assert (
        z.eval(1, 2, 3) == 0
    ), "Z should always evaluate to 0 regardless of arguments"


def test_S_eval():
    s = S()
    assert s.eval(0) == 1, "S(0) should eval to 1"
    assert s.eval(5) == 6, "S(5) should eval to 6"


def test_P_eval():
    p = P(3, 2)
    assert p.eval(1, 2, 3) == 2, "P(3,2) should return the second argument"


def test_invalid_P_eval():
    p_func = P(3, 2)
    logging.debug("P(3, 2)")
    with pytest.raises(InputSizeError):
        p_func.eval(10, 20)


def test_C_eval():
    c = C(S(), Z())
    assert c.eval(1) == 1, "C(S(), Z()) should eval to 1 when input is 1"
    assert c.eval(5) == 1, "C(S(), Z()) should eval to 1 when input is 5"


def test_R_eval():
    r = R(P(1, 1), P(2, 1), C(S(), Z()))
    assert (
        r.eval(0) == 1
    ), "R(C(S(), Z()), P(2, 1)).eval(0) should return base value 0"
    assert r.eval(1) == 0, "R(C(S(), Z()), P(2, 1)) should return step value 1"
    assert r.eval(2) == 1, "R(C(S(), Z()), P(2, 1)) should return step value 2"


def test_R_eval_complex_1():
    add = R(P(1, 1), C(S(), P(3, 2)), P(1, 1))
    logging.debug("Test Sample: R(P(1, 1), C(S(), P(3, 2)))")
    assert (
        add.eval(2, 3) == 5
    ), "Error: R(P(1, 1), C(S(), P(3, 2))) evaluation is wrong"


def test_add():
    test = R(P(1, 1), C(S(), P(3, 2)), P(1, 1))
    assert test.is_valid is True
    assert test.arity == 2
    assert test.eval(2, 3) == 5
    assert test.eval(3, 4) == 7


def test_fib():
    fib = R(
        P(2, 1),
        P(3, 3),
        C(R(P(1, 1), C(S(), P(3, 2)), P(1, 1)), P(3, 2), P(3, 3)),
        Z(),
        C(S(), Z()),
    )
    assert fib.is_valid is True
    assert fib.eval(0) == 0
    assert fib.eval(1) == 1
    assert fib.eval(2) == 1
    assert fib.eval(3) == 2
    assert fib.eval(4) == 3
    assert fib.eval(5) == 5


def test_C_eval_complex_2():
    when0_then1_else0 = C(R(P(1, 1), P(3, 3), S()), P(1, 1), Z())
    logging.debug("Test Sample: C(R(S(), P(3, 3)), P(1, 1), Z())")
    ans_sequence = [when0_then1_else0.eval(i) for i in range(5)]
    assert ans_sequence == [
        1,
        0,
        0,
        0,
        0,
    ], "Error: C(R(S(), P(3, 3)), P(1, 1), Z()) evaluation is wrong"


def test_oveflow():
    with pytest.raises(OverflowError):
        S().eval(2**64)


def test_S_inavlid_size():
    with pytest.raises(InputSizeError):
        S().eval(1, 2)


def test_C_invalid():
    syn_inv = C(P(2, 1), Z())
    with pytest.raises(SemanticsError):
        syn_inv.eval(1)


def test_C_invalid_size():
    with pytest.raises(InputSizeError):
        C(S(), S()).eval(1, 1)


def test_invalid_R_eval():
    rss_invalid_arity = R(P(1, 1), S(), S())
    logging.debug("Test Sample: R(S(), S())")
    with pytest.raises(SemanticsError):
        rss_invalid_arity.eval(1)


def test_invalid_inner_eval():
    inner_invalid = R(P(1, 1), C(S(), P(2, 2), P(2, 2)), Z())
    logging.debug("R(Z(), C(S(), P(2, 2), P(2, 2)))")
    with pytest.raises(SemanticsError):
        inner_invalid.eval(1)


def test_invalid_R_eval_2():
    inner_invalid2 = R(P(1, 1), Z(), S())
    logging.debug("R(S(), Z())")
    with pytest.raises(InputSizeError):
        inner_invalid2.eval(1)


def test_temp_eval():
    expr = R(
        P(2, 1),
        R(
            R(S(), P(3, 2), Z()),
            P(4, 2),
            P(2, 2),
        ),
        P(3, 1),
        Z(),
        Z(),
    )
    assert expr.is_valid is False


# ---- is_valid ----


def test_z_is_valid():
    assert Z().is_valid


def test_s_is_valid():
    assert S().is_valid


def test_p_is_valid():
    assert P(2, 1).is_valid


def test_c_is_valid():
    assert C(S(), Z()).is_valid
    assert C(Z(), S()).is_valid
    assert C(P(2, 1), Z(), S()).is_valid
    assert C(P(2, 2), P(2, 1), C(S(), P(2, 2))).is_valid

    assert not C(C(P(2, 1), S()), S()).is_valid
    assert not C(Z(), C(P(2, 1), S())).is_valid
    assert not C(P(2, 1), P(3, 1), S()).is_valid


def test_r_is_valid():
    assert R(S(), Z(), P(1, 1)).is_valid
    assert R(S(), P(2, 1), Z()).is_valid
    assert R(P(2, 1), P(3, 2), P(3, 1), Z(), Z()).is_valid
    assert R(P(2, 1), P(4, 2), P(4, 1), S(), P(1, 1)).is_valid
    assert R(P(2, 1), Z(), Z(), Z(), Z()).is_valid
    assert R(P(2, 1), Z(), Z(), Z(), P(2, 1)).is_valid

    # sub expr is invalid
    assert not R(S(), Z(), C(P(2, 2), S())).is_valid
    assert not R(P(2, 1), P(4, 2), C(P(3, 1), S()), S(), P(1, 1)).is_valid

    # step arity is invalid for base
    assert not R(S(), P(2, 1), S()).is_valid
    assert not R(P(2, 1), P(3, 2), P(3, 1), Z(), S()).is_valid
    assert not R(P(2, 1), Z(), S(), Z(), Z()).is_valid

    # termianl function is invalid
    assert not R(P(3, 1), Z(), Z(), Z(), S()).is_valid


def test_overflow_trace_back():
    """
    Show details of Overflow Error traceback
    when executed in standard python file
    """
    plus = R(P(1, 1), C(S(), P(3, 2)), P(1, 1))
    mult = R(P(1, 1), C(plus, P(3, 2), P(3, 3)), Z())
    power = R(P(1, 1), C(mult, P(3, 2), P(3, 3)), C(S(), Z()))
    tetra = R(P(1, 1), C(power, P(3, 2), P(3, 3)), C(S(), Z()))
    with pytest.raises(OverflowError):
        tetra.eval(5, 2)
