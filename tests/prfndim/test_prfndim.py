import pytest

from prfndim.prfndim import C, Expr, P, PrfSyntaxError, R, S, Z


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
