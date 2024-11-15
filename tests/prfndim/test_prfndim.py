from prfndim.prfndim import Z


def test_Z():
    z1 = Z()
    z2 = Z()
    assert z1 is z2
    assert z1.eval(1, 2, 3) == 0
    assert str(z1) == "Z()"
    assert z1.arity is None
