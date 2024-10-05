from AlphaMu.pytest_check import pytest_check


def test_pytest_check():
    assert pytest_check(1, 2) == 3, "Error"
