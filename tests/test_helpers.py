import pytest
import math
from kop2024.helpers import product, monomial_exponents


def test_product_empty():
    assert product([]) == 1.0


def test_product():
    assert product(range(1, 5 + 1)) == math.factorial(5)


def test_monomial_exponents_2_2():
    exponents = monomial_exponents(2, 2)
    assert len(exponents) == 3
    assert set(exponents) == {(0, 0), (0, 1), (1, 1)}


@pytest.mark.parametrize("n,d", [(1, 4), (2, 2), (3, 2), (2, 3), (2, 5), (5, 2)])
def test_monomial_exponents(n, d):
    exponents = monomial_exponents(n, d)
    assert len(exponents) == math.comb(n + d - 1, n - 1)
    for e in exponents:
        assert all(0 <= i < n for i in e)
        counts = [e.count(i) for i in set(e)]
        assert sum(counts) == d
