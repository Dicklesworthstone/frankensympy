"""Public factorization and composite modular roots; use the real extension."""

import pytest

from sympy import (
    Float, Integer, Rational, S, divisor_count, divisor_sigma, factorint,
    is_quad_residue, mobius, sqrt_mod, totient,
)
from sympy.ntheory import carmichael, is_nthpow_residue, primefactors


def test_large_semiprime_factorization_and_existing_consumers():
    p, q = 1_000_000_007, 1_000_000_009
    n = p*q
    assert factorint(n) == {p: 1, q: 1}
    assert factorint(-n) == {-1: 1, p: 1, q: 1}
    assert factorint(0) == {0: 1}
    assert factorint(1) == {}
    assert factorint(S.Zero) == {0: 1}
    assert factorint(S.One) == {}
    assert factorint(S.NegativeOne) == {-1: 1}
    assert factorint(4_294_967_291**2) == {4_294_967_291: 2}
    assert primefactors(n) == [p, q]
    assert totient(n) == (p - 1)*(q - 1)
    assert divisor_count(n) == 4
    assert divisor_sigma(n) == (p + 1)*(q + 1)
    assert mobius(n) == 1
    assert carmichael(n) == (p - 1)//2*(q - 1)


@pytest.mark.parametrize("a,modulus,expected", [
    (0, 9, [0, 3, 6]),
    (4, 8, [2, 6]),
    (1, 8, [1, 3, 5, 7]),
    (9, 27, [3, 6, 12, 15, 21, 24]),
    (0, 1, [0]),
    (2, 1, [0]),
    (2, 3, []),
    (-1, 5, [2, 3]),
    (1, -8, [1, 3, 5, 7]),
])
def test_complete_prime_power_and_composite_roots(a, modulus, expected):
    assert sqrt_mod(a, modulus, all_roots=True) == expected
    one = sqrt_mod(a, modulus)
    if expected:
        assert one in expected
        assert one <= abs(modulus)//2
    else:
        assert one is None
    if modulus > 0:
        assert is_quad_residue(a, modulus) is bool(expected)
        assert is_nthpow_residue(a, 2, modulus) is bool(expected)


def test_small_moduli_match_complete_exhaustive_reference():
    for modulus in range(1, 65):
        for a in (-2, 0, 1, 2, modulus - 1, modulus + 1):
            expected = [r for r in range(modulus) if r*r % modulus == a % modulus]
            assert sqrt_mod(a, modulus, all_roots=True) == expected
            assert is_quad_residue(a, modulus) is bool(expected)
            one = sqrt_mod(a, modulus)
            assert (one is None) is (not expected)
            if one is not None:
                assert one in expected and one <= modulus//2


def test_large_crt_system_and_high_two_adic_precision():
    n = 1_000_000_007*1_000_000_009
    roots = sqrt_mod(4, n, all_roots=True)
    assert len(roots) == 4
    assert roots == sorted(set(roots))
    assert all(r*r % n == 4 for r in roots)
    assert sqrt_mod(1, 2**63, all_roots=True) == [
        1, 2**62 - 1, 2**62 + 1, 2**63 - 1,
    ]


def test_one_root_and_existence_do_not_enumerate_large_solution_sets():
    # There are 2**31 roots. Scalar/existence requests must remain usable.
    assert sqrt_mod(0, 2**62) == 0
    assert is_quad_residue(0, 2**62) is True
    with pytest.raises(ValueError):
        sqrt_mod(0, 2**62, all_roots=True)


def test_exact_input_admission_and_modulus_conventions():
    assert sqrt_mod(Integer(4), Integer(8), all_roots=True) == [2, 6]
    assert sqrt_mod(S.Zero, S.One, all_roots=True) == [0]
    assert sqrt_mod(S.One, 8, all_roots=True) == [1, 3, 5, 7]
    assert sqrt_mod(S.NegativeOne, 5, all_roots=True) == [2, 3]
    assert is_quad_residue(S.Zero, S.One) is True
    assert sqrt_mod(Rational(4, 2), 7) == 3
    assert sqrt_mod((1 << 200)*7 + 2, 7) == 3
    for invalid in (True, 2.0, Float(2), Rational(3, 2), "2"):
        for operation in (sqrt_mod, is_quad_residue):
            with pytest.raises(ValueError):
                operation(invalid, 7)
            with pytest.raises(ValueError):
                operation(1, invalid)
    with pytest.raises(ZeroDivisionError):
        sqrt_mod(0, 0)
    for modulus in (0, -8):
        with pytest.raises(ValueError):
            is_quad_residue(1, modulus)
    with pytest.raises(OverflowError):
        sqrt_mod(1, 2**64)


def test_custom_integer_hooks_are_not_executed_by_exact_native_admission():
    class CustomInteger(int):
        def __int__(self):
            raise AssertionError("custom integer conversion executed")

    for operation in (sqrt_mod, is_quad_residue):
        with pytest.raises(ValueError):
            operation(CustomInteger(4), 8)
