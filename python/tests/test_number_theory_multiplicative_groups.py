"""Multiplicative groups through the real native number-theory extension."""

import math

import pytest

from sympy import Float, Integer, Rational, S, is_primitive_root, n_order, primitive_root


@pytest.mark.parametrize("modulus, expected", [
    (2, 1), (3, 2), (4, 3), (9, 2), (18, 5), (25, 2), (27, 2), (50, 3),
])
def test_least_generators_cover_prime_powers_and_twice_prime_powers(modulus, expected):
    assert primitive_root(modulus) == expected
    assert is_primitive_root(expected, modulus) is True
    assert type(primitive_root(modulus)) is int
    alternative = primitive_root(modulus, smallest=False)
    assert is_primitive_root(alternative, modulus) is True


@pytest.mark.parametrize("modulus", [8, 12, 15, 16, 24, 45, 72, 105, 2**63])
def test_noncyclic_groups_have_no_primitive_root(modulus):
    assert primitive_root(modulus) is None
    assert primitive_root(modulus, smallest=False) is None
    assert is_primitive_root(modulus - 1, modulus) is False


def test_orders_match_independent_complete_small_group_enumeration():
    for modulus in range(2, 49):
        for a in range(1, modulus):
            if math.gcd(a, modulus) != 1:
                continue
            expected = next(k for k in range(1, modulus) if pow(a, k, modulus) == 1)
            assert n_order(a, modulus) == expected
            assert n_order(a - 13*modulus, modulus) == expected


def test_large_orders_and_prime_power_lifting_use_exact_integer_arithmetic():
    assert n_order(2, 3**40) == 2*3**39
    assert n_order(5, 2**63) == 2**61
    assert n_order(2**64 - 2, 2**64 - 1) == 2
    assert primitive_root(40487) == 5
    assert primitive_root(40487**2, smallest=False) == 40492
    assert is_primitive_root(40492, 40487**2) is True
    assert n_order(18_446_744_073_709_551_556, 18_446_744_073_709_551_557) == 2


def test_exact_integer_normalization_and_public_module_identity():
    from sympy.ntheory import n_order as number_order
    from sympy.ntheory.residue_ntheory import (
        is_primitive_root as residue_is_root,
        n_order as residue_order,
        primitive_root as residue_root,
    )

    assert number_order is residue_order is n_order
    assert residue_root is primitive_root
    assert residue_is_root is is_primitive_root
    assert n_order(S.NegativeOne, Integer(4)) == 2
    assert n_order((1 << 200)*7 + 4, 7) == 3
    assert n_order(Rational(6, 2), 10) == 4
    assert primitive_root(Integer(18)) == 5
    assert is_primitive_root(-1, 4) is True


def test_invalid_domains_are_errors_instead_of_false_or_none():
    for modulus in (0, 1, -7):
        with pytest.raises(ValueError):
            n_order(1, modulus)
        with pytest.raises(ValueError):
            primitive_root(modulus)
        with pytest.raises(ValueError):
            is_primitive_root(1, modulus)
    for a, modulus in ((0, 2), (2, 4), (3, 9), (12, 18)):
        with pytest.raises(ValueError, match="relatively prime"):
            n_order(a, modulus)
        with pytest.raises(ValueError, match="relatively prime"):
            is_primitive_root(a, modulus)
    for invalid in (True, 2.0, Float(2), Rational(3, 2), "2"):
        with pytest.raises(ValueError):
            primitive_root(invalid)
        for operation in (n_order, is_primitive_root):
            with pytest.raises(ValueError):
                operation(invalid, 7)
            with pytest.raises(ValueError):
                operation(1, invalid)
    with pytest.raises(OverflowError):
        n_order(1, 2**64)
    with pytest.raises(OverflowError):
        primitive_root(2**64)
    with pytest.raises(OverflowError):
        is_primitive_root(1, 2**64)


def test_custom_integer_hooks_do_not_run_during_native_admission():
    class CustomInteger(int):
        def __int__(self):
            raise AssertionError("custom conversion hook ran")

    for operation in (n_order, is_primitive_root):
        with pytest.raises(ValueError):
            operation(CustomInteger(2), 7)
    with pytest.raises(ValueError):
        primitive_root(CustomInteger(7))
