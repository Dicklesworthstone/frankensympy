"""Least discrete logarithms through the real bounded native implementation."""

import pytest

from sympy import Float, Integer, Rational, S, discrete_log


@pytest.mark.parametrize("modulus, target, base, expected", [
    (17, 16, 16, 1),
    (41, 40, 40, 1),
    (41, 15, 7, 3),
    (17, 13, 4, 3),
    (9, 0, 3, 2),
    (12, 4, 2, 2),
    (12, 8, 2, 3),
    (24, 0, 6, 3),
    (8, 0, 0, 1),
    (8, 1, 0, 0),
    (1, 0, 0, 0),
])
def test_least_exponents_include_low_order_bases_and_nonunit_tails(modulus, target, base, expected):
    assert discrete_log(modulus, target, base) == expected
    assert discrete_log(modulus, target - 11*modulus, base + 13*modulus) == expected
    assert type(discrete_log(modulus, target, base)) is int


def test_small_rings_match_complete_independent_power_enumeration():
    for modulus in range(1, 25):
        for base in range(modulus):
            powers = {}
            for exponent in range(modulus + 1):
                powers.setdefault(pow(base, exponent, modulus), exponent)
            for target in range(modulus):
                if target in powers:
                    assert discrete_log(modulus, target, base) == powers[target]
                else:
                    with pytest.raises(ValueError, match="Log does not exist"):
                        discrete_log(modulus, target, base)


def test_order_hints_are_verified_and_do_not_change_least_exponents():
    assert discrete_log(17, 16, 16, order=2) == 1
    assert discrete_log(41, 15, 7, order=40) == 3
    assert discrete_log(41, 15, 7, order=80) == 3
    assert discrete_log(17, 13, 4, order=8) == 3
    assert discrete_log(17, 16, 16, order=16) == 1
    assert discrete_log(15, 4, 4, order=4) == 1
    assert discrete_log(8, 1, 2, order=2) == 0
    assert discrete_log(9, 0, 3, order=6) == 2
    with pytest.raises(ValueError, match="Log does not exist"):
        discrete_log(8, 5, 3, order=2)
    for order in (0, -1, 1, 2, 3, 6):
        with pytest.raises(ValueError):
            discrete_log(17, 13, 4, order=order)
    with pytest.raises(ValueError, match="supplied order"):
        discrete_log(12, 8, 2, order=3)


def test_large_groups_and_word_boundary_nonunits_use_exact_arithmetic():
    modulus = 1_000_000_007
    exponent = 1_234_567
    target = pow(5, exponent, modulus)
    assert discrete_log(modulus, target, 5) == exponent
    assert discrete_log(modulus, target, 5, order=modulus - 1) == exponent
    assert discrete_log(2**63, 1 + 2**62, 1 + 2**61) == 2
    assert discrete_log(2**63, 2**61, 2) == 61
    assert discrete_log(2**63, 0, 2) == 63


def test_search_resource_refusal_is_distinct_from_mathematical_nonexistence():
    # The solution is exactly 2, but the complete configured search would need
    # 92682 baby steps. Refusal must not be reported as a nonexistent logarithm.
    with pytest.raises(RuntimeError, match="baby steps"):
        discrete_log(2**35, 25, 5)
    with pytest.raises(ValueError, match="Log does not exist"):
        discrete_log(8, 5, 3)


def test_exact_admission_and_residue_module_identity():
    from sympy.ntheory.residue_ntheory import discrete_log as residue_log

    assert residue_log is discrete_log
    assert discrete_log(Integer(17), S.NegativeOne, Integer(-1), order=Integer(2)) == 1
    assert discrete_log(7, (1 << 200)*7 + 4, 2) == 2
    for invalid in (True, 2.0, Float(2), Rational(3, 2), "2"):
        for arguments in ((invalid, 1, 1), (7, invalid, 1), (7, 1, invalid)):
            with pytest.raises(ValueError):
                discrete_log(*arguments)
        with pytest.raises(ValueError):
            discrete_log(17, 16, 16, order=invalid)
    for modulus in (0, -7):
        with pytest.raises(ValueError, match="positive"):
            discrete_log(modulus, 1, 1)
    with pytest.raises(OverflowError):
        discrete_log(2**64, 1, 1)
