"""Exact native power-residue decisions with no composite residue scan."""

import pytest

from sympy import Float, Integer, Rational, S, is_nthpow_residue


def test_every_small_ring_and_power_matches_complete_enumeration():
    for modulus in range(1, 33):
        for degree in range(1, 9):
            residues = {pow(x, degree, modulus) for x in range(modulus)}
            for target in range(modulus):
                expected = target in residues
                assert is_nthpow_residue(target, degree, modulus) is expected
                assert is_nthpow_residue(target - 7*modulus, degree, modulus) is expected


@pytest.mark.parametrize("target,degree,modulus,expected", [
    (3**5, 5, 3**40, True),
    (3**6, 5, 3**40, False),
    (8, 3, 3**40, True),
    (2, 3, 3**40, False),
    (8, 3, 7**22, True),
    (3, 3, 7**22, False),
    (3, 5, 7**22, True),
    (7**3, 3, 7**22, True),
    (7**4, 3, 7**22, False),
    (64, 3, 2**63, True),
    (32, 3, 2**63, False),
])
def test_large_prime_power_valuation_and_unit_conditions(target, degree, modulus, expected):
    assert is_nthpow_residue(target, degree, modulus) is expected


def test_two_adic_units_distinguish_squares_fourth_and_eighth_powers():
    assert is_nthpow_residue(9, 2, 2**63) is True
    assert is_nthpow_residue(9, 4, 2**63) is False
    assert is_nthpow_residue(17, 4, 2**63) is True
    assert is_nthpow_residue(17, 8, 2**63) is False
    assert is_nthpow_residue(33, 8, 2**63) is True
    assert is_nthpow_residue(1, 2**63, 2**63) is True
    assert is_nthpow_residue(1 + 2**62, 2**63, 2**63) is False
    assert is_nthpow_residue(2**64 - 1, 2**64 - 1, 2**63) is True


def test_mixed_composites_and_large_prime_moduli_use_exact_criteria():
    for modulus in (2**64 - 1, 1_000_000_007*1_000_000_009):
        target = pow(123_456_789, 5, modulus)
        assert is_nthpow_residue(target, 5, modulus) is True
    modulus = 3**12*5**8
    assert is_nthpow_residue(pow(45, 6, modulus), 6, modulus) is True
    assert is_nthpow_residue(3**5*5**6, 6, modulus) is False
    prime = 18_446_744_073_709_551_557
    assert is_nthpow_residue(1, prime - 1, prime) is True
    assert is_nthpow_residue(2, prime - 1, prime) is False


def test_zero_degree_preserves_the_explicit_pinned_public_convention():
    assert is_nthpow_residue(0, 0, 9) is False
    assert is_nthpow_residue(1, 0, 9) is True
    assert is_nthpow_residue(2, 0, 9) is False
    assert is_nthpow_residue(0, 0, 1) is False
    assert is_nthpow_residue(1, 0, 1) is False
    assert is_nthpow_residue(0, 1, 1) is True
    assert is_nthpow_residue(0, 3, 9) is True


def test_exact_admission_and_public_module_identity():
    from sympy.ntheory.residue_ntheory import is_nthpow_residue as residue_test

    assert residue_test is is_nthpow_residue
    assert is_nthpow_residue(Integer(8), Integer(3), Integer(27)) is True
    assert is_nthpow_residue(S.NegativeOne, 3, 27) is True
    assert is_nthpow_residue((1 << 200)*27 + 8, 3, 27) is True
    for invalid in (True, 2.0, Float(2), Rational(3, 2), "2"):
        for arguments in ((invalid, 3, 7), (1, invalid, 7), (1, 3, invalid)):
            with pytest.raises(ValueError):
                is_nthpow_residue(*arguments)
    with pytest.raises(ValueError):
        is_nthpow_residue(1, -1, 7)
    for modulus in (0, -7):
        with pytest.raises(ValueError):
            is_nthpow_residue(1, 3, modulus)
    with pytest.raises(OverflowError):
        is_nthpow_residue(1, 3, 2**64)
    with pytest.raises(OverflowError):
        is_nthpow_residue(1, 2**64, 7)
