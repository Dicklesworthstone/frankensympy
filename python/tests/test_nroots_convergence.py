"""Public numerical-root convergence, scaling and component-retention regressions."""

import math

import pytest
import sympy
from sympy.polys._numeric_roots import NoConvergence


def _assert_roots(actual, expected, relative=5e-13):
    remaining = [complex(root) for root in actual]
    assert len(remaining) == len(expected)
    for root in expected:
        nearest = min(range(len(remaining)), key=lambda i: abs(remaining[i] - root))
        candidate = remaining.pop(nearest)
        if root == 0:
            assert candidate == 0
        else:
            assert abs(candidate - root) / abs(root) <= relative


@pytest.mark.parametrize("large", [10**10, 10**20])
def test_large_root_does_not_turn_imaginary_pair_into_real_roots(large):
    x = sympy.Symbol("x")
    roots = sympy.nroots((x - large) * (x**2 + 1))
    _assert_roots(roots, [large, -1j, 1j])
    assert sum(complex(root).imag == 0 for root in roots) == 1


def test_tiny_nonreal_roots_survive_formatting_and_cleanup():
    x = sympy.Symbol("x")
    roots = sympy.nroots(x**2 + sympy.Rational(1, 10**24))
    _assert_roots(roots, [-1e-12j, 1e-12j])
    assert all(complex(root).imag != 0 for root in roots)


@pytest.mark.parametrize("domain", ["ZZ", "QQ", None])
def test_checked_square_free_path_retains_exact_multiplicities(domain):
    x = sympy.Symbol("x")
    p = sympy.Poly((x - 1)**3 * (x**2 + 1)**2, x, domain=domain)
    _assert_roots(p.nroots(), [1, 1, 1, -1j, -1j, 1j, 1j])


def test_repeated_tiny_imaginary_roots_remain_nonzero():
    x = sympy.Symbol("x")
    p = (x - 1)**3 * (x**2 + sympy.Rational(1, 10**24))**2
    _assert_roots(sympy.nroots(p), [1, 1, 1, -1e-12j, -1e-12j, 1e-12j, 1e-12j])


@pytest.mark.parametrize("method", ["module", "Poly"])
def test_requested_iteration_exhaustion_is_not_published(method):
    x = sympy.Symbol("x")
    p = sympy.Poly(x**3 - 2*x - 5, x)
    with pytest.raises(NoConvergence, match="maxsteps"):
        if method == "module":
            sympy.nroots(p, maxsteps=1)
        else:
            p.nroots(maxsteps=1)
    assert len(sympy.nroots(p, maxsteps=50)) == 3


@pytest.mark.parametrize("scalar", [10**400, sympy.Rational(1, 10**400)])
def test_exact_common_coefficient_scale_never_overflows_before_normalization(scalar):
    x = sympy.Symbol("x")
    _assert_roots(sympy.nroots(scalar * (x**2 - 2)), [-math.sqrt(2), math.sqrt(2)])


def test_unrepresentable_exact_coefficient_is_refused_before_zero_deflation():
    x = sympy.Symbol("x")
    with pytest.raises(ValueError, match="binary64 precision"):
        sympy.nroots(x + sympy.Rational(1, 10**1000))


def test_subnormal_exact_coefficient_cannot_claim_fifteen_digit_accuracy():
    x = sympy.Symbol("x")
    with pytest.raises(ValueError, match="binary64 precision"):
        sympy.nroots(x + sympy.Rational(1, 10**321))


def test_component_conversion_cannot_drop_an_exact_imaginary_part():
    # The conversion boundary is exercised directly because the current
    # Poly coefficient extractor itself admits a narrower rational slice.
    with pytest.raises(ValueError, match="binary64 precision"):
        sympy._nroots_coefficients([sympy.Integer(1), sympy.I / 10**1000])
    with pytest.raises(ValueError, match="binary64 precision"):
        sympy._nroots_coefficients([sympy.Integer(1), 1 + sympy.I / 10**1000])


def test_shared_legacy_helper_forwards_cleanup_without_chopping():
    raw = sympy._numeric_poly_roots([1, complex(-1, -1e-17)], cleanup=False)
    cleaned = sympy._numeric_poly_roots([1, complex(-1, -1e-17)], cleanup=True)
    assert raw == [complex(1, 1e-17)]
    assert cleaned == [complex(1, 0)]


def test_square_free_failure_does_not_restart_the_unchecked_path(monkeypatch):
    x = sympy.Symbol("x")
    p = sympy.Poly((x - 1)**2, x)

    def reject(_self, *args, **kwargs):
        raise RuntimeError("square-free source admission failed")

    monkeypatch.setattr(sympy.Poly, "sqf_list", reject)
    with pytest.raises(RuntimeError, match="square-free source admission failed"):
        sympy.nroots(p)


def test_field_polynomial_cannot_be_reinterpreted_as_complex_integer_roots():
    x = sympy.Symbol("x")
    p = sympy.Poly(x**2 + 1, x, modulus=5)
    with pytest.raises(NotImplementedError):
        sympy.nroots(p)


def test_zero_and_constant_polynomials_have_no_roots():
    x = sympy.Symbol("x")
    assert sympy.nroots(sympy.Poly(0, x)) == []
    assert sympy.nroots(sympy.Poly(7, x)) == []


def test_binary64_precision_limit_remains_explicit():
    x = sympy.Symbol("x")
    with pytest.raises(NotImplementedError, match="at most 15"):
        sympy.nroots(x**2 - 2, n=30)
    for precision in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            sympy.nroots(x**2 - 2, n=precision)
