"""Exact public real-root identities and rendering; require the actual extension."""

from decimal import Decimal

import pytest

import sympy
from sympy import CRootOf, Float, Integer, Poly, Rational, Symbol, all_roots, real_roots, sqrt
from sympy.core import _native, _native_expr


def _coefficients(values):
    return [_native_expr(value) for value in values]


def test_real_index_and_complete_real_roots_do_not_use_floating_root_finding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real-root identity must not use approximate roots")

    monkeypatch.setattr(sympy, "_numeric_poly_roots", forbidden)
    x = Symbol("x")
    f = x**5 - x + 1
    root = CRootOf(f, 0)
    assert root.is_real is True
    assert str(root.evalf(30)) == "-1.16730397826141868425604589985"
    assert real_roots((x - 1)*f, radicals=False) == [root, Integer(1)]
    assert all_roots(x**4 - 5*x**2 + 6, radicals=False) == [
        CRootOf(x**2 - 3, 0), CRootOf(x**2 - 2, 0),
        CRootOf(x**2 - 2, 1), CRootOf(x**2 - 3, 1),
    ]


def test_real_root_decimal_rendering_preserves_fifty_correct_digits():
    x = Symbol("x")
    expected = "1.4142135623730950488016887242096980785696718753769"
    value = CRootOf(x**2 - 2, 1).evalf(50)
    assert isinstance(value, Float)
    assert value.dps == 50
    assert str(value) == expected
    assert str(CRootOf(x**2 - 2, 0).evalf(50)) == "-" + expected
    # This checks the decimal rendering, not arbitrary-precision arithmetic
    # on the current shell Float backend.


def test_distinct_real_roots_closer_than_binary64_are_ordered_exactly():
    x = Symbol("x")
    scale = 10**40
    roots = real_roots((scale*x - scale)**2 - 2, radicals=False)
    assert len(roots) == 2
    assert roots[0] != roots[1]
    lo, hi = [Decimal(str(root.evalf(70))) for root in roots]
    assert lo < Decimal(1) < hi
    assert roots[0].is_real is True and roots[1].is_real is True


def test_near_axis_complex_pair_is_not_mistaken_for_two_real_roots(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real-root absence must be exact")

    monkeypatch.setattr(sympy, "_numeric_poly_roots", forbidden)
    x = Symbol("x")
    f = (x - 1)**2 + Rational(1, 10**80)
    assert real_roots(f, radicals=False) == []


def test_original_multiplicities_and_multiple_false_are_preserved():
    x = Symbol("x")
    f = (x**2 - 2)**3 * (x - 1)**2
    negative, positive = CRootOf(x**2 - 2, 0), CRootOf(x**2 - 2, 1)
    assert real_roots(f, radicals=False, multiple=False) == [
        (negative, 3), (Integer(1), 2), (positive, 3),
    ]
    assert real_roots(f, radicals=False) == [negative]*3 + [Integer(1)]*2 + [positive]*3
    assert CRootOf(f, -1) == positive


def test_real_radicals_use_exact_signs_for_order_and_odd_negative_roots():
    x = Symbol("x")
    assert real_roots(-x**2 + 2) == [-sqrt(2), sqrt(2)]
    assert CRootOf(x**3 + 2, 0, radicals=True) == -Integer(2)**Rational(1, 3)
    assert real_roots(x**4 - 16) == [Integer(-2), Integer(2)]


def test_bound_generator_follows_pinned_alpha_equivalence():
    x = Symbol("x", real=True)
    other = Symbol("t", positive=True)
    root = CRootOf(x**5 - x + 1, x, 0)
    renamed = CRootOf(other**5 - other + 1, other, 0)
    assert root.free_symbols == set()
    assert root.subs(x, other) == root
    assert root.poly.gen == Symbol("x")
    assert renamed.poly.gen == Symbol("t")
    assert root == renamed and hash(root) == hash(renamed)
    assert str(root.evalf(30)) == "-1.16730397826141868425604589985"


def test_typed_bridge_rejects_nonliteral_and_oversized_coefficients():
    with pytest.raises((ValueError, TypeError)):
        _native.real_roots_isolate_native(_coefficients([Float(1.5), 1]))
    with pytest.raises(ValueError):
        _native.real_roots_isolate_native(_coefficients([1]*66))
    with pytest.raises(ValueError):
        _native.real_roots_isolate_native(_coefficients([Integer(1 << 4096), 1]))
    with pytest.raises(ValueError):
        _native.real_roots_isolate_native(_coefficients([0]))


def test_typed_factor_association_rejects_unrelated_or_incomplete_factors():
    with pytest.raises(ValueError):
        _native.indexed_real_roots_native(
            _coefficients([-1, 0, 1]), [_coefficients([-2, 1])], [1],
        )
    with pytest.raises(ValueError):
        _native.indexed_real_roots_native(
            _coefficients([1, -2, 1]),
            [_coefficients([-1, 1]), _coefficients([-1, 1])], [1, 1],
        )


def test_native_rounding_resolves_exact_zero_and_nonlinear_half_even_ties():
    assert _native.real_root_decimal_native(_coefficients([0, 1, 0, -1]), 1, 100) == "0"
    # (x-5/4)(x-3); the first root is exactly at a two-digit rounding tie.
    assert _native.real_root_decimal_native(
        _coefficients([Rational(15, 4), Rational(-17, 4), 1]), 0, 2,
    ) == "1.2e0"


def test_invalid_indices_precision_and_nonreal_precision_refuse():
    x = Symbol("x")
    with pytest.raises(ValueError):
        CRootOf(x**2 - 2, Rational(1, 2))
    with pytest.raises(IndexError):
        CRootOf(x**2 - 2, 2)
    for digits in (0, 513):
        with pytest.raises(ValueError):
            CRootOf(x**2 - 2, 1).evalf(digits)
    with pytest.raises(NotImplementedError):
        CRootOf(x**2 + 1, 0).evalf(30)
    with pytest.raises(TypeError):
        float(CRootOf(x**2 + 1, 0))


def test_nonzero_constant_has_no_real_roots():
    x = Symbol("x")
    assert real_roots(Poly(7, x)) == []


def test_poly_methods_and_public_positional_multiplicity_flags():
    x = Symbol("x")
    f = (x**2 - 2)**3 * (x - 1)**2
    expected = [(CRootOf(x**2 - 2, 0), 3), (Integer(1), 2), (CRootOf(x**2 - 2, 1), 3)]
    assert real_roots(f, False, False) == expected
    assert Poly(f, x).real_roots(False, False) == expected
    assert all_roots(f, False, False) == expected
    assert Poly(f, x).all_roots(False, False) == expected
    assert real_roots(f, False, False, True) == expected


def test_original_held_tree_is_checked_before_polynomial_expansion_or_cancellation():
    from sympy import Add, Mul, Pow

    x = Symbol("x")
    undefined = Pow(Integer(0), Integer(0), evaluate=False)
    inputs = [
        Pow(x, Integer(10**30), evaluate=False),
        Add(x, Mul(0, undefined, evaluate=False), evaluate=False),
        Pow(Add(x, -x, evaluate=False), Integer(0), evaluate=False),
        Pow(Pow(x, -1, evaluate=False), -1, evaluate=False),
    ]
    for expression in inputs:
        with pytest.raises((ValueError, NotImplementedError)):
            CRootOf(expression, x, 0)
        with pytest.raises((ValueError, NotImplementedError)):
            real_roots(Poly(expression, x))
