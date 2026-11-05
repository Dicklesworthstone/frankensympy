"""Strip-aware bilateral rational inverse Mellin transforms.

The mathematical comparator is exact symbolic equality on x > 0; the
public-object checks separately cover retained strips and typed variables.
"""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


def test_one_pole_has_different_bilateral_support_on_opposite_strips():
    z, x = s.symbols("mellin_s mellin_x")
    below_one = s.Heaviside(1 - x)
    above_one = s.Heaviside(x - 1)
    assert s.inverse_mellin_transform(1/z, z, x, (0, s.oo)) == below_one
    assert s.inverse_mellin_transform(1/z, z, x, (-s.oo, 0)) == -above_one
    # Strict sub-strips choose the same inverse; no contour midpoint guess.
    assert s.inverse_mellin_transform(1/z, z, x, (s.Rational(1, 3), 2)) == below_one
    assert s.inverse_mellin_transform(1/z, z, x, (-2, s.Rational(-1, 3))) == -above_one


def test_three_strips_of_two_real_poles_select_every_required_mode():
    z, x = s.symbols("mellin_two_s mellin_two_x")
    source = 1/(z*z - 1)
    left, right = s.Heaviside(1-x), s.Heaviside(x-1)
    cases = [
        ((1, s.oo), (1/x - x)*left/2),
        ((-1, 1), -x*left/2 - right/(2*x)),
        ((-s.oo, -1), (x - 1/x)*right/2),
    ]
    for strip, expected in cases:
        equal(s.inverse_mellin_transform(source, z, x, strip), expected)


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_repeated_poles_retain_logarithmic_orders_and_bilateral_signs(order):
    z, x = s.symbols("mellin_repeated_s mellin_repeated_x")
    polynomial = (-s.log(x))**(order-1)/s.factorial(order-1)
    equal(s.inverse_mellin_transform(1/z**order, z, x, (0, s.oo)),
          polynomial*s.Heaviside(1-x))
    equal(s.inverse_mellin_transform(1/z**order, z, x, (-s.oo, 0)),
          -polynomial*s.Heaviside(x-1))


def test_scaled_and_shifted_repeated_poles_keep_exact_coefficients():
    z, x = s.symbols("mellin_shift_s mellin_shift_x")
    equal(s.inverse_mellin_transform(1/(2*z+3)**3, z, x, (-1, s.oo)),
          x**s.Rational(3, 2)*s.log(x)**2*s.Heaviside(1-x)/16)
    equal(s.inverse_mellin_transform(3/(z-2)**4, z, x, (2, s.oo)),
          -s.log(x)**3*s.Heaviside(1-x)/(2*x*x))


def test_complex_quadratic_poles_produce_real_logarithmic_oscillations():
    z, x = s.symbols("mellin_quadratic_s mellin_quadratic_x")
    equal(s.inverse_mellin_transform(1/(z*z+1), z, x, (0, s.oo)),
          -s.sin(s.log(x))*s.Heaviside(1-x))
    equal(s.inverse_mellin_transform(1/(z*z+1), z, x, (-s.oo, 0)),
          s.sin(s.log(x))*s.Heaviside(x-1))
    equal(s.inverse_mellin_transform((z+2)/((z+2)**2+9), z, x, (-2, s.oo)),
          x*x*s.cos(3*s.log(x))*s.Heaviside(1-x))


def test_repeated_quadratics_keep_all_logarithmic_terms():
    z, x = s.symbols("mellin_repeated_quadratic_s mellin_repeated_quadratic_x")
    logarithm = s.log(x)
    inverse = (logarithm*s.cos(logarithm) - s.sin(logarithm))/2
    equal(s.inverse_mellin_transform(1/(z*z+1)**2, z, x, (0, s.oo)),
          inverse*s.Heaviside(1-x))
    equal(s.inverse_mellin_transform(1/(z*z+1)**2, z, x, (-s.oo, 0)),
          -inverse*s.Heaviside(x-1))


def test_real_irrational_quadratic_can_split_across_a_rational_strip():
    z, x = s.symbols("mellin_split_s mellin_split_x")
    root = s.sqrt(2)
    expected = -(x**root*s.Heaviside(1-x)
                 + x**(-root)*s.Heaviside(x-1))/(2*root)
    equal(s.inverse_mellin_transform(1/(z*z-2), z, x, (-1, 1)), expected)


def test_repeated_real_quadratic_split_keeps_each_pole_polynomial():
    z, x = s.symbols("mellin_split_repeated_s mellin_split_repeated_x")
    root = s.sqrt(2)
    expected = (
        x**root*(root/s.Integer(16)-s.log(x)/8)*s.Heaviside(1-x)
        + x**(-root)*(root/s.Integer(16)+s.log(x)/8)*s.Heaviside(x-1)
    )
    equal(s.inverse_mellin_transform(1/(z*z-2)**2, z, x, (-1, 1)), expected)


def test_mixed_pole_families_share_one_strip_and_keep_their_supports():
    z, x = s.symbols("mellin_mixed_s mellin_mixed_x")
    source = 1/(z+2)**2 + 3/(z-2) + 1/(z*z+2*z+2)
    expected = (-x*x*s.log(x)-x*s.sin(s.log(x)))*s.Heaviside(1-x)
    expected -= 3*x**-2*s.Heaviside(x-1)
    equal(s.inverse_mellin_transform(source, z, x, (0, 1)), expected)


def test_noconds_and_unevaluated_class_reuse_the_original_strip():
    z, x = s.symbols("mellin_doit_s mellin_doit_x")
    transform = s.InverseMellinTransform(1/z, z, x, -s.oo, 0)
    assert transform.args == (1/z, z, x, -s.oo, 0)
    assert transform.fundamental_strip == (-s.oo, 0)
    assert transform.free_symbols == {x}
    assert transform.func(*transform.args) == transform
    assert transform.doit() == -s.Heaviside(x-1)
    assert transform.doit(noconds=False) == (-s.Heaviside(x-1), True)
    assert s.InverseMellinTransform(1/z, z, x, (0, s.oo)).doit() == s.Heaviside(1-x)
    assert s.InverseMellinTransform(1/z, z, x, strip=(0, s.oo)).doit() == s.Heaviside(1-x)


def test_unsupported_inverses_remain_unevaluated_with_both_endpoints():
    z, x = s.symbols("mellin_unsupported_s mellin_unsupported_x")
    for source in (s.Integer(1), z/(z+1), 1/(z**3-z+1)):
        for noconds in (True, False):
            result = s.inverse_mellin_transform(source, z, x, (2, s.oo), noconds=noconds)
            assert type(result) is s.InverseMellinTransform
            assert result.args == (source, z, x, 2, s.oo)
            assert result.fundamental_strip == (2, s.oo)


def test_unspecified_strip_is_preserved_without_guessing_a_contour(monkeypatch):
    z, x = s.symbols("mellin_unknown_strip_s mellin_unknown_strip_x")

    def forbidden(*args):
        raise AssertionError("unknown strip entered the exact native lane")

    monkeypatch.setattr(_native, "inverse_rational_mellin_native", forbidden)
    for strip in (None, (None, 1), (-1, None)):
        result = s.inverse_mellin_transform(1/z, z, x, strip)
        expected = (None, None) if strip is None else strip
        assert result.fundamental_strip == expected
        assert result.doit() == result
        assert result.free_symbols == {x}


@pytest.mark.parametrize("strip", [(0, 0), (1, -1), (s.oo, s.oo), (-1, 1)])
def test_empty_or_pole_crossing_strips_are_rejected(strip):
    z, x = s.symbols("mellin_invalid_strip_s mellin_invalid_strip_x")
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(1/z, z, x, strip)


def test_source_holes_survive_cancellation_and_zero_shortcuts():
    z, x = s.symbols("mellin_held_s mellin_held_x")
    inverse = s.Pow(z, -1, evaluate=False)
    held_one = s.Pow(inverse, 0, evaluate=False)
    source = s.Mul(held_one, 1/(z-2), evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(source, z, x, (-1, 1))
    equal(s.inverse_mellin_transform(source, z, x, (0, 1)),
          -x**-2*s.Heaviside(x-1))
    undefined = s.Pow(s.Integer(0), -1, evaluate=False)
    hidden = s.Mul(0, undefined, evaluate=False)
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(hidden, z, x, (0, 1))


def test_typed_same_named_variables_and_dummy_owners_are_preserved():
    z = s.Symbol("mellin_same")
    x = s.Symbol("mellin_same", positive=True)
    result = s.inverse_mellin_transform(1/z, z, x, (0, s.oo))
    assert result.free_symbols == {x}
    assert next(iter(result.free_symbols)) is x
    first, second = s.Dummy("mellin_dummy"), s.Dummy("mellin_dummy")
    result = s.inverse_mellin_transform(1/first, first, second, (-s.oo, 0))
    assert result.free_symbols == {second}
    assert next(iter(result.free_symbols)) is second
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(1/z, z, z, (0, s.oo))


def test_output_assumptions_keep_the_public_heaviside_constructor_semantics():
    z = s.Symbol("mellin_axis_s")
    x = s.Symbol("mellin_axis_x", negative=True)
    result = s.inverse_mellin_transform(1/z, z, x, (0, s.oo))
    # Isolated SymPy 1.14 also simplifies this Heaviside to one. The native
    # Mellin identity itself is explicitly scoped to the positive x axis.
    assert result == s.Heaviside(1-x) == 1
    assert result.free_symbols == set()


def test_zero_transform_and_direct_native_bridge_agree():
    z, x = s.symbols("mellin_bridge_s mellin_bridge_x")
    assert s.inverse_mellin_transform(0, z, x, (-s.oo, s.oo)) == 0
    raw = _native.inverse_rational_mellin_native(
        _native_expr(1/z), _symbol_binding(z), _symbol_binding(x),
        _native_expr(-s.oo), _native_expr(0),
    )
    assert s.sympify(raw) == -s.Heaviside(x-1)


def test_custom_sources_and_strip_containers_do_not_run_python_hooks():
    z, x = s.symbols("mellin_hooks_s mellin_hooks_x")
    events = []

    class Converter:
        def _sympy_(self):
            events.append("convert")
            return 1/z

    class Strip(list):
        def __len__(self):
            events.append("length")
            return 2

    class ClassSpoof:
        @property
        def __class__(self):
            events.append("class")
            return s.Symbol

    with pytest.raises(TypeError):
        s.inverse_mellin_transform(Converter(), z, x, (0, s.oo))
    with pytest.raises(TypeError):
        s.inverse_mellin_transform(ClassSpoof(), z, x, (0, s.oo))
    with pytest.raises(TypeError):
        s.inverse_mellin_transform(1/z, ClassSpoof(), x, (0, s.oo))
    with pytest.raises(TypeError):
        s.inverse_mellin_transform(1/z, z, ClassSpoof(), (0, s.oo))
    with pytest.raises(TypeError):
        s.inverse_mellin_transform(1/z, z, x, Strip([0, s.oo]))
    assert events == []


def test_source_bounds_and_inexact_strips_do_not_publish_an_exact_inverse():
    z, x = s.symbols("mellin_bound_s mellin_bound_x")
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(1 << 4097, z, x, (0, s.oo))
    with pytest.raises(ValueError):
        s.inverse_mellin_transform(s.Pow(z, -10**9, evaluate=False), z, x, (0, s.oo))
    result = s.inverse_mellin_transform(1/z, z, x, (0.5, s.oo))
    assert type(result) is s.InverseMellinTransform
    assert result.fundamental_strip == (s.Float(0.5), s.oo)


def test_operational_failure_is_propagated_without_residue_or_solver_retry(monkeypatch):
    z, x = s.symbols("mellin_failure_s mellin_failure_x")
    failure = ValueError("Mellin verifier budget exhausted")
    calls = []

    def exhausted(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(_native, "inverse_rational_mellin_native", exhausted)
    with pytest.raises(ValueError) as caught:
        s.inverse_mellin_transform(1/z, z, x, (0, s.oo))
    assert caught.value is failure
    assert len(calls) == 1
