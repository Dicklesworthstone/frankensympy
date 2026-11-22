"""Conditional elementary primitives pinned to SymPy 1.14.0.

The admitted slope is a rational multiple of one typed parameter. Exact public
forms and independent branch derivatives are checked separately. Preview-only
tests cover deliberate refusals outside that profile; no definite convergence
or arbitrary symbolic-affine completeness claim is made here.
"""

import pytest
import sympy as s


def _expand_entire_exponential_products(expression):
    # Exact exp(u+v) = exp(u)*exp(v) normalization is the mathematical
    # derivative comparator here. Primitive comparisons remain structural.
    return s.expand(expression, mul=False, multinomial=False, power_exp=True,
                    power_base=False, log=False)


@pytest.mark.parametrize("name", ["exp", "sin", "cos", "sinh", "cosh"])
@pytest.mark.parametrize("scale", [1, -3, s.Rational(1, 2)])
def test_parameter_slope_has_exact_generic_and_degenerate_branches(name, scale):
    x, a, b = s.symbols("conditional_x conditional_a conditional_b")
    kernel = getattr(s, name)
    argument = scale*a*x + (0 if name == "exp" else b)
    generic = {
        "exp": s.exp(argument),
        "sin": -s.cos(argument),
        "cos": s.sin(argument),
        "sinh": s.cosh(argument),
        "cosh": s.sinh(argument),
    }[name] / (scale*a)
    degenerate = x*kernel(0 if name == "exp" else b)
    source = kernel(argument)
    # Odd-function construction extracts a negative argument sign before
    # integration, and the pinned profile retains that scalar outside.
    sign = -1 if scale < 0 and name in ("sin", "sinh") else 1
    expected = sign*s.Piecewise((sign*generic, s.Ne(a, 0)),
                                (sign*degenerate, True))
    actual = s.integrate(source, x)

    assert actual == expected
    conditional = sign*actual
    assert type(conditional) is s.Piecewise
    assert actual.free_symbols == source.free_symbols
    assert s.Integral(source, x).doit() == expected
    assert actual.subs(a, 0) == degenerate
    assert s.diff(conditional.args[0].expr, x) == sign*source
    assert s.diff(conditional.args[1].expr, x) == sign*source.subs(a, 0)
    for value in [-2, s.Rational(2, 3), 3]:
        specialized = actual.subs(a, value)
        assert specialized == generic.subs(a, value)
        assert s.diff(specialized, x) == source.subs(a, value)


@pytest.mark.parametrize("name", ["sin", "cos", "sinh", "cosh"])
def test_correlated_intercept_is_specialized_in_the_zero_branch(name):
    x, a, b = s.symbols("correlated_x correlated_a correlated_b")
    kernel = getattr(s, name)
    argument = 2*a*x + a + b
    generic = {
        "sin": -s.cos(argument), "cos": s.sin(argument),
        "sinh": s.cosh(argument), "cosh": s.sinh(argument),
    }[name] / (2*a)
    expected = s.Piecewise((generic, s.Ne(a, 0)), (x*kernel(b), True))
    actual = s.integrate(kernel(argument), x)
    assert actual == expected
    assert actual.subs(a, 0) == x*kernel(b)
    assert s.diff(actual.subs(a, 0), x) == kernel(b)


def test_exponential_with_a_vanishing_correlated_shift_has_the_pinned_form():
    x, a = s.symbols("correlated_exp_x correlated_exp_a")
    source = s.exp(a*x + a)
    expected = s.Piecewise((source/a, s.Ne(a, 0)), (x, True))
    actual = s.integrate(source, x)
    assert actual == expected
    assert actual.subs(a, 0) == x
    assert s.diff(actual.args[0].expr, x) == source


def test_exponential_coefficients_are_inside_the_conditional_branches():
    x, a, b = s.symbols("scaled_exp_x scaled_exp_a scaled_exp_b")
    for coefficient in [2, b, s.exp(b), a + b, a + 1, s.sin(a)]:
        source = coefficient*s.exp(a*x)
        expected = s.Piecewise(
            (source/a, s.Ne(a, 0)), (coefficient*x, True),
        )
        actual = s.integrate(source, x)
        assert actual == expected
        assert (_expand_entire_exponential_products(s.diff(actual.args[0].expr, x))
                == _expand_entire_exponential_products(source))
        assert s.diff(actual.subs(a, 0), x) == source.subs(a, 0)


def test_a_cancelled_exponential_scale_has_an_entire_unconditional_primitive():
    x, a, b = s.symbols("cancelled_exp_x cancelled_exp_a cancelled_exp_b")
    for coefficient in [a, a*b, a*a]:
        source = coefficient*s.exp(2*a*x)
        expected = coefficient*s.exp(2*a*x)/(2*a)
        actual = s.integrate(source, x)
        assert actual == expected
        assert not isinstance(actual, s.Piecewise)
        assert s.diff(actual, x) == source
        assert s.diff(actual.subs(a, 0), x) == source.subs(a, 0)


@pytest.mark.parametrize("name", ["sin", "cos", "sinh", "cosh"])
def test_trigonometric_and_hyperbolic_scalars_stay_outside_piecewise(name):
    x, a, b = s.symbols("scaled_trig_x scaled_trig_a scaled_trig_b")
    kernel = getattr(s, name)
    inner = kernel(a*x + b)
    base = s.integrate(inner, x)
    for coefficient in [2, b, a]:
        assert s.integrate(coefficient*inner, x) == coefficient*base


@pytest.mark.parametrize("name", ["sin", "sinh"])
def test_odd_kernel_sign_uses_the_add_profile_not_just_the_slope(name):
    x, a, b, c = s.symbols("odd_x odd_a odd_b odd_c")
    kernel = getattr(s, name)
    for shift, sign in [(b, -1), (b + c, 1)]:
        argument = -3*a*x + shift
        generic = (s.cos(argument) if name == "sin" else -s.cosh(argument))/(3*a)
        degenerate = x*kernel(shift)
        expected = sign*s.Piecewise((sign*generic, s.Ne(a, 0)),
                                    (sign*degenerate, True))
        actual = s.integrate(kernel(argument), x)
        assert actual == expected
        assert type(sign*actual) is s.Piecewise
        assert (sign*actual).args[0].cond.lhs is a
        assert actual.subs(a, 0) == degenerate


def test_linearity_retains_both_parameter_conditions():
    x, a, b = s.symbols("sum_x sum_a sum_b")
    expected = s.Piecewise((-s.cos(a*x)/a, s.Ne(a, 0)), (0, True))
    expected += s.Piecewise((s.sin(b*x)/b, s.Ne(b, 0)), (x, True))
    actual = s.integrate(s.sin(a*x) + s.cos(b*x), x)
    assert actual == expected
    assert actual.subs({a: 0, b: 0}) == x


@pytest.mark.parametrize("kind", ["assumptions", "dummy"])
def test_equal_names_do_not_capture_the_axis_slope_or_shift(kind):
    if kind == "assumptions":
        x = s.Symbol("shared_parameter", positive=True)
        a = s.Symbol("shared_parameter", real=True)
        b = s.Symbol("shared_parameter")
    else:
        x, a, b = (s.Dummy("shared_parameter") for _ in range(3))
    source = s.sin(a*x + b)
    expected = s.Piecewise((-s.cos(a*x + b)/a, s.Ne(a, 0)), (x*s.sin(b), True))
    actual = s.integrate(source, x)
    assert actual == expected
    assert actual.free_symbols == {x, a, b}
    assert actual.args[0].cond.lhs is a
    assert actual.subs(a, 0) == x*s.sin(b)
    assert actual.subs(b, 0) == s.Piecewise((-s.cos(a*x)/a, s.Ne(a, 0)), (0, True))


def test_nonzero_assumptions_keep_unconditional_shifted_primitives():
    x, b = s.symbols("nonzero_x nonzero_b")
    a = s.Symbol("nonzero_a", nonzero=True)
    for kernel, primitive in [
        (s.exp, s.exp(a*x + b)/a),
        (s.sin, -s.cos(a*x + b)/a),
        (s.cos, s.sin(a*x + b)/a),
        (s.sinh, s.cosh(a*x + b)/a),
        (s.cosh, s.sinh(a*x + b)/a),
    ]:
        actual = s.integrate(kernel(a*x + b), x)
        assert actual == primitive
        assert not isinstance(actual, s.Piecewise)
        assert s.diff(actual, x) == kernel(a*x + b)


def test_existing_finite_definite_route_with_proved_slope_is_unchanged():
    x = s.Symbol("definite_conditional_x")
    assert s.integrate(s.exp(2*x), (x, 0, 1)) == (s.exp(2) - 1)/2
    a = s.Symbol("definite_conditional_a", positive=True)
    # The structural oracle form is expanded across the endpoint subtraction.
    assert s.integrate(s.exp(a*x), (x, 0, 1)) == s.exp(a)/a - 1/a


def test_preview_unknown_shifted_exponential_keeps_its_formal_integral():
    # The pinned SymPy profile drops exp(b) in this zero-slope branch. This
    # excluded grammar stays formal instead of publishing an incorrect value.
    x, a, b = s.symbols("excluded_exp_x excluded_exp_a excluded_exp_b")
    for source in [s.exp(a*x + b), s.exp(a*x + 2), 2*s.exp(a*x + b)]:
        assert s.integrate(source, x) == s.Integral(source, x)


def test_preview_held_source_domains_cannot_be_erased_by_fallback_extraction():
    x, a, b = s.symbols("held_affine_x held_affine_a held_affine_b")
    invalid_coefficient = s.Mul(b, s.Pow(b, -1, evaluate=False), evaluate=False)
    source = s.Mul(invalid_coefficient, s.exp(a*x), evaluate=False)
    assert s.integrate(source, x) == s.Integral(source, x)
    hidden_pole = s.Mul(0, s.Pow(x - 1, -1, evaluate=False), evaluate=False)
    argument = s.Add(a*x + b, hidden_pole, evaluate=False)
    source = s.sin(argument, evaluate=False)
    assert s.integrate(source, x) == s.Integral(source, x)


@pytest.mark.parametrize("owner_name", ["Expr", "Integer"])
def test_preview_modified_sign_method_refuses_without_invoking_the_hook(owner_name, monkeypatch):
    x, a, b = s.symbols("sign_hook_x sign_hook_a sign_hook_b")
    source = s.sin(-3*a*x + b)
    calls = []

    def changed_sign(self):
        calls.append(self)
        raise AssertionError("a changed sign method must not execute in the closed profile lane")

    with monkeypatch.context() as patch:
        # The Integer case introduces a new shadow of the inherited method;
        # the Expr case replaces the originally recorded implementation.
        patch.setattr(getattr(s, owner_name), "could_extract_minus_sign", changed_sign)
        actual = s.integrate(source, x)
    assert calls == []
    assert actual == s.Integral(source, x)
