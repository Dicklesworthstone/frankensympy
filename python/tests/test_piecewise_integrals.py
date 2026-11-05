"""Exact first-match Piecewise integration through the public native path."""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import Basic, _native, _symbol_binding
from sympy.integrals._piecewise import _rational_components


def _equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


def _forbid_fallback(*args, **kwargs):
    raise AssertionError("admitted Piecewise integration must not use another integral lane")


@pytest.mark.parametrize("reverse", [False, True])
def test_polynomial_branches_and_reversed_limits(reverse):
    x = s.Symbol("piecewise_integral_polynomial", real=True)
    source = s.Piecewise((x, x < 0), (x**2, True))
    bounds, expected = ((3, -2), -7) if reverse else ((-2, 3), 7)
    assert s.integrate(source, (x, *bounds)) == expected
    assert s.Integral(source, (x, *bounds)).doit() == expected
    assert s.integrate(source, x, *bounds) == expected


def test_overlapping_conditions_keep_first_match_and_original_input():
    x = s.Symbol("piecewise_integral_first_match")
    source = s.Piecewise((1, x < 1), (2, x < 2), (3, True), evaluate=False)
    original = source.args
    assert s.integrate(source, (x, 0, 3)) == 6
    assert source.args == original
    duplicate = s.Piecewise((1, x >= 0), (99, x >= 0), (2, True), evaluate=False)
    assert s.integrate(duplicate, (x, -1, 1)) == 3


def test_checked_rational_primitives_compose_across_the_breakpoint():
    x = s.Symbol("piecewise_integral_rational")
    source = s.Piecewise((1/(x + 1), x < 1), (x, True))
    _equal(s.integrate(source, (x, 0, 2)), s.log(2) + s.Rational(3, 2))
    quadratic = s.Piecewise((1/(x*x + 1), x <= 0), (x, True))
    _equal(s.integrate(quadratic, (x, -1, 2)), s.pi/4 + 2)


def test_rational_endpoints_and_intervals_inside_one_branch():
    x = s.Symbol("piecewise_integral_bounds")
    source = s.Piecewise((2*x, x < 0), (3*x*x, True))
    assert s.integrate(source, (x, Fraction(-1, 2), Fraction(1, 2))) == -s.Rational(1, 8)
    assert s.integrate(source, (x, 1, 2)) == 7


def test_nonlinear_rational_guard_has_complete_rational_cuts(monkeypatch):
    x = s.Symbol("piecewise_integral_nonlinear_guard")
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_fallback)
    source = s.Piecewise((x*x, x*x < 1), (2, True))
    assert s.integrate(source, (x, -2, 2)) == s.Rational(14, 3)


def test_boolean_conditions_integrate_all_selected_components():
    x = s.Symbol("piecewise_integral_boolean")
    outside = s.Not(s.And(x >= -1, x <= 1), evaluate=False)
    source = s.Piecewise((1, outside), (x*x, True), evaluate=False)
    assert s.integrate(source, (x, -2, 2)) == s.Rational(8, 3)
    separate = s.Piecewise((2, s.Or(x < -1, x > 1)), (1, True), evaluate=False)
    assert s.integrate(separate, (x, -2, 2)) == 6


def test_rational_guard_with_unreachable_pole_is_integrated_exactly():
    x = s.Symbol("piecewise_integral_rational_guard")
    guard = s.Gt(1/(x + 2), s.Rational(1, 2), evaluate=False)
    source = s.Piecewise((x, guard), (2, True), evaluate=False)
    assert s.integrate(source, (x, -1, 1)) == s.Rational(3, 2)


@pytest.mark.parametrize("closed", [False, True])
def test_single_breakpoint_ownership_does_not_change_the_integral(closed):
    x = s.Symbol("piecewise_integral_breakpoint")
    condition = x <= 0 if closed else x < 0
    source = s.Piecewise((1, condition), (2, True))
    assert s.integrate(source, (x, -1, 1)) == 3


@pytest.mark.parametrize("point_value", ["finite", "pole", "undefined"])
def test_isolated_selected_values_have_zero_integral(point_value):
    x = s.Symbol("piecewise_integral_isolated")
    value = {
        "finite": lambda: s.Integer(99),
        "pole": lambda: s.Pow(x, -1, evaluate=False),
        "undefined": lambda: s.Pow(0, -1, evaluate=False),
    }[point_value]()
    source = s.Piecewise((value, s.Eq(x, 0, evaluate=False)), (x*x, True), evaluate=False)
    assert s.integrate(source, (x, -1, 1)) == s.Rational(2, 3)


def test_uncovered_singleton_contributes_zero_but_uncovered_interval_is_undefined(monkeypatch):
    x = s.Symbol("piecewise_integral_uncovered")
    point_gap = s.Piecewise((1, x < 0), (2, x > 0), evaluate=False)
    assert s.integrate(point_gap, (x, -1, 1)) == 3
    assert s.integrate(point_gap, (x, 1, -1)) == -3
    monkeypatch.setattr(_native, "exact_rational_integral_native", _forbid_fallback)
    interval_gap = s.Piecewise((x, x < 0), evaluate=False)
    assert s.integrate(interval_gap, (x, -1, 1)) is s.nan
    point_only = s.Piecewise((x, s.Eq(x, 0, evaluate=False)), evaluate=False)
    assert s.integrate(point_only, (x, -1, 1)) is s.nan


@pytest.mark.parametrize("kind", ["regular", "pole", "uncovered"])
def test_zero_width_interval_after_bounded_rational_admission(kind, monkeypatch):
    x = s.Symbol("piecewise_integral_zero")
    sources = {
        "regular": s.Piecewise((x, x < 0), (x*x, True), evaluate=False),
        "pole": s.Piecewise((s.Pow(x, -1, evaluate=False), x >= 0), (0, True), evaluate=False),
        "uncovered": s.Piecewise((x, x < 0), evaluate=False),
    }
    monkeypatch.setattr(_native, "exact_rational_integral_native", _forbid_fallback)
    assert s.integrate(sources[kind], (x, 0, 0)) == 0


def test_unreachable_branch_poles_do_not_block_an_integral():
    x = s.Symbol("piecewise_integral_unreachable")
    source = s.Piecewise((1/(x - 2), x < 0), (x, True), evaluate=False)
    _equal(s.integrate(source, (x, -1, 1)), s.log(2) - s.log(3) + s.Rational(1, 2))
    unreachable = s.Piecewise((s.Pow(0, -1, evaluate=False), x < -2), (x*x, True), evaluate=False)
    assert s.integrate(unreachable, (x, -1, 1)) == s.Rational(2, 3)


@pytest.mark.parametrize("kind", ["interior", "endpoint", "held_hole"])
def test_active_poles_and_original_holes_cannot_publish_a_finite_integral(kind):
    x = s.Symbol("piecewise_integral_active_pole")
    integrand = {
        "interior": lambda: 1/(x - 1),
        "endpoint": lambda: 1/x,
        "held_hole": lambda: s.Mul(0, s.Pow(x - 1, -1, evaluate=False), evaluate=False),
    }[kind]()
    source = s.Piecewise((integrand, x >= 0), (1, True), evaluate=False)
    with pytest.raises(ValueError):
        s.integrate(source, (x, -1, 2))


def test_explicit_piecewise_split_preserves_convergent_removable_endpoints():
    x = s.Symbol("piecewise_integral_removable")
    inverse = s.Pow(x, -1, evaluate=False)
    held = s.Mul(x, inverse, evaluate=False)
    source = s.Piecewise((0, s.Eq(x, 0, evaluate=False)), (held, True), evaluate=False)
    assert s.integrate(source, (x, -1, 1)) == 2


def test_reachable_undefined_guard_preserves_the_original_integral(monkeypatch):
    x = s.Symbol("piecewise_integral_guard_pole")
    guard = s.Gt(s.Pow(x, -1, evaluate=False), 0, evaluate=False)
    source = s.Piecewise((x, guard), (0, True), evaluate=False)
    monkeypatch.setattr(_native, "exact_rational_integral_native", _forbid_fallback)
    result = s.integrate(source, (x, -1, 1))
    assert isinstance(result, s.Integral)
    assert result.function == source
    assert result.limits == ((x, s.Integer(-1), s.Integer(1)),)


@pytest.mark.parametrize("kind", ["branch", "parameter", "unknown_guard", "algebraic_cut"])
def test_unsupported_piecewise_families_stay_unevaluated(kind, monkeypatch):
    x, parameter = s.symbols("piecewise_integral_unsupported parameter")
    source = {
        "branch": lambda: s.Piecewise((s.sin(x), x < 0), (0, True), evaluate=False),
        "parameter": lambda: s.Piecewise((parameter*x, x < 0), (0, True), evaluate=False),
        "unknown_guard": lambda: s.Piecewise((x, parameter), (0, True), evaluate=False),
        "algebraic_cut": lambda: s.Piecewise((x, x*x < 2), (0, True), evaluate=False),
    }[kind]()
    monkeypatch.setattr(_native, "exact_rational_integral_native", _forbid_fallback)
    monkeypatch.setattr(_native, "integrate_definite_native", _forbid_fallback)
    result = s.integrate(source, (x, -2, 2))
    assert isinstance(result, s.Integral)
    assert result.function == source


@pytest.mark.parametrize("bound", ["symbolic", "algebraic", "float", "infinite"])
def test_nonrational_or_infinite_bounds_remain_unevaluated(bound):
    x, parameter = s.symbols("piecewise_integral_endpoint parameter")
    upper = {"symbolic": parameter, "algebraic": s.sqrt(2), "float": 0.5, "infinite": s.oo}[bound]
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    assert isinstance(s.integrate(source, (x, -1, upper)), s.Integral)


@pytest.mark.parametrize("native_name", [
    "solve_real_rational_native", "solve_real_rational_inequality_native",
    "exact_rational_integral_native",
])
@pytest.mark.parametrize("failure", [ValueError("resource exhausted"), NotImplementedError("native refusal")])
def test_admitted_native_failure_is_terminal(native_name, failure, monkeypatch):
    x = s.Symbol("piecewise_integral_failure")
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    calls = []

    def fail(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(_native, native_name, fail)
    monkeypatch.setattr(_native, "integrate_definite_native", _forbid_fallback)
    with pytest.raises(type(failure)) as error:
        s.integrate(source, (x, -1, 1))
    assert error.value is failure
    assert len(calls) == 1


@pytest.mark.parametrize("dummy", [False, True])
def test_every_checked_component_uses_the_original_typed_variable(dummy, monkeypatch):
    x = s.Dummy("piecewise_integral_owner", real=True) if dummy else s.Symbol("piecewise_integral_owner", real=True)
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    native = _native.exact_rational_integral_native
    seen = []

    def record(expression, binding, lower, upper):
        seen.append((binding.name, binding.identity_hex))
        return native(expression, binding, lower, upper)

    monkeypatch.setattr(_native, "exact_rational_integral_native", record)
    assert s.integrate(source, (x, -1, 1)) == -s.Rational(1, 6)
    binding = _symbol_binding(x)
    assert len(seen) == 2
    assert all(item == (binding.name, binding.identity_hex) for item in seen)


def test_mutation_during_a_checked_component_prevents_publication(monkeypatch):
    x = s.Symbol("piecewise_integral_mutation")
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    native = _native.exact_rational_integral_native
    slot = s.Piecewise.__dict__["_piecewise_branches"]

    def mutate(*args):
        result = native(*args)
        slot.__set__(source, ((s.Integer(99), s.true),))
        return result

    monkeypatch.setattr(_native, "exact_rational_integral_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.integrate(source, (x, -1, 1))


@pytest.mark.parametrize("slot_name", ["_args", "_struct_args", "_surface_args"])
@pytest.mark.parametrize("position", ["branch", "guard"])
def test_preexisting_held_argument_mismatch_refuses_before_generation(slot_name, position, monkeypatch):
    x = s.Symbol("piecewise_integral_stale_" + position + slot_name)
    held = s.Add(x, 1, evaluate=False)
    source = (s.Piecewise((held, x < 0), (0, True), evaluate=False) if position == "branch"
              else s.Piecewise((x, s.Lt(held, 0, evaluate=False)), (0, True), evaluate=False))
    owner = s.Add if slot_name == "_args" else Basic
    owner.__dict__[slot_name].__set__(held, (x, s.Integer(2)))
    for name in ("solve_real_rational_native", "solve_real_rational_inequality_native",
                 "exact_rational_integral_native", "integrate_definite_native"):
        monkeypatch.setattr(_native, name, _forbid_fallback)
    with pytest.raises(ValueError, match="retained surface arguments"):
        s.integrate(source, (x, -1, 1))


@pytest.mark.parametrize("kind", ["add", "mul", "pow"])
def test_held_argument_mutation_during_integral_is_not_published(kind, monkeypatch):
    x = s.Symbol("piecewise_integral_held_mutation_" + kind)
    held, replacement = {
        "add": lambda: (s.Add(x, 1, evaluate=False), (x, s.Integer(2))),
        "mul": lambda: (s.Mul(2, x, evaluate=False), (s.Integer(3), x)),
        "pow": lambda: (s.Pow(x + 2, -1, evaluate=False), (x + 2, s.Integer(-2))),
    }[kind]()
    source = s.Piecewise((held, x < 0), (0, True), evaluate=False)
    native = _native.exact_rational_integral_native

    def mutate(*args):
        result = native(*args)
        type(held).__dict__["_args"].__set__(held, replacement)
        return result

    monkeypatch.setattr(_native, "exact_rational_integral_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.integrate(source, (x, -1, 1))


@pytest.mark.parametrize("kind", ["contents", "identity"])
def test_symbol_assumptions_mutation_during_integral_is_not_published(kind, monkeypatch):
    x = s.Symbol("piecewise_integral_assumptions_" + kind, real=True)
    source = s.Piecewise((x, x < 0), (0, True), evaluate=False)
    slot = s.Symbol.__dict__["_assumptions"]
    assumptions = slot.__get__(x, s.Symbol)
    original = tuple(assumptions.items())
    native = _native.exact_rational_integral_native

    def mutate(*args):
        result = native(*args)
        if kind == "contents":
            assumptions["positive"] = True
        else:
            slot.__set__(x, dict(original))
        return result

    monkeypatch.setattr(_native, "exact_rational_integral_native", mutate)
    try:
        with pytest.raises(RuntimeError, match="source mutated"):
            s.integrate(source, (x, -1, 1))
    finally:
        assumptions.clear()
        assumptions.update(original)
        slot.__set__(x, assumptions)


@pytest.mark.parametrize("kind", ["identity_add", "duplicate_add", "identity_mul", "one_power", "zero_power", "inverse"])
def test_consistent_held_rational_arguments_remain_supported(kind):
    x = s.Symbol("piecewise_integral_held_consistency_" + kind)
    held, expected = {
        "identity_add": lambda: (s.Add(x, 0, evaluate=False), -s.Rational(1, 2)),
        "duplicate_add": lambda: (s.Add(x, x, evaluate=False), s.Integer(-1)),
        "identity_mul": lambda: (s.Mul(1, x, evaluate=False), -s.Rational(1, 2)),
        "one_power": lambda: (s.Pow(x, 1, evaluate=False), -s.Rational(1, 2)),
        "zero_power": lambda: (s.Pow(x, 0, evaluate=False), s.Integer(1)),
        "inverse": lambda: (s.Pow(x + 2, -1, evaluate=False), s.log(2)),
    }[kind]()
    source = s.Piecewise((held, x < 0), (0, True), evaluate=False)
    _equal(s.integrate(source, (x, -1, 1)), expected)


def test_exact_rational_set_components_keep_punctures_and_merge_covered_boundaries():
    def q(value):
        return ("rational", Fraction(value))

    whole = ("Interval", q(-2), q(2), False, False)
    point = ("FiniteSet", (q(0),))
    punctured = ("Intersection", (whole, ("Complement", (point,))))
    assert _rational_components(punctured, Fraction(-2), Fraction(2)) == [
        (Fraction(-2), Fraction(0)), (Fraction(0), Fraction(2)),
    ]
    filled = ("Union", (punctured, point))
    assert _rational_components(filled, Fraction(-2), Fraction(2)) == [(Fraction(-2), Fraction(2))]
    removed = ("Interval", q(-1), q(1), False, False)
    outside = ("Intersection", (whole, ("Complement", (removed,))))
    assert _rational_components(outside, Fraction(-2), Fraction(2)) == [
        (Fraction(-2), Fraction(-1)), (Fraction(1), Fraction(2)),
    ]


def test_exact_rational_set_components_clip_unions_and_ignore_isolated_points():
    def q(value):
        return ("rational", Fraction(value))

    overlapping = ("Union", (
        ("Interval", ("infinity", -1), q(1), True, True),
        ("Interval", q(0), ("infinity", 1), True, True),
        ("FiniteSet", (q(-1), q(4))),
    ))
    assert _rational_components(overlapping, Fraction(-2), Fraction(2)) == [(Fraction(-2), Fraction(2))]
    assert _rational_components(("FiniteSet", (q(0),)), Fraction(-1), Fraction(1)) == []
    algebraic = ("Interval", q(-2), ("root", (), 0, 1), False, False)
    assert _rational_components(algebraic, Fraction(-2), Fraction(2)) is None
