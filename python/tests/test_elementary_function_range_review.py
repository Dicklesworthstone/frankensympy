"""Independent exact-image and trust-boundary tests for elementary ranges.

The main fixture table uses exact SymPy 1.14 set/endpoint comparisons.
Attainment and held-source fixtures exercise the explicit exact-image and
original-domain contracts, including poles that cancellation would remove.
No floating samples establish an image.
"""

import importlib

import pytest
import sympy as s

from sympy.calculus.util import function_range, maximum, minimum


@pytest.mark.parametrize("source, domain, expected", [
    pytest.param(
        lambda x: s.exp(1/x), lambda: s.S.Reals,
        lambda: s.Union(s.Interval.open(0, 1), s.Interval.open(1, s.oo)),
        id="exp-punctured-image"),
    pytest.param(
        lambda x: s.exp(x*x), lambda: s.Interval.open(-1, 1),
        lambda: s.Interval.Ropen(1, s.exp(1)), id="exp-attained-critical-value"),
    pytest.param(
        lambda x: s.log(x), lambda: s.Interval(-2, 2),
        lambda: s.Interval(-s.oo, s.log(2)), id="log-clips-negative-inputs"),
    pytest.param(
        lambda x: s.log(x*x), lambda: s.S.Reals,
        lambda: s.S.Reals, id="log-unbounded-both-ways"),
    pytest.param(
        lambda x: s.log(1/x), lambda: s.Interval(-2, 2),
        lambda: s.Interval(s.log(s.Rational(1, 2)), s.oo),
        id="log-reciprocal-domain"),
    pytest.param(
        lambda x: s.sqrt(x), lambda: s.Interval(-2, 2),
        lambda: s.Interval(0, s.sqrt(2)), id="square-root-closed-branch-point"),
    pytest.param(
        lambda x: s.sqrt(1-x*x), lambda: s.Interval(-2, 2),
        lambda: s.Interval(0, 1), id="square-root-both-turning-points"),
    pytest.param(
        lambda x: s.sqrt((x-1)/(x+1)), lambda: s.S.Reals,
        lambda: s.Union(s.Interval.Ropen(0, 1), s.Interval.open(1, s.oo)),
        id="square-root-retains-unattained-asymptote"),
    pytest.param(
        lambda x: s.atan(x), lambda: s.S.Reals,
        lambda: s.Interval.open(-s.pi/2, s.pi/2), id="atan-open-asymptotes"),
    pytest.param(
        lambda x: s.atan(1/x), lambda: s.S.Reals,
        lambda: s.Union(s.Interval.open(-s.pi/2, 0),
                        s.Interval.open(0, s.pi/2)), id="atan-punctured-image"),
    pytest.param(
        lambda x: s.asin(x), lambda: s.Interval(-2, 2),
        lambda: s.Interval(-s.pi/2, s.pi/2), id="asin-clips-both-branch-ends"),
    pytest.param(
        lambda x: s.acos(x), lambda: s.Interval.Lopen(-1, 1),
        lambda: s.Interval.Ropen(0, s.pi), id="acos-reverses-open-end"),
    pytest.param(
        lambda x: s.asinh(x), lambda: s.S.Reals,
        lambda: s.S.Reals, id="asinh-unbounded-image"),
    pytest.param(
        lambda x: s.acosh(x), lambda: s.Interval(-2, 3),
        lambda: s.Interval(0, s.acosh(3)), id="acosh-closed-branch-point"),
    pytest.param(
        lambda x: s.atanh(x), lambda: s.Interval(-2, 2),
        lambda: s.S.Reals, id="atanh-open-branch-limits"),
    pytest.param(
        lambda x: s.sinh(x), lambda: s.S.Reals,
        lambda: s.S.Reals, id="sinh-unbounded-image"),
    pytest.param(
        lambda x: s.tanh(x), lambda: s.S.Reals,
        lambda: s.Interval.open(-1, 1), id="tanh-open-asymptotes"),
    pytest.param(
        lambda x: s.cosh(x), lambda: s.Interval.open(-2, 1),
        lambda: s.Interval.Ropen(1, s.cosh(2)), id="cosh-folds-negative-arm"),
    pytest.param(
        lambda x: s.cosh(1/x), lambda: s.S.Reals,
        lambda: s.Interval.open(1, s.oo), id="cosh-missing-critical-value"),
    pytest.param(
        lambda x: s.asech(x), lambda: s.Interval(0, 1),
        lambda: s.Interval(0, s.oo), id="asech-one-sided-singularity"),
    pytest.param(
        lambda x: s.asech(x), lambda: s.Interval.Lopen(s.Rational(1, 2), 1),
        lambda: s.Interval.Ropen(0, s.asech(s.Rational(1, 2))),
        id="asech-reverses-open-end"),
    pytest.param(
        lambda x: s.Pow(x, s.Rational(1, 3)), lambda: s.Interval(0, 8),
        lambda: s.Interval(0, 2), id="positive-principal-cube-root"),
    pytest.param(
        lambda x: s.Pow(x, s.Rational(-1, 3)), lambda: s.Interval(0, 8),
        lambda: s.Interval(s.Rational(1, 2), s.oo),
        id="negative-principal-power-excludes-zero"),
    pytest.param(
        lambda x: s.Pow(x, s.Rational(-3, 2)), lambda: s.Interval(-1, 4),
        lambda: s.Interval(s.Rational(1, 8), s.oo),
        id="negative-even-root-power-clips-domain"),
    pytest.param(
        lambda x: 2-3*s.exp(1/x), lambda: s.S.Reals,
        lambda: s.Union(s.Interval.open(-s.oo, -1), s.Interval.open(-1, 2)),
        id="negative-affine-wrapper-preserves-hole"),
    pytest.param(
        lambda x: s.Rational(2, 3)*s.log(x)-s.Rational(1, 5),
        lambda: s.Interval.Ropen(1, 4),
        lambda: s.Interval.Ropen(-s.Rational(1, 5),
                                 s.Rational(2, 3)*s.log(4)-s.Rational(1, 5)),
        id="rational-affine-wrapper-preserves-open-end"),
])
def test_exact_elementary_images(source, domain, expected):
    x = s.Symbol("independent_elementary_image")
    assert function_range(source(x), x, domain()) == expected()


@pytest.mark.parametrize("domain, expected", [
    (lambda: s.Interval.open(-1, 1), lambda: s.Interval.Ropen(1, s.cosh(1))),
    (lambda: s.Interval.Lopen(-1, 1), lambda: s.Interval(1, s.cosh(1))),
    (lambda: s.Interval.Ropen(-1, 1), lambda: s.Interval(1, s.cosh(1))),
    (lambda: s.Interval.open(-1, 0), lambda: s.Interval.open(1, s.cosh(1))),
])
def test_cosh_equal_endpoint_images_preserve_attainment(domain, expected):
    # SymPy 1.14 incorrectly drops cosh(1) for the two half-open intervals.
    # The included endpoint +/-1 is an explicit exact attainment witness.
    # These are mathematical correctness cases, not an oracle-parity claim.
    x = s.Symbol("independent_cosh_attainment")
    assert function_range(s.cosh(x), x, domain()) == expected()


def test_absolute_value_folds_without_filling_a_missing_zero():
    x = s.Symbol("independent_absolute_image")
    assert function_range(s.Abs(x), x, s.Interval.open(-2, 1)) == s.Interval.Ropen(0, 2)
    source = s.Abs(s.Pow(x, -1, evaluate=False), evaluate=False)
    assert function_range(source, x, s.S.Reals) == s.Interval.open(0, s.oo)


@pytest.mark.parametrize("source, domain, infimum, supremum", [
    (lambda x: s.exp(1/x), lambda: s.S.Reals, lambda: 0, lambda: s.oo),
    (lambda x: s.log(x), lambda: s.Interval.Ropen(0, 2),
     lambda: -s.oo, lambda: s.log(2)),
    (lambda x: s.acos(x), lambda: s.Interval.open(-1, 1),
     lambda: 0, lambda: s.pi),
    (lambda x: s.cosh(1/x), lambda: s.S.Reals, lambda: 1, lambda: s.oo),
    (lambda x: 7-2*s.tanh(x), lambda: s.S.Reals, lambda: 5, lambda: 9),
])
def test_extrema_are_exact_infimum_and_supremum_even_when_unattained(
        source, domain, infimum, supremum):
    x = s.Symbol("independent_elementary_extrema")
    expression, interval = source(x), domain()
    assert minimum(expression, x, interval) == infimum()
    assert maximum(expression, x, interval) == supremum()


@pytest.mark.parametrize("power", [s.Rational(1, 3), s.Rational(2, 3), s.Rational(-1, 3)])
def test_principal_odd_powers_do_not_become_real_cube_roots(power):
    x = s.Symbol("independent_principal_branch")
    with pytest.raises((NotImplementedError, ValueError)):
        function_range(s.Pow(x, power, evaluate=False), x, s.Interval(-1, 1))


@pytest.mark.parametrize("source", [
    lambda x: s.exp(x)+x,
    lambda x: x*s.exp(x),
    lambda x: s.exp(x)+s.log(x),
])
def test_correlated_varying_children_are_not_combined_as_independent_ranges(source):
    x = s.Symbol("independent_correlated_source")
    with pytest.raises(NotImplementedError):
        function_range(source(x), x, s.Interval(1, 2))


def test_typed_axes_with_the_same_name_do_not_alias():
    x = s.Dummy("independent_typed_axis")
    other = s.Dummy("independent_typed_axis")
    with pytest.raises(NotImplementedError):
        function_range(s.exp(other), x, s.Interval(0, 1))


def test_an_original_held_pole_survives_a_constant_outer_image():
    x = s.Symbol("independent_constant_pole")
    ratio = s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.exp(ratio, evaluate=False)
    assert function_range(source, x, s.Interval(-1, 1)) == s.FiniteSet(s.exp(1))
    assert function_range(source, x, s.Interval(0, 0)) == s.EmptySet()


def test_zero_affine_scale_does_not_erase_the_logarithm_domain():
    x = s.Symbol("independent_zero_scale_domain")
    source = s.Mul(0, s.log(x, evaluate=False), evaluate=False)
    assert function_range(source, x, s.Interval(-1, 2)) == s.FiniteSet(0)
    assert function_range(source, x, s.Interval(-2, -1)) == s.EmptySet()
    with pytest.raises(ValueError, match="empty domain"):
        minimum(source, x, s.Interval(-2, -1))
    with pytest.raises(ValueError, match="empty domain"):
        maximum(source, x, s.Interval(-2, -1))


def test_endpoint_ordering_and_extrema_never_sample_floats(monkeypatch):
    x = s.Symbol("independent_no_numeric_ordering")
    source = 2-3*s.exp(1/x)
    expected = s.Union(s.Interval.open(-s.oo, -1), s.Interval.open(-1, 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("an exact elementary range sampled floating values")

    monkeypatch.setattr(s.Expr, "__float__", forbidden)
    monkeypatch.setattr(s.Expr, "evalf", forbidden)
    assert function_range(source, x, s.S.Reals) == expected
    assert minimum(source, x, s.S.Reals) == -s.oo
    assert maximum(source, x, s.S.Reals) == 2


def test_custom_function_refuses_without_invoking_a_descriptor(monkeypatch):
    x = s.Symbol("independent_custom_range")

    class Custom(s.Function):
        pass

    source = Custom(x, evaluate=False)
    calls = []

    def forbidden(_):
        calls.append("descriptor")
        raise AssertionError("an unadmitted custom function was inspected")

    monkeypatch.setattr(Custom, "args", property(forbidden))
    with pytest.raises(NotImplementedError):
        function_range(source, x, s.Interval(0, 1))
    assert not calls


def test_foreign_metaclass_equality_is_not_an_admission_callback():
    calls = []

    class Hostile(type):
        def __eq__(cls, other):
            calls.append("metaclass equality")
            raise AssertionError("foreign type equality was used for admission")

        __hash__ = type.__hash__

    class Foreign(metaclass=Hostile):
        pass

    x = s.Symbol("independent_foreign_range")
    with pytest.raises(NotImplementedError):
        function_range(Foreign(), x, s.Interval(0, 1))
    assert not calls


def test_changed_builtin_evaluator_refuses_before_callback(monkeypatch):
    x = s.Symbol("independent_changed_elementary")
    source = s.exp(x, evaluate=False)
    domain = s.Interval(0, 1)
    calls = []

    def forbidden(cls, *args):
        calls.append("eval")
        raise AssertionError("a replaced built-in evaluator was invoked")

    monkeypatch.setattr(s.exp, "eval", classmethod(forbidden))
    with pytest.raises(NotImplementedError):
        function_range(source, x, domain)
    assert not calls


def test_original_held_arguments_are_checked_before_native_range_work(monkeypatch):
    from sympy.core import Function, _native

    x = s.Symbol("independent_stale_elementary")
    source = s.exp(x, evaluate=False)
    slot = Function.__dict__["_args"]
    original = slot.__get__(source, Function)
    calls = []

    def forbidden(*args):
        calls.append("native")
        raise AssertionError("a stale source reached the native range engine")

    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", forbidden)
    slot.__set__(source, (x+1,))
    try:
        with pytest.raises((NotImplementedError, ValueError)):
            function_range(source, x, s.Interval(0, 1))
    finally:
        slot.__set__(source, original)
    assert not calls


def test_source_mutation_while_computing_the_inner_range_cannot_publish(monkeypatch):
    from sympy.core import Function, _native

    x = s.Symbol("independent_mutating_elementary")
    source = s.exp(x, evaluate=False)
    slot = Function.__dict__["_args"]
    original = slot.__get__(source, Function)
    solve = _native.solve_real_rational_inequality_native

    def changed(*args):
        result = solve(*args)
        slot.__set__(source, (x+1,))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", changed)
    try:
        with pytest.raises((NotImplementedError, RuntimeError, ValueError)):
            function_range(source, x, s.Interval(0, 1))
    finally:
        slot.__set__(source, original)


@pytest.mark.parametrize("failure", [ValueError("independent resource exhaustion"),
                                    RuntimeError("independent cancellation")])
def test_native_failures_cannot_fall_back_to_a_sampled_image(monkeypatch, failure):
    from sympy.core import _native

    x = s.Symbol("independent_failed_range")
    source = s.exp(x)

    def failed(*args):
        raise failure

    def forbidden(*args, **kwargs):
        raise AssertionError("a failed exact check fell back to another solver")

    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", failed)
    monkeypatch.setattr(importlib.import_module("sympy.solvers"), "solveset", forbidden)
    with pytest.raises((type(failure), NotImplementedError), match="independent"):
        function_range(source, x, s.Interval(0, 1))
