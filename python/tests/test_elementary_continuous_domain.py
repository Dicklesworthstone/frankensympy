"""Elementary continuity domains through the actual checked native partition.

Ordinary cases follow the pinned SymPy 1.14 real-axis convention. Held-source
cases deliberately retain original exclusions, including under zero products
and zeroth powers. No sampled endpoint or numerical comparison is a comparator.
"""

import pytest
import sympy as s

from sympy.core import Basic, Function, _native


def _punctured(point):
    return s.Union(s.Interval.open(-s.oo, point), s.Interval.open(point, s.oo))


@pytest.mark.parametrize("source, expected", [
    (lambda x: s.log(x), lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.log((x - 1)/(x + 2)),
     lambda: s.Union(s.Interval.open(-s.oo, -2), s.Interval.open(1, s.oo))),
    (lambda x: s.log((x - 1)**2), lambda: _punctured(1)),
    (lambda x: s.sqrt(1 - x*x), lambda: s.Interval(-1, 1)),
    (lambda x: s.sqrt((x - 1)/(x + 2)),
     lambda: s.Union(s.Interval.open(-s.oo, -2), s.Interval(1, s.oo))),
    (lambda x: (x - 1)**s.Rational(-3, 2), lambda: s.Interval.open(1, s.oo)),
    (lambda x: x**s.Rational(1, 3), lambda: s.S.Reals),
    (lambda x: x**s.Rational(2, 3), lambda: s.S.Reals),
    (lambda x: x**s.Rational(-1, 3), lambda: _punctured(0)),
    (lambda x: s.asin((x - 1)/2), lambda: s.Interval(-1, 3)),
    (lambda x: s.acos(x), lambda: s.Interval(-1, 1)),
    (lambda x: s.atan(1/(x - 3)), lambda: _punctured(3)),
    (lambda x: s.acot(x), lambda: _punctured(0)),
    (lambda x: s.asinh(1/(x - 3)), lambda: _punctured(3)),
    (lambda x: s.acosh((x + 1)/2), lambda: s.Interval(1, s.oo)),
    (lambda x: s.atanh((x - 1)/2), lambda: s.Interval.open(-1, 3)),
    (lambda x: s.acoth(x/2),
     lambda: s.Union(s.Interval.open(-s.oo, -2), s.Interval.open(2, s.oo))),
    (lambda x: s.asec(x),
     lambda: s.Union(s.Interval(-s.oo, -1), s.Interval(1, s.oo))),
    (lambda x: s.acsc(x),
     lambda: s.Union(s.Interval(-s.oo, -1), s.Interval(1, s.oo))),
    (lambda x: s.asech(x), lambda: s.Interval.Lopen(0, 1)),
    (lambda x: s.acsch(x), lambda: _punctured(0)),
    (lambda x: s.coth(x), lambda: _punctured(0)),
    (lambda x: s.csch(x), lambda: _punctured(0)),
    (lambda x: s.sech(x), lambda: s.S.Reals),
    (lambda x: s.exp(1/x), lambda: _punctured(0)),
    (lambda x: s.sin(s.sqrt(x)), lambda: s.Interval(0, s.oo)),
    (lambda x: s.cosh(s.log(x)), lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.tanh(s.log(x)), lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.Abs(s.log(x)), lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.sqrt(s.log(x)), lambda: s.Interval(1, s.oo)),
    (lambda x: s.log(s.sqrt(x)), lambda: s.Interval.open(0, s.oo)),
    (lambda x: 1/s.log(x),
     lambda: s.Union(s.Interval.open(0, 1), s.Interval.open(1, s.oo))),
    (lambda x: s.sqrt(s.exp(x)), lambda: s.S.Reals),
    (lambda x: s.log(s.exp(x), evaluate=False), lambda: s.S.Reals),
    (lambda x: 1/s.exp(1/x), lambda: _punctured(0)),
    (lambda x: s.log(x - 1) + s.sqrt(5 - x), lambda: s.Interval.Lopen(1, 5)),
])
def test_elementary_domains_and_branch_endpoints(source, expected):
    x = s.Symbol("elementary_domain_x")
    assert s.continuous_domain(source(x), x, s.S.Reals) == expected()


def test_every_obligation_uses_one_common_checked_partition(monkeypatch):
    x = s.Symbol("elementary_domain_partition")
    expression = s.sqrt(2 - x*x) + s.log(x*x - 3)
    calls = []
    solve = _native.solve_real_rational_constraints_native

    def checked(*args):
        calls.append(args)
        return solve(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("continuity conditions must not use scalar or numerical fallback")

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", checked)
    monkeypatch.setattr(s, "solveset", forbidden)
    monkeypatch.setattr(_native, "solve_inequality_native", forbidden)
    assert s.continuous_domain(expression, x, s.S.Reals) == s.EmptySet()
    assert len(calls) == 1
    assert ">=" in calls[0][2] and ">" in calls[0][2]


def test_algebraic_endpoints_and_poles_share_the_complete_partition():
    x = s.Symbol("elementary_domain_algebraic")
    expression = s.sqrt(x*x - 2) + s.log(3 - x)
    expected = s.Union(s.Interval(-s.oo, -s.sqrt(2)), s.Interval.Ropen(s.sqrt(2), 3))
    assert s.continuous_domain(expression, x, s.S.Reals) == expected
    assert s.continuous_domain(s.log(1/(x*x - 2)), x, s.S.Reals) == s.Union(
        s.Interval.open(-s.oo, -s.sqrt(2)), s.Interval.open(s.sqrt(2), s.oo),
    )


@pytest.mark.parametrize("source, expected", [
    (lambda x: s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False),
     lambda: _punctured(0)),
    (lambda x: s.Mul(0, s.Pow(x - 1, -1, evaluate=False), evaluate=False),
     lambda: _punctured(1)),
    (lambda x: s.Pow(s.log(x, evaluate=False), 0, evaluate=False),
     lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.Mul(0, s.log(s.Pow(x, -1, evaluate=False), evaluate=False), evaluate=False),
     lambda: s.Interval.open(0, s.oo)),
    (lambda x: s.log(s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False), evaluate=False),
     lambda: _punctured(0)),
    (lambda x: s.Pow(s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False),
                     s.Rational(1, 2), evaluate=False),
     lambda: _punctured(0)),
])
def test_original_held_source_exclusions_survive_identities(source, expected):
    x = s.Symbol("elementary_domain_held")
    assert s.continuous_domain(source(x), x, s.S.Reals) == expected()


def test_restricted_intervals_finite_points_and_disconnected_domains():
    x = s.Symbol("elementary_domain_restricted")
    assert s.continuous_domain(s.log(x), x, s.Interval(-2, 2)) == s.Interval.Lopen(0, 2)
    assert s.continuous_domain(s.sqrt(1 - x*x), x, s.FiniteSet(-2, -1, 0, 1, 2)) == s.FiniteSet(-1, 0, 1)
    domain = s.Union(s.Interval(-3, -1), s.Interval(1, 3))
    assert s.continuous_domain(s.log(x), x, domain) == s.Interval(1, 3)
    assert s.continuous_domain(s.log(x), x, s.EmptySet()) == s.EmptySet()


def test_typed_variables_and_inferred_coordinates_keep_their_identity():
    x = s.Dummy("domain_coordinate", real=True)
    assert s.continuous_domain(s.log(x), x, s.S.Reals) == s.Interval.open(0, s.oo)
    assert s.continuous_domain(s.log(x)) == s.Interval.open(0, s.oo)
    assert s.continuous_domain(7) == s.S.Reals
    other = s.Dummy("domain_coordinate", real=True)
    with pytest.raises(NotImplementedError):
        s.continuous_domain(s.log(other), x, s.S.Reals)
    ordinary = s.Symbol("domain_typed", real=True)
    positive = s.Symbol("domain_typed", positive=True)
    with pytest.raises(NotImplementedError):
        s.continuous_domain(s.log(positive), ordinary, s.S.Reals)


@pytest.mark.parametrize("source", [
    lambda x: s.log(s.sin(x)),
    lambda x: s.tan(x),
    lambda x: s.atan(x**s.Rational(1, 3)),
    lambda x: s.log(x**s.Rational(1, 3)),
    lambda x: x**s.Symbol("domain_exponent"),
])
def test_unsupported_conditions_do_not_infer_real_values_or_sample(source):
    x = s.Symbol("elementary_domain_unsupported")
    with pytest.raises(NotImplementedError):
        s.continuous_domain(source(x), x, s.S.Reals)


def test_custom_function_and_changed_implementations_do_not_run_hooks(monkeypatch):
    x = s.Symbol("elementary_domain_custom")
    calls = []

    class Custom(s.Function):
        pass

    source = Custom(x)

    def forbidden_args(self):
        calls.append("args")
        raise AssertionError("custom source must not be traversed")

    monkeypatch.setattr(Custom, "args", property(forbidden_args))
    with pytest.raises(NotImplementedError):
        s.continuous_domain(source, x, s.S.Reals)
    assert not calls
    built_in = s.log(x, evaluate=False)
    monkeypatch.setattr(s.log, "_eval_is_real", lambda self: True, raising=False)
    with pytest.raises(NotImplementedError):
        s.continuous_domain(built_in, x, s.S.Reals)


def test_integer_surface_cannot_disguise_a_noninteger_native_payload():
    number = object.__new__(s.Integer)
    Basic.__dict__["_value"].__set__(number, _native.py_rational(1, 2))
    with pytest.raises((ValueError, NotImplementedError)):
        s.continuous_domain(number, s.Symbol("elementary_domain_scalar"), s.S.Reals)


def test_foreign_metaclass_equality_is_not_an_admission_hook():
    class Hostile(type):
        def __eq__(cls, other):
            raise AssertionError("foreign class equality must not run")

        __hash__ = type.__hash__

    class Foreign(metaclass=Hostile):
        pass

    x = s.Symbol("elementary_domain_foreign")
    with pytest.raises(NotImplementedError):
        s.continuous_domain(Foreign(), x, s.S.Reals)
    with pytest.raises(TypeError):
        s.continuous_domain(s.log(x), Foreign(), s.S.Reals)


def test_inconsistent_original_arguments_refuse_before_solver(monkeypatch):
    x = s.Dummy("elementary_domain_stale")
    source = s.log(x, evaluate=False)
    slot = Function.__dict__["_args"]
    original = slot.__get__(source, Function)

    def forbidden(*args, **kwargs):
        raise AssertionError("stale original source must not reach a solver")

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", forbidden)
    slot.__set__(source, (x + 1,))
    try:
        with pytest.raises((ValueError, NotImplementedError)):
            s.continuous_domain(source, x, s.S.Reals)
    finally:
        slot.__set__(source, original)


def test_source_mutation_during_checking_cannot_publish_a_domain(monkeypatch):
    x = s.Dummy("elementary_domain_mutation")
    source = s.log(x, evaluate=False)
    slot = Function.__dict__["_args"]
    original = slot.__get__(source, Function)
    solve = _native.solve_real_rational_constraints_native

    def changed(*args):
        result = solve(*args)
        slot.__set__(source, (x + 1,))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", changed)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.continuous_domain(source, x, s.S.Reals)
    finally:
        slot.__set__(source, original)


@pytest.mark.parametrize("failure", [ValueError("native resource refusal"), RuntimeError("cancelled")])
def test_admitted_operational_errors_propagate_without_fallback(monkeypatch, failure):
    x = s.Symbol("elementary_domain_failure")

    def failed(*args):
        raise failure

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", failed)
    with pytest.raises(type(failure)) as caught:
        s.continuous_domain(s.log(x), x, s.S.Reals)
    assert caught.value is failure


def test_intrinsic_source_and_condition_bounds_refuse_explicitly():
    x = s.Symbol("elementary_domain_bound")
    with pytest.raises(ValueError, match="exponent|integer exponent"):
        s.continuous_domain(s.Pow(x, 65, evaluate=False), x, s.S.Reals)
    expression = s.Add(*(s.log(x - index, evaluate=False) for index in range(20)), evaluate=False)
    with pytest.raises(ValueError, match="32 independent"):
        s.continuous_domain(expression, x, s.S.Reals)
