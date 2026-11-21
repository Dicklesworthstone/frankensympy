"""Exact public root counts, isolators, refinement and original-source guards."""

from unittest.mock import patch

import pytest
from sympy import (
    Dummy, Integer, Mul, Poly, Pow, PurePoly, Rational, S, Symbol,
    count_roots, intervals, refine_root,
)
from sympy.core import (
    SupervisedCallbackSession, _native, _native_expr, _symbol_binding,
    supervised_callback_session,
)
from sympy.polys.polyerrors import DomainError, PolynomialError


CLASSES = (Poly, PurePoly)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("lower,upper,expected", [
    (None, None, 4), (-2, 1, 3), (1, 1, 1), (-2, -2, 1),
    (None, 0, 2), (0, None, 2), (Rational(3, 2), 2, 0),
    (S.NegativeInfinity, S.Infinity, 4),
])
def test_counts_are_distinct_and_include_both_closed_endpoints(cls, lower, upper, expected):
    x = Symbol("x")
    p = cls((x + 2)**2 * (x - 1)**3 * (x**2 - 2), x)
    result = p.count_roots(lower, upper)
    assert result == Integer(expected)
    assert result.is_Integer


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("expression", [0, 7])
def test_constant_profile_conventions_do_not_construct_zero_root_certificates(cls, expression):
    x = Symbol("x")
    p = cls(expression, x)
    assert p.count_roots() == Integer(0)
    assert p.intervals() == []
    assert p.intervals(sqf=True) == []


@pytest.mark.parametrize("cls", CLASSES)
def test_intervals_are_complete_ordered_and_retain_original_multiplicities(cls):
    x = Symbol("x")
    p = cls((x + 2)**2 * (x - 1)**3 * (x**2 - 2), x)
    values = p.intervals(eps=Rational(1, 100))
    assert len(values) == 4
    assert [multiplicity for _, multiplicity in values] == [2, 1, 3, 1]
    for (lo, hi), multiplicity in values:
        assert lo <= hi and hi - lo <= Rational(1, 100)
        assert p.count_roots(lo, hi) == 1
        assert type(multiplicity) is int
    assert all(values[i][0][1] <= values[i + 1][0][0] for i in range(3))
    restricted = p.intervals(inf=-2, sup=1)
    assert len(restricted) == 3
    assert restricted[0] == ((Integer(-2), Integer(-2)), 2)
    assert restricted[-1] == ((Integer(1), Integer(1)), 3)


@pytest.mark.parametrize("cls", CLASSES)
def test_square_free_result_shape_and_nonreal_repetition_obligation(cls):
    x = Symbol("x")
    p = cls(x**2 - 2, x)
    plain = p.intervals(sqf=True)
    assert len(plain) == 2 and all(len(interval) == 2 for interval in plain)
    for lo, hi in plain:
        assert p.count_roots(lo, hi) == 1
    with pytest.raises(ValueError, match="square-free"):
        cls((x**2 + 1)**2, x).intervals(sqf=True)


@pytest.mark.parametrize("cls", CLASSES)
def test_precision_and_nearby_roots_remain_exact(cls):
    x = Symbol("x")
    epsilon = Rational(1, 10**30)
    p = cls(x**2 - 2, x)
    (lo, hi), multiplicity = p.intervals(inf=0, eps=epsilon)[0]
    assert multiplicity == 1 and hi - lo <= epsilon
    assert lo**2 <= 2 <= hi**2
    scale = Integer(10**25)
    nearby = cls((scale*x - scale) * (scale*x - scale - 1), x)
    assert nearby.count_roots() == 2
    assert nearby.count_roots(1, 1) == 1
    assert nearby.count_roots(1 + 1/scale, 1 + 1/scale) == 1


@pytest.mark.parametrize("cls", CLASSES)
def test_supplied_isolator_refinement_honors_width_or_explicit_step_cap(cls):
    x = Symbol("x")
    p = cls(x**2 - 2, x)
    assert p.refine_root(1, 2) == (Integer(1), Rational(3, 2))
    assert p.refine_root(2, 1) == (Integer(1), Rational(3, 2))
    assert p.refine_root(1, 2, steps=0) == (Integer(1), Integer(2))
    assert p.refine_root(1, 2, steps=-1) == (Integer(1), Integer(2))
    assert p.refine_root(1, 2, eps=Rational(1, 1000), steps=0) == (Integer(1), Integer(2))
    lo, hi = p.refine_root(1, 2, eps=Rational(1, 10**20))
    assert hi - lo <= Rational(1, 10**20)
    assert lo**2 <= 2 <= hi**2
    assert cls((x - 1)**2, x).refine_root(1, 1) == (Integer(1), Integer(1))
    with pytest.raises(ValueError, match="square-free"):
        cls((x - 1)**2, x).refine_root(0, 2, check_sqf=True)


@pytest.mark.parametrize("bounds", [(-2, 2), (2, 3), (1, 1)])
def test_nonisolating_or_false_singleton_intervals_are_refused(bounds):
    x = Symbol("x")
    with pytest.raises(ValueError, match="exactly one"):
        Poly(x**2 - 2, x).refine_root(*bounds)


def test_module_exports_and_immutable_float_tolerances():
    from sympy.polys import count_roots as poly_count
    from sympy.polys import intervals as poly_intervals
    from sympy.polys import refine_root as poly_refine

    x = Symbol("x")
    assert count_roots(x**2 - 2) == poly_count(x**2 - 2) == 2
    assert intervals(x**2 - 2) == poly_intervals(x**2 - 2)
    assert refine_root(x**2 - 2, 1, 2) == poly_refine(x**2 - 2, 1, 2)
    lo, hi = refine_root(x**2 - 2, 1.0, 2.0, eps=0.001)
    assert hi - lo <= Rational(*float(0.001).as_integer_ratio())


@pytest.mark.parametrize("variable", [Symbol("x", positive=True), Dummy("x")])
def test_typed_generator_identity_is_retained(variable):
    p = Poly(variable**2 - 2, variable)
    assert p.count_roots() == 2
    assert len(p.intervals()) == 2
    other = Symbol("x", negative=True)
    with pytest.raises((ValueError, DomainError)):
        Poly(variable**2 + other, variable).count_roots()


def test_unsupported_fields_shapes_and_complex_rectangles_are_explicit():
    x, y = Symbol("x"), Symbol("y")
    with pytest.raises(DomainError):
        Poly(x**2 - 2, x, modulus=5).count_roots()
    with pytest.raises(DomainError):
        Poly(x**2 - 2, x, domain="EX").intervals()
    with pytest.raises(PolynomialError):
        Poly(x + y, x, y).count_roots()
    with pytest.raises(NotImplementedError, match="complex"):
        Poly(x**2 + 1, x).intervals(all=True)
    with pytest.raises(NotImplementedError, match="multiple"):
        intervals([x**2 - 2, x - 1])
    with pytest.raises(ValueError, match="ordered"):
        Poly(x**2 - 2, x).count_roots(2, -2)


@pytest.mark.parametrize("epsilon", [0, -1])
def test_invalid_precision_and_step_limits_fail_before_publication(epsilon):
    x = Symbol("x")
    p = Poly(x**2 - 2, x)
    with pytest.raises(ValueError, match="positive"):
        p.intervals(eps=epsilon)
    with pytest.raises(ValueError, match="32768"):
        p.refine_root(1, 2, steps=32769)


@pytest.mark.parametrize("operation", ["count_roots", "intervals"])
def test_original_held_source_cannot_hide_an_undefined_child(operation):
    x = Symbol("x")
    expression = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        getattr(Poly(expression, x), operation)()


def test_mutated_polynomial_metadata_cannot_publish_a_result(monkeypatch):
    x = Symbol("x")
    p = Poly(x**2 - 2, x)
    native = _native.real_root_analysis_native

    def mutate(*args, **kwargs):
        result = native(*args, **kwargs)
        p._domain = "ZZ"
        return result

    monkeypatch.setattr(_native, "real_root_analysis_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        p.count_roots()


@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_refuse_before_source_capture(cancelled):
    from sympy.polys import _root_analysis

    x = Symbol("x")
    p = Poly(x**2 - 2, x)
    session = SupervisedCallbackSession(max_nested_steps=1_000_000 if cancelled else 0)
    if cancelled:
        session.cancel()
    with patch.object(_root_analysis.retained, "_PolyState",
                      side_effect=AssertionError("terminal query captured source")):
        with supervised_callback_session(session):
            with pytest.raises(RuntimeError if cancelled else ValueError,
                               match="cancelled|budget exhausted"):
                p.count_roots()


def test_native_root_queries_share_cumulative_active_request_and_cancel_terminally():
    x = Symbol("x")
    expression, variable = _native_expr(x**2 - 2), _symbol_binding(x)
    measured = SupervisedCallbackSession(max_nested_steps=10_000_000)
    assert len(_native.real_root_analysis_native(expression, variable, "count", session=measured)) == 2
    cost = measured.consumed_nested_steps
    assert cost > 0
    session = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    _native.real_root_analysis_native(expression, variable, "count", session=session)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        _native.real_root_analysis_native(expression, variable, "count", session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps and session.cancelled
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        _native.real_root_analysis_native(expression, variable, "count", session=session)
    assert session.consumed_nested_steps == consumed


def test_cancellation_after_native_acceptance_prevents_publication(monkeypatch):
    x = Symbol("x")
    native = _native.real_root_analysis_native
    with supervised_callback_session(max_nested_steps=10_000_000) as session:
        def cancel(*args, **kwargs):
            result = native(*args, **kwargs)
            session.cancel()
            return result
        monkeypatch.setattr(_native, "real_root_analysis_native", cancel)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            Poly(x**2 - 2, x).count_roots()


@pytest.mark.parametrize("slot", ["_struct_args", "_surface_args"])
@pytest.mark.parametrize("kind", ["coordinate", "bound", "coefficient"])
def test_atomic_retained_graphs_cannot_be_hidden_by_literal_native_storage(slot, kind, monkeypatch):
    x = Symbol("root_atomic_" + kind + slot)
    atom = Integer(17)
    if kind == "coefficient":
        expression = Mul(atom, x, evaluate=False)
    else:
        expression = x**2 - 2
    p = Poly(expression, x)
    target = x if kind == "coordinate" else atom
    monkeypatch.setattr(target, slot, (Pow(0, -1, evaluate=False),), raising=False)
    with pytest.raises(ValueError, match="retained"):
        if kind == "bound":
            p.count_roots(atom, 20)
        else:
            p.count_roots()


@pytest.mark.parametrize("constant", [S.Zero, S.One, S.NegativeOne, S.Half])
def test_fixed_numeric_bound_owner_cannot_be_rebound_to_another_native_atom(constant, monkeypatch):
    x = Symbol("root_fixed_bound")
    p = Poly(x**2 - 2, x)
    monkeypatch.setattr(constant, "_value", _native.py_integer(17))
    with pytest.raises(ValueError, match="native owner"):
        p.count_roots(constant, 20)
