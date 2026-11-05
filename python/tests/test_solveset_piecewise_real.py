"""First-match rational Piecewise solving through the checked public path."""

import pytest

import sympy as s
from sympy.core import Basic, _native, _surface_symbol, _symbol_binding
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys import rootoftools


def _forbid_fallback(*args, **kwargs):
    raise AssertionError("admitted Piecewise solving must not use a fallback")


@pytest.mark.parametrize("closed", [True, False])
def test_undefined_selected_branch_never_falls_through(closed):
    x = s.Symbol("x")
    condition = s.Ge(x, 0, evaluate=False) if closed else s.Gt(x, 0, evaluate=False)
    source = s.Piecewise((s.Pow(x, -1, evaluate=False), condition), (0, True), evaluate=False)
    expected = s.Interval(-s.oo, 0, right_open=closed)
    assert s.solveset(source, x, s.S.Reals) == expected
    assert s.solveset(source, x, s.Interval(-1, 1)) == s.Interval(-1, 0, right_open=closed)


def test_first_match_excludes_prior_conditions_not_prior_solution_sets():
    x = s.Symbol("x")
    source = s.Piecewise((1, x <= 2), (x-1, x < 4), (0, True), evaluate=False)
    assert s.solveset(source, x, s.S.Reals) == s.Interval(4, s.oo)
    duplicate = s.Piecewise((1, x >= 0), (0, x >= 0), (0, True), evaluate=False)
    assert s.solveset(duplicate, x, s.S.Reals) == s.Interval.open(-s.oo, 0)
    roots = s.Piecewise((x-2, x < 3), (x-4, x > 1), (x, True), evaluate=False)
    assert s.solveset(roots, x, s.S.Reals) == s.FiniteSet(2, 4)


def test_missing_default_leaves_unmatched_points_undefined():
    x = s.Symbol("x")
    source = s.Piecewise((0, x < 0), (x-1, x < 2), evaluate=False)
    assert s.solveset(source, x, s.S.Reals) == s.Union(s.Interval.open(-s.oo, 0), s.FiniteSet(1))


def test_false_branches_and_held_zero_expression_domains_are_distinct():
    x = s.Symbol("x")
    nowhere = s.Pow(0, -1, evaluate=False)
    hidden_hole = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.Piecewise((nowhere, False), (hidden_hole, x >= 0), (0, True), evaluate=False)
    assert s.solveset(source, x, s.S.Reals) == s.Complement(s.S.Reals, s.FiniteSet(0))
    assert s.solveset(source, x, s.FiniteSet(-1, 0, 1)) == s.FiniteSet(-1, 1)


def test_rational_branch_cancellation_retains_all_selected_source_holes():
    x = s.Symbol("x")
    quotient = s.Mul(x-1, s.Pow(x-1, -1, evaluate=False), evaluate=False)
    residual = s.Add(quotient, -1, evaluate=False)
    source = s.Piecewise((residual, x >= 0), (1, True), evaluate=False)
    assert s.solveset(source, x, s.Interval(-2, 2)) == s.Complement(s.Interval(0, 2), s.FiniteSet(1))


@pytest.mark.parametrize("reverse", [False, True])
def test_held_equality_keeps_common_side_holes_and_orientation(reverse):
    x = s.Symbol("x")
    inverse = s.Pow(x, -1, evaluate=False)
    source = s.Piecewise((inverse, x >= 0), (0, True), evaluate=False)
    equation = s.Eq(inverse, source, evaluate=False) if reverse else s.Eq(source, inverse, evaluate=False)
    assert s.solveset(equation, x, s.S.Reals) == s.Interval.open(0, s.oo)
    absolute_negative = s.Piecewise((x, x < 0), (-x, True), evaluate=False)
    assert s.solveset(s.Eq(absolute_negative, -1, evaluate=False), x, s.S.Reals) == s.FiniteSet(-1, 1)


def test_boolean_guards_keep_exact_boundaries_and_point_conditions():
    x = s.Symbol("x")
    source = s.Piecewise(
        (0, s.And(x >= -2, x < 0)),
        (0, s.Or(s.Eq(x, 1, evaluate=False), s.Eq(x, 3, evaluate=False))),
        (1, True), evaluate=False,
    )
    assert s.solveset(source, x, s.S.Reals) == s.Union(s.Interval.Ropen(-2, 0), s.FiniteSet(1, 3))
    outside = s.Piecewise((0, s.Not(s.And(x >= -1, x <= 1), evaluate=False)), (1, True), evaluate=False)
    assert s.solveset(outside, x, s.S.Reals) == s.Union(
        s.Interval.open(-s.oo, -1), s.Interval.open(1, s.oo),
    )


@pytest.mark.parametrize("condition_kind,selected", [("eq", True), ("ne", False), ("not", False)])
def test_finite_algebraic_aliases_remain_decided_in_boolean_guards(condition_kind, selected):
    x, bound = s.symbols("x bound")
    point = s.CRootOf(bound**2-2, 0, radicals=False)
    equality = s.Eq(x**2-2, 0, evaluate=False)
    condition = (equality if condition_kind == "eq" else
                 s.Ne(x**2-2, 0, evaluate=False) if condition_kind == "ne" else
                 s.Not(s.And(equality, s.Ge(x, -2, evaluate=False)), evaluate=False))
    source = s.Piecewise((0, condition), (1, True), evaluate=False)
    domain = s.FiniteSet(point)
    assert s.solveset(source, x, domain) == (domain if selected else s.S.EmptySet)
    later = s.Piecewise((1, equality), (0, True), evaluate=False)
    assert s.solveset(later, x, domain) == s.S.EmptySet


def test_algebraic_guard_endpoints_retain_alias_holes_in_later_branches():
    x, bound = s.symbols("x bound")
    low = s.CRootOf(bound**2-2, 0, radicals=False)
    high = s.CRootOf(bound**2-2, 1, radicals=False)
    source = s.Piecewise((1, s.Eq(x**2-2, 0, evaluate=False)), (0, True), evaluate=False)
    result = s.solveset(source, x, s.Interval(low, high))
    for excluded in (low, high, -s.sqrt(2), s.sqrt(2)):
        assert result.contains(excluded) is False
    assert result.contains(0) is not False


def test_irreducible_guards_and_branch_roots_are_complete_and_exact(monkeypatch):
    x = s.Symbol("x")
    polynomial = x**5-x+1
    root = s.CRootOf(polynomial, 0)
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_fallback)
    monkeypatch.setattr(rootoftools, "_approximate_factor_roots", _forbid_fallback)
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid_fallback)
    source = s.Piecewise((0, s.Le(polynomial, 0, evaluate=False)), (1, True), evaluate=False)
    assert s.solveset(source, x, s.S.Reals) == s.Interval(-s.oo, root)
    equation = s.Piecewise((polynomial, x < 0), (x-2, True), evaluate=False)
    assert s.solveset(equation, x, s.S.Reals) == s.FiniteSet(root, 2)


def test_restricted_unions_and_finite_domains_compose_with_branch_partitioning():
    x = s.Symbol("x")
    source = s.Piecewise((0, x < 0), (x**2-2, True), evaluate=False)
    domain = s.Union(s.Interval(-2, -1), s.FiniteSet(-3, 0, s.sqrt(2), 2))
    assert s.solveset(source, x, domain) == s.Union(s.Interval(-2, -1), s.FiniteSet(-3, s.sqrt(2)))


def test_rational_guards_are_solved_when_original_poles_are_unreachable():
    x = s.Symbol("x")
    guard = s.Gt(s.Pow(x, -1, evaluate=False), 0, evaluate=False)
    source = s.Piecewise((0, guard), (1, True), evaluate=False)
    domain = s.Complement(s.Interval(-2, 2), s.FiniteSet(0))
    assert s.solveset(source, x, domain) == s.Interval.Lopen(0, 2)
    unreachable = s.Piecewise((1, s.Eq(x, 0, evaluate=False)), (0, guard), (1, True), evaluate=False)
    assert s.solveset(unreachable, x, s.S.Reals) == s.Interval.open(0, s.oo)


def test_reachable_or_undecidable_guard_poles_keep_the_whole_query_conditional():
    x, endpoint = s.symbols("x endpoint")
    guard = s.Gt(s.Pow(x, -1, evaluate=False), 0, evaluate=False)
    source = s.Piecewise((x, guard), (0, True), evaluate=False)
    for domain in (s.S.Reals, s.Interval(endpoint, 1)):
        result = s.solveset(source, x, domain)
        assert type(result) is s.ConditionSet
        assert result.sym == x
        assert result.condition == s.Eq(source, 0, evaluate=False)
        assert result.base_set is domain


def test_unknown_guards_and_nonrational_branches_never_publish_a_partial_set(monkeypatch):
    x, unknown = s.symbols("x unknown")
    sources = (
        s.Piecewise((0, x < 0), (x, unknown), (0, True), evaluate=False),
        s.Piecewise((0, x < 0), (s.sin(x), True), evaluate=False),
        s.Piecewise((0, s.Symbol("True")), (1, True), evaluate=False),
    )
    monkeypatch.setattr(_native, "solve_real_rational_native", _forbid_fallback)
    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", _forbid_fallback)
    monkeypatch.setattr(_native, "solve_native", _forbid_fallback)
    for source in sources:
        result = s.solveset(source, x, s.S.Reals)
        assert type(result) is s.ConditionSet
        assert result.condition == s.Eq(source, 0, evaluate=False)


def test_typed_symbols_are_used_for_every_branch_and_guard(monkeypatch):
    ordinary = _native.solve_real_rational_native
    inequality = _native.solve_real_rational_inequality_native
    seen = []

    def record_plain(source, binding):
        seen.append((binding.name, binding.identity_hex))
        return ordinary(source, binding)

    def record_guard(source, binding, operation):
        seen.append((binding.name, binding.identity_hex))
        return inequality(source, binding, operation)

    monkeypatch.setattr(_native, "solve_real_rational_native", record_plain)
    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", record_guard)
    monkeypatch.setattr(_native, "solve_native", _forbid_fallback)
    for x in (s.Symbol("x", real=True), s.Dummy("x", real=True)):
        seen.clear()
        source = s.Piecewise((x-1, s.Ge(x, 0, evaluate=False)), (x+1, True), evaluate=False)
        assert s.solveset(source, domain=s.S.Reals) == s.FiniteSet(-1, 1)
        binding = _symbol_binding(x)
        assert seen and all(item == (binding.name, binding.identity_hex) for item in seen)
    other = s.Symbol("x", positive=True)
    source = s.Piecewise((0, s.Gt(other, 0, evaluate=False)), (1, True), evaluate=False)
    assert type(s.solveset(source, s.Symbol("x", real=True), s.S.Reals)) is s.ConditionSet


@pytest.mark.parametrize("native_name", ["solve_real_rational_native", "solve_real_rational_inequality_native"])
@pytest.mark.parametrize("failure", [ValueError("resource exhausted"), OverflowError("coefficient limit"),
                                      MemoryError("allocation refused"), NotImplementedError("native refusal")])
def test_admitted_native_failures_propagate_without_fallback(monkeypatch, native_name, failure):
    x = s.Symbol("x")
    source = s.Piecewise((x, x >= 0), (0, True), evaluate=False)
    calls = []

    def fail(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(_native, native_name, fail)
    monkeypatch.setattr(_native, "solve_native", _forbid_fallback)
    with pytest.raises(type(failure)) as error:
        s.solveset(source, x, s.S.Reals)
    assert error.value is failure
    assert len(calls) == 1


def test_wrong_ordered_guard_roots_are_not_published(monkeypatch):
    x = s.Symbol("x")
    source = s.Piecewise((0, x**2 >= 2), (1, True), evaluate=False)
    monkeypatch.setattr(rootoftools, "real_roots", lambda *args, **kwargs: [(s.Integer(1), 1), (s.Integer(2), 1)])
    with pytest.raises(ValueError, match="association changed"):
        s.solveset(source, x, s.S.Reals)


def test_retained_piecewise_mutation_during_native_work_is_not_published(monkeypatch):
    x = s.Symbol("x")
    source = s.Piecewise((x, x >= 0), (0, True), evaluate=False)
    slot = Piecewise.__dict__["_piecewise_branches"]
    native = _native.solve_real_rational_native

    def mutate(*args):
        result = native(*args)
        slot.__set__(source, ((s.Integer(1), s.true),))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.solveset(source, x, s.S.Reals)


def test_plain_symbol_native_handle_refresh_preserves_the_captured_source(monkeypatch):
    name = "piecewise_plain_refresh"
    x = s.Symbol(name)
    source = s.Piecewise((x-1, x >= 0), (x+1, True), evaluate=False)
    native = _native.solve_real_rational_native
    slot = Basic.__dict__["_value"]
    refreshed = []

    def refresh(*args):
        result = native(*args)
        old = slot.__get__(x, Basic)
        new = _native.py_symbol(name)
        assert new is not old and new == old
        assert _surface_symbol(new) is x
        assert slot.__get__(x, Basic) is new
        refreshed.append(new)
        return result

    monkeypatch.setattr(_native, "solve_real_rational_native", refresh)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(-1, 1)
    assert refreshed


def test_same_named_symbol_binding_mutation_is_still_rejected(monkeypatch):
    name = "piecewise_changed_binding"
    x = s.Symbol(name)
    replacement = s.Symbol(name, positive=True)
    source = s.Piecewise((0, True), (x, False), evaluate=False)
    native = _native.solve_real_rational_native
    slot = Basic.__dict__["_value"]
    previous = slot.__get__(x, Basic)

    def mutate(*args):
        result = native(*args)
        slot.__set__(x, slot.__get__(replacement, Basic))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_native", mutate)
    try:
        with pytest.raises(RuntimeError, match="source mutated"):
            s.solveset(source, x, s.S.Reals)
    finally:
        slot.__set__(x, previous)


def test_foreign_and_custom_retained_sources_do_not_execute_hooks():
    class Foreign:
        @property
        def __class__(self):
            raise AssertionError("foreign class getter must not run")

        def _sympy_(self):
            raise AssertionError("foreign conversion must not run")

    class CustomSymbol(s.Symbol):
        def __sub__(self, other):
            raise AssertionError("custom arithmetic must not run")

        def __str__(self):
            raise AssertionError("custom printer must not run")

    x = s.Symbol("x")
    source = s.Piecewise((x, x >= 0), (0, True), evaluate=False)
    with pytest.raises(TypeError, match="variable must be a Symbol"):
        s.solveset(source, Foreign(), s.S.Reals)
    slot = Piecewise.__dict__["_piecewise_branches"]
    for branch, error in ((Foreign(), TypeError), (CustomSymbol("x"), NotImplementedError)):
        slot.__set__(source, ((branch, s.true),))
        with pytest.raises(error):
            s.solveset(source, x, s.S.Reals)


def test_branch_and_guard_bounds_refuse_before_any_solver_call(monkeypatch):
    x = s.Symbol("x")
    too_many = s.Piecewise(*((x-i, x < i) for i in range(17)), evaluate=False)
    crowded = s.Piecewise((0, s.And(*(s.Ge(x, i, evaluate=False) for i in range(64)), evaluate=False)),
                         (1, True), evaluate=False)
    monkeypatch.setattr(_native, "solve_real_rational_native", _forbid_fallback)
    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", _forbid_fallback)
    with pytest.raises(ValueError, match="sixteen retained branches"):
        s.solveset(too_many, x, s.S.Reals)
    with pytest.raises(ValueError, match="too many guard nodes"):
        s.solveset(crowded, x, s.S.Reals)


def test_piecewise_resource_refusal_remains_terminal_in_the_real_domain():
    x = s.Symbol("x")
    source = s.Piecewise((s.Pow(x, 65, evaluate=False), True), (0, False), evaluate=False)
    with pytest.raises(ValueError, match="resource limit"):
        s.solveset(source, x, s.S.Reals)
    assert type(s.solveset(s.Pow(x, 65, evaluate=False), x, s.S.Reals)) is s.ConditionSet
