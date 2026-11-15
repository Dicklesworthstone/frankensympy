"""Complete real Abs solutions, including domains lost by naive squaring."""

import pytest
import sympy as s

from sympy.core import Basic, Function, _native


def _no_fallback(*args, **kwargs):
    raise AssertionError("admitted Abs sources must use complete checked branches")


@pytest.mark.parametrize("case, expected", [
    (lambda x: s.Abs(x) - 3, lambda: s.FiniteSet(-3, 3)),
    (lambda x: s.Eq(s.Abs(2*x - 1), 3, evaluate=False), lambda: s.FiniteSet(-1, 2)),
    (lambda x: s.Eq(-2, -s.Abs(x), evaluate=False), lambda: s.FiniteSet(-2, 2)),
    (lambda x: s.Abs(x - 1) + s.Abs(x + 1) - 4, lambda: s.FiniteSet(-2, 2)),
    (lambda x: s.Abs(x - 1) + s.Abs(x + 1) - 2, lambda: s.Interval(-1, 1)),
    (lambda x: (s.Abs(x) - 1)*(s.Abs(x) - 2), lambda: s.FiniteSet(-2, -1, 1, 2)),
    (lambda x: s.Abs(s.Abs(x) - 2) - 1, lambda: s.FiniteSet(-3, -1, 1, 3)),
    (lambda x: s.Abs(x) + 1, lambda: s.EmptySet()),
    (lambda x: s.Abs(x), lambda: s.FiniteSet(0)),
    (lambda x: s.Abs(x) - x, lambda: s.Interval(0, s.oo)),
    (lambda x: s.Abs(x) + x, lambda: s.Interval(-s.oo, 0)),
])
def test_complete_equations_and_nonfinite_solution_sets(case, expected, monkeypatch):
    x = s.Symbol("absolute_equation", real=True)
    source = case(x)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, domain=s.S.Reals) == expected()


def test_rational_absolute_values_keep_original_poles():
    x = s.Symbol("absolute_rational")
    source = s.Abs((x - 1)/(x + 1)) - 2
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(-3, -s.Rational(1, 3))
    assert s.solveset(s.Abs(x)/x - 1, x, s.S.Reals) == s.Interval.open(0, s.oo)


@pytest.mark.parametrize("form", ["zero", "power", "equality"])
def test_undefined_absolute_arguments_survive_held_identities(form):
    x = s.Symbol("absolute_held_" + form)
    atom = s.Abs(s.Pow(x, -1, evaluate=False), evaluate=False)
    source = {
        "zero": lambda: s.Mul(0, atom, evaluate=False),
        "power": lambda: s.Add(s.Pow(atom, 0, evaluate=False), -1, evaluate=False),
        "equality": lambda: s.Eq(atom, atom, evaluate=False),
    }[form]()
    result = s.solveset(source, x, s.S.Reals)
    assert result == s.Union(s.Interval.open(-s.oo, 0), s.Interval.open(0, s.oo))
    assert result.contains(0) is False


@pytest.mark.parametrize("operation, expected", [
    (lambda a: a < 2, lambda: s.Interval.open(-1, 3)),
    (lambda a: a <= 2, lambda: s.Interval(-1, 3)),
    (lambda a: a > 2, lambda: s.Union(s.Interval.open(-s.oo, -1), s.Interval.open(3, s.oo))),
    (lambda a: a >= 2, lambda: s.Union(s.Interval(-s.oo, -1), s.Interval(3, s.oo))),
])
def test_absolute_inequalities_are_complete_exact_sets(operation, expected, monkeypatch):
    x = s.Symbol("absolute_inequality", real=True)
    source = operation(s.Abs(x - 1))
    monkeypatch.setattr(_native, "solve_inequality_native", _no_fallback)
    assert s.solve_univariate_inequality(source, x, relational=False) == expected()
    assert s.solveset(source, x, s.S.Reals) == expected()


def test_rational_and_algebraic_inequality_boundaries():
    x = s.Symbol("absolute_sign_chart")
    rational = s.Abs((x - 1)/(x + 1)) < 1
    assert s.solve_univariate_inequality(rational, x, relational=False) == s.Interval.open(0, s.oo)
    polynomial = s.Abs(x*x - 2) <= 1
    result = s.solve_univariate_inequality(polynomial, x, relational=False)
    assert result == s.Union(s.Interval(-s.sqrt(3), -1), s.Interval(1, s.sqrt(3)))


def test_absolute_disequality_returns_the_complete_real_complement():
    x = s.Symbol("absolute_disequality", real=True)
    source = s.Ne(s.Abs(x), 2, evaluate=False)
    expected = s.Union(s.Interval.open(-s.oo, -2),
                       s.Interval.open(-2, 2), s.Interval.open(2, s.oo))
    assert s.solveset(source, x, s.S.Reals) == expected


def test_absolute_equations_and_disequalities_compose_with_real_constraints():
    x = s.Symbol("absolute_constraints", real=True)
    result = s.reduce_inequalities([
        s.Ne(s.Abs(x), 2, evaluate=False), x >= 0, x <= 3,
    ], x)
    assert result == s.Or(s.And(s.Le(0, x), s.Lt(x, 2)),
                          s.And(s.Lt(2, x), s.Le(x, 3)))
    assert s.reduce_inequalities([s.Eq(s.Abs(x), 2, evaluate=False), x > 0], x) == s.Eq(x, 2)


def test_absolute_constraints_intersect_checked_rational_disequalities():
    x = s.Symbol("absolute_rational_disequality", real=True)
    result = s.reduce_inequalities([s.Abs(x) <= 2, s.Ne(x, 1, evaluate=False)], x)
    assert result == s.Or(s.And(s.Le(-2, x), s.Lt(x, 1)),
                          s.And(s.Lt(1, x), s.Le(x, 2)))


def test_mixed_rational_disequality_keeps_its_held_pole():
    x = s.Symbol("absolute_rational_disequality_pole", real=True)
    quotient = s.Mul(x, s.Pow(x, -1, evaluate=False), evaluate=False)
    result = s.reduce_inequalities([
        s.Abs(x) <= 2, s.Ne(quotient, 0, evaluate=False),
    ], x)
    assert result == s.Or(s.And(s.Le(-2, x), s.Lt(x, 0)),
                          s.And(s.Lt(0, x), s.Le(x, 2)))


def test_restricted_domains_and_inferred_typed_variable():
    x = s.Symbol("absolute_restricted", real=True)
    source = s.Abs(x) - 2
    assert s.solveset(source, x, s.Interval(0, 3)) == s.FiniteSet(2)
    assert s.solveset(source, x, s.FiniteSet(-2, -1, 0, 2)) == s.FiniteSet(-2, 2)
    assert s.solveset(source, domain=s.S.Reals) == s.FiniteSet(-2, 2)
    assert s.solveset(source, x, s.EmptySet()) == s.EmptySet()


def test_unsupported_functions_and_parameters_preserve_the_original_condition():
    x, a = s.symbols("absolute_unknown_x absolute_unknown_a")
    for source in (s.Abs(x) - a, s.exp(s.Abs(x)) - 2, s.Abs(x)**x - 1):
        result = s.solveset(source, x, s.S.Reals)
        assert isinstance(result, s.ConditionSet)
        assert result.condition == s.Eq(source, 0, evaluate=False)
        assert result.base_set == s.S.Reals


def test_distinct_absolute_value_limit_precedes_candidate_generation(monkeypatch):
    x = s.Symbol("absolute_branch_limit")
    source = s.Add(*(s.Abs(x - k) for k in range(5)), evaluate=False)
    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", _no_fallback)
    with pytest.raises(ValueError, match="four distinct absolute"):
        s.solveset(source, x, s.S.Reals)


def test_repeated_absolute_values_share_the_same_case_assignment():
    x = s.Symbol("absolute_repeated")
    source = s.Add(*(s.Abs(x) for _ in range(10)), -20, evaluate=False)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(-2, 2)


@pytest.mark.parametrize("held", [False, True])
def test_sign_and_nested_canonical_forms_preserve_the_same_real_roots(held):
    x = s.Symbol("absolute_canonical_" + str(held))
    negative = s.Abs(-x, evaluate=not held)
    nested = s.Abs(s.Abs(x, evaluate=not held), evaluate=not held)
    for atom in (negative, nested):
        assert s.solveset(atom - 1, x, s.S.Reals) == s.FiniteSet(-1, 1)


def test_bounded_dispatch_does_not_apply_absolute_exponent_limits_to_other_grammars():
    from sympy.solvers._absolute_real import solve_absolute_real

    x = s.Symbol("absolute_unrelated_grammar")
    source = s.exp(s.Pow(x, 1000, evaluate=False))
    assert solve_absolute_real(source, x, s.S.Reals) is None


def test_oversized_original_power_is_refused_before_any_branch(monkeypatch):
    x = s.Symbol("absolute_power_limit")
    source = s.Abs(s.Pow(x, 65, evaluate=False), evaluate=False)
    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", _no_fallback)
    with pytest.raises(ValueError, match="integer exponent"):
        s.solveset(source, x, s.S.Reals)


@pytest.mark.parametrize("slot_owner, slot_name", [(Function, "_args"), (Basic, "_surface_args")])
def test_stale_absolute_surface_cannot_change_the_claim(slot_owner, slot_name):
    x = s.Symbol("absolute_stale_" + slot_name)
    atom = s.Abs(x, evaluate=False)
    slot = slot_owner.__dict__[slot_name]
    slot.__set__(atom, (x + 1,))
    with pytest.raises(ValueError, match="retained arguments"):
        s.solveset(atom, x, s.S.Reals)


def test_opaque_absolute_name_does_not_acquire_native_semantics():
    x = s.Symbol("absolute_opaque")
    impostor = s.Function("Abs")
    with pytest.raises(NotImplementedError):
        s.solveset(impostor(x), x, s.S.Reals)


def test_later_branch_failure_is_not_a_partial_solution(monkeypatch):
    x = s.Symbol("absolute_failure")
    original = _native.solve_real_rational_constraints_native
    calls = []

    def stop_second(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("injected later branch budget exhaustion")
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", stop_second)
    with pytest.raises(ValueError, match="later branch budget"):
        s.solveset(s.Abs(x) - 1, x, s.S.Reals)


def test_source_work_charges_the_active_caller_allowance(monkeypatch):
    from sympy.core import supervised_callback_session

    x = s.Symbol("absolute_caller_budget")
    source = s.Abs(x) - 1
    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", _no_fallback)
    with supervised_callback_session(max_nested_steps=1):
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            s.solveset(source, x, s.S.Reals)


def test_a_forged_branch_is_rejected_before_rational_verification(monkeypatch):
    from sympy.solvers import _absolute_real

    x = s.Symbol("absolute_forged_branch")
    monkeypatch.setattr(_absolute_real, "_rebuild", lambda *args: _native.py_integer(0))
    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", _no_fallback)
    with pytest.raises(ValueError, match="original source|original operator"):
        s.solveset(s.Abs(x) - 1, x, s.S.Reals)


def test_missing_sign_branches_cannot_publish_an_incomplete_set(monkeypatch):
    from sympy.solvers import _absolute_real

    x = s.Symbol("absolute_missing_branch")
    monkeypatch.setattr(_absolute_real, "product", lambda *args, **kwargs: iter(((1,),)))
    with pytest.raises(ValueError, match="coverage is incomplete"):
        s.solveset(s.Abs(x) - 1, x, s.S.Reals)


def test_source_mutation_during_verification_is_terminal(monkeypatch):
    x = s.Symbol("absolute_mutation")
    atom = s.Abs(x, evaluate=False)
    original = _native.solve_real_rational_constraints_native

    def mutate_source(*args, **kwargs):
        result = original(*args, **kwargs)
        Function.__dict__["_args"].__set__(atom, (x + 1,))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", mutate_source)
    with pytest.raises(RuntimeError, match="mutated"):
        s.solveset(atom, x, s.S.Reals)
