"""Complete principal-square-root sets, branch signs and original domains."""

import importlib

import pytest
import sympy as s

from sympy.core import Basic, _native, _native_expr


def _no_fallback(*args, **kwargs):
    raise AssertionError("admitted radical equations must not use generic root candidates")


@pytest.mark.parametrize("source, expected", [
    (lambda x: s.sqrt(x**2)-x, lambda x: s.Interval(0, s.oo)),
    (lambda x: s.sqrt(x**2)+x, lambda x: s.Interval(-s.oo, 0)),
    (lambda x: s.sqrt((x-1)**2)-(x-1), lambda x: s.Interval(1, s.oo)),
    (lambda x: -2*s.sqrt(x**2)+2*x, lambda x: s.Interval(0, s.oo)),
    (lambda x: s.sqrt((x-1)**2)/(x-1)-1, lambda x: s.Interval.open(1, s.oo)),
    (lambda x: (x+1)*s.sqrt(x), lambda x: s.FiniteSet(-1, 0)),
    (lambda x: s.sqrt(x)+1, lambda x: s.EmptySet()),
    (lambda x: s.sqrt(-x)-x, lambda x: s.FiniteSet(0)),
    (lambda x: s.sqrt(-x)+x, lambda x: s.FiniteSet(-1, 0)),
    (lambda x: s.sqrt(x+1)+x-1, lambda x: s.FiniteSet(0)),
    (lambda x: s.sqrt(x)+x-1, lambda x: s.FiniteSet((3-s.sqrt(5))/2)),
    (lambda x: s.sqrt(2)*x-1, lambda x: s.FiniteSet(s.sqrt(2)/2)),
])
def test_complete_radical_equations_include_intervals_and_correct_signs(source, expected, monkeypatch):
    x = s.Symbol("radical_complete_axis")
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source(x), x, s.S.Reals) == expected(x)


def test_high_degree_radical_uses_the_complete_checked_algebraic_partition(monkeypatch):
    x = s.Symbol("radical_quintic_axis")
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    source = s.sqrt(x**5+x**2-x-1)-x
    expected = s.FiniteSet(s.CRootOf(x**5-x-1, 0))
    assert s.solveset(source, x, s.S.Reals) == expected


@pytest.mark.parametrize("orientation", ["forward", "reverse", "both_sides"])
def test_equation_sides_are_decomposed_before_source_cancellation(orientation, monkeypatch):
    x = s.Symbol("radical_equality_axis")
    root = s.sqrt(x**2)
    source = {
        "forward": lambda: s.Eq(root, x, evaluate=False),
        "reverse": lambda: s.Eq(x, root, evaluate=False),
        "both_sides": lambda: s.Eq(3*root+2*x, root+4*x, evaluate=False),
    }[orientation]()
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.Interval(0, s.oo)


@pytest.mark.parametrize("form", ["equality", "zero", "power"])
def test_original_rational_holes_survive_every_radical_identity_shortcut(form, monkeypatch):
    x = s.Symbol("radical_retained_hole")
    inverse = s.Pow(x-2, -1, evaluate=False)
    residual = s.sqrt(x**2)-x
    source = {
        "equality": lambda: s.Eq(inverse, s.Add(inverse, residual, evaluate=False), evaluate=False),
        "zero": lambda: s.Add(residual, s.Mul(0, inverse, evaluate=False), evaluate=False),
        "power": lambda: s.Add(residual, s.Pow(inverse, 0, evaluate=False), -1, evaluate=False),
    }[form]()
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    expected = s.Union(s.Interval.Ropen(0, 2), s.Interval.open(2, s.oo))
    assert s.solveset(source, x, s.S.Reals) == expected


@pytest.mark.parametrize("form", ["zero_coefficient", "zeroth_power", "equal_sides"])
def test_radicand_poles_remain_when_the_radical_coefficient_vanishes(form, monkeypatch):
    x = s.Symbol("radical_zero_coefficient")
    radical = s.Pow(s.Pow(x, -1, evaluate=False), s.Rational(1, 2), evaluate=False)
    source = {
        "zero_coefficient": lambda: s.Mul(0, radical, evaluate=False),
        "zeroth_power": lambda: s.Add(s.Pow(radical, 0, evaluate=False), -1, evaluate=False),
        "equal_sides": lambda: s.Eq(radical, radical, evaluate=False),
    }[form]()
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    expected = s.Union(s.Interval.open(-s.oo, 0), s.Interval.open(0, s.oo))
    assert s.solveset(source, x, s.S.Reals) == expected


def test_zero_coefficient_does_not_require_the_radical_itself_to_be_real(monkeypatch):
    x = s.Symbol("radical_negative_radicand")
    source = s.Mul(0, s.Pow(x, s.Rational(1, 2), evaluate=False), evaluate=False)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.S.Reals


@pytest.mark.parametrize("domain, expected", [
    (s.Interval(-2, 2), s.Interval(0, 2)),
    (s.Interval.open(0, 2), s.Interval.open(0, 2)),
    (s.FiniteSet(-1, 0, 1), s.FiniteSet(0, 1)),
    (s.Union(s.Interval(-3, -1), s.Interval(1, 2)), s.Interval(1, 2)),
    (s.Complement(s.Interval(-2, 2), s.FiniteSet(0)), s.Interval.Lopen(0, 2)),
])
def test_complete_radical_sets_respect_compound_real_domains(domain, expected, monkeypatch):
    x = s.Symbol("radical_restricted_axis")
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(s.sqrt(x**2)-x, x, domain) == expected


def test_symbolic_domain_boundary_keeps_undecided_order(monkeypatch):
    x, endpoint = s.symbols("radical_symbolic_domain endpoint", real=True)
    domain = s.Interval(-2, endpoint)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    expected = s.Intersection(domain, s.Interval(0, s.oo))
    assert s.solveset(s.sqrt(x**2)-x, x, domain) == expected


def test_typed_and_inferred_coordinates_are_preserved(monkeypatch):
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    for x in (s.Symbol("radical_typed", real=True), s.Dummy("radical_typed")):
        source = s.Pow(x*x, s.Rational(1, 2), evaluate=False)-x
        assert s.solveset(source, x, s.S.Reals) == s.Interval(0, s.oo)
        assert s.solveset(source, domain=s.S.Reals) == s.Interval(0, s.oo)


@pytest.mark.parametrize("kind", ["multiple", "nested", "denominator", "nonrational", "parameter"])
def test_unsupported_radical_grammars_retain_the_original_condition(kind, monkeypatch):
    x, parameter = s.symbols("radical_unsupported_axis radical_parameter")
    root = s.Pow(x, s.Rational(1, 2), evaluate=False)
    source = {
        "multiple": lambda: root+s.sqrt(x+1)-2,
        "nested": lambda: s.Pow(root, s.Rational(1, 2), evaluate=False)-1,
        "denominator": lambda: s.Pow(root+1, -1, evaluate=False)-1,
        "nonrational": lambda: s.pi*root-1,
        "parameter": lambda: root-parameter,
    }[kind]()
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    result = s.solveset(source, x, s.S.Reals)
    assert type(result) is s.ConditionSet
    assert result.sym is x
    assert result.condition == s.Eq(source, 0, evaluate=False)
    assert result.base_set == s.S.Reals


def test_same_name_parameter_is_not_rebound_to_the_coordinate():
    from sympy.solvers._radical_real import solve_radical_real

    x = s.Symbol("radical_same_name")
    other = s.Symbol("radical_same_name", positive=True)
    source = s.Pow(other, s.Rational(1, 2), evaluate=False)-x
    result = solve_radical_real(source, x, s.S.Reals)
    assert type(result) is s.ConditionSet
    assert result.condition == s.Eq(source, 0, evaluate=False)


def test_other_domains_and_non_square_powers_keep_their_existing_dispatch():
    from sympy.solvers._radical_real import solve_radical_real

    x = s.Symbol("radical_separate_routes")
    assert solve_radical_real(s.sqrt(x)-1, x, s.S.Complexes) is None
    assert solve_radical_real(x**s.Rational(1, 3)-1, x, s.S.Reals) is None
    assert solve_radical_real(x*x-1, x, s.S.Reals) is None


def test_wrong_proposed_coefficients_cannot_publish_a_solution_set(monkeypatch):
    module = importlib.import_module("sympy.solvers._radical_real")
    x = s.Symbol("radical_wrong_coefficients")
    monkeypatch.setattr(module, "_decompose_equation", lambda trees, work: (module._ZERO, module._ZERO))
    with pytest.raises(ValueError, match="reconstruct"):
        s.solveset(s.sqrt(x**2)-x, x, s.S.Reals)


def test_equivalent_coefficients_cannot_invent_a_new_domain_hole(monkeypatch):
    module = importlib.import_module("sympy.solvers._radical_real")
    x = s.Symbol("radical_invented_pole")
    original = module._decompose_equation

    def invented_pole(trees, work):
        p, q = original(trees, work)
        coordinate = module._Value(_native_expr(x))
        inverse = work.make("Pow", (coordinate, module._NEGATIVE_ONE))
        return p, work.make("Mul", (q, coordinate, inverse))

    monkeypatch.setattr(module, "_decompose_equation", invented_pole)
    with pytest.raises(ValueError, match="domain hole"):
        s.solveset(s.sqrt(x**2)-x, x, s.S.Reals)


def test_forged_domain_shadow_cannot_erase_original_poles(monkeypatch):
    module = importlib.import_module("sympy.solvers._radical_real")
    x = s.Symbol("radical_forged_shadow")
    original = module._replay

    def erase(tree, replacement, work):
        if replacement is None:
            return module._ONE
        return original(tree, replacement, work)

    monkeypatch.setattr(module, "_replay", erase)
    with pytest.raises(ValueError, match="domain shadow"):
        s.solveset(s.sqrt((x-1)**2)/(x-1)-1, x, s.S.Reals)


def test_source_mutation_during_checked_solving_is_terminal(monkeypatch):
    x = s.Symbol("radical_source_mutation")
    radical = s.Pow(x*x, s.Rational(1, 2), evaluate=False)
    source = s.Add(radical, -x, evaluate=False)
    original = _native.solve_real_rational_constraints_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        s.Pow.__dict__["_args"].__set__(radical, (x+1, s.Rational(1, 2)))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        s.solveset(source, x, s.S.Reals)


@pytest.mark.parametrize("slot_name", ["_struct_args", "_surface_args"])
@pytest.mark.parametrize("kind", ["integer", "symbol", "native_wrapper"])
def test_atomic_owners_cannot_hide_a_second_source_graph(slot_name, kind):
    x = s.Symbol("radical_atomic_"+kind+slot_name)
    if kind == "symbol":
        atom = x
        source = s.Pow(x*x, s.Rational(1, 2), evaluate=False)-x
    else:
        atom = s.Integer(17) if kind == "integer" else object.__new__(s.Expr)
        if kind == "native_wrapper":
            Basic.__dict__["_value"].__set__(atom, _native.py_integer(17))
        source = s.Mul(atom, s.sqrt(x**2)-x, evaluate=False)
    slot = Basic.__dict__[slot_name]
    missing = object()
    try:
        original = slot.__get__(atom, Basic)
    except AttributeError:
        original = missing
    try:
        slot.__set__(atom, (s.Pow(x, -1, evaluate=False),))
        with pytest.raises((ValueError, NotImplementedError), match="retained|surface"):
            s.solveset(source, x, s.S.Reals)
    finally:
        if original is missing:
            slot.__delete__(atom)
        else:
            slot.__set__(atom, original)


def test_stale_power_arguments_are_rejected_before_solving():
    x = s.Symbol("radical_stale_power")
    radical = s.Pow(x*x, s.Rational(1, 2), evaluate=False)
    source = s.Add(radical, -x, evaluate=False)
    s.Pow.__dict__["_args"].__set__(radical, (x+1, s.Rational(1, 2)))
    with pytest.raises((ValueError, NotImplementedError), match="retained|surface"):
        s.solveset(source, x, s.S.Reals)


@pytest.mark.parametrize("failure", [
    ValueError("radical constraint budget exhausted"),
    RuntimeError("radical request cancelled"),
])
def test_native_operational_failures_remain_terminal(failure, monkeypatch):
    x = s.Symbol("radical_failure_axis")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", fail)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with pytest.raises(type(failure)) as caught:
        s.solveset(s.sqrt(x**2)-x, x, s.S.Reals)
    assert caught.value is failure


def test_construction_work_limit_is_not_a_fallback(monkeypatch):
    module = importlib.import_module("sympy.solvers._radical_real")
    x = s.Symbol("radical_work_axis")
    monkeypatch.setattr(module, "_MAX_WORK", 1)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with pytest.raises(ValueError, match="work bound"):
        s.solveset(s.sqrt(x**2)-x, x, s.S.Reals)


def test_source_work_charges_the_active_caller_budget(monkeypatch):
    from sympy.core import supervised_callback_session

    x = s.Symbol("radical_caller_budget")
    source = s.sqrt(x**2)-x
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with supervised_callback_session(max_nested_steps=1):
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            s.solveset(source, x, s.S.Reals)
