"""Complete real elementary inverses with original domains and typed axes."""

import importlib

import pytest
import sympy as s

from sympy.core import Basic, Function, _native


def _no_fallback(*args, **kwargs):
    raise AssertionError("admitted elementary equations must not use principal-root fallback")


@pytest.mark.parametrize("function, target, values", [
    (s.exp, 2, lambda: [s.log(2)]),
    (s.exp, 1, lambda: [s.Integer(0)]),
    (s.exp, 0, lambda: []),
    (s.exp, -2, lambda: []),
    (s.log, 0, lambda: [s.Integer(1)]),
    (s.log, 2, lambda: [s.exp(2)]),
    (s.log, -2, lambda: [s.exp(-2)]),
    (s.sinh, 0, lambda: [s.Integer(0)]),
    (s.sinh, 2, lambda: [s.asinh(2)]),
    (s.sinh, -2, lambda: [s.asinh(-2)]),
    (s.cosh, 2, lambda: [-s.acosh(2), s.acosh(2)]),
    (s.cosh, 1, lambda: [s.Integer(0)]),
    (s.cosh, s.Rational(1, 2), lambda: []),
    (s.cosh, -2, lambda: []),
    (s.tanh, s.Rational(1, 2), lambda: [s.atanh(s.Rational(1, 2))]),
    (s.tanh, -s.Rational(1, 2), lambda: [s.atanh(-s.Rational(1, 2))]),
    (s.tanh, 0, lambda: [s.Integer(0)]),
    (s.tanh, 1, lambda: []),
    (s.tanh, -1, lambda: []),
    (s.tanh, 2, lambda: []),
    (s.tanh, -2, lambda: []),
    (s.asinh, 1, lambda: [s.sinh(1)]),
    (s.asinh, -1, lambda: [s.sinh(-1)]),
    (s.acosh, 0, lambda: [s.Integer(1)]),
    (s.acosh, 1, lambda: [s.cosh(1)]),
    (s.acosh, -1, lambda: []),
    (s.atanh, 1, lambda: [s.tanh(1)]),
    (s.atanh, -1, lambda: [s.tanh(-1)]),
])
def test_complete_real_ranges_and_every_inverse_branch(function, target, values, monkeypatch):
    x = s.Symbol("elementary_real_axis")
    source = s.Eq(function(x), target, evaluate=False)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(*values())


@pytest.mark.parametrize("slope, shift, scale, offset", [
    (2, 3, 5, 7),
    (-2, 3, -5, 7),
    (s.Rational(2, 3), -s.Rational(4, 5), s.Rational(7, 3), -2),
])
def test_rational_affine_inner_and_outer_maps_preserve_both_cosh_roots(
    slope, shift, scale, offset, monkeypatch,
):
    x = s.Symbol("elementary_affine")
    source = s.Eq(scale*s.cosh(slope*x+shift)+offset, 2*scale+offset, evaluate=False)
    expected = s.FiniteSet((-s.acosh(2)-shift)/slope, (s.acosh(2)-shift)/slope)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == expected


def test_repeated_elementary_values_have_one_exact_affine_identity(monkeypatch):
    x = s.Symbol("elementary_repeated")
    source = s.Add(s.cosh(x), s.cosh(x), -4, evaluate=False)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(-s.acosh(2), s.acosh(2))


@pytest.mark.parametrize("form", ["zero", "power", "equality", "cancelled_fraction"])
def test_original_poles_survive_outer_rational_normalization(form, monkeypatch):
    x = s.Symbol("elementary_hole_" + form)
    hole = s.Pow(x, -1, evaluate=False)
    residual = s.cosh(x)-1
    source = {
        "zero": lambda: s.Add(residual, s.Mul(0, hole, evaluate=False), evaluate=False),
        "power": lambda: s.Add(residual, s.Pow(hole, 0, evaluate=False), -1, evaluate=False),
        "equality": lambda: s.Eq(hole, s.Add(hole, residual, evaluate=False), evaluate=False),
        "cancelled_fraction": lambda: s.Add(
            s.Mul(x, hole, s.cosh(x), evaluate=False), -1, evaluate=False,
        ),
    }[form]()
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.EmptySet()


def test_original_holes_can_leave_a_valid_elementary_solution(monkeypatch):
    x = s.Symbol("elementary_valid_hole")
    hole = s.Pow(x-2, -1, evaluate=False)
    source = s.Eq(hole, s.Add(hole, s.log(x), evaluate=False), evaluate=False)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(1)


def test_affine_argument_identity_retains_its_original_pole(monkeypatch):
    x = s.Symbol("elementary_argument_hole")
    argument = s.Mul(x*x, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.cosh(argument, evaluate=False)-1
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, s.S.Reals) == s.EmptySet()


@pytest.mark.parametrize("domain, expected", [
    (s.Interval(-1, 1), s.FiniteSet(0)),
    (s.Interval.open(0, 1), s.EmptySet()),
    (s.FiniteSet(-2, 0, 2), s.FiniteSet(0)),
    (s.Union(s.Interval(-2, -1), s.Interval(0, 1)), s.FiniteSet(0)),
    (s.Intersection(s.Interval(-1, 1), s.Complement(s.S.Reals, s.FiniteSet(0))), s.EmptySet()),
])
def test_complete_roots_are_filtered_by_compound_real_domains(domain, expected, monkeypatch):
    x = s.Symbol("elementary_domain")
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(s.cosh(x)-1, x, domain) == expected


def test_unknown_transcendental_order_retains_an_exact_intersection(monkeypatch):
    x, endpoint = s.symbols("elementary_order endpoint", real=True)
    source = s.cosh(x)-2
    domain = s.Interval(0, endpoint)
    expected = s.Intersection(domain, s.FiniteSet(-s.acosh(2), s.acosh(2)))
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    assert s.solveset(source, x, domain) == expected


def test_inferred_variable_and_assumed_or_dummy_coordinates(monkeypatch):
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    for x in (s.Symbol("elementary_typed", positive=True), s.Dummy("elementary_typed")):
        source = s.cosh(x)-2
        expected = s.FiniteSet(-s.acosh(2), s.acosh(2))
        assert s.solveset(source, x, s.S.Reals) == expected
        assert s.solveset(source, domain=s.S.Reals) == expected


def test_different_symbol_identity_never_becomes_the_solved_coordinate():
    from sympy.solvers._elementary_real import solve_elementary_real

    x = s.Symbol("elementary_same_name")
    other = s.Symbol("elementary_same_name", positive=True)
    assert solve_elementary_real(s.cosh(other)-2, x, s.S.Reals) is None


def test_complex_and_unsupported_families_keep_their_separate_routes():
    from sympy.solvers._elementary_real import solve_elementary_real

    x, parameter = s.symbols("elementary_other parameter")
    for source, domain in (
        (s.cosh(x)-2, s.S.Complexes),
        (s.cosh(x*x)-2, s.S.Reals),
        (s.cosh(x)+s.sinh(x)-2, s.S.Reals),
        (s.cosh(x)-parameter, s.S.Reals),
        (s.cosh(parameter*x)-2, s.S.Reals),
        (s.sin(x)-1, s.S.Reals),
    ):
        assert solve_elementary_real(source, x, domain) is None


@pytest.mark.parametrize("slot_owner, slot_name", [(Function, "_args"), (Basic, "_surface_args")])
def test_stale_function_storage_is_rejected(slot_owner, slot_name):
    x = s.Symbol("elementary_stale_"+slot_name)
    atom = s.cosh(x, evaluate=False)
    source = s.Add(atom, -2, evaluate=False)
    slot_owner.__dict__[slot_name].__set__(atom, (x+1,))
    with pytest.raises((ValueError, NotImplementedError), match="retained|surface"):
        s.solveset(source, x, s.S.Reals)


def test_same_name_custom_function_has_no_elementary_authority():
    x = s.Symbol("elementary_custom")
    impostor = s.Function("cosh")
    with pytest.raises(NotImplementedError):
        s.solveset(impostor(x)-2, x, s.S.Reals)


@pytest.mark.parametrize("kind", ["integer", "symbol", "native_wrapper"])
@pytest.mark.parametrize("slot_name", ["_struct_args", "_surface_args"])
def test_atomic_owners_cannot_hide_a_second_source_graph(kind, slot_name):
    x = s.Symbol("elementary_forged_atom_"+kind+slot_name)
    if kind == "integer":
        atom = s.Integer(17)
        source = s.Mul(atom, s.cosh(x)-1, evaluate=False)
    elif kind == "symbol":
        atom = x
        source = s.cosh(x, evaluate=False)-1
    else:
        atom = object.__new__(s.Expr)
        Basic.__dict__["_value"].__set__(atom, _native.py_integer(17))
        source = s.Mul(atom, s.cosh(x)-1, evaluate=False)
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


def test_source_mutation_during_inversion_is_terminal(monkeypatch):
    module = importlib.import_module("sympy.solvers._elementary_real")
    x = s.Symbol("elementary_mutation")
    atom = s.cosh(x, evaluate=False)
    source = s.Add(atom, -2, evaluate=False)
    original = module._inverse_values

    def mutate(*args):
        result = original(*args)
        Function.__dict__["_args"].__set__(atom, (x+1,))
        return result

    monkeypatch.setattr(module, "_inverse_values", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        s.solveset(source, x, s.S.Reals)


def test_source_work_charges_the_active_caller_allowance(monkeypatch):
    from sympy.core import supervised_callback_session

    x = s.Symbol("elementary_caller_budget")
    source = s.cosh(x)-2
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with supervised_callback_session(max_nested_steps=1):
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            s.solveset(source, x, s.S.Reals)


def test_algebra_budget_exhaustion_is_not_a_fallback(monkeypatch):
    module = importlib.import_module("sympy.solvers._elementary_real")
    x = s.Symbol("elementary_algebra_budget")
    source = s.cosh(x)-2
    monkeypatch.setattr(module, "_MAX_WORK", 1)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with pytest.raises(ValueError, match="algebra work bound"):
        s.solveset(source, x, s.S.Reals)


def test_original_domain_verification_failure_is_terminal(monkeypatch):
    checker = importlib.import_module("sympy.solvers._checksol")
    x = s.Symbol("elementary_domain_failure")
    source = s.Add(s.cosh(x)-1, s.Mul(0, s.Pow(x-2, -1, evaluate=False), evaluate=False),
                   evaluate=False)
    failure = ValueError("injected domain verifier exhaustion")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(checker, "_check_with_work", fail)
    monkeypatch.setattr(_native, "solve_native", _no_fallback)
    with pytest.raises(ValueError) as caught:
        s.solveset(source, x, s.S.Reals)
    assert caught.value is failure
