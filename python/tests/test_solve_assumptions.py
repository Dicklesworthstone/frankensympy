"""Scalar solve admission respects exact declared facts without guessing.

Expected roots and output conventions are pinned to isolated SymPy 1.14.0.
The candidate package runs these cases through its actual public solve path.
"""

import importlib

import pytest
import sympy as s


@pytest.mark.parametrize("facts,expected", [
    ({"positive": True}, [1]),
    ({"negative": True}, [-1]),
    ({"nonnegative": True}, [0, 1]),
    ({"nonpositive": True}, [-1, 0]),
    ({"nonzero": True}, [-1, 1]),
    ({"zero": False}, [-1, 1]),
    ({"integer": True}, [-1, 0, 1]),
    ({"even": True}, [0]),
    ({"odd": True}, [-1, 1]),
    ({"positive": False}, [-1, 0]),
    ({"negative": False}, [0, 1]),
    ({"nonzero": False}, [0]),
    ({"integer": False}, []),
])
def test_signed_integer_and_parity_declarations_filter_exact_polynomial_roots(facts, expected):
    x = s.Symbol("solve_declared_root", **facts)
    source = x*(x - 1)*(x + 1)
    assert s.solve(source, x) == expected
    assert s.solve(source, x, check=False) == [-1, 0, 1]


@pytest.mark.parametrize("facts,expected", [
    ({"positive": True}, lambda: [s.sqrt(2)]),
    ({"negative": True}, lambda: [-s.sqrt(2)]),
    ({"integer": True}, lambda: []),
    ({"rational": True}, lambda: []),
    ({"integer": False}, lambda: [-s.sqrt(2), s.sqrt(2)]),
    ({"real": True}, lambda: [-s.sqrt(2), s.sqrt(2)]),
    ({"real": False}, lambda: []),
])
def test_quadratic_radicals_use_exact_sign_and_integrality(facts, expected):
    x = s.Symbol("solve_radical_root", **facts)
    assert s.solve(x*x - 2, x) == expected()
    assert s.solve(x*x - 2, x, check=False) == [-s.sqrt(2), s.sqrt(2)]


@pytest.mark.parametrize("facts,expected", [
    ({"positive": True}, lambda: []),
    ({"negative": True}, lambda: []),
    ({"real": True}, lambda: []),
    ({"integer": True}, lambda: []),
    # SymPy nonzero denotes a nonzero real number, not arbitrary nonvanishing.
    ({"nonzero": True}, lambda: []),
    ({"complex": True}, lambda: [-s.I, s.I]),
    ({"real": False}, lambda: [-s.I, s.I]),
    ({"positive": False}, lambda: [-s.I, s.I]),
])
def test_nonreal_roots_do_not_satisfy_real_numeric_tower_predicates(facts, expected):
    x = s.Symbol("solve_complex_root", **facts)
    assert s.solve(x*x + 1, x) == expected()
    assert s.solve(x*x + 1, x, check=False) == [-s.I, s.I]


@pytest.mark.parametrize("factory", [s.Symbol, s.Dummy])
@pytest.mark.parametrize("mode", ["scalar", "sequence", "inferred"])
def test_scalar_sign_filter_keeps_output_shapes_and_caller_variable_identity(factory, mode):
    x = factory("solve_output_root", positive=True)
    source = x*x - 1
    args = {"scalar": (x,), "sequence": ([x],), "inferred": ()}[mode]
    assert s.solve(source, *args) == [1]
    mappings = s.solve(source, *args, dict=True)
    assert mappings == [{x: 1}]
    assert next(iter(mappings[0])) is x
    variables, roots = s.solve(source, *args, set=True)
    assert variables == [x] and variables[0] is x
    assert roots == {(s.Integer(1),)}
    assert s.solve(source, *args, dict=True, check=False) == [{x: -1}, {x: 1}]
    assert s.solve(source, *args, set=True, check=False) == ([x], {(-1,), (1,)})


def test_linear_roots_check_every_declared_constraint_and_float_rationalization():
    x = s.Symbol("solve_integral_positive", positive=True, integer=True)
    assert s.solve(2*x - 1, x) == []
    assert s.solve(2*x - 1, x, dict=True) == []
    assert s.solve(2*x - 1, x, set=True) == ([x], set())
    assert s.solve(2*x - 1, x, check=False) == [s.Rational(1, 2)]
    assert s.solve(x - s.Float("0.5"), x) == []
    assert s.solve(x - s.Float("0.5"), x, check=False) == [s.Float("0.5")]


def test_symbolic_unknowns_remain_roots_and_same_name_symbols_keep_their_facts():
    x = s.Symbol("solve_same_name", positive=True)
    unknown = s.Symbol("solve_same_name")
    negative = s.Symbol("solve_same_name", negative=True)
    assert s.solve(x - unknown, x) == [unknown]
    assert s.solve(x - unknown, x)[0] is unknown
    assert s.solve(x - negative, x) == []
    assert s.solve(x - negative, x, check=False) == [negative]
    y = s.Symbol("solve_unknown_parameter")
    assert s.solve(x*x - y, x) == [-s.sqrt(y), s.sqrt(y)]


def test_unknown_integrality_is_preserved_and_explicit_contradiction_is_removed():
    x = s.Symbol("solve_integer_variable", integer=True)
    unknown = s.Symbol("solve_integer_parameter")
    integral = s.Symbol("solve_integer_parameter", integer=True)
    nonintegral = s.Symbol("solve_integer_parameter", integer=False)
    assert s.solve(x - unknown, x) == [unknown]
    assert s.solve(x - integral, x) == [integral]
    assert s.solve(x - nonintegral, x) == []


def test_scalar_solve_does_not_import_global_context_into_declared_assumptions():
    x = s.Symbol("solve_global_context_variable", positive=True)
    y = s.Symbol("solve_global_context_parameter")
    with s.assuming(s.Q.negative(y)):
        assert s.solve(x - y, x) == [y]


def test_check_false_bypasses_only_the_new_assumption_filter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("check=False must not invoke root assumption admission")

    monkeypatch.setattr(s, "_filter_solve_assumptions", forbidden)
    x = s.Symbol("solve_unchecked_integer", integer=True)
    assert s.solve(2*x - 1, x, check=False) == [s.Rational(1, 2)]


def test_mutated_solved_symbol_declarations_are_not_silently_rebound(monkeypatch):
    x = s.Symbol("solve_mutated_variable", positive=True)
    source = x*x - 1
    native_solve = s._native.solve_native

    def changed_declaration(*args, **kwargs):
        roots = native_solve(*args, **kwargs)
        x._assumptions["positive"] = False
        return roots

    monkeypatch.setattr(s._native, "solve_native", changed_declaration)
    try:
        with pytest.raises((ValueError, NotImplementedError), match="assumptions|identity"):
            s.solve(source, x)
    finally:
        x._assumptions["positive"] = True


def test_assumption_budget_failure_propagates_without_retry_or_root_publication(monkeypatch):
    module = importlib.import_module("sympy.assumptions._query")
    failure = ValueError("assumptions query exceeds its work/depth limit")

    def exhausted(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.QueryProblem, "answer", exhausted)
    x = s.Symbol("solve_assumption_budget", positive=True)
    with pytest.raises(ValueError) as caught:
        s.solve(x*x - 1, x)
    assert caught.value is failure
