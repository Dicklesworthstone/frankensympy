"""System solve admits complete tuples against the requested declarations.

Values and public return shapes are pinned to isolated SymPy 1.14.0. These
tests run the candidate's real linear, polynomial and scalar fallback paths;
the oracle is never imported into the candidate process.
"""

import importlib

import pytest
import sympy as s


@pytest.mark.parametrize("facts,coefficient,constant,unchecked", [
    ({"positive": True}, 1, 1, -1),
    ({"negative": True}, 1, -2, 2),
    ({"positive": False}, 1, -2, 2),
    ({"integer": True}, 2, -1, lambda: s.Rational(1, 2)),
    ({"even": True}, 1, -1, 1),
    ({"odd": True}, 1, -2, 2),
])
@pytest.mark.parametrize("inferred", [False, True])
def test_linear_system_rejects_a_definite_coordinate_contradiction(
        facts, coefficient, constant, unchecked, inferred):
    x = s.Symbol("system_declared_coordinate", **facts)
    expected = unchecked() if callable(unchecked) else s.Integer(unchecked)
    source = [coefficient*x + constant]
    variables = () if inferred else (x,)
    assert s.solve(source, *variables) == []
    assert s.solve(source, *variables, dict=True) == []
    returned, roots = s.solve(source, *variables, set=True)
    assert returned == [x] and returned[0] is x and roots == set()
    assert s.solve(source, *variables, check=False) == {x: expected}
    assert s.solve(source, *variables, check=False, dict=True) == [{x: expected}]
    assert s.solve(source, *variables, check=False, set=True) == ([x], {(expected,)})


def test_linear_system_checks_every_coordinate_before_returning_any_mapping():
    x = s.Symbol("system_positive_coordinate", positive=True)
    y = s.Symbol("system_integer_coordinate", integer=True)
    source = [x-1, 2*y-1]
    assert s.solve(source, x, y) == []
    assert s.solve(source, x, y, dict=True) == []
    assert s.solve(source, x, y, set=True) == ([x, y], set())
    assert s.solve(source, x, y, check=False) == {x: 1, y: s.Rational(1, 2)}


def test_polynomial_system_filters_whole_rows_in_every_public_shape():
    x = s.Symbol("system_positive_polynomial", positive=True)
    y = s.Symbol("system_negative_polynomial", negative=True)
    source = [x*x-1, y*y-1]
    assert s.solve(source, x, y) == [(1, -1)]
    assert s.solve(source, x, y, dict=True) == [{x: 1, y: -1}]
    assert s.solve(source, x, y, set=True) == ([x, y], {(1, -1)})
    assert set(s.solve(source, x, y, check=False)) == {
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    }


@pytest.mark.parametrize("facts,constant,expected", [
    ({"real": True}, 1, []),
    ({"real": False}, 1, lambda: [(-s.I, -s.I), (s.I, s.I)]),
    ({"nonzero": True}, 1, []),
    ({"rational": True}, -2, []),
    ({"integer": False}, -2, lambda: [(-s.sqrt(2), -s.sqrt(2)), (s.sqrt(2), s.sqrt(2))]),
])
def test_finite_system_sign_domain_and_integrality_are_exact(facts, constant, expected):
    x = s.Symbol("system_exact_root", **facts)
    y = s.Symbol("system_matching_root")
    source = [x*x+constant, y-x]
    result = expected() if callable(expected) else expected
    assert s.solve(source, x, y) == result
    assert len(s.solve(source, x, y, check=False)) == 2


@pytest.mark.parametrize("factory", [s.Symbol, s.Dummy])
def test_same_name_coordinates_preserve_declarations_order_and_exact_owners(factory):
    positive = factory("system_same_name", positive=True)
    negative = factory("system_same_name", negative=True)
    source = [positive*positive-1, negative*negative-1]
    result = s.solve(source, [negative, positive], dict=True)
    assert result == [{negative: -1, positive: 1}]
    keys = list(result[0])
    assert next(key for key in keys if key == negative) is negative
    assert next(key for key in keys if key == positive) is positive
    variables, roots = s.solve(source, [negative, positive], set=True)
    assert variables[0] is negative and variables[1] is positive
    assert roots == {(-1, 1)}


@pytest.mark.parametrize("container", [lambda value: value, lambda value: [value]])
def test_free_coordinate_assumptions_can_disprove_a_linear_family(container):
    x = s.Symbol("system_positive_dependent", positive=True)
    y = s.Symbol("system_positive_free", positive=True)
    source = container(x+y)
    assert s.solve(source, x, y) == []
    assert s.solve(source, x, y, dict=True) == []
    assert s.solve(source, x, y, set=True) == ([x, y], set())
    assert s.solve(source, x, y, check=False, dict=True) == [{x: -y}]
    assert s.solve(source, x, y, check=False, set=True) == ([x, y], {(-y, y)})


def test_undecided_free_parameters_are_retained_without_importing_global_context():
    x = s.Symbol("system_unknown_dependent", positive=True, integer=True)
    y = s.Symbol("system_unknown_free")
    assert s.solve([x-y], x, y) == {x: y}
    assert s.solve([x-y], x, y, dict=True) == [{x: y}]
    variables, roots = s.solve([x-y], x, y, set=True)
    assert variables[0] is x and variables[1] is y
    row = next(iter(roots))
    assert row[0] is y and row[1] is y
    with s.assuming(s.Q.negative(y)):
        assert s.solve([x-y], x, y) == {x: y}


def test_float_rationalization_still_applies_exact_assumption_admission():
    x = s.Symbol("system_float_integral", integer=True)
    y = s.Symbol("system_float_other")
    source = [x-s.Float("0.5"), y-s.Float("2.0")]
    assert s.solve(source, x, y) == []
    assert s.solve(source, x, y, check=False) == {
        x: s.Float("0.5"), y: s.Float("2.0"),
    }


@pytest.mark.parametrize("equation", [False, True])
def test_multivariable_scalar_fallback_keeps_check_false_and_partial_mapping(equation):
    x = s.Symbol("system_fallback_positive", positive=True)
    y = s.Symbol("system_fallback_free", negative=True)
    source = s.Eq(x*x, 1, evaluate=False) if equation else x*x-1
    assert s.solve(source, x, y) == [(1, y)]
    assert s.solve(source, x, y, dict=True) == [{x: 1}]
    assert s.solve(source, x, y, set=True) == ([x, y], {(1, y)})
    assert s.solve(source, x, y, check=False) == [(-1, y), (1, y)]
    assert s.solve(source, x, y, check=False, dict=True) == [{x: -1}, {x: 1}]
    assert s.solve(source, x, y, check=False, set=True) == ([x, y], {(-1, y), (1, y)})


@pytest.mark.parametrize("path", ["linear", "polynomial", "scalar_fallback"])
def test_check_false_does_not_invoke_system_or_scalar_admission(path, monkeypatch):
    x = s.Symbol("system_unchecked_positive", positive=True)
    y = s.Symbol("system_unchecked_other")

    def forbidden(*args, **kwargs):
        raise AssertionError("check=False must not invoke declaration admission")

    monkeypatch.setattr(s, "_filter_solve_system_assumptions", forbidden)
    monkeypatch.setattr(s, "_filter_solve_assumptions", forbidden)
    source = {"linear": [x+1, y-2], "polynomial": [x*x-1, y-2],
              "scalar_fallback": x*x-1}[path]
    result = s.solve(source, x, y, check=False, dict=True)
    assert any(row.get(x) == -1 for row in result)


@pytest.mark.parametrize("path,empty", [
    ("linear", False), ("linear", True),
    ("polynomial", False), ("polynomial", True),
])
def test_solved_coordinate_mutation_is_detected_even_after_empty_generation(path, empty, monkeypatch):
    module = importlib.import_module("sympy.solvers." + ("solvers" if path == "linear" else "polysys"))
    name = "linsolve" if path == "linear" else "solve_poly_system"
    original = getattr(module, name)
    x = s.Symbol("system_mutated_declared", positive=True)
    y = s.Symbol("system_mutated_other")
    source = ([x-1, x-2] if empty else [x-1, y-2]) if path == "linear" else (
        [x*x-1, x*x-2] if empty else [x*x-1, y-2]
    )

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        x._assumptions["positive"] = False
        return result

    monkeypatch.setattr(module, name, mutate)
    try:
        with pytest.raises((ValueError, NotImplementedError), match="assumptions|identity"):
            s.solve(source, x, y)
    finally:
        x._assumptions["positive"] = True


@pytest.mark.parametrize("scalar", [False, True])
def test_admission_failure_never_retries_as_a_polynomial_system(scalar, monkeypatch):
    query = importlib.import_module("sympy.assumptions._query")
    polysys = importlib.import_module("sympy.solvers.polysys")
    x = s.Symbol("system_admission_failure", positive=True)
    y = s.Symbol("system_admission_failure_free")
    failure = ValueError("assumptions query exceeds its work/depth limit")

    def exhausted(*args, **kwargs):
        raise failure

    def forbidden(*args, **kwargs):
        raise AssertionError("an admitted linear result must not restart another solver")

    monkeypatch.setattr(query.QueryProblem, "answer", exhausted)
    monkeypatch.setattr(polysys, "solve_poly_system", forbidden)
    source = x-y if scalar else [x-y]
    with pytest.raises(ValueError) as caught:
        s.solve(source, x, y)
    assert caught.value is failure


def test_one_cumulative_assumption_budget_covers_all_system_coordinates(monkeypatch):
    query = importlib.import_module("sympy.assumptions._query")
    x = s.Symbol("system_budget_positive", positive=True)
    y = s.Symbol("system_budget_negative", negative=True)
    original = query.QueryProblem.answer
    calls = {}
    failure = ValueError("bounded whole-system assumption ledger exhausted")

    def bounded(problem, *args, **kwargs):
        calls[problem] = calls.get(problem, 0) + 1
        if calls[problem] > 1:
            raise failure
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(query.QueryProblem, "answer", bounded)
    with pytest.raises(ValueError) as caught:
        s.solve([x-1, y+1], x, y)
    assert caught.value is failure


def test_post_admission_rechecks_coordinates_skipped_by_a_rejected_row(monkeypatch):
    query = importlib.import_module("sympy.assumptions._query")
    x = s.Symbol("system_rejected_positive", positive=True)
    y = s.Symbol("system_rejected_negative", negative=True)
    original = query.QueryProblem.answer

    def mutate_other(problem, *args, **kwargs):
        result = original(problem, *args, **kwargs)
        y._assumptions["negative"] = False
        return result

    monkeypatch.setattr(query.QueryProblem, "answer", mutate_other)
    try:
        with pytest.raises((ValueError, NotImplementedError), match="assumptions|identity"):
            s.solve([x+1, y+1], x, y)
    finally:
        y._assumptions["negative"] = True


def test_multivariable_scalar_dispatch_retains_the_original_equality_owner(monkeypatch):
    module = importlib.import_module("sympy.solvers.solvers")
    x = s.Symbol("system_retained_positive", positive=True)
    y = s.Symbol("system_retained_free")
    source = s.Eq(x+y, 1, evaluate=False)
    original = module.linsolve
    observed = []

    def retained(equations, *variables):
        observed.append(equations[0])
        return original(equations, *variables)

    monkeypatch.setattr(module, "linsolve", retained)
    assert s.solve(source, x, y, dict=True) == [{x: 1-y}]
    assert observed == [source] and observed[0] is source
