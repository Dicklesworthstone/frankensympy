"""Root search candidates acquire evidence only from the independent checker."""

import importlib
import math

import pytest

from sympy import (
    Add, Dummy, Eq, Float, Function, Matrix, Mul, Pow, Rational, Symbol,
    compile_residual_system, exp, log, nsolve, sin, sqrt,
)
from sympy.core import _symbol_binding
from sympy.solvers.numerical import RootSearchError, nsolve_certified
from sympy.utilities import RootBoxReceipt


def test_scalar_and_equality_search_return_exact_checked_root_enclosures():
    x = Symbol("automatic_scalar_x")
    for expression in (x*x-2, Eq(x*x, 2)):
        receipt = nsolve_certified(expression, x, 1)
        assert type(receipt) is RootBoxReceipt
        lo, hi = receipt.root_enclosure[0]
        assert 0 < lo <= hi
        assert lo*lo <= 2 <= hi*hi
        assert 0 <= receipt.contraction_bound < 1
        assert receipt.variables[0].name == x.name
        assert "unique nonsingular real root" in receipt.claim
    assert type(nsolve(x*x-2, x, 1)) is Float


@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_supported_square_dimensions_certify_known_exact_affine_roots(dimension):
    variables = [Symbol(f"automatic_affine_{dimension}_{i}") for i in range(dimension)]
    receipt = nsolve_certified([x-i for i, x in enumerate(variables)], variables, [0]*dimension)
    assert receipt.root_enclosure == tuple((i, i) for i in range(dimension))
    assert receipt.contraction_bound == 0


def test_coupled_polynomial_system_has_a_local_receipt_for_every_original_row():
    x, y = Symbol("automatic_nonlinear_x"), Symbol("automatic_nonlinear_y")
    residuals = [3*x*x-2*y*y-1, x*x-2*x+y*y+2*y-8]
    receipt = nsolve_certified(residuals, (x, y), (-1, 1))
    expected = (-1.1928730993524608, 1.278444111699106)
    for (lo, hi), reference in zip(receipt.root_enclosure, expected):
        assert float(lo) <= reference <= float(hi)
    enclosed = compile_residual_system(residuals, [x, y]).enclose(receipt.root_enclosure)
    assert all(lo <= 0 <= hi for lo, hi in enclosed.residual_values)
    assert receipt.contraction_bound < 1


def test_transcendental_system_reaches_the_independent_original_source_checker():
    x, y = Symbol("automatic_transcendental_x"), Symbol("automatic_transcendental_y")
    receipt = nsolve_certified((exp(x)-2, y-sin(x)), (x, y), (0, 0))
    expected = (math.log(2), math.sin(math.log(2)))
    for (lo, hi), reference in zip(receipt.root_enclosure, expected):
        assert float(lo) <= reference <= float(hi)
    assert all(a < lo <= hi < b for (a, b), (lo, hi) in zip(receipt.domain, receipt.root_enclosure))


def test_domain_crossing_newton_trials_backtrack_before_certification():
    x = Symbol("automatic_backtracking_x")
    receipt = nsolve_certified(log(x), x, 10)
    lo, hi = receipt.root_enclosure[0]
    assert lo <= 1 <= hi
    assert receipt.domain[0][0] > 0
    receipt = nsolve_certified(sqrt(x)-2, x, 1)
    lo, hi = receipt.root_enclosure[0]
    assert lo <= 4 <= hi


def test_compiled_search_reuses_nested_layout_and_immutable_matrix_snapshot():
    x, y = Symbol("automatic_snapshot_x"), Symbol("automatic_snapshot_y")
    source = Matrix([x+y-3, x-y+1])
    program = compile_residual_system(source, [(x, y)])
    source[0, 0] = x+y-300
    receipt = program.solve_certified([0, 0])
    assert receipt.root_enclosure == ((1, 1), (2, 2))
    assert receipt.residuals == (x+y-3, x-y+1)


def test_assumed_symbols_and_dummies_keep_distinct_typed_coordinates():
    x = Symbol("automatic_same")
    positive = Symbol("automatic_same", positive=True)
    a, b = Dummy("same"), Dummy("same")
    variables = [x, positive, a, b]
    receipt = nsolve_certified([v-i-1 for i, v in enumerate(variables)], variables, [0]*4)
    assert receipt.root_enclosure == ((1, 1), (2, 2), (3, 3), (4, 4))
    assert receipt.variables[0].plain and not receipt.variables[1].plain
    assert receipt.variables[2].name != receipt.variables[3].name


def test_captured_assumptions_cannot_be_relabelled_between_search_and_check(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("automatic_assumption_snapshot", positive=True)
    facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    program = compile_residual_system([x-1], [x])
    iterate = numerical._iterate_system

    def mutate_after_candidate(*args):
        candidate = iterate(*args)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return candidate

    monkeypatch.setattr(numerical, "_iterate_system", mutate_after_candidate)
    try:
        receipt = program.solve_certified(0)
        assert receipt.root_enclosure == ((1, 1),)
        assert receipt.variables[0].identity_hex == binding.identity_hex
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_a_candidate_outside_its_declared_assumptions_never_becomes_accepted():
    x = Symbol("automatic_positive_x", positive=True)
    with pytest.raises(RootSearchError) as failed:
        nsolve_certified(x*x-2, x, -1, max_attempts=3)
    assert failed.value.stage == "certification"
    assert failed.value.attempts == 3
    assert all("positive" in message for message in failed.value.failures)


@pytest.mark.parametrize("case", ["multiple_root", "dependent_system", "no_root"])
def test_singular_or_nonexistent_roots_never_receive_nonsingular_receipts(case):
    x, y = Symbol("automatic_invalid_x"), Symbol("automatic_invalid_y")
    with pytest.raises(RootSearchError):
        if case == "multiple_root":
            nsolve_certified(x*x, x, 0, max_attempts=2)
        elif case == "dependent_system":
            nsolve_certified((x+y, 2*x+2*y), (x, y), (0, 0), max_attempts=2)
        else:
            nsolve_certified(x*x+1, x, 1, max_attempts=2)


def test_rounded_zero_residual_is_only_a_candidate():
    x = Symbol("automatic_rounded_zero_x")
    # At binary64 x=1e20, (x+1)-x rounds to zero. The original expression is
    # identically one, so no independently accepted root can exist.
    expression = Add(Add(x, 1, evaluate=False), Mul(-1, x, evaluate=False), evaluate=False)
    with pytest.raises(RootSearchError) as failed:
        nsolve_certified(expression, x, 1e20, max_attempts=2)
    assert failed.value.stage == "certification"


def test_held_singularities_are_checked_before_any_root_claim():
    x = Symbol("automatic_held_domain_x")
    hidden = Add(x, Mul(0, Pow(x, -1, evaluate=False), evaluate=False), evaluate=False)
    with pytest.raises((RootSearchError, ValueError)):
        nsolve_certified(hidden, x, 0, max_attempts=2)


@pytest.mark.parametrize("candidate", [[math.nan], [math.inf], [0], [0.0, 1.0], [999.0]])
def test_forged_candidate_outputs_are_revalidated_or_rejected(monkeypatch, candidate):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("automatic_forged_x")
    monkeypatch.setattr(numerical, "_iterate_system", lambda *args: candidate)
    with pytest.raises(RootSearchError):
        nsolve_certified(x*x-2, x, 1, max_attempts=3)


def test_imperfect_candidate_can_be_recovered_only_after_actual_native_acceptance(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = Symbol("automatic_imperfect_x")
    monkeypatch.setattr(numerical, "_iterate_system", lambda *args: [math.log(2)+0.0001])
    checked, precisions = certified.certify_root_box, []

    def observe(program, box, center, preconditioner, precision):
        precisions.append(precision)
        return checked(program, box, center, preconditioner, precision)

    monkeypatch.setattr(certified, "certify_root_box", observe)
    receipt = nsolve_certified(exp(x)-2, x, 0, precision=8, max_attempts=8)
    lo, hi = receipt.root_enclosure[0]
    assert float(lo) <= math.log(2) <= float(hi)
    assert len(precisions) > 1
    assert precisions[0] == 8 and max(precisions) > 8
    assert receipt.precision_digits == precisions[-1]


def test_every_rejected_native_trial_consumes_the_same_outer_attempt_budget(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = Symbol("automatic_attempt_limit_x")
    monkeypatch.setattr(numerical, "_iterate_system", lambda *args: [1.0])
    checked, calls = certified.certify_root_box, []

    def observe(*args):
        calls.append(args[-1])
        return checked(*args)

    monkeypatch.setattr(certified, "certify_root_box", observe)
    with pytest.raises(RootSearchError) as failed:
        nsolve_certified(x*x+1, x, 1, max_attempts=5, precision=8)
    assert calls == [8, 8, 16, 16, 32]
    assert failed.value.stage == "certification"
    assert failed.value.attempts == 5
    assert len(failed.value.failures) == 5


@pytest.mark.parametrize("option,value", [
    ("precision", 7), ("precision", 129), ("precision", True),
    ("max_attempts", 0), ("max_attempts", 25), ("max_attempts", True),
    ("maxiter", 0), ("maxiter", 10_001), ("tol", 0),
])
def test_invalid_search_controls_refuse_before_compilation(monkeypatch, option, value):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("automatic_bad_controls_x")

    def forbidden(*args):
        raise AssertionError("invalid controls must refuse before compilation")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    with pytest.raises((ValueError, TypeError)):
        nsolve_certified(x-1, x, 0, **{option: value})


def test_non_square_and_oversized_systems_refuse_before_compilation(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x, y = Symbol("automatic_shape_x"), Symbol("automatic_shape_y")

    def forbidden(*args):
        raise AssertionError("invalid dimensions must refuse before compilation")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    for expressions, variables in (([x, y], [x]), ([x], [x, y]), ([x]*17, [x]*17), ([], [])):
        with pytest.raises(ValueError):
            nsolve_certified(expressions, variables, [0])


def test_inexact_source_and_unsupported_assumptions_refuse_before_newton(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("automatic_inexact_x")
    integer = Symbol("automatic_integer_x", integer=True)

    def forbidden(*args):
        raise AssertionError("ineligible certificate sources must refuse before Newton work")

    monkeypatch.setattr(numerical, "_iterate_system", forbidden)
    with pytest.raises(ValueError, match="exact source"):
        nsolve_certified(Add(x, Float(0.5), evaluate=False), x, 0)
    with pytest.raises(NotImplementedError, match="assumptions"):
        nsolve_certified(integer-1, integer, 0)


def test_numerical_work_and_iteration_limits_do_not_turn_into_certificates(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = Symbol("automatic_candidate_budget_x")
    with pytest.raises(RootSearchError, match="failed to converge") as failed:
        nsolve_certified(x*x-2, x, 1, maxiter=1)
    assert failed.value.stage == "candidate"
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 1)
    with pytest.raises(RootSearchError, match="evaluation budget"):
        nsolve_certified(x*x-2, x, 1)
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 4096)
    monkeypatch.setattr(numerical, "_MAX_LINEAR_WORK", 0)
    with pytest.raises(RootSearchError, match="linear-algebra work budget"):
        nsolve_certified(x*x-2, x, 1)


def test_custom_hooks_are_never_executed_by_search_or_certification():
    x = Symbol("automatic_hook_x")
    calls = []

    class UserFunction(Function):
        def _eval_derivative(self, variable):
            calls.append("derivative")
            return 1

    class UserNumber:
        def __float__(self):
            calls.append("float")
            return 0.0

    with pytest.raises((TypeError, NotImplementedError)):
        nsolve_certified(UserFunction(x, evaluate=False), x, 0)
    with pytest.raises(TypeError):
        nsolve_certified(x-1, x, UserNumber())
    assert calls == []
