"""Only independent native strict-minimum receipts may finish minimum search."""

import importlib
import math
from types import SimpleNamespace

import pytest
import sympy as s


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_coupled_quadratic_returns_an_exact_source_bound_strict_minimum_receipt():
    x, y = s.symbols("certified_minimize_coupled_x certified_minimize_coupled_y")
    source = 2*(x-1)**2 + (x-1)*(y+2) + 3*(y+2)**2
    program = s.compile_objective(source, ((x,), y))
    receipt = program.minimize_certified([7, 9])
    assert type(receipt) is s.StationaryPointReceipt
    assert receipt.objective == source and receipt.kind == "strict_minimum"
    assert receipt.stationary_enclosure == ((1, 1), (-2, -2))
    assert receipt.hessian_enclosure == (((4, 4), (1, 1)), ((1, 1), (6, 6)))
    assert receipt.contraction_bound == 0
    assert all(a < lo <= hi < b
               for (a, b), (lo, hi) in zip(receipt.domain, receipt.stationary_enclosure))
    assert "declared closed box" in receipt.claim


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonconvex_descent_certifies_the_minimum_reached_on_either_side(sign):
    x = s.Symbol("certified_minimize_nonconvex")
    source = x**4-x*x
    program = s.compile_objective(source, x)
    receipt = program.minimize_certified(sign*0.1)
    assert receipt.kind == "strict_minimum" and receipt.objective == source
    lo, hi = receipt.stationary_enclosure[0]
    reference = sign*math.sqrt(0.5)
    assert float(lo) <= reference <= float(hi)
    assert sign*float(lo) > 0 and sign*float(hi) > 0
    assert receipt.hessian_enclosure[0][0][0] > 0
    assert 0 <= receipt.contraction_bound < 1


@pytest.mark.parametrize("kind", ["strict_maximum", "saddle"])
def test_stationary_maxima_and_saddles_never_publish_minimum_receipts(kind):
    x, y = s.symbols("certified_minimize_kind_x certified_minimize_kind_y")
    source = -x*x-y*y if kind == "strict_maximum" else x*x-y*y
    program = s.compile_objective(source, (x, y))
    assert program.minimize([0, 0], maxiter=0).evidence == "approximate"
    with pytest.raises(s.MinimizationSearchError) as failed:
        program.minimize_certified([0, 0], maxiter=0)
    assert failed.value.stage == "classification"
    assert failed.value.attempts == 1
    assert kind in failed.value.failures[-1]
    assert not hasattr(failed.value, "candidate") and not hasattr(failed.value, "receipt")


def test_unclassified_curvature_retries_within_one_attempt_cap_and_budget():
    x, y = s.symbols("certified_minimize_unclassified_x certified_minimize_unclassified_y")
    program = s.compile_objective(x*y, (x, y))
    budget = ample_budget()
    before = budget.remaining_steps
    with pytest.raises(s.MinimizationSearchError) as failed:
        program.minimize_certified([0, 0], maxiter=0, max_attempts=2, budget=budget)
    assert failed.value.stage == "classification"
    assert failed.value.attempts == 2
    assert all("unclassified" in failure for failure in failed.value.failures)
    assert budget.remaining_steps < before


def test_singular_minimum_is_an_inconclusive_search_not_a_false_certificate():
    x = s.Symbol("certified_minimize_singular")
    program = s.compile_objective(x**4, x)
    with pytest.raises(s.MinimizationSearchError) as failed:
        program.minimize_certified(0, maxiter=0, max_attempts=2)
    assert failed.value.stage == "certification" and failed.value.attempts == 2
    assert not hasattr(failed.value, "receipt")


@pytest.mark.parametrize("point", [(10.0,), (float("nan"),), ()])
def test_forged_numerical_success_metadata_cannot_bypass_exact_source_acceptance(monkeypatch, point):
    x = s.Symbol("certified_minimize_forged_candidate")
    program = s.compile_objective((x-1)**2, x)
    forged = SimpleNamespace(
        point=point, value=0.0, gradient=(0.0,), gradient_norm=0.0,
        evidence="certified", kind="strict_minimum",
    )
    monkeypatch.setattr(type(program), "minimize", lambda *args, **kwargs: forged)
    with pytest.raises(s.MinimizationSearchError):
        program.minimize_certified(0, max_attempts=1)


def test_mutable_shell_classification_cannot_upgrade_native_maximum_evidence(monkeypatch):
    # Importing the helper captures the native classification descriptor before
    # presentation properties can be changed. Native evidence remains decisive.
    importlib.import_module("sympy.utilities._objective_minimization")
    x = s.Symbol("certified_minimize_native_kind")
    program = s.compile_objective(-x*x, x)
    monkeypatch.setattr(s.StationaryPointReceipt, "kind", property(lambda self: "strict_minimum"))
    with pytest.raises(s.MinimizationSearchError) as failed:
        program.minimize_certified(0, maxiter=0)
    assert failed.value.stage == "classification"
    assert "strict_maximum" in failed.value.failures[-1]


@pytest.mark.parametrize("case", [
    "float_source", "integer_assumption", "precision_low", "precision_high",
    "attempts_zero", "attempts_high", "bad_gtol", "bad_maxiter", "bad_step",
])
def test_static_certification_eligibility_precedes_initial_conversion_and_work(case):
    x = s.Symbol("certified_minimize_admission", integer=True) if case == "integer_assumption" else s.Symbol("certified_minimize_admission")
    source = x*x + s.Float("0.5") if case == "float_source" else (x-1)**2
    program = s.compile_objective(source, x)
    options = {
        "precision_low": {"precision": 7},
        "precision_high": {"precision": 129},
        "attempts_zero": {"max_attempts": 0},
        "attempts_high": {"max_attempts": 25},
        "bad_gtol": {"gtol": -1},
        "bad_maxiter": {"maxiter": 1025},
        "bad_step": {"max_step": 0},
    }.get(case, {})
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("static ineligibility must not convert initial values")

    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    expected = NotImplementedError if case == "integer_assumption" else ValueError
    with pytest.raises(expected):
        program.minimize_certified(Initial(), budget=budget, **options)
    assert not effects and budget.remaining_steps == before


def test_captured_positive_assumption_cannot_be_mutated_into_a_negative_minimum(monkeypatch):
    x = s.Symbol("certified_minimize_captured_assumption", positive=True)
    facts = dict(x._assumptions)
    program = s.compile_objective((x+1)**2, x)
    original = type(program).minimize

    def mutate_after_candidate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(type(program), "minimize", mutate_after_candidate)
    try:
        with pytest.raises(s.MinimizationSearchError) as failed:
            program.minimize_certified(-1, maxiter=0, max_attempts=2)
        assert failed.value.stage == "certification"
        assert failed.value.attempts == 2
        assert all("positive" in failure for failure in failed.value.failures)
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_numerical_failure_is_distinguished_from_certificate_failure():
    x = s.Symbol("certified_minimize_candidate_failure")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(s.MinimizationSearchError) as failed:
        program.minimize_certified(0, maxiter=0)
    assert failed.value.stage == "candidate" and failed.value.attempts == 0


def test_one_budget_covers_candidate_and_independent_certification_without_reset():
    x = s.Symbol("certified_minimize_budget")
    program = s.compile_objective((x-1)**2, x)
    numerical = ample_budget()
    before = numerical.remaining_steps
    program.minimize(0, budget=numerical)
    numerical_cost = before-numerical.remaining_steps
    measured = ample_budget()
    before = measured.remaining_steps
    receipt = program.minimize_certified(0, budget=measured)
    cost = before-measured.remaining_steps
    assert receipt.kind == "strict_minimum" and cost > numerical_cost > 0
    exact = ample_budget(max_steps=cost)
    assert program.minimize_certified(0, budget=exact).kind == "strict_minimum"
    assert exact.remaining_steps == 0
    short = ample_budget(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.minimize_certified(0, budget=short)
    assert short.remaining_steps < cost-1
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.minimize_certified(0, budget=cancelled)


def test_existing_stationary_search_still_publishes_nonminimum_stationary_evidence():
    x = s.Symbol("certified_minimize_stationary_compatibility")
    receipt = s.compile_objective(-x*x, x).solve_stationary_certified(2)
    assert receipt.kind == "strict_maximum"


def test_minimization_public_exports_share_the_same_types():
    from sympy.utilities import MinimizationSearchError, NumericMinimizationCandidate
    assert MinimizationSearchError is s.MinimizationSearchError
    assert NumericMinimizationCandidate is s.NumericMinimizationCandidate
    assert issubclass(MinimizationSearchError, s.StationarySearchError)
