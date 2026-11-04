"""Automatic objective search publishes only independently accepted receipts."""

import builtins
import importlib
import math

import pytest
import sympy as s

from sympy.core import _symbol_binding


def search_module():
    return importlib.import_module("sympy.utilities._stationary_search")


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_coupled_quadratic_search_certifies_exact_minimum_and_original_source():
    x, y = s.symbols("stationary_coupled_x stationary_coupled_y")
    source = 2*(x-1)**2 + (x-1)*(y+2) + 3*(y+2)**2
    program = s.compile_objective(source, ((x,), y))
    receipt = program.solve_stationary_certified([7, 9])
    assert type(receipt) is s.StationaryPointReceipt
    assert receipt.objective == source
    assert receipt.kind == "strict_minimum"
    assert receipt.stationary_enclosure == ((1, 1), (-2, -2))
    assert receipt.hessian_enclosure == (((4, 4), (1, 1)), ((1, 1), (6, 6)))
    assert receipt.contraction_bound == 0
    assert all(a < lo <= hi < b
               for (a, b), (lo, hi) in zip(receipt.domain, receipt.stationary_enclosure))
    assert "declared closed box" in receipt.claim


@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_supported_dimensions_certify_known_stationary_coordinates(dimension):
    variables = tuple(s.Symbol(f"stationary_dimension_{dimension}_{i}")
                      for i in range(dimension))
    # Keep the full 16-coordinate fixture within the checker's intrinsic
    # work envelope while exercising a real, uncoupled dense solve.
    source = s.Add(*(s.Pow(s.Add(variable, -(i+1), evaluate=False), 2, evaluate=False)
                     for i, variable in enumerate(variables)), evaluate=False)
    receipt = s.compile_objective(source, variables).solve_stationary_certified([0]*dimension)
    assert receipt.kind == "strict_minimum"
    assert receipt.stationary_enclosure == tuple((i+1, i+1) for i in range(dimension))
    assert receipt.contraction_bound == 0


@pytest.mark.parametrize("kind", ["strict_maximum", "saddle", "unclassified"])
def test_stationary_search_retains_local_curvature_scope(kind):
    x, y = s.symbols("stationary_curvature_x stationary_curvature_y")
    source = {
        "strict_maximum": -x*x-2*y*y,
        "saddle": x*x-y*y,
        "unclassified": x*y,
    }[kind]
    receipt = s.compile_objective(source, (x, y)).solve_stationary_certified([3, -2])
    assert receipt.kind == kind
    assert receipt.stationary_enclosure == ((0, 0), (0, 0))
    assert receipt.contraction_bound == 0


def test_coupled_nonlinear_minimum_is_checked_against_the_exact_source():
    x, y = s.symbols("stationary_nonlinear_x stationary_nonlinear_y")
    source = (x*x-2)**2 + (y-x)**2
    program = s.compile_objective(source, (x, y))
    receipt = program.solve_stationary_certified([1.5, 1])
    assert receipt.kind == "strict_minimum"
    for lo, hi in receipt.stationary_enclosure:
        assert lo > 0 and lo*lo <= 2 <= hi*hi
    enclosed = program.enclose(receipt.stationary_enclosure)
    assert all(lo <= 0 <= hi for lo, hi in enclosed.gradient)
    assert 0 <= receipt.contraction_bound < 1
    assert receipt.objective == source


@pytest.mark.parametrize("case", ["exponential", "positive_dynamic_power"])
def test_transcendental_stationary_search_uses_real_original_source_bounds(case):
    x = s.Symbol("stationary_transcendental_" + case)
    source, initial, reference = (
        (s.exp(x)-2*x, 0, math.log(2)) if case == "exponential"
        else (x**x, 0.5, math.exp(-1))
    )
    receipt = s.compile_objective(source, x).solve_stationary_certified(initial)
    lo, hi = receipt.stationary_enclosure[0]
    assert float(lo) <= reference <= float(hi)
    assert receipt.kind == "strict_minimum"
    assert receipt.hessian_enclosure[0][0][0] > 0
    assert receipt.objective == source


def test_domain_crossing_newton_trials_backtrack_on_the_original_objective():
    x = s.Symbol("stationary_log_x")
    # Gradient log(x), Hessian 1/x. The first full Newton step from 10 is
    # negative, so success requires rejecting that source-domain trial.
    receipt = s.compile_objective(x*s.log(x)-x, x).solve_stationary_certified(10)
    lo, hi = receipt.stationary_enclosure[0]
    assert lo <= 1 <= hi
    assert receipt.domain[0][0] > 0
    assert receipt.kind == "strict_minimum"


def test_tiny_exact_stationary_displacement_is_not_snapped_to_the_candidate():
    x = s.Symbol("stationary_tiny_x")
    coefficient = s.Rational(1, 10**100)
    source = x*x + coefficient*x
    receipt = s.compile_objective(source, x).solve_stationary_certified(0)
    # The default floating residual/step tolerance admits the initial zero.
    # Independent exact checking still returns the nonzero stationary point.
    expected = -s.Rational(1, 2*10**100)
    assert receipt.center == (0,)
    assert receipt.stationary_enclosure == ((expected, expected),)
    assert receipt.kind == "strict_minimum"


def test_local_search_does_not_claim_other_stationary_points_are_absent():
    x = s.Symbol("stationary_multiple_x")
    program = s.compile_objective((x*x-1)**2, x)
    negative = program.solve_stationary_certified(-1.5)
    positive = program.solve_stationary_certified(1.5)
    for receipt, point in ((negative, -1), (positive, 1)):
        lo, hi = receipt.stationary_enclosure[0]
        assert lo <= point <= hi
        assert receipt.kind == "strict_minimum"
    assert negative.domain[0][1] < 0 < positive.domain[0][0]


def test_checked_program_is_reused_without_symbolic_recompilation(monkeypatch):
    x = s.Symbol("stationary_captured_program_x")
    program = s.compile_objective((x-2)**2, x)

    def forbidden(*args, **kwargs):
        raise AssertionError("stationary iteration must reuse the immutable objective")

    core = importlib.import_module("sympy.core")
    numerical = importlib.import_module("sympy.solvers.numerical")
    monkeypatch.setattr(core, "diff", forbidden)
    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert program.solve_stationary_certified(9).stationary_enclosure == ((2, 2),)


def test_typed_coordinates_and_nested_argument_layout_remain_bound():
    plain = s.Symbol("stationary_same")
    positive = s.Symbol("stationary_same", positive=True)
    left, right = s.Dummy("stationary_same"), s.Dummy("stationary_same")
    variables = (plain, positive, left, right)
    source = sum((value-i-1)**2 for i, value in enumerate(variables))
    program = s.compile_objective(source, ((plain, positive), (left, right)))
    receipt = program.solve_stationary_certified([0, 1, 0, 0])
    assert receipt.stationary_enclosure == ((1, 1), (2, 2), (3, 3), (4, 4))
    assert receipt.variables[0].plain and not receipt.variables[1].plain
    assert receipt.variables[2].name != receipt.variables[3].name


def test_mutated_shell_assumptions_cannot_relabel_captured_search_identity(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_assumption_snapshot_x", positive=True)
    facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    program = s.compile_objective((x-1)**2, x)
    original = module._iterate_stationary

    def mutate_after_candidate(*args, **kwargs):
        result = original(*args, **kwargs)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(module, "_iterate_stationary", mutate_after_candidate)
    try:
        receipt = program.solve_stationary_certified(2)
        assert receipt.stationary_enclosure == ((1, 1),)
        assert receipt.variables[0].identity_hex == binding.identity_hex
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_candidate_outside_captured_assumptions_has_only_failed_trials(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_positive_x", positive=True)
    program = s.compile_objective((x+1)**2, x)
    monkeypatch.setattr(module, "_iterate_stationary", lambda *args: [-1.0])
    with pytest.raises(s.StationarySearchError) as failed:
        program.solve_stationary_certified(1, max_attempts=3)
    assert failed.value.stage == "certification"
    assert failed.value.attempts == 3
    assert all("positive" in message for message in failed.value.failures)


@pytest.mark.parametrize("case", ["flat", "singular", "absent"])
def test_singular_or_nonexistent_stationary_points_never_receive_receipts(case):
    x = s.Symbol("stationary_invalid_" + case)
    source = {"flat": s.Integer(7), "singular": x**4, "absent": x+x**3}[case]
    with pytest.raises(s.StationarySearchError) as failed:
        s.compile_objective(source, x).solve_stationary_certified(0, max_attempts=3)
    if case in ("flat", "singular"):
        assert failed.value.stage == "certification"
        assert failed.value.attempts == 3
    else:
        assert failed.value.stage == "candidate"


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_forged_stationary_candidate_cannot_erase_an_original_domain_hole(monkeypatch, hidden):
    x = s.Symbol("stationary_original_domain_" + hidden)
    hole = (s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
            if hidden == "zero_product" else
            s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False))
    source = s.Add(x*x, hole, evaluate=False)
    program = s.compile_objective(source, x)
    monkeypatch.setattr(search_module(), "_iterate_stationary", lambda *args: [0.0])
    with pytest.raises(s.StationarySearchError) as failed:
        program.solve_stationary_certified(1, max_attempts=2)
    assert failed.value.stage == "certification"
    assert failed.value.attempts == 2


@pytest.mark.parametrize("candidate", [[math.nan], [math.inf], [0], [0.0, 1.0], [999.0]])
def test_forged_candidates_require_independent_acceptance(monkeypatch, candidate):
    x = s.Symbol("stationary_forged_candidate_x")
    program = s.compile_objective((x-1)**2, x)
    monkeypatch.setattr(search_module(), "_iterate_stationary", lambda *args: candidate)
    with pytest.raises(s.StationarySearchError):
        program.solve_stationary_certified(0, max_attempts=3)


def test_actual_box_and_precision_recovery_share_one_attempt_and_evaluation_budget(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_recovery_x")
    program = s.compile_objective(s.exp(x)-2*x, x)
    budget = ample_budget()
    checked, observations = module.certify_stationary_point, []

    def inaccurate(*args, budget):
        budget._charge(19)
        return [math.log(2)+0.0001]

    def observe(program, box, center, preconditioner, precision, *, budget):
        observations.append((precision, budget, budget.remaining_steps))
        return checked(program, box, center, preconditioner, precision, budget=budget)

    monkeypatch.setattr(module, "_iterate_stationary", inaccurate)
    monkeypatch.setattr(module, "certify_stationary_point", observe)
    receipt = program.solve_stationary_certified(
        0, precision=8, max_attempts=8, budget=budget,
    )
    lo, hi = receipt.stationary_enclosure[0]
    assert float(lo) <= math.log(2) <= float(hi)
    assert len(observations) > 1
    assert observations[0][0] == 8 and max(row[0] for row in observations) > 8
    assert all(row[1] is budget for row in observations)
    assert all(left[2] > right[2] for left, right in zip(observations, observations[1:]))
    assert budget.remaining_steps < observations[-1][2]
    assert receipt.precision_digits == observations[-1][0]
    assert receipt.kind == "strict_minimum"


def test_every_rejected_trial_consumes_the_same_outer_attempt_count(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_trial_bound_x")
    program = s.compile_objective(x**4, x)
    checked, precisions = module.certify_stationary_point, []

    def observe(program, box, center, preconditioner, precision):
        precisions.append(precision)
        return checked(program, box, center, preconditioner, precision)

    monkeypatch.setattr(module, "certify_stationary_point", observe)
    with pytest.raises(s.StationarySearchError) as failed:
        program.solve_stationary_certified(0, precision=8, max_attempts=5)
    assert precisions == [8, 8, 16, 16, 32]
    assert failed.value.stage == "certification"
    assert failed.value.attempts == len(failed.value.failures) == 5


@pytest.mark.parametrize("option,value", [
    ("precision", 7), ("precision", 129), ("precision", True),
    ("max_attempts", 0), ("max_attempts", 25), ("max_attempts", True),
    ("maxiter", 0), ("maxiter", 10_001), ("maxiter", True),
    ("tol", 0), ("tol", math.nan), ("tol", math.inf),
])
def test_invalid_search_controls_refuse_before_candidate_work(monkeypatch, option, value):
    x = s.Symbol("stationary_controls_x")
    program = s.compile_objective((x-1)**2, x)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid controls must refuse before candidate work")

    monkeypatch.setattr(search_module(), "_iterate_stationary", forbidden)
    with pytest.raises((TypeError, ValueError)):
        program.solve_stationary_certified(0, **{option: value})


def test_static_source_dimension_and_assumption_failures_refuse_before_newton(monkeypatch):
    x = s.Symbol("stationary_admission_x")
    integer = s.Symbol("stationary_admission_integer", integer=True)
    floating = s.Add(x*x, s.Mul(s.Float(0.5), x, evaluate=False), evaluate=False)
    inexact = s.compile_objective(floating, x)
    discrete = s.compile_objective((integer-1)**2, integer)
    constant = s.compile_objective(1, ())
    oversized = s.compile_objective(1, tuple(s.Symbol(f"stationary_large_{i}") for i in range(17)))

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible exact claims must refuse before candidate work")

    monkeypatch.setattr(search_module(), "_iterate_stationary", forbidden)
    with pytest.raises(ValueError, match="exact source"):
        inexact.solve_stationary_certified(0)
    with pytest.raises(NotImplementedError, match="assumptions"):
        discrete.solve_stationary_certified(0)
    for program in (constant, oversized):
        with pytest.raises(ValueError, match="1..16"):
            program.solve_stationary_certified(0)


def test_numerical_iteration_and_work_limits_remain_candidate_failures(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("stationary_work_limits_x")
    program = s.compile_objective(s.exp(x)-2*x, x)
    with pytest.raises(s.StationarySearchError, match="failed to converge") as failed:
        program.solve_stationary_certified(0, maxiter=1)
    assert failed.value.stage == "candidate"
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 1)
    with pytest.raises(s.StationarySearchError, match="evaluation budget"):
        program.solve_stationary_certified(0)
    monkeypatch.setattr(numerical, "_MAX_SYSTEM_EVALUATIONS", 4096)
    monkeypatch.setattr(numerical, "_MAX_LINEAR_WORK", 0)
    with pytest.raises(s.StationarySearchError, match="linear-algebra work budget"):
        program.solve_stationary_certified(0)


def test_custom_initial_conversions_and_budget_lookalikes_never_execute():
    x = s.Symbol("stationary_custom_input_x")
    program = s.compile_objective((x-1)**2, x)
    calls = []

    class Number:
        def __float__(self):
            calls.append("float")
            return 0.0

    class Sequence(list):
        def __iter__(self):
            calls.append("iterate")
            return super().__iter__()

    class Budget:
        def _checkpoint(self):
            calls.append("budget")

    for initial in (Number(), Sequence([0])):
        with pytest.raises(TypeError):
            program.solve_stationary_certified(initial)
    with pytest.raises(TypeError, match="EvaluationBudget"):
        program.solve_stationary_certified(0, budget=Budget())
    assert calls == []


def test_initial_vector_snapshot_cannot_grow_past_its_admitted_dimension(monkeypatch):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("stationary_input_snapshot_x")
    program = s.compile_objective((x-1)**2, x)
    initial, grown, real_len = [0], [], builtins.len

    def grow_after_size(value):
        size = real_len(value)
        if value is initial and not grown:
            grown.append(True)
            initial.extend([0]*1000)
        return size

    monkeypatch.setattr(certified, "len", grow_after_size, raising=False)
    with pytest.raises(ValueError, match="length changed"):
        program.solve_stationary_certified(initial)
    assert grown == [True]


def test_repeated_searches_cannot_reset_the_callers_measured_budget():
    x = s.Symbol("stationary_shared_budget_x")
    program = s.compile_objective((x-1)**2, x)
    probe = ample_budget()
    expected = program.solve_stationary_certified(0, budget=probe).stationary_enclosure
    cost = (1 << 40)-probe.remaining_memory_bytes
    assert cost >= 2*64*1024*1024
    budget = ample_budget(max_memory_bytes=cost)
    assert program.solve_stationary_certified(0, budget=budget).stationary_enclosure == expected
    assert budget.remaining_memory_bytes == 0
    after_success = budget.remaining_steps
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_stationary_certified(0, budget=budget)
    assert budget.remaining_steps <= after_success
    assert budget.remaining_memory_bytes == 0


def test_exhausted_and_cancelled_budgets_refuse_without_candidate_publication(monkeypatch):
    x = s.Symbol("stationary_cancelled_x")
    program = s.compile_objective((x-1)**2, x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_stationary_certified(0, budget=ample_budget(max_steps=0))
    budget = ample_budget()
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled work must not enter the candidate engine")

    monkeypatch.setattr(search_module(), "_iterate_stationary", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_stationary_certified(0, budget=budget)
    assert budget.remaining_steps == 1 << 45


def test_cancellation_after_newton_prevents_all_exact_witness_work(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_cancel_after_newton_x")
    program = s.compile_objective((x-1)**2, x)
    budget = ample_budget()

    def cancelled(*args, budget):
        budget.cancel()
        return [1.0]

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled candidates must not prepare a witness")

    monkeypatch.setattr(module, "_iterate_stationary", cancelled)
    monkeypatch.setattr(module, "_search_stationary_box", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_stationary_certified(0, budget=budget)


def test_cancellation_after_native_acceptance_still_refuses_publication(monkeypatch):
    module = search_module()
    x = s.Symbol("stationary_cancel_after_check_x")
    program = s.compile_objective((x-1)**2, x)
    budget = ample_budget()
    checked, accepted = module.certify_stationary_point, []

    def cancel_after_check(*args, budget):
        receipt = checked(*args, budget=budget)
        accepted.append(receipt)
        budget.cancel()
        return receipt

    monkeypatch.setattr(module, "certify_stationary_point", cancel_after_check)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_stationary_certified(0, budget=budget)
    assert len(accepted) == 1


@pytest.mark.parametrize("failure", ["budget", "memory", "unsupported"])
def test_operational_or_unsupported_inverse_failures_are_never_retried(monkeypatch, failure):
    certified = importlib.import_module("sympy.utilities._certified_objective")
    x = s.Symbol("stationary_inverse_failure_x")
    program = s.compile_objective((x-1)**2, x)
    budget, calls = ample_budget(), []
    error_type = {
        "budget": s.EvaluationBudgetExceeded,
        "memory": MemoryError,
        "unsupported": NotImplementedError,
    }[failure]

    def exhausted(matrix, supplied):
        calls.append(supplied)
        raise error_type("bounded inverse cannot continue")

    monkeypatch.setattr(certified, "_bounded_inverse", exhausted)
    with pytest.raises(error_type, match="bounded inverse"):
        program.solve_stationary_certified(0, budget=budget)
    assert calls == [budget]
    assert budget.remaining_steps < 1 << 45


def test_stationary_search_error_is_exported_from_both_public_namespaces():
    from sympy.utilities import StationarySearchError
    assert StationarySearchError is s.StationarySearchError
