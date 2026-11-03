"""One caller ledger governs proposals and independent interval acceptance."""

from fractions import Fraction
import builtins
import importlib
import math
import threading
import time

import pytest
import sympy as s

from sympy.core import _native


def ample_budget(**limits):
    # Core interval work reserves 64 MiB per invocation. These are fuel
    # allowances, not allocations made by constructing an EvaluationBudget.
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_one_budget_composes_approximation_enclosure_and_root_acceptance():
    x = s.Symbol("certified_budget_shared_x")
    program = s.compile_residual_system([x*x-2], [x])
    budget = ample_budget()
    assert program.residuals(1, budget=budget) == [-1]
    after_approximation = budget.remaining_steps
    enclosed = program.enclose([(1, 2)], precision=16, budget=budget)
    lo, hi = enclosed.residual_values[0]
    assert lo <= -1 and hi >= 2
    lo, hi = enclosed.jacobian[0][0]
    assert lo <= 2 and hi >= 4
    after_enclosure = budget.remaining_steps
    receipt = program.certify_root_box([(Fraction(7, 5), Fraction(3, 2))],
                                       precision=16, budget=budget)
    lo, hi = receipt.root_enclosure[0]
    assert lo*lo <= 2 <= hi*hi
    assert 0 <= receipt.contraction_bound < 1
    assert budget.remaining_steps < after_enclosure < after_approximation < 1 << 45
    assert (1 << 40) - budget.remaining_memory_bytes >= 3 * 64 * 1024 * 1024
    assert receipt.residuals == (x*x-2,)


def test_repeated_enclosures_cannot_reset_the_measured_memory_ledger():
    x = s.Symbol("certified_budget_repeat_x")
    program = s.compile_residual_system([x-1], [x])
    probe = ample_budget()
    expected = program.enclose([(0, 2)], budget=probe).residual_values
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 64 * 1024 * 1024
    budget = ample_budget(max_memory_bytes=cost)
    assert program.enclose([(0, 2)], budget=budget).residual_values == expected
    assert budget.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.enclose([(0, 2)], budget=budget)
    assert budget.remaining_memory_bytes == 0


@pytest.mark.parametrize("limit", ["max_memory_bytes", "max_allocations"])
def test_native_exact_input_storage_refuses_an_atomic_zero_allowance(limit):
    x = s.Symbol("certified_budget_input_x")
    native = s.compile_residual_system([x], [x]).native_system
    domain = [(_native.py_rational(0, 1), _native.py_rational(1, 1))]
    budget = ample_budget(**{limit: 0})
    before = budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations
    with pytest.raises(s.EvaluationBudgetExceeded):
        native.enclose(domain, budget=budget)
    assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


def test_actual_interval_reservation_error_keeps_its_operational_type():
    x = s.Symbol("certified_budget_interval_x")
    native = s.compile_residual_system([x], [x]).native_system
    domain = [(_native.py_rational(0, 1), _native.py_rational(1, 1))]
    budget = ample_budget(max_memory_bytes=64 * 1024 * 1024 - 1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        native.enclose(domain, budget=budget)
    assert 0 < budget.remaining_memory_bytes < 64 * 1024 * 1024 - 1
    assert not issubclass(s.EvaluationBudgetExceeded, ValueError)
    assert not issubclass(s.EvaluationCancelled, ValueError)


def test_domain_rejection_retains_actual_work_before_later_valid_enclosure():
    x = s.Symbol("certified_budget_domain_x")
    program = s.compile_residual_system([s.log(x)], [x])
    budget = ample_budget()
    with pytest.raises(ValueError):
        program.enclose([(-1, 2)], budget=budget)
    after_failure = budget.remaining_steps
    assert after_failure < 1 << 45
    assert (1 << 40) - budget.remaining_memory_bytes >= 64 * 1024 * 1024
    assert program.enclose([(1, 1)], budget=budget).residual_values == ((0, 0),)
    assert budget.remaining_steps < after_failure


@pytest.mark.parametrize("dimension", [1, 3])
def test_budgeted_search_preserves_exact_source_bound_root_receipts(dimension):
    variables = [s.Symbol(f"certified_budget_affine_{dimension}_{i}") for i in range(dimension)]
    residuals = [value-i-1 for i, value in enumerate(variables)]
    budget = ample_budget()
    receipt = s.nsolve_certified(residuals, variables, [0]*dimension, budget=budget)
    assert receipt.root_enclosure == tuple((i+1, i+1) for i in range(dimension))
    assert receipt.residuals == tuple(residuals)
    assert receipt.contraction_bound == 0
    assert budget.remaining_steps < 1 << 45


def test_cancelled_budget_refuses_all_certified_entry_points_before_new_work(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("certified_budget_cancelled_x")
    program = s.compile_residual_system([x-1], [x])
    budget = ample_budget()
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled search must not compile a new program")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    for call in (
        lambda: program.enclose([(0, 2)], budget=budget),
        lambda: program.certify_root_box([(0, 2)], budget=budget),
        lambda: program.solve_certified(0, budget=budget),
        lambda: s.nsolve_certified(x-1, x, 0, budget=budget),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()
    assert budget.remaining_steps == 1 << 45


def test_candidate_budget_failure_is_not_wrapped_as_a_root_search_error():
    x = s.Symbol("certified_budget_candidate_x")
    program = s.compile_residual_system([x*x-2], [x])
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_certified(1, budget=ample_budget(max_steps=0))


def test_cancellation_after_newton_prevents_all_witness_preparation(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_newton_publication_x")
    program = s.compile_residual_system([x-1], [x])
    budget = ample_budget()

    def cancelled(*args, budget):
        budget.cancel()
        return [1.0]

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled candidate must not enter certificate search")

    monkeypatch.setattr(numerical, "_iterate_system", cancelled)
    monkeypatch.setattr(certified, "search_root_box", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_certified(0, budget=budget)


def test_real_box_and_precision_retries_share_the_same_cumulative_ledger(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_retry_x")
    program = s.compile_residual_system([s.exp(x)-2], [x])
    budget = ample_budget()
    monkeypatch.setattr(numerical, "_iterate_system", lambda *args, **kwargs: [math.log(2)+0.0001])
    original, observations = certified.certify_root_box, []

    def observe(*args, budget):
        before = budget.remaining_steps
        try:
            return original(*args, budget=budget)
        finally:
            observations.append((budget, args[-1], before, budget.remaining_steps))

    monkeypatch.setattr(certified, "certify_root_box", observe)
    receipt = program.solve_certified(0, precision=8, max_attempts=8, budget=budget)
    lo, hi = receipt.root_enclosure[0]
    assert float(lo) <= math.log(2) <= float(hi)
    assert len(observations) > 1
    assert max(row[1] for row in observations) > 8
    assert all(row[0] is budget and row[3] < row[2] for row in observations)
    assert all(later[2] < earlier[3] for earlier, later in zip(observations, observations[1:]))


def test_exhausted_interval_work_stops_retries_without_returning_a_candidate(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_stop_retry_x")
    program = s.compile_residual_system([x*x+1], [x])
    monkeypatch.setattr(numerical, "_iterate_system", lambda *args, **kwargs: [1.0])
    original, attempts = certified.certify_root_box, []

    def observe(*args, budget):
        attempts.append(args[-1])
        return original(*args, budget=budget)

    monkeypatch.setattr(certified, "certify_root_box", observe)
    # One real point enclosure + rejected root check fits; the next full
    # pair cannot. Resetting the native ledger would run all 24 attempts.
    budget = ample_budget(max_memory_bytes=200 * 1024 * 1024)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_certified(1, max_attempts=24, budget=budget)
    assert len(attempts) == 2
    assert budget.remaining_memory_bytes < 64 * 1024 * 1024


def test_fixed_checker_work_limit_is_not_retried_as_failed_inclusion(monkeypatch):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_fixed_cap_x")
    program = s.compile_residual_system([x**65-1], [x])
    original, attempts = certified.certify_root_box, []

    def observe(*args, **kwargs):
        attempts.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(certified, "certify_root_box", observe)
    with pytest.raises(s.EvaluationBudgetExceeded, match="integer exponent"):
        program.solve_certified(1, max_attempts=12)
    assert len(attempts) == 1


def test_exact_inverse_proposal_uses_real_cumulative_compute_fuel():
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    matrix = ((Fraction(2), Fraction(1)), (Fraction(1), Fraction(3)))
    budget = ample_budget()
    assert certified._bounded_inverse(matrix, budget) == (
        (Fraction(3, 5), Fraction(-1, 5)), (Fraction(-1, 5), Fraction(2, 5)),
    )
    cost = (1 << 45) - budget.remaining_steps
    limited = ample_budget(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        certified._bounded_inverse(matrix, limited)
    assert limited.remaining_steps < cost-1


def test_exact_inverse_cancellation_publishes_no_partial_preconditioner(monkeypatch):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    budget = ample_budget()
    original, calls = certified._charge_exact_work, []

    def cancel_during_work(actual, *values, **kwargs):
        calls.append(values)
        original(actual, *values, **kwargs)
        if len(calls) == 3:
            actual.cancel()

    monkeypatch.setattr(certified, "_charge_exact_work", cancel_during_work)
    with pytest.raises(s.EvaluationCancelled):
        certified._bounded_inverse(((Fraction(2), Fraction(1)), (Fraction(1), Fraction(3))), budget)
    assert len(calls) <= 4
    assert budget.remaining_steps < 1 << 45


@pytest.mark.parametrize("kind", ["enclosure", "root"])
def test_cancellation_after_real_receipt_wrapper_assembly_refuses_publication(monkeypatch, kind):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_receipt_x")
    program = s.compile_residual_system([x-1], [x])
    budget = ample_budget()
    name = "RootBoxReceipt" if kind == "root" else "ResidualJacobianEnclosure"
    wrapper, created = getattr(certified, name), []

    def cancel_after_construction(native_receipt):
        private = wrapper(native_receipt)
        created.append(private)
        budget.cancel()
        return private

    monkeypatch.setattr(certified, name, cancel_after_construction)
    with pytest.raises(s.EvaluationCancelled):
        if kind == "root":
            program.certify_root_box([(0, 2)], budget=budget)
        else:
            program.enclose([(0, 2)], budget=budget)
    assert len(created) == 1
    assert (1 << 40) - budget.remaining_memory_bytes >= 64 * 1024 * 1024


def test_native_interval_work_releases_gil_for_external_cancellation():
    x = s.Symbol("certified_budget_thread_x")
    program = s.compile_residual_system([s.exp(s.Rational(i+1, 64)*x) for i in range(64)], [x])
    native = program.native_system
    domain = [(_native.py_rational(1, 1), _native.py_rational(2, 1))]
    budget = ample_budget()
    outcomes = []
    started = threading.Event()

    def evaluate():
        started.set()
        try:
            # Native input snapshots retain the GIL. A busy ledger observed
            # by another Python thread therefore identifies detached interval work,
            # not a Python-side proposal/preparation charge.
            outcomes.append(native.enclose(domain, 128, budget=budget))
        except Exception as error:
            outcomes.append(error)

    worker = threading.Thread(target=evaluate)
    worker.start()
    assert started.wait(5)
    observed_busy = False
    deadline = time.monotonic() + 5
    try:
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                budget.remaining_steps
            except RuntimeError as error:
                assert "already in use" in str(error)
                observed_busy = True
                break
            time.sleep(0.001)
    finally:
        budget.cancel()
        worker.join(10)
    assert not worker.is_alive()
    assert observed_busy
    assert len(outcomes) == 1 and isinstance(outcomes[0], s.EvaluationCancelled)


def test_invalid_budget_objects_are_refused_without_custom_hooks():
    x = s.Symbol("certified_budget_type_x")
    program = s.compile_residual_system([x-1], [x])

    class HostileBudget:
        def _checkpoint(self):
            raise AssertionError("a budget lookalike must not execute")

    for call in (
        lambda b: program.enclose([(0, 2)], budget=b),
        lambda b: program.certify_root_box([(0, 2)], budget=b),
        lambda b: program.solve_certified(0, budget=b),
        lambda b: s.nsolve_certified(x-1, x, 0, budget=b),
    ):
        with pytest.raises(TypeError, match="EvaluationBudget"):
            call(HostileBudget())


@pytest.mark.parametrize("kind", ["bounds", "candidate"])
def test_mutable_input_growth_cannot_enlarge_a_preflighted_snapshot(monkeypatch, kind):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    x = s.Symbol("certified_budget_snapshot_x")
    program = s.compile_residual_system([x-1], [x])
    values = [[0, 2]] if kind == "bounds" else [1.0]
    real_len = builtins.len
    grown = []

    def grow_after_measurement(value):
        length = real_len(value)
        if value is values and not grown:
            grown.append(True)
            values.extend([values[0]] * 1000)
        return length

    monkeypatch.setattr(certified, "len", grow_after_measurement, raising=False)
    if kind == "bounds":
        with pytest.raises(ValueError, match="length changed"):
            program.enclose(values, budget=ample_budget())
    else:
        with pytest.raises(s.RootSearchError, match="invalid finite-real candidate"):
            certified.search_root_box(program, values, 16, 2, budget=ample_budget())
    assert grown == [True]


def test_certified_newton_vector_snapshot_cannot_grow_after_size_admission(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("certified_budget_initial_snapshot_x")
    program = s.compile_residual_system([x-1], [x])
    initial = [0.0]
    real_len = builtins.len
    grown = []

    def grow_after_measurement(value):
        length = real_len(value)
        if value is initial and not grown:
            grown.append(True)
            initial.extend([0.0] * 1000)
        return length

    monkeypatch.setattr(numerical, "len", grow_after_measurement, raising=False)
    with pytest.raises(ValueError, match="length changed"):
        program.solve_certified(initial, budget=ample_budget())
    assert grown == [True]
