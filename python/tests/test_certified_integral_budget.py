"""A complete integral request shares fuel through independent acceptance."""

from fractions import Fraction
import importlib
import threading
import time

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding
from sympy.utilities.lambdify import _functions, _numeric_source


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def remaining(budget):
    return budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations


def native_integral(source, variable, *, budget, precision=32, max_intervals=512):
    return _native.certified_integral_native(
        source, _symbol_binding(variable), _native.py_integer(0), _native.py_integer(1),
        _native.py_rational(1, 10000), precision, max_intervals, budget=budget,
    )


def test_shared_budget_composes_numeric_evaluation_and_exact_integral_acceptance():
    x = s.Symbol("integral_budget_compose_x")
    program = s.compile_numeric(x, x*x)
    budget = ample_budget()
    assert program(3, budget=budget) == 9
    before = remaining(budget)
    receipt = s.certified_integral(x*x, (x, 0, 1), budget=budget)
    assert receipt.integrand == x*x
    assert receipt.bounds == (s.Rational(1, 3), s.Rational(1, 3))
    assert receipt.error_bound == 0
    assert receipt.interval_count == 1
    assert all(after < start for after, start in zip(remaining(budget), before))
    # Each independent phase reserves its bounded interval workspace.
    assert before[1] - budget.remaining_memory_bytes >= 2 * 72 * 1024 * 1024


def test_repeated_integrals_cannot_refresh_the_measured_memory_allowance():
    x = s.Symbol("integral_budget_repeat_x")
    probe = ample_budget()
    expected = s.certified_integral(x, (x, 0, 1), budget=probe).bounds
    cost = (1 << 40) - probe.remaining_memory_bytes
    shared = ample_budget(max_memory_bytes=cost)
    assert s.certified_integral(x, (x, 0, 1), budget=shared).bounds == expected
    assert shared.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.certified_integral(x, (x, 0, 1), budget=shared)
    assert shared.remaining_memory_bytes == 0


def test_generator_success_cannot_bypass_the_verifier_memory_allowance():
    x = s.Symbol("integral_budget_verifier_x")
    # The affine proposal fits its ~72 MiB workspace reservation. Its
    # independent verifier cannot reserve another workspace from this ledger.
    budget = ample_budget(max_memory_bytes=128 * 1024 * 1024)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.certified_integral(x, (x, 0, 1), budget=budget)
    assert 0 < budget.remaining_memory_bytes < 56 * 1024 * 1024
    assert not issubclass(s.EvaluationBudgetExceeded, ValueError)
    assert not issubclass(s.EvaluationCancelled, ValueError)


@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_direct_native_input_copy_refuses_zero_allowances_atomically(limit):
    x = s.Symbol("integral_budget_native_zero_x")
    budget = ample_budget(**{limit: 0})
    before = remaining(budget)
    with pytest.raises(s.EvaluationBudgetExceeded):
        native_integral(x._value, x, budget=budget)
    assert remaining(budget) == before


def test_source_domain_and_accuracy_failures_retain_their_work_on_reuse():
    x = s.Symbol("integral_budget_failed_work_x")
    budget = ample_budget()
    source = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.certified_integral(source, (x, 0, 1), budget=budget)
    after_domain = remaining(budget)
    assert after_domain[0] < 1 << 45
    assert after_domain[1] < 1 << 40
    with pytest.raises(ValueError, match="width"):
        s.certified_integral(
            x**4, (x, 0, 1), tolerance=Fraction(1, 10**20), max_intervals=1, budget=budget,
        )
    after_accuracy = remaining(budget)
    assert all(after < before for after, before in zip(after_accuracy, after_domain))
    assert s.certified_integral(x, (x, 0, 1), budget=budget).bounds == (s.Rational(1, 2),)*2
    assert all(after < before for after, before in zip(remaining(budget), after_accuracy))


@pytest.mark.parametrize("reverse", [False, True])
def test_budgeted_corner_integrals_preserve_exact_partition_and_orientation(reverse):
    x = s.Symbol("integral_budget_corner_x", real=True)
    limits = (x, 1, -1) if reverse else (x, -1, 1)
    receipt = s.certified_integral(
        s.Abs(x), limits, tolerance=Fraction(1, 10**100), max_intervals=2,
        budget=ample_budget(),
    )
    expected = -1 if reverse else 1
    assert receipt.bounds == (expected, expected)
    assert receipt.partition == (-1, 0, 1)
    assert receipt.interval == limits[1:]


def test_private_budget_is_captured_once_before_preparation_and_forwarded(monkeypatch):
    parametric = importlib.import_module("sympy.utilities._parametric_root")
    real_factory = parametric._fallback_budget
    created = []

    def capture():
        budget = real_factory()
        created.append(budget)
        return budget

    monkeypatch.setattr(parametric, "_fallback_budget", capture)
    real_native = _native.certified_integral_native
    seen = []

    def observe(*args, **kwargs):
        budget = kwargs["budget"]
        seen.append(budget)
        assert budget.remaining_steps < 512 * 1024 * 1024
        return real_native(*args, **kwargs)

    monkeypatch.setattr(_native, "certified_integral_native", observe)
    x = s.Symbol("integral_budget_private_x")
    assert s.certified_integral(x, (x, 0, 1)).bounds == (s.Rational(1, 2),)*2
    assert len(created) == 1 and seen == created
    assert (512 * 1024 * 1024 - created[0].remaining_memory_bytes) >= 2 * 72 * 1024 * 1024


def test_cancelled_request_stops_before_source_or_limit_preparation(monkeypatch):
    certified = importlib.import_module("sympy.integrals.certified")
    budget = ample_budget()
    before = remaining(budget)
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled integral must not inspect its inputs")

    monkeypatch.setattr(certified, "_sequence", forbidden)
    monkeypatch.setattr(certified, "_numeric_source", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        s.certified_integral(object(), object(), budget=budget)
    assert remaining(budget) == before


def test_cancellation_after_receipt_construction_prevents_publication(monkeypatch):
    certified = importlib.import_module("sympy.integrals.certified")
    real_receipt = certified.IntegralEnclosureReceipt
    budget = ample_budget()
    accepted = []

    def cancel_after_acceptance(native):
        receipt = real_receipt(native)
        accepted.append(receipt)
        budget.cancel()
        return receipt

    monkeypatch.setattr(certified, "IntegralEnclosureReceipt", cancel_after_acceptance)
    x = s.Symbol("integral_budget_publication_x")
    with pytest.raises(s.EvaluationCancelled):
        s.certified_integral(x, (x, 0, 1), budget=budget)
    assert len(accepted) == 1
    assert accepted[0].bounds == (s.Rational(1, 2),)*2
    assert (1 << 40) - budget.remaining_memory_bytes >= 2 * 72 * 1024 * 1024


def test_direct_native_work_releases_gil_for_external_cancellation():
    x = s.Symbol("integral_budget_thread_x")
    source = s.Add(*(s.exp(s.Rational(i+1, 64)*x) for i in range(64)), evaluate=False)
    native = _numeric_source(source, _functions(), [0], inexact=[False])
    budget = ample_budget()
    outcomes = []
    started = threading.Event()

    def integrate():
        started.set()
        try:
            outcomes.append(native_integral(native, x, budget=budget, precision=128))
        except Exception as error:
            outcomes.append(error)

    worker = threading.Thread(target=integrate)
    worker.start()
    observed_busy = False
    try:
        assert started.wait(5)
        deadline = time.monotonic() + 5
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                budget.remaining_steps
            except RuntimeError as error:
                # Input snapshots retain the GIL. A busy ledger observed by
                # this Python thread therefore identifies detached arithmetic.
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


def test_invalid_budget_refuses_without_executing_lookalike_hooks():
    class HostileBudget:
        def _checkpoint(self):
            raise AssertionError("lookalike budgets must not execute")

    x = s.Symbol("integral_budget_type_x")
    for call in (
        lambda: s.certified_integral(x, (x, 0, 1), budget=HostileBudget()),
        lambda: native_integral(x._value, x, budget=HostileBudget()),
    ):
        with pytest.raises(TypeError, match="EvaluationBudget"):
            call()


@pytest.mark.parametrize("options", [{"precision": 7}, {"max_intervals": 0}])
def test_native_configuration_refusals_remain_value_errors(options):
    x = s.Symbol("integral_budget_options_x")
    budget = ample_budget()
    before = remaining(budget)
    with pytest.raises(ValueError):
        native_integral(x._value, x, budget=budget, **options)
    assert remaining(budget) == before
