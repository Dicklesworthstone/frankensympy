"""Real donor proposals acquire root evidence only through the existing checker."""

import importlib
import math
from types import SimpleNamespace

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding


BACKENDS = ("frankennumpy", "frankenscipy")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entrypoint", ["compiled", "nsolve"])
def test_both_public_searches_certify_real_donor_transcendental_proposals(backend, entrypoint):
    x, y = s.symbols("certified_donor_x certified_donor_y")
    residuals = (s.exp(x) - 2, y - s.sin(x))
    program = s.compile_residual_system(residuals, [(x, y)])
    budget = s.EvaluationBudget()
    if entrypoint == "compiled":
        receipt = program.solve_certified([0, 0], backend=backend, budget=budget)
    else:
        receipt = s.nsolve_certified(residuals, (x, y), (0, 0), backend=backend, budget=budget)
    assert type(receipt) is s.RootBoxReceipt
    assert receipt.residuals == residuals
    for (lo, hi), expected in zip(receipt.root_enclosure, (math.log(2), math.sin(math.log(2)))):
        assert float(lo) <= expected <= float(hi)
    assert all(a < lo <= hi < b for (a, b), (lo, hi) in zip(receipt.domain, receipt.root_enclosure))
    assert receipt.contraction_bound < 1
    assert "unique nonsingular real root" in receipt.claim
    assert budget.remaining_steps < 10_000_000
    assert budget.remaining_memory_bytes < 256 * 1024 * 1024


@pytest.mark.parametrize("backend", BACKENDS)
def test_transport_typed_coordinates_and_source_order_remain_bound_to_the_receipt(backend):
    plain = s.Symbol("certified_donor_same")
    positive = s.Symbol("certified_donor_same", positive=True)
    dummy = s.Dummy("certified_donor_same")
    variables = (dummy, positive, plain)
    bindings = tuple(_symbol_binding(value) for value in variables)
    # Zero-fact bindings lower to plain native atoms, with no typed digest.
    # Receipts project that absence as 64 zero hex characters. The initial
    # shell handle's empty-facts digest is only a diagnostic placeholder;
    # declared identities retain their exact bytes, even an all-zero digest.
    expected_bindings = tuple(
        (value.name, value.plain, "0" * 64 if value.plain else value.identity_hex)
        for value in bindings
    )
    residuals = [plain + positive - 5, dummy + plain - 4, positive - 2]
    original = s.compile_residual_system(residuals, variables)
    restored = s.load_residual_system(original.to_bytes(), residuals=residuals, args=variables)
    receipt = restored.solve_certified([0, 0, 0], backend=backend)
    assert receipt.root_enclosure == ((1, 1), (2, 2), (3, 3))
    assert all(left is right for left, right in zip(restored.variables, variables))
    assert all(type(value) is _native.SymbolBinding for value in receipt.variables)
    assert tuple((value.name, value.plain, value.identity_hex) for value in receipt.variables) == expected_bindings


def test_default_python_controls_and_original_iterator_call_shape_are_preserved(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    original = numerical._iterate_system
    calls = []

    def observe(program, values, tolerance, maxiter):
        calls.append((tolerance, maxiter))
        return original(program, values, tolerance, maxiter)

    monkeypatch.setattr(numerical, "_iterate_system", observe)
    x = s.Symbol("certified_python_default_x")
    assert s.nsolve_certified(x-1, x, 0).root_enclosure == ((1, 1),)
    assert s.compile_residual_system([x-1], [x]).solve_certified(0).root_enclosure == ((1, 1),)
    assert calls == [(1e-12, 200), (1e-12, 200)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_ineligible_sources_and_assumptions_refuse_before_any_donor_or_initial_conversion(monkeypatch, backend):
    x = s.Symbol("certified_donor_preflight_x")
    integer = s.Symbol("certified_donor_integer", integer=True)
    programs = (
        s.compile_residual_system([s.Add(x, s.Float(-1.25), evaluate=False)], [x]),
        s.compile_residual_system([integer-1], [integer]),
        s.compile_residual_system([x, x+1], [x]),
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("donor")
        raise AssertionError("ineligible source reached candidate work")

    class Initial:
        def __float__(self):
            calls.append("initial")
            return 0.0

    monkeypatch.setattr(type(programs[0]), "solve_candidate", forbidden)
    budget = s.EvaluationBudget(max_steps=0)
    for program in programs:
        with pytest.raises((ValueError, NotImplementedError)):
            program.solve_certified(Initial(), backend=backend, budget=budget)
    assert calls == [] and budget.remaining_steps == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_assumptions_cannot_be_relabelled_after_a_real_donor_finishes(monkeypatch, backend):
    x = s.Symbol("certified_donor_snapshot_x", positive=True)
    facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    program = s.compile_residual_system([x-1], [x])
    solve = type(program).solve_candidate

    def change_caller_facts(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(type(program), "solve_candidate", change_caller_facts)
    try:
        receipt = program.solve_certified(0, backend=backend)
        assert receipt.root_enclosure == ((1, 1),)
        assert receipt.variables[0].identity_hex == binding.identity_hex
        assert receipt.domain[0][0] > 0
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["tiny_positive_residual", "singular_root", "wrong_assumption"])
def test_approximate_donor_success_cannot_assert_nonexistent_or_ineligible_roots(backend, kind):
    x = s.Symbol("certified_donor_false_root_x", positive=True) if kind == "wrong_assumption" else s.Symbol("certified_donor_false_root_x")
    source, initial = {
        "tiny_positive_residual": (s.exp(x), -40),
        "singular_root": (x*x, 0),
        "wrong_assumption": (x+1, 0),
    }[kind]
    program = s.compile_residual_system([source], [x])
    candidate = program.solve_candidate(initial, backend=backend)
    assert candidate.residual_norm <= 1e-10
    assert candidate.evidence == "approximate"
    with pytest.raises(s.RootSearchError) as failed:
        program.solve_certified(initial, backend=backend, max_attempts=2)
    assert failed.value.stage == "certification"
    assert failed.value.attempts == 2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("root", [(math.nan,), (0,), (999.0,)])
def test_forged_donor_success_still_requires_original_source_verification(monkeypatch, backend, root):
    x = s.Symbol("certified_donor_forged_x")
    program = s.compile_residual_system([x*x-2], [x])
    monkeypatch.setattr(type(program), "solve_candidate", lambda *args, **kwargs: SimpleNamespace(root=root))
    with pytest.raises(s.RootSearchError):
        program.solve_certified(1, backend=backend, max_attempts=2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_candidate_and_certificate_work_cannot_reset_a_shared_ledger(monkeypatch, backend):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    search = certified.search_root_box
    x = s.Symbol("certified_donor_budget_x")
    program = s.compile_residual_system([x-1], [x])
    measured = s.EvaluationBudget(max_steps=1_000_000)
    program.solve_candidate(0, backend=backend, budget=measured)
    candidate_cost = 1_000_000 - measured.remaining_steps
    budget = s.EvaluationBudget(max_steps=candidate_cost)
    calls = []

    def observe(captured, root, precision, max_attempts, *, budget):
        assert captured is program
        calls.append((budget, budget.remaining_steps, tuple(root)))
        return search(captured, root, precision, max_attempts, budget=budget)

    monkeypatch.setattr(certified, "search_root_box", observe)
    with pytest.raises(s.EvaluationBudgetExceeded) as failed:
        program.solve_certified(0, backend=backend, budget=budget)
    assert not isinstance(failed.value, s.RootSearchError)
    assert calls == [(budget, 0, (1.0,))]
    assert budget.remaining_steps == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancellation_after_real_native_receipt_assembly_prevents_publication(monkeypatch, backend):
    certified = importlib.import_module("sympy.utilities._certified_numeric")
    constructor = certified.RootBoxReceipt
    x = s.Symbol("certified_donor_late_cancel_x")
    budget = s.EvaluationBudget()

    def cancel_after_assembly(receipt):
        result = constructor(receipt)
        budget.cancel()
        return result

    monkeypatch.setattr(certified, "RootBoxReceipt", cancel_after_assembly)
    with pytest.raises(s.EvaluationCancelled):
        s.nsolve_certified(x-1, x, 0, backend=backend, budget=budget)
    assert budget.cancelled


@pytest.mark.parametrize("exception", [s.EvaluationCancelled, s.EvaluationBudgetExceeded, MemoryError, RuntimeError])
def test_operational_donor_failures_are_never_reclassified_as_candidate_search_failure(monkeypatch, exception):
    x = s.Symbol("certified_donor_operational_x")
    program = s.compile_residual_system([x-1], [x])
    error = exception("original operational failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(type(program), "solve_candidate", fail)
    with pytest.raises(exception) as failed:
        program.solve_certified(0, backend="frankennumpy")
    assert failed.value is error


@pytest.mark.parametrize("options", [
    {"backend": "frankenscipy", "tol": 1e-10},
    {"backend": "frankenscipy", "maxiter": 200},
    {"backend": "frankenscipy", "max_step_norm": 1e6},
    {"backend": "python", "max_step_norm": 1e6},
    {"backend": "frankennumpy", "maxiter": 1025},
    {"backend": "missing"}, {"backend": 1},
])
def test_invalid_backend_controls_refuse_before_compilation_or_candidate_work(monkeypatch, options):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("certified_donor_options_x")
    program = s.compile_residual_system([x-1], [x])

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid controls started compilation or a donor")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    monkeypatch.setattr(type(program), "solve_candidate", forbidden)
    for call in (
        lambda: s.nsolve_certified(x-1, x, 0, **options),
        lambda: program.solve_certified(0, **options),
    ):
        with pytest.raises((ValueError, TypeError)):
            call()


def test_numpy_controls_apply_before_the_same_independent_certificate_search():
    x = s.Symbol("certified_donor_controls_x")
    receipt = s.nsolve_certified(x-1, x, 1, backend="frankennumpy", tol=0, maxiter=0)
    assert receipt.root_enclosure == ((1, 1),)
    with pytest.raises(s.RootSearchError) as failed:
        s.nsolve_certified(x-1, x, 0, backend="frankennumpy", maxiter=0)
    assert failed.value.stage == "candidate"
