"""One caller ledger covers original-source decimal evaluation and publication.

These tests execute the real mathematical PyO3 bridge. Decimal constants and
exact identities supply expected values; budget failures must stay terminal.
"""

from decimal import Decimal, localcontext

import pytest
import sympy as s
from sympy import core


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def remaining(budget):
    return budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations


def evaluate(source, digits, budget, method):
    if method == "N":
        return s.N(source, digits, budget=budget)
    if method == "evalf":
        return source.evalf(digits, budget=budget)
    if method == "n":
        return source.n(digits, budget=budget)
    return s.Basic.evalf(source, digits, budget=budget)


def components(value):
    return tuple(Decimal(str(part)) for part in value.as_real_imag())


@pytest.mark.parametrize("method", ["N", "evalf", "n", "Basic.evalf"])
def test_public_entry_points_round_original_complex_sources_on_one_ledger(method):
    source = s.Add(s.Rational(1, 3), s.Mul(s.Rational(-2, 7), s.I, evaluate=False), evaluate=False)
    budget = ample_budget()
    before = remaining(budget)
    value = evaluate(source, 60, budget, method)
    with localcontext() as context:
        context.prec = 60
        expected = Decimal(1) / 3, -Decimal(2) / 7
    assert components(value) == expected
    assert all(part.dps == 60 for part in value.as_real_imag())
    assert all(after < prior for after, prior in zip(remaining(budget), before))


def test_default_budget_handles_a_high_precision_elementary_composition():
    budget = s.EvaluationBudget()
    value = s.N(s.exp(1 + s.I, evaluate=False), 50, budget=budget)
    assert components(value) == (
        Decimal("1.4686939399158851571389675973266042613269567366290"),
        Decimal("2.2873552871788423912081719067005018089555862566684"),
    )
    assert 0 < budget.remaining_steps < 10_000_000


@pytest.mark.parametrize("source, expected", [
    (s.asin(s.Rational(1, 2), evaluate=False), "0.5235987755982988730771072305465838140329"),
    (s.LambertW(s.E, evaluate=False), "1.000000000000000000000000000000000000000"),
    (s.log(-1, evaluate=False), None),
])
def test_real_only_kernels_and_principal_branches_keep_the_shared_budget(source, expected):
    value = s.N(source, 40, budget=ample_budget())
    if expected is None:
        assert components(value) == (0, Decimal("3.141592653589793238462643383279502884197"))
    else:
        assert Decimal(str(value)) == Decimal(expected)


def test_pi_and_elementary_sources_do_not_escape_to_unbudgeted_routes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an explicit budget entered an unbudgeted fallback")
    monkeypatch.setattr(core, "_machin_pi_floor", forbidden)
    monkeypatch.setattr(core, "_certified_float", forbidden)
    monkeypatch.setattr(core, "_certified_complex_float", forbidden)
    monkeypatch.setattr(core._native, "evalf_decimal_native", forbidden)
    assert str(s.N(s.pi, 40, budget=ample_budget())) == "3.141592653589793238462643383279502884197"
    assert components(s.I.evalf(40, budget=ample_budget())) == (0, 1)


@pytest.mark.parametrize("source", [7, 0.125, s.Float(0.125), core._ExactDecimalFloat("0.125", 40)])
def test_exact_builtin_inputs_and_existing_float_owners_use_the_metered_route(source):
    budget = ample_budget()
    value = s.N(source, 40, budget=budget)
    assert Decimal(str(value)) == (7 if type(source) is int else Decimal("0.125"))
    assert budget.remaining_steps < 1 << 45


def test_float_same_precision_shortcuts_cannot_bypass_cancellation():
    for source in (s.Float(0.125), core._ExactDecimalFloat("0.125", 40)):
        budget = ample_budget()
        budget.cancel()
        before = remaining(budget)
        for method in (source.evalf, source.n):
            with pytest.raises(s.EvaluationCancelled):
                method(source.dps, budget=budget)
        assert remaining(budget) == before


def test_decimal_calls_and_compiled_evaluation_compose_without_replenishing_fuel():
    x = s.Symbol("shared_decimal_budget_x")
    program = s.compile_numeric(x, x + 1)
    probe = ample_budget()
    s.N(s.I, 30, budget=probe)
    decimal_cost = (1 << 45) - probe.remaining_steps
    before = probe.remaining_steps
    assert program(2, budget=probe) == 3
    program_cost = before - probe.remaining_steps
    budget = ample_budget(max_steps=decimal_cost + program_cost)
    assert components(s.I.n(30, budget=budget)) == (0, 1)
    assert program(2, budget=budget) == 3
    assert budget.remaining_steps == 0
    for operation in (lambda: s.N(s.I, 30, budget=budget), lambda: program(2, budget=budget)):
        with pytest.raises(s.EvaluationBudgetExceeded):
            operation()
    assert budget.remaining_steps == 0


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_every_exhausted_dimension_refuses_before_any_source_publication(dimension):
    budget = ample_budget(**{dimension: 0})
    before = remaining(budget)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.N(s.log(0, evaluate=False), 40, budget=budget)
    assert remaining(budget) == before


def test_domain_failure_keeps_spent_work_and_cannot_hide_held_zero_factors():
    budget = ample_budget()
    source = s.Mul(0, s.log(0, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.N(source, 40, budget=budget)
    after_failure = budget.remaining_steps
    assert after_failure < 1 << 45
    assert components(s.N(s.I, 40, budget=budget)) == (0, 1)
    assert budget.remaining_steps < after_failure


def test_cancellation_precedes_precision_validation_and_unknown_source_hooks():
    calls = []
    class Hostile:
        @property
        def args(self):
            calls.append("args")
            raise AssertionError
        def evalf(self, *args, **kwargs):
            calls.append("evalf")
            raise AssertionError
        def __float__(self):
            calls.append("float")
            raise AssertionError
        def __str__(self):
            calls.append("str")
            raise AssertionError
    source = Hostile()
    budget = ample_budget()
    budget.cancel()
    before = remaining(budget)
    with pytest.raises(s.EvaluationCancelled):
        s.N(source, 0, budget=budget)
    assert remaining(budget) == before
    with pytest.raises(NotImplementedError):
        s.N(source, 40, budget=ample_budget())
    assert calls == []


def test_custom_evalf_hooks_and_algebraic_wrappers_are_explicitly_outside_the_lane(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported source entered native decimal evaluation")
    monkeypatch.setattr(core._native, "evalf_complex_decimal_native", forbidden)
    class Custom(s.Function):
        def _eval_evalf(self, prec):
            calls.append(prec)
            return s.Integer(1)
    source = Custom(1, evaluate=False)
    with pytest.raises(NotImplementedError):
        s.N(source, 40, budget=ample_budget())
    x = s.Symbol("budgeted_algebraic_root_x")
    root = s.CRootOf(x*x - 2, 1)
    for source in (root, s.Mul(0, root, evaluate=False)):
        with pytest.raises(NotImplementedError):
            s.N(source, 40, budget=ample_budget())
    assert calls == []


@pytest.mark.parametrize("precision, exception", [(True, TypeError), (1.0, TypeError), (0, ValueError), (1001, ValueError)])
def test_budgeted_precision_has_a_bounded_exact_integer_contract(precision, exception):
    with pytest.raises(exception):
        s.N(s.I, precision, budget=ample_budget())


def test_budget_and_integer_subclasses_cannot_supply_conversion_or_charge_hooks():
    calls = []
    class FakeBudget:
        def _checkpoint(self):
            calls.append("checkpoint")
    class CustomInt(int):
        def __index__(self):
            calls.append("index")
            return 40
    with pytest.raises(TypeError):
        s.N(s.I, 40, budget=FakeBudget())
    with pytest.raises(TypeError):
        s.N(s.I, CustomInt(40), budget=ample_budget())
    with pytest.raises(NotImplementedError):
        s.N(CustomInt(2), 40, budget=ample_budget())
    assert calls == []


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, MemoryError, s.EvaluationBudgetExceeded, s.EvaluationCancelled])
def test_native_failure_is_terminal_without_a_replacement_ledger(monkeypatch, failure):
    calls = []
    budget = ample_budget()
    def refuse(source, digits, *, budget):
        calls.append(budget)
        raise failure("terminal decimal refusal")
    monkeypatch.setattr(core._native, "evalf_complex_decimal_native", refuse)
    with pytest.raises(failure, match="terminal decimal refusal"):
        s.N(s.I, 40, budget=budget)
    assert calls == [budget]
    assert budget.remaining_steps < 1 << 45


def test_cancellation_after_output_assembly_prevents_publication(monkeypatch):
    budget = ample_budget()
    original = core._assemble_decimal_pair
    calls = []
    def cancel_after_assembly(value, digits):
        result = original(value, digits)
        calls.append(result)
        budget.cancel()
        return result
    monkeypatch.setattr(core, "_assemble_decimal_pair", cancel_after_assembly)
    with pytest.raises(s.EvaluationCancelled):
        s.N(s.I, 40, budget=budget)
    assert len(calls) == 1 and budget.cancelled


def test_mutation_after_output_assembly_is_rejected(monkeypatch):
    source = core._ExactDecimalFloat("0.125", 40)
    slot = core._ExactDecimalFloat.__dict__["_decimal"]
    original = core._assemble_decimal_pair
    def mutate_after_assembly(value, digits):
        result = original(value, digits)
        slot.__set__(source, "0.25")
        return result
    monkeypatch.setattr(core, "_assemble_decimal_pair", mutate_after_assembly)
    with pytest.raises(RuntimeError, match="mutated"):
        source.evalf(40, budget=ample_budget())


def test_borrowed_native_cost_counts_original_nodes_and_charges_only_inspection():
    source = core._native.py_rational(1 << 4000, 3)
    budget = ample_budget()
    before = remaining(budget)
    nodes, size, allocations = core._native.CompiledFunction._evalf_source_cost(source, budget=budget)
    assert nodes == 1 and size > 500 and allocations == 2
    consumed = tuple(a-b for a, b in zip(before, remaining(budget)))
    assert consumed[0] >= nodes and 0 < consumed[1] < size and consumed[2] > 0
    held = core._native.Add(source, core._native_expr(s.I), evaluate=False).as_expr()
    larger = core._native.CompiledFunction._evalf_source_cost(held, budget=budget)
    assert larger[0] == 3 and larger[1] > size and larger[2] > allocations


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_borrowed_native_cost_uses_the_same_operational_budget_errors(dimension):
    with pytest.raises(s.EvaluationBudgetExceeded):
        core._native.CompiledFunction._evalf_source_cost(
            core._native_expr(s.I), budget=ample_budget(**{dimension: 0}),
        )


def test_borrowed_native_cost_refuses_oversized_literals_names_and_depth():
    budget = ample_budget()
    deep = core._native_expr(s.I)
    one = core._native_expr(s.Integer(1))
    for _ in range(130):
        deep = core._native.Pow(deep, one, evaluate=False).as_expr()
    for source in (core._native.py_integer(1 << 4096),
                   core._native.py_function("x" * 4097, one), deep):
        with pytest.raises(ValueError):
            core._native.CompiledFunction._evalf_source_cost(source, budget=budget)
    with pytest.raises(TypeError):
        core._native.CompiledFunction._evalf_source_cost(object(), budget=budget)
    budget.cancel()
    before = remaining(budget)
    with pytest.raises(s.EvaluationCancelled):
        core._native.CompiledFunction._evalf_source_cost(object(), budget=budget)
    assert remaining(budget) == before


@pytest.mark.parametrize("method", ["evalf", "n"])
def test_algebraic_root_methods_refuse_explicit_budgets_before_unmetered_work(method, monkeypatch):
    x = s.Symbol("direct_budgeted_root_x")
    root = s.CRootOf(x*x - 2, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported root entered native decimal evaluation")
    monkeypatch.setattr(core._native, "evalf_complex_decimal_native", forbidden)
    budget = ample_budget()
    with pytest.raises(NotImplementedError):
        getattr(root, method)(40, budget=budget)
    assert budget.remaining_steps < 1 << 45
    budget.cancel()
    before = remaining(budget)
    # Cancellation is observed even before invalid precision can be inspected.
    with pytest.raises(s.EvaluationCancelled):
        getattr(root, method)(0, budget=budget)
    assert remaining(budget) == before
