"""Actual native fragments compose with supervised, non-replayed Python effects."""

import importlib
import math
import pickle
import threading
import time
import warnings

import pytest
import sympy as s

from sympy.core import FunctionClass, SupervisedCallbackSession, _native, supervised_callback_session


def _budget(**changes):
    options = dict(max_steps=1 << 30, max_memory_bytes=1 << 30, max_allocations=1 << 20)
    options.update(changes)
    return s.EvaluationBudget(**options)


def test_mixed_native_fragments_capture_callbacks_and_publish_explicit_provenance():
    x, y = s.symbols("hybrid_x hybrid_y")
    calls = []

    class Law(s.Function):
        nargs = 1

    def law(value):
        calls.append(value)
        return value**3

    callbacks = {Law: law}
    source = s.Add(s.Pow(x, 2, evaluate=False), Law(x, evaluate=False), s.sin(y), evaluate=False)
    program = s.compile_numeric_hybrid((x, y), source, callbacks=callbacks)
    callbacks[Law] = lambda value: 999
    Law._imp_ = lambda value: 999
    assert calls == []
    assert program(2, 0) == 12.0
    assert calls == [2.0]
    assert program.variables == (x, y)
    assert program.callback_slots == ((0, Law, law),)
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.native_fragment_count >= 3
    assert program.operation_count > 0
    with pytest.raises(AttributeError):
        program._actions = ()
    with pytest.raises(TypeError):
        pickle.dumps(program)
    for name in ("enclose", "certify_root_box", "solve_certified", "to_bytes", "native_system"):
        assert not hasattr(program, name)


def test_every_repeated_occurrence_and_invocation_executes_without_purity_memoization():
    x = s.Symbol("hybrid_occurrence_x")
    calls = []

    class Counter(s.Function):
        nargs = 1

    def count(value):
        calls.append(value)
        return 10 * len(calls) + value

    node = Counter(x, evaluate=False)
    program = s.compile_numeric_hybrid(x, s.Add(node, node, evaluate=False), callbacks={Counter: count})
    with supervised_callback_session() as session:
        assert program(1) == 32
        assert program(1, session=session) == 72
        assert session.invocations == 4
        assert session.cached_effect_count() == 0
        successful = [entry for entry in session.trace() if entry[3] == "ok"]
        assert len(successful) == 4
        assert len({entry[2] for entry in successful}) == 4
    assert calls == [1.0] * 4


def test_nested_callbacks_keep_stored_child_order_and_exact_class_slots():
    x = s.Symbol("hybrid_nested_x")
    calls = []

    class Inner(s.Function):
        nargs = 1

    class Outer(s.Function):
        nargs = 2

    def inner(value):
        calls.append(("inner", value))
        return value + 1

    def outer(left, right):
        calls.append(("outer", left, right))
        return 10 * left + right

    source = Outer(Inner(x, evaluate=False), Inner(x, evaluate=False), evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Inner: inner, Outer: outer})
    assert program(2) == 33
    assert calls == [("inner", 2.0), ("inner", 2.0), ("outer", 3.0, 3.0)]
    assert tuple(slot[1] for slot in program.callback_slots) == (Inner, Inner, Outer)


@pytest.mark.parametrize("held_zero", [False, True])
def test_earlier_native_domain_failure_prevents_a_later_callback(held_zero):
    x = s.Symbol("hybrid_domain_x")
    calls = []

    class Law(s.Function):
        pass

    children = (s.log(x), Law(x, evaluate=False))
    source = s.Mul(0, *children, evaluate=False) if held_zero else s.Add(*children, evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: lambda value: calls.append(value) or 1})
    with pytest.raises(ValueError):
        program(-1)
    assert calls == []


def test_a_later_domain_failure_does_not_rollback_an_already_completed_effect():
    x = s.Symbol("hybrid_later_domain_x")
    calls = []

    class Law(s.Function):
        pass

    source = s.Add(Law(x, evaluate=False), s.log(x), evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: lambda value: calls.append(value) or 1})
    with pytest.raises(ValueError):
        program(-1)
    assert calls == [-1.0]


@pytest.mark.parametrize("exponent, expected", [(2, 4.0), (-1, -0.5), (0, 1.0)])
def test_literal_power_combiner_preserves_the_original_negative_base_semantics(exponent, expected):
    x = s.Symbol("hybrid_literal_power_x")

    class Law(s.Function):
        pass

    source = s.Pow(Law(x, evaluate=False), exponent, evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: lambda value: -2})
    assert program(1) == expected


def test_variable_power_retains_positive_base_guard_and_large_literal_refuses_before_effects():
    x, y = s.symbols("hybrid_power_x hybrid_power_y")
    calls = []

    class Law(s.Function):
        pass

    def negative(value):
        calls.append(value)
        return -2

    node = Law(x, evaluate=False)
    dynamic = s.compile_numeric_hybrid((x, y), s.Pow(node, y, evaluate=False), callbacks={Law: negative})
    with pytest.raises(ValueError):
        dynamic(1, 2)
    assert calls == [1]
    calls.clear()
    with pytest.raises(ValueError):
        s.compile_numeric_hybrid(x, s.Pow(node, 1 << 53, evaluate=False), callbacks={Law: negative})
    assert calls == []


def test_held_power_zero_cannot_hide_a_source_domain_failure():
    x = s.Symbol("hybrid_power_zero_x")
    calls = []

    class Law(s.Function):
        pass

    base = s.Add(s.log(x), Law(x, evaluate=False), evaluate=False)
    source = s.Pow(base, 0, evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: lambda value: calls.append(value) or 1})
    with pytest.raises(ValueError):
        program(0)
    assert calls == []


def test_same_printed_class_names_and_builtin_names_cannot_capture_each_other():
    x = s.Symbol("hybrid_identity_x")
    first = FunctionClass("SameName", (s.Function,), {"__module__": __name__})
    first_node = first(x, evaluate=False)
    second = FunctionClass("SameName", (s.Function,), {"__module__": __name__})
    second_node = second(x, evaluate=False)

    class sin(s.Function):
        pass

    source = [first_node, second_node, sin(x, evaluate=False), s.sin(x)]
    program = s.compile_numeric_hybrid(
        x, source, callbacks={first: lambda value: 11, second: lambda value: 22, sin: lambda value: 33},
    )
    assert program(0) == [11, 22, 33, 0]
    assert tuple(slot[1] for slot in program.callback_slots) == (first, second, sin)
    with pytest.raises(ValueError, match="built-in"):
        s.compile_numeric_hybrid(x, s.sin(x), callbacks={s.sin: lambda value: 44})


def test_custom_properties_printers_hash_and_float_hooks_are_not_used_during_compilation():
    x = s.Symbol("hybrid_hook_x")
    calls = []

    class Guard(s.Function):
        @property
        def args(self):
            calls.append("args")
            raise AssertionError("read stored Function arguments directly")

        def __str__(self):
            calls.append("str")
            raise AssertionError("names are not numerical IR")

        def __hash__(self):
            calls.append("hash")
            raise AssertionError("opaque nodes are not effect-cache keys")

        def __float__(self):
            calls.append("float")
            raise AssertionError("opaque nodes require explicit callbacks")

    node = Guard(x, evaluate=False)
    source = s.Add(1, node, evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Guard: lambda value: value + 1})
    assert calls == []
    assert program(2) == 4
    assert calls == []


def test_native_only_unknown_function_recovery_refuses_without_running_surface_hooks():
    x = s.Symbol("hybrid_erased_x")
    calls = []

    class Erased(s.Function):
        def __setattr__(self, name, value):
            calls.append(name)
            object.__setattr__(self, name, value)

    # This explicit native syntax contains a name, but no exact Python class.
    source = _native.py_function("Erased", _native.py_symbol("hybrid_erased_x"))
    with pytest.raises(NotImplementedError, match="retained exact Python class"):
        s.compile_numeric_hybrid(x, source, callbacks={Erased: lambda value: value})
    assert calls == []


@pytest.mark.parametrize("name", ["sin", "__fsymFloat"])
def test_native_only_same_name_intrinsic_is_refused_when_its_surface_identity_is_ambiguous(name):
    x = s.Symbol("hybrid_erased_sin_x")
    custom = FunctionClass(name, (s.Function,), {"__module__": __name__})
    argument = _native.py_symbol("hybrid_erased_sin_x") if name == "sin" else _native.py_integer(0)
    native_sine = _native.py_function(name, argument)
    with pytest.raises(NotImplementedError, match="retained exact Python class"):
        s.compile_numeric_hybrid(x, native_sine, callbacks={custom: lambda value: 99})


def test_matrix_snapshot_and_nested_argument_output_shapes_survive_source_mutation():
    x, y = s.symbols("hybrid_matrix_x hybrid_matrix_y")

    class Law(s.Function):
        pass

    matrix = s.Matrix([[Law(x, evaluate=False), s.Add(x, y, evaluate=False)]])
    source = (matrix, [y, []])
    program = s.compile_numeric_hybrid([(x, y)], source, callbacks={Law: lambda value: 2 * value})
    matrix[0, 0] = 999
    assert program((2, 3)) == ([[4, 5]], [3, []])
    with pytest.raises(TypeError):
        program(2, 3)


def test_custom_symbolic_jacobian_connects_to_one_mixed_residual_and_jacobian_evaluator():
    x, y = s.symbols("hybrid_campaign_x hybrid_campaign_y")
    effects = []

    class ConstitutiveLaw(s.Function):
        nargs = 1

        def fdiff(self, argindex=1):
            effects.append(("derivative", argindex))
            return 3 * self.args[0]**2

    residual = s.Matrix([
        s.Add(ConstitutiveLaw(x, evaluate=False), s.sin(y), evaluate=False),
        s.Add(x, y, evaluate=False),
    ])
    jacobian = residual.jacobian([x, y])
    symbolic_effects = tuple(effects)

    def constitutive(value):
        effects.append(("numeric", value))
        return value**3

    program = s.compile_numeric_hybrid(
        (x, y), (residual, jacobian), callbacks={ConstitutiveLaw: constitutive},
    )
    assert tuple(effects) == symbolic_effects
    assert program(2, 0) == ([[8], [2]], [[12, 1], [1, 1]])
    assert tuple(effects) == symbolic_effects + (("numeric", 2.0),)
    assert program.evidence == "approximate_with_user_callbacks"


def test_batch_uses_one_session_and_exposes_no_partial_row_list_after_later_failure():
    x = s.Symbol("hybrid_batch_x")
    calls = []

    class Law(s.Function):
        pass

    def law(value):
        calls.append(value)
        return value if value < 3 else float("nan")

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: law})
    sentinel = object()
    published = sentinel
    with supervised_callback_session() as session:
        with pytest.raises(ValueError, match="finite"):
            published = program.evaluate_batch([[1], [2], [3]])
        assert published is sentinel
        assert session.invocations == 3
        assert session.unknown_effect_tainted
        with pytest.raises(RuntimeError):
            program(1)
    assert calls == [1, 2, 3]


def test_batch_preflights_all_numeric_inputs_before_committing_any_effect():
    x = s.Symbol("hybrid_batch_input_x")
    calls = []

    class Law(s.Function):
        pass

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: lambda value: calls.append(value) or 1})
    with pytest.raises(ValueError, match="finite"):
        program.evaluate_batch([[1], [math.inf]])
    assert calls == []


@pytest.mark.parametrize("value, error", [
    (complex(1, 0), TypeError), ([1], TypeError), (s.Integer(1), TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (1 << 4096, ValueError), (1 << 1024, OverflowError),
])
def test_invalid_callback_returns_taint_the_session_without_retry(value, error):
    x = s.Symbol("hybrid_bad_output_x")
    calls = []

    class Law(s.Function):
        pass

    def invalid(argument):
        calls.append(argument)
        return value

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: invalid})
    with supervised_callback_session() as session:
        with pytest.raises(error):
            program(1)
        assert session.unknown_effect_tainted
        with pytest.raises(RuntimeError):
            program(1)
        assert session.current_reentrancy_depth == 0
    assert calls == [1]


def test_custom_float_conversion_never_runs_outside_or_inside_return_admission():
    x = s.Symbol("hybrid_output_hook_x")
    calls = []

    class Law(s.Function):
        pass

    class Result:
        def __float__(self):
            calls.append("float")
            return 1.0

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: lambda value: Result()})
    with pytest.raises(TypeError, match="exact built-in"):
        program(1)
    assert calls == []


def test_warning_and_original_exception_object_are_preserved():
    x = s.Symbol("hybrid_warning_x")

    class Law(s.Function):
        pass

    error = LookupError("original hybrid callback failure")

    def effect(value):
        warnings.warn("hybrid visible warning", UserWarning)
        raise error

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: effect})
    with warnings.catch_warnings(record=True) as records, supervised_callback_session() as session:
        warnings.simplefilter("always")
        with pytest.raises(LookupError) as caught:
            program(1)
        assert caught.value is error
        assert session.unknown_effect_tainted
    assert len(records) == 1
    assert records[0].category is UserWarning
    assert str(records[0].message) == "hybrid visible warning"


def test_cancellation_during_callback_result_admission_taints_and_never_publishes():
    x = s.Symbol("hybrid_cancel_x")
    budget, calls = _budget(), []

    class Law(s.Function):
        pass

    def cancel(value):
        calls.append(value)
        budget.cancel()
        return 42

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: cancel})
    with supervised_callback_session() as session:
        with pytest.raises(s.EvaluationCancelled):
            program(1, budget=budget)
        assert session.unknown_effect_tainted
        with pytest.raises(RuntimeError):
            program(1)
    assert calls == [1]


def test_one_budget_is_consumed_across_repeated_hybrid_invocations():
    x = s.Symbol("hybrid_ledger_x")
    calls = []

    class Law(s.Function):
        pass

    def effect(value):
        calls.append(value)
        return value + 1

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: effect})
    probe = _budget()
    assert program(1, budget=probe) == 2
    cost = (1 << 30) - probe.remaining_steps
    calls.clear()
    budget = _budget(max_steps=2 * cost)
    assert program(1, budget=budget) == 2
    assert program(1, budget=budget) == 2
    assert budget.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program(1, budget=budget)
    assert calls == [1, 1]


def test_reentrant_hybrid_evaluation_inherits_budget_and_owning_session():
    x = s.Symbol("hybrid_reenter_x")

    class Inner(s.Function):
        pass

    class Outer(s.Function):
        pass

    inner = s.compile_numeric_hybrid(x, Inner(x, evaluate=False), callbacks={Inner: lambda value: value + 1})
    outer = s.compile_numeric_hybrid(x, Outer(x, evaluate=False), callbacks={Outer: lambda value: inner(value) + 1})
    budget = _budget()
    with supervised_callback_session() as session:
        assert outer(1, budget=budget) == 3
        assert session.invocations == 2
        assert session.current_reentrancy_depth == 0
    assert budget.remaining_steps < 1 << 30
    budget.cancel()
    # The inherited ledger is request-scoped, not leaked into later calls.
    assert inner(1) == 2


def test_nested_hybrid_call_cannot_reset_the_active_budget():
    x = s.Symbol("hybrid_reset_x")

    class Inner(s.Function):
        pass

    class Outer(s.Function):
        pass

    inner = s.compile_numeric_hybrid(x, Inner(x, evaluate=False), callbacks={Inner: lambda value: value})
    replacement = _budget()
    outer = s.compile_numeric_hybrid(
        x, Outer(x, evaluate=False), callbacks={Outer: lambda value: inner(value, budget=replacement)},
    )
    with supervised_callback_session() as session:
        with pytest.raises(ValueError, match="active EvaluationBudget"):
            outer(1, budget=_budget())
        assert session.unknown_effect_tainted
        assert session.invocations == 1
    assert replacement.remaining_steps == 1 << 30


def test_caught_nested_supervision_failure_still_prevents_outer_publication():
    x = s.Symbol("hybrid_caught_x")

    class Inner(s.Function):
        pass

    class Outer(s.Function):
        pass

    inner = s.compile_numeric_hybrid(x, Inner(x, evaluate=False), callbacks={Inner: lambda value: []})

    def catches(value):
        try:
            inner(value)
        except TypeError:
            return 1
        raise AssertionError("invalid nested result must refuse")

    outer = s.compile_numeric_hybrid(x, Outer(x, evaluate=False), callbacks={Outer: catches})
    with supervised_callback_session() as session:
        with pytest.raises(RuntimeError, match="unknown effect"):
            outer(1)
        assert session.unknown_effect_tainted
        assert session.current_reentrancy_depth == 0


def test_cancelled_conflicting_and_wrong_thread_sessions_refuse_before_native_work():
    x = s.Symbol("hybrid_owner_x")

    class Law(s.Function):
        pass

    program = s.compile_numeric_hybrid(
        x, s.Add(x*x, Law(x, evaluate=False), evaluate=False), callbacks={Law: lambda value: value},
    )
    cancelled = SupervisedCallbackSession()
    cancelled.cancel()
    budget = _budget()
    with pytest.raises(RuntimeError, match="cancelled"):
        program(2, budget=budget, session=cancelled)
    assert budget.remaining_steps == 1 << 30
    other = SupervisedCallbackSession()
    with supervised_callback_session():
        with pytest.raises(ValueError, match="conflicts"):
            program(2, budget=budget, session=other)
    assert budget.remaining_steps == 1 << 30
    failures = []

    def foreign():
        try:
            program(2, budget=budget, session=other)
        except RuntimeError as error:
            failures.append(str(error))

    worker = threading.Thread(target=foreign)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(failures) == 1 and "ownership" in failures[0]
    assert other.invocations == 0
    assert budget.remaining_steps == 1 << 30


def test_an_active_duck_typed_session_cannot_replace_native_supervision(monkeypatch):
    core = importlib.import_module("sympy.core")
    x = s.Symbol("hybrid_fake_session_x")

    class Law(s.Function):
        pass

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: lambda value: value})
    touched = []

    class FakeSession:
        unknown_effect_tainted = False
        cancelled = False

        def verify_ownership(self):
            touched.append("ownership")

        def charge_nested_steps(self, steps):
            touched.append("charge")

        def invoke_fresh(self, kind, target, count, callback, args):
            touched.append("effect")
            return callback(*args)

    monkeypatch.setattr(core._callback_tls, "session", FakeSession(), raising=False)
    with pytest.raises(TypeError, match="exact SupervisedCallbackSession"):
        program(1)
    assert touched == []


def test_session_invocation_nested_work_and_wall_time_limits_are_live():
    x = s.Symbol("hybrid_limits_x")
    calls = []

    class Law(s.Function):
        pass

    program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: lambda value: calls.append(value) or value})
    with supervised_callback_session(max_invocations=1) as session:
        assert program(1) == 1
        with pytest.raises(ValueError, match="invocation"):
            program(1)
        assert session.unknown_effect_tainted
    assert calls == [1]
    with supervised_callback_session(max_nested_steps=0):
        with pytest.raises(ValueError, match="nested native"):
            program(2)
    assert calls == [1]

    def slow(value):
        time.sleep(0.02)
        calls.append(value)
        return value

    slow_program = s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={Law: slow})
    with supervised_callback_session(max_wall_time_ms=1) as session:
        with pytest.raises(RuntimeError, match="wall-time"):
            slow_program(3)
        assert session.unknown_effect_tainted
        assert session.current_reentrancy_depth == 0
    assert calls == [1, 3]


def test_native_source_admission_precedes_literal_expansion_and_custom_mapping_is_explicit():
    x = s.Symbol("hybrid_admission_x")

    class Law(s.Function):
        pass

    with pytest.raises(TypeError, match="exact built-in dict"):
        s.compile_numeric_hybrid(x, x, callbacks=[])
    with pytest.raises(TypeError, match="FunctionClass"):
        s.compile_numeric_hybrid(x, x, callbacks={"Law": lambda value: value})
    with pytest.raises(NotImplementedError, match="explicit exact-class"):
        s.compile_numeric_hybrid(x, Law(x, evaluate=False), callbacks={})
    huge = _native.py_integer(1 << 4096)
    with pytest.raises(ValueError, match="literal"):
        s.compile_numeric_hybrid((), huge, callbacks={})
    malformed_float = _native.py_function("__fsymFloat", huge)
    with pytest.raises(ValueError, match="literal"):
        s.compile_numeric_hybrid((), malformed_float, callbacks={})


@pytest.mark.parametrize("kind", ["symbol", "function"])
def test_native_identifier_limits_precede_name_cloning_and_graph_construction(kind):
    name = "n" * 4097
    source = (_native.py_symbol(name) if kind == "symbol"
              else _native.py_function(name, _native.py_integer(1)))
    with pytest.raises(ValueError, match="name|identifier"):
        s.compile_numeric_hybrid((), source, callbacks={})


@pytest.mark.parametrize("sparse", [False, True])
def test_native_only_matrix_literals_are_borrowed_checked_before_entry_snapshots(sparse):
    coefficient = s.Integer(1 << 4096)
    matrix = (s.SparseMatrix(1, 1, {(0, 0): coefficient}) if sparse
              else s.Matrix(_native.Matrix(1, 1, [_native.py_integer(1 << 4096)])))
    if not sparse:
        assert object.__getattribute__(matrix, "_surface") is None
    with pytest.raises(ValueError, match="literal"):
        s.compile_numeric_hybrid((), matrix, callbacks={})


def test_pure_native_source_and_floating_coefficients_remain_supported():
    x = s.Symbol("hybrid_pure_x")
    expression = s.Float(1.5)*x + s.sin(x)
    program = s.compile_numeric_hybrid(x, expression, callbacks={})
    assert program(2) == pytest.approx(3 + math.sin(2))
    assert program.callback_slots == ()
    assert program.evaluate_batch([[0], [2]]) == pytest.approx([0, 3 + math.sin(2)])


def test_public_exports_refer_to_the_same_explicit_preview_api():
    module = importlib.import_module("sympy.utilities.hybrid")
    from sympy.utilities import compile_numeric_hybrid

    assert s.compile_numeric_hybrid is compile_numeric_hybrid is module.compile_numeric_hybrid
