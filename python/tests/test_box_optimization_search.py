"""Resumable global candidates require source binding and independent finish."""

from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def ample_budget(**changes):
    limits = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def remaining(budget):
    return budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations


def identities(variables):
    return tuple((value.name, value.plain, None if value.plain else value.identity_hex)
                 for value in variables)


def stats(progress):
    return (progress.status, progress.leaf_count, progress.split_count,
            progress.evaluation_count, progress.ready_for_verification)


def complete(search, batch=4):
    for _ in range(1024):
        if search.progress.ready_for_verification:
            break
        search.advance(batch)
    assert search.progress.ready_for_verification
    result = search.finish()
    assert type(result) is s.BoxOptimizationReceipt
    assert 0 <= result.gap <= result.absolute_tolerance
    return result


@pytest.fixture(scope="module")
def partial_checkpoint():
    x = s.Symbol("resumable_global_x")
    program = s.compile_objective((x*x-2)**2, x)
    search = program.begin_box_optimization(
        [(-2, 2)], tolerance=Fraction(1, 1000), precision=16,
        max_leaves=256, budget=ample_budget(),
    )
    search.advance(1)
    assert not search.progress.ready_for_verification
    return x, program, search.checkpoint(), stats(search.progress)


def restore_fixture(fixture, **changes):
    _, program, data, _ = fixture
    options = dict(tolerance=Fraction(1, 1000), precision=16,
                   max_leaves=256, budget=ample_budget())
    options.update(changes)
    return program.restore_box_optimization(data, [(-2, 2)], **options)


def test_partial_global_search_resumes_and_matches_the_blocking_checked_cover(partial_checkpoint):
    x, program, data, saved_stats = partial_checkpoint
    search = restore_fixture(partial_checkpoint)
    assert type(search) is s.BoxOptimizationSearch
    assert type(search.progress) is s.BoxOptimizationProgress
    # Restoration re-evaluates the frontier, so evaluation counts can grow.
    assert stats(search.progress)[:3] == saved_stats[:3]
    assert search.progress.evaluation_count >= saved_stats[3]
    assert search.objective == (x*x-2)**2
    assert identities(search.variables) == identities((_symbol_binding(x),))
    assert search.domain == ((-2, 2),) and search.kind == "minimum"
    assert search.absolute_tolerance == s.Rational(1, 1000)
    assert search.precision_digits == 16 and search.max_leaves == 256
    assert search.max_advance_splits == 1024
    assert search.max_checkpoint_bytes == 1024*1024
    assert type(data) is bytes and len(data) < search.max_checkpoint_bytes
    with pytest.raises(ValueError, match="ready|complete|pending"):
        search.finish()
    old = search.progress
    old_stats = stats(old)
    new = search.advance(2)
    assert old.split_count <= new.split_count <= old.split_count+2
    assert stats(old) == old_stats
    result = complete(search, batch=3)
    blocking = program.optimize_box_certified(
        [(-2, 2)], tolerance=Fraction(1, 1000), precision=16,
        max_leaves=256, budget=ample_budget(),
    )
    assert result.point == blocking.point
    assert result.value == blocking.value
    assert result.point_value == blocking.point_value
    assert result.gap == blocking.gap
    assert result.leaf_count == blocking.leaf_count
    assert result.value[0] <= 0 <= result.value[1]


@pytest.mark.parametrize("kind,point,value", [
    ("minimum", (-1, 4), -9), ("maximum", (2, -3), 8),
])
def test_immediately_ready_boundary_search_still_needs_independent_finish(kind, point, value):
    x, y = s.symbols("resumable_affine_x resumable_affine_y")
    search = s.compile_objective(x-2*y, ((x,), y)).begin_box_optimization(
        [(-1, 2), (-3, 4)], kind=kind,
    )
    assert search.progress.status == "ready_for_verification"
    assert search.progress.ready_for_verification
    assert not hasattr(search, "value") and not hasattr(search, "gap")
    assert not hasattr(search.progress, "value") and not hasattr(search.progress, "claim")
    result = search.finish()
    assert result.point == point and result.value == (value, value)
    assert result.kind == kind and result.gap == 0


def test_source_only_absolute_value_corner_survives_checkpoint_and_resume():
    x = s.Symbol("resumable_absolute_x", real=True)
    source = s.Abs(x-s.Rational(1, 3))
    search = s.begin_box_optimization(
        source, x, [(-2, 2)], tolerance=Fraction(1, 1000), precision=16,
        budget=ample_budget(),
    )
    restored = s.restore_box_optimization(
        search.checkpoint(), objective=source, args=x, box=[(-2, 2)],
        tolerance=Fraction(1, 1000), precision=16, budget=ample_budget(),
    )
    result = complete(restored)
    assert result.objective == source
    assert result.value[0] <= 0 <= result.value[1]
    assert result.gap <= s.Rational(1, 1000)
    assert not hasattr(result, "stationary_enclosure")


def test_fixed_coordinates_and_flat_objectives_retain_the_exact_box():
    x, y = s.symbols("resumable_fixed_x resumable_fixed_y")
    search = s.begin_box_optimization(x+y, (x, y), [(2, 2), (3, 3)])
    assert search.progress.ready_for_verification
    result = complete(search)
    assert result.domain == ((2, 2), (3, 3))
    assert result.point == (2, 3) and result.value == (5, 5)


@pytest.mark.parametrize("changed", [
    "source", "typed_axis", "domain", "kind", "tolerance", "precision", "max_leaves",
])
def test_restore_binds_every_part_of_the_receivers_expected_statement(partial_checkpoint, changed):
    x, program, data, _ = partial_checkpoint
    box = [(-2, 2)]
    options = dict(kind="minimum", tolerance=Fraction(1, 1000), precision=16,
                   max_leaves=256, budget=ample_budget())
    if changed == "source":
        program = s.compile_objective((x*x-3)**2, x)
    elif changed == "typed_axis":
        x = s.Symbol(x.name, real=True)
        program = s.compile_objective((x*x-2)**2, x)
    elif changed == "domain":
        box = [(-3, 3)]
    elif changed == "kind":
        options["kind"] = "maximum"
    elif changed == "tolerance":
        options["tolerance"] = Fraction(1, 100)
    elif changed == "precision":
        options["precision"] = 17
    else:
        options["max_leaves"] = 257
    with pytest.raises(ValueError):
        program.restore_box_optimization(data, box, **options)


def test_coordinate_order_and_captured_assumptions_are_not_recaptured():
    positive = s.Symbol("resumable_typed_axis", positive=True)
    real = s.Symbol("resumable_typed_axis", real=True)
    source = (positive-1)**2+(real-2)**2
    forward = s.compile_objective(source, (positive, real))
    reverse = s.compile_objective(source, (real, positive))
    box = [[Fraction(1, 2), 3], [Fraction(1, 2), 3]]
    old_facts = dict(positive._assumptions)
    try:
        positive._assumptions.clear()
        positive._assumptions["negative"] = True
        search = forward.begin_box_optimization(box, precision=16, budget=ample_budget())
        data = search.checkpoint()
        box[0][:] = [-3, -1]
        assert search.domain == ((s.Rational(1, 2), 3),)*2
        with pytest.raises(ValueError):
            reverse.restore_box_optimization(
                data, [(Fraction(1, 2), 3)]*2, precision=16, budget=ample_budget(),
            )
        restored = forward.restore_box_optimization(
            data, [(Fraction(1, 2), 3)]*2, precision=16, budget=ample_budget(),
        )
        assert identities(restored.variables) == identities(search.variables)
        result = complete(restored)
        assert result.value[0] <= 0 <= result.value[1]
        with pytest.raises(ValueError, match="positive"):
            forward.begin_box_optimization([(-1, 3), (1, 3)])
    finally:
        positive._assumptions.clear()
        positive._assumptions.update(old_facts)


@pytest.mark.parametrize("damage", [
    "truncated", "noncanonical", "unknown_field", "version", "context",
    "understated_counters", "exhausted_counters", "missing_cover", "wrong_order",
])
def test_corrupted_checkpoint_bytes_never_grant_acceptance(partial_checkpoint, damage):
    _, program, data, _ = partial_checkpoint
    if damage == "truncated":
        data = data[:-1]
    elif damage == "noncanonical":
        data += b"\n"
    else:
        wire = json.loads(data)
        if damage == "unknown_field":
            wire["accepted"] = True
        elif damage == "version":
            wire["schema_version"] += 1
        elif damage == "context":
            wire["context_digest"][0] ^= 1
        elif damage == "understated_counters":
            wire["counters"] = dict.fromkeys(wire["counters"], 0)
        elif damage == "exhausted_counters":
            wire["counters"]["steps"] = (1 << 64)-1
        elif damage == "missing_cover":
            wire["partition"] = []
        else:
            wire["split_order"] = [1 << 63]
        data = json.dumps(wire, separators=(",", ":")).encode()
    with pytest.raises((ValueError, s.EvaluationBudgetExceeded)):
        program.restore_box_optimization(
            data, [(-2, 2)], tolerance=Fraction(1, 1000), precision=16,
            max_leaves=256, budget=ample_budget(),
        )


def test_checkpoint_restore_and_finish_spend_the_same_live_ledger(partial_checkpoint):
    ledger = ample_budget()
    search = restore_fixture(partial_checkpoint, budget=ledger)
    assert search.budget is ledger
    before = remaining(ledger)
    first = search.checkpoint()
    first_counters = json.loads(first)["counters"]
    second = search.checkpoint()
    second_counters = json.loads(second)["counters"]
    assert all(second_counters[key] >= value for key, value in first_counters.items())
    assert second_counters["steps"] > first_counters["steps"]
    assert all(new <= old for new, old in zip(remaining(ledger), before))
    assert remaining(ledger) != before
    _, program, _, _ = partial_checkpoint
    before = remaining(ledger)
    restored = program.restore_box_optimization(
        second, [(-2, 2)], tolerance=Fraction(1, 1000), precision=16,
        max_leaves=256, budget=ledger,
    )
    assert restored.budget is ledger
    assert remaining(ledger) != before
    third = json.loads(restored.checkpoint())["counters"]
    assert all(third[key] >= value for key, value in second_counters.items())
    assert third["steps"] > second_counters["steps"]
    complete(restored)
    assert restored.budget is ledger


def test_private_budget_is_retained_and_cannot_be_replaced():
    x = s.Symbol("resumable_private_global_x")
    search = s.compile_objective(x, x).begin_box_optimization([(-1, 1)])
    ledger = search.budget
    assert type(ledger) is s.EvaluationBudget
    before = remaining(ledger)
    assert search.finish().value == (-1, -1)
    assert search.budget is ledger and remaining(ledger) != before
    for method in (search.advance, search.checkpoint, search.finish):
        with pytest.raises(TypeError):
            method(budget=ample_budget())


@pytest.mark.parametrize("refusal", ["cancelled", "exhausted"])
def test_ready_candidates_cannot_publish_on_a_refused_ledger(refusal):
    x = s.Symbol("resumable_global_refusal_x")
    ledger = ample_budget()
    search = s.compile_objective(x, x).begin_box_optimization([(-1, 1)], budget=ledger)
    before = stats(search.progress)
    if refusal == "cancelled":
        ledger.cancel()
        expected = s.EvaluationCancelled
    else:
        ledger._charge(ledger.remaining_steps)
        expected = s.EvaluationBudgetExceeded
    for method in (search.finish, search.checkpoint):
        with pytest.raises(expected):
            method()
    if refusal == "cancelled":
        with pytest.raises(s.EvaluationCancelled):
            search.advance()
    else:
        # A ready search can report existing progress without spending work.
        # It still cannot publish a receipt or encode a new checkpoint.
        assert stats(search.advance()) == before
    assert stats(search.progress) == before


def test_native_search_failure_is_terminal_and_previous_checkpoint_remains_a_candidate():
    x = s.Symbol("resumable_terminal_global_x")
    program = s.compile_objective((x*x-2)**2, x)
    search = program.begin_box_optimization(
        [(-2, 2)], tolerance=Fraction(1, 1000), max_leaves=2, budget=ample_budget(),
    )
    saved = search.checkpoint()
    with pytest.raises(ValueError):
        search.advance()
    failed = stats(search.progress)
    assert failed[0] == "failed" and not failed[-1]
    for method in (search.advance, search.finish, search.checkpoint):
        with pytest.raises(ValueError):
            method()
        assert stats(search.progress) == failed
    restored = program.restore_box_optimization(
        saved, [(-2, 2)], tolerance=Fraction(1, 1000), max_leaves=2, budget=ample_budget(),
    )
    assert restored.progress.status == "searching"
    with pytest.raises(ValueError):
        restored.finish()


def test_held_original_domain_holes_cannot_be_erased_by_restore():
    x = s.Symbol("resumable_global_hole_x")
    source = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    search = s.begin_box_optimization(source, x, [(-1, 1)], max_leaves=4, budget=ample_budget())
    restored = s.restore_box_optimization(
        search.checkpoint(), objective=source, args=x, box=[(-1, 1)],
        max_leaves=4, budget=ample_budget(),
    )
    with pytest.raises(ValueError):
        restored.advance(1024)
    assert not restored.progress.ready_for_verification
    with pytest.raises(ValueError):
        restored.finish()


def test_cancellation_after_native_acceptance_still_prevents_publication(monkeypatch):
    x = s.Symbol("resumable_global_publication_x")
    ledger = ample_budget()
    search = s.compile_objective(x, x).begin_box_optimization([(-1, 1)], budget=ledger)
    module = importlib.import_module("sympy.utilities._box_optimization_search")
    original = module.BoxOptimizationReceipt

    def cancel_after_wrapping(receipt):
        result = original(receipt)
        ledger.cancel()
        return result

    monkeypatch.setattr(module, "BoxOptimizationReceipt", cancel_after_wrapping)
    with pytest.raises(s.EvaluationCancelled):
        search.finish()


@pytest.mark.parametrize("max_splits", [0, 1025, -1, 10**100, True, 1.0])
def test_advance_requires_an_exact_bounded_builtin_integer(max_splits, partial_checkpoint):
    search = restore_fixture(partial_checkpoint)
    before = stats(search.progress)
    with pytest.raises(ValueError, match="max_splits"):
        search.advance(max_splits)
    assert stats(search.progress) == before


def test_checkpoint_input_admission_precedes_untrusted_source_or_box_hooks(partial_checkpoint):
    _, program, data, _ = partial_checkpoint
    calls = []

    class BytesSubclass(bytes):
        pass

    class Source:
        def _sympy_(self):
            calls.append("source")
            raise AssertionError("checkpoint admission must come first")

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("checkpoint admission must come first")

    for invalid in ("{}", bytearray(data), memoryview(data), BytesSubclass(data), object()):
        with pytest.raises(TypeError, match="bytes"):
            program.restore_box_optimization(invalid, Source())
        with pytest.raises(TypeError, match="bytes"):
            s.restore_box_optimization(invalid, objective=Source(), args=Source(), box=Source())
    with pytest.raises(ValueError, match="1 MiB"):
        program.restore_box_optimization(b" "*(1024*1024+1), Source())
    with pytest.raises(ValueError):
        program.restore_box_optimization(b"\x80\x04N.", [(-2, 2)])
    assert calls == []


def test_begin_and_restore_keep_exact_source_and_option_admission(partial_checkpoint):
    x, program, data, _ = partial_checkpoint
    floating = s.compile_objective(s.Add(x, s.Float(0.5), evaluate=False), x)
    for method, args in ((floating.begin_box_optimization, ([(-2, 2)],)),
                         (floating.restore_box_optimization, (data, [(-2, 2)]))):
        with pytest.raises(ValueError, match="exact source"):
            method(*args)
    for invalid in (True, 16.0, 7, 129):
        with pytest.raises(ValueError, match="precision"):
            program.begin_box_optimization([(-2, 2)], precision=invalid)
    for invalid in (True, 1.0, 0, 1025):
        with pytest.raises(ValueError, match="max_leaves"):
            program.restore_box_optimization(data, [(-2, 2)], max_leaves=invalid)
    for invalid in (0, -1, Fraction(-1, 2), 0.1):
        with pytest.raises((ValueError, TypeError)):
            program.begin_box_optimization([(-2, 2)], tolerance=invalid)


def test_search_and_progress_are_immutable_exact_native_owners():
    from sympy.utilities import BoxOptimizationProgress, BoxOptimizationSearch

    assert BoxOptimizationSearch is s.BoxOptimizationSearch
    assert BoxOptimizationProgress is s.BoxOptimizationProgress
    x = s.Symbol("resumable_global_immutable_x")
    search = s.begin_box_optimization(x, x, [(-1, 1)])
    progress = search.progress
    for value, field in ((search, "_search"), (progress, "_progress")):
        with pytest.raises(AttributeError):
            setattr(value, field, object())
        with pytest.raises(AttributeError):
            delattr(value, field)
        with pytest.raises(AttributeError):
            value.__init__(getattr(value, field))
    for constructor in (BoxOptimizationSearch, BoxOptimizationProgress):
        with pytest.raises(TypeError):
            constructor(object())
    for constructor in (_native.BoxOptimizationSearch, _native.BoxOptimizationProgress):
        with pytest.raises(TypeError):
            constructor()
    assert "status" in repr(progress) and "BoxOptimizationSearch" in repr(search)


def test_native_source_and_compiled_entries_share_canonical_candidates():
    x = s.Symbol("resumable_global_native_x")
    source = _native.SourceObjective(_native_expr(x), [_symbol_binding(x)])
    compiled = s.compile_objective(x, x).native_objective
    box = [(_native_expr(-1), _native_expr(1))]
    tolerance = _native_expr(s.Rational(1, 1000))
    search = source.begin_box_optimization(box, tolerance, "minimum", 16, 1)
    assert type(search) is _native.BoxOptimizationSearch
    with pytest.raises(TypeError):
        search.advance(True)
    restored = compiled.restore_box_optimization(
        search.checkpoint(), box, tolerance, "minimum", 16, 1, budget=ample_budget(),
    )
    assert restored.progress.ready_for_verification
    assert type(restored.finish()) is _native.BoxOptimizationReceipt


def test_checkpoint_resumes_in_a_fresh_python_process(partial_checkpoint):
    _, _, data, _ = partial_checkpoint
    shell_root = Path(s.__file__).resolve().parents[1]
    code = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])
native = importlib.util.module_from_spec(spec)
sys.modules[sys.argv[1]] = native
spec.loader.exec_module(native)
sys.path.insert(0, sys.argv[3])
import sympy as s
from sympy.core import _native
assert _native is native
assert Path(s.__file__).resolve() == Path(sys.argv[3]) / 'sympy' / '__init__.py'
x = s.Symbol('resumable_global_x')
program = s.compile_objective((x*x-2)**2, x)
budget = s.EvaluationBudget(max_steps=1<<45, max_memory_bytes=1<<40, max_allocations=1<<30)
search = program.restore_box_optimization(sys.stdin.buffer.read(), [(-2, 2)],
    tolerance=s.Rational(1, 1000), precision=16, max_leaves=256, budget=budget)
assert not search.progress.ready_for_verification
for _ in range(1024):
    if search.progress.ready_for_verification:
        break
    search.advance(4)
receipt = search.finish()
assert receipt.value[0] <= 0 <= receipt.value[1]
assert 0 <= receipt.gap <= s.Rational(1, 1000)
print(receipt.kind)
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, _native.__name__, _native.__file__, str(shell_root)],
        input=data, capture_output=True, timeout=90,
    )
    assert process.returncode == 0, process.stdout.decode()+process.stderr.decode()
    assert process.stdout.strip() == b"minimum"
