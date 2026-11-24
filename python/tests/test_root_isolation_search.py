"""Resumable candidates require source-bound replay and final independent proof."""

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


def progress_tuple(progress):
    return (progress.processed_cells, progress.pending_cells, progress.candidate_root_count,
            progress.excluded_cells, progress.leaf_count, progress.ready_for_verification)


def complete(search, batch=4):
    # The fixed search envelope has at most 2047 transitions. This assertion
    # prevents an erroneous nonprogressing continuation from hanging the test.
    for _ in range(2047):
        if search.progress.ready_for_verification:
            break
        search.advance(batch)
    assert search.progress.ready_for_verification
    result = search.finish()
    assert type(result) is s.RootIsolationReceipt
    return result


@pytest.fixture(scope="module")
def partial_checkpoint():
    x = s.Symbol("resumable_partial_x")
    program = s.compile_residual_system([x*x-1], x)
    search = program.begin_root_isolation(
        [(-2, 2)], precision=16, max_leaves=64, budget=ample_budget(),
    )
    progress = search.advance(1)
    assert progress.processed_cells == 1 and progress.pending_cells == 2
    assert not progress.ready_for_verification
    return x, program, search.checkpoint(), progress_tuple(progress)


def restore_fixture(partial_checkpoint, **changes):
    _, program, data, _ = partial_checkpoint
    options = dict(precision=16, max_leaves=64, budget=ample_budget())
    options.update(changes)
    return program.restore_root_isolation(data, [(-2, 2)], **options)


def test_partial_steps_resume_all_roots_and_match_a_blocking_complete_cover(partial_checkpoint):
    x, program, data, original_progress = partial_checkpoint
    search = restore_fixture(partial_checkpoint)
    assert type(search) is s.RootIsolationSearch
    assert type(search.progress) is s.RootIsolationProgress
    assert progress_tuple(search.progress) == original_progress
    assert search.residuals == (x*x-1,)
    assert identities(search.variables) == identities((_symbol_binding(x),))
    assert search.domain == ((-2, 2),)
    assert search.precision_digits == 16 and search.max_leaves == 64
    assert search.max_advance_cells == 2047
    assert search.max_checkpoint_bytes == 1024*1024
    assert type(data) is bytes and len(data) < search.max_checkpoint_bytes
    with pytest.raises(ValueError, match="ready|pending|complete"):
        search.finish()

    first = search.progress
    progress = search.advance(1)
    assert progress.processed_cells <= first.processed_cells + 1
    assert progress_tuple(first) == original_progress
    result = complete(search, batch=2)
    blocking = program.isolate_roots_certified(
        [(-2, 2)], precision=16, max_leaves=64, budget=ample_budget(),
    )
    assert result.root_count == blocking.root_count == 2
    assert result.domain == blocking.domain
    assert result.leaf_count == blocking.leaf_count
    assert result.excluded_leaf_count == blocking.excluded_leaf_count
    assert tuple(root.root_enclosure for root in result.roots) == tuple(
        root.root_enclosure for root in blocking.roots)
    for expected in (-1, 1):
        assert sum(bool(root.root_enclosure[0][0] <= expected <= root.root_enclosure[0][1])
                   for root in result.roots) == 1


def test_zero_candidate_roots_is_not_a_root_free_claim_until_finish():
    x = s.Symbol("resumable_empty_x")
    search = s.compile_residual_system([x*x+1], x).begin_root_isolation([(-2, 2)])
    initial = search.progress
    assert initial.candidate_root_count == 0 and initial.pending_cells == 1
    assert not initial.ready_for_verification
    with pytest.raises(ValueError):
        search.finish()
    progress = search.advance()
    assert progress.ready_for_verification and progress.candidate_root_count == 0
    assert not hasattr(progress, "root_count") and not hasattr(progress, "is_root_free")
    assert not hasattr(search, "root_count") and not hasattr(search, "roots")
    result = search.finish()
    assert result.is_root_free and result.root_count == 0 and result.roots == ()
    assert result.leaf_count == result.excluded_leaf_count == 1


def test_original_source_domain_and_captured_assumptions_survive_mutation():
    x = s.Symbol("resumable_owned_x", positive=True)
    y = s.Symbol("resumable_owned_y", real=True)
    binding = _symbol_binding(x)
    original_facts = dict(x._assumptions)
    source = s.Matrix([x-1, y-2])
    program = s.compile_residual_system(source, [(x,), y])
    box = [[Fraction(1, 2), 3], [0, 4]]
    try:
        source[0, 0] = x-100
        x._assumptions.clear()
        x._assumptions["negative"] = True
        search = program.begin_root_isolation(box, precision=16, budget=ample_budget())
        data = search.checkpoint()
        box[0][:] = [-3, -1]
        assert search.domain == ((s.Rational(1, 2), 3), (0, 4))
        assert search.residuals == (x-1, y-2)
        assert identities(search.variables[:1]) == identities((binding,))
        restored = program.restore_root_isolation(
            data, [(Fraction(1, 2), 3), (0, 4)], precision=16, budget=ample_budget(),
        )
        result = complete(restored)
        assert result.root_count == 1
        assert result.roots[0].root_enclosure == ((1, 1), (2, 2))
        with pytest.raises(ValueError, match="positive"):
            program.begin_root_isolation([(-1, 3), (0, 4)])
        with pytest.raises(ValueError, match="positive"):
            program.restore_root_isolation(data, [(-1, 3), (0, 4)], precision=16)
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)


@pytest.mark.parametrize("changed", ["source", "typed_axis", "domain", "precision", "max_leaves"])
def test_restore_requires_the_receivers_exact_expected_statement(partial_checkpoint, changed):
    x, program, data, _ = partial_checkpoint
    domain = [(-2, 2)]
    options = dict(precision=16, max_leaves=64, budget=ample_budget())
    if changed == "source":
        program = s.compile_residual_system([x*x-2], x)
    elif changed == "typed_axis":
        x = s.Symbol(x.name, real=True)
        program = s.compile_residual_system([x*x-1], x)
    elif changed == "domain":
        domain = [(-3, 3)]
    elif changed == "precision":
        options["precision"] = 17
    else:
        options["max_leaves"] = 65
    with pytest.raises(ValueError):
        program.restore_root_isolation(data, domain, **options)


def test_coordinate_order_and_same_name_typed_axes_are_not_remapped():
    positive = s.Symbol("resumable_same_name", positive=True)
    real = s.Symbol("resumable_same_name", real=True)
    sources = [positive-1, real-2]
    forward = s.compile_residual_system(sources, (positive, real))
    reversed_axes = s.compile_residual_system(sources, (real, positive))
    box = [(Fraction(1, 2), 3)]*2
    search = forward.begin_root_isolation(box, precision=16, budget=ample_budget())
    data = search.checkpoint()
    with pytest.raises(ValueError):
        reversed_axes.restore_root_isolation(data, box, precision=16, budget=ample_budget())
    restored = forward.restore_root_isolation(data, box, precision=16, budget=ample_budget())
    assert identities(restored.variables) == identities((_symbol_binding(positive), _symbol_binding(real)))
    assert complete(restored).roots[0].root_enclosure == ((1, 1), (2, 2))


@pytest.mark.parametrize("damage", [
    "truncated", "noncanonical", "unknown_field", "version", "context",
    "understated_counters", "exhausted_counters", "false_exclusion",
])
def test_corrupt_checkpoints_never_recover_progress_or_acceptance(partial_checkpoint, damage):
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
        else:
            # Canonical syntax alone cannot certify an exclusion of two roots.
            wire["witness"]["partition"] = ["exclude"]
        data = json.dumps(wire, separators=(",", ":")).encode()
    with pytest.raises((ValueError, s.EvaluationBudgetExceeded)):
        program.restore_root_isolation(
            data, [(-2, 2)], precision=16, max_leaves=64, budget=ample_budget(),
        )


def test_checkpoint_and_restore_retain_intrinsic_counters_and_the_live_ledger():
    x = s.Symbol("resumable_counters_x")
    program = s.compile_residual_system([x*x-1], x)
    ledger = ample_budget()
    before = remaining(ledger)
    search = program.begin_root_isolation(
        [(-2, 2)], precision=16, max_leaves=64, budget=ledger,
    )
    assert search.budget is ledger
    assert all(new <= old for new, old in zip(remaining(ledger), before))
    search.advance(1)
    first = search.checkpoint()
    first_counters = json.loads(first)["counters"]
    before = remaining(ledger)
    second = search.checkpoint()
    second_counters = json.loads(second)["counters"]
    assert second_counters["steps"] > first_counters["steps"]
    assert all(second_counters[key] >= value for key, value in first_counters.items())
    assert all(new <= old for new, old in zip(remaining(ledger), before))
    assert remaining(ledger) != before
    before = remaining(ledger)
    restored = program.restore_root_isolation(
        second, [(-2, 2)], precision=16, max_leaves=64, budget=ledger,
    )
    assert restored.budget is ledger and search.budget is ledger
    assert all(new <= old for new, old in zip(remaining(ledger), before))
    assert remaining(ledger) != before
    third_counters = json.loads(restored.checkpoint())["counters"]
    assert third_counters["steps"] > second_counters["steps"]
    assert all(third_counters[key] >= value for key, value in second_counters.items())
    assert complete(restored).root_count == 2


def test_default_private_budget_is_retained_and_cannot_be_replaced_between_steps():
    x = s.Symbol("resumable_private_budget_x")
    search = s.compile_residual_system([x], x).begin_root_isolation([(-1, 1)])
    ledger = search.budget
    assert type(ledger) is s.EvaluationBudget and search.budget is ledger
    before = remaining(ledger)
    search.advance()
    assert search.budget is ledger and remaining(ledger) != before
    before = remaining(ledger)
    assert search.finish().root_count == 1
    assert search.budget is ledger and remaining(ledger) != before
    for method in (search.advance, search.checkpoint, search.finish):
        with pytest.raises(TypeError):
            method(budget=ample_budget())


@pytest.mark.parametrize("refusal", ["cancelled", "exhausted"])
def test_ready_candidates_cannot_publish_or_checkpoint_on_a_refused_ledger(refusal):
    x = s.Symbol("resumable_refusal_x")
    ledger = ample_budget()
    search = s.compile_residual_system([x], x).begin_root_isolation([(-1, 1)], budget=ledger)
    search.advance()
    progress = progress_tuple(search.progress)
    if refusal == "cancelled":
        ledger.cancel()
        expected = s.EvaluationCancelled
    else:
        ledger._charge(ledger.remaining_steps)
        expected = s.EvaluationBudgetExceeded
    for method in (search.finish, search.checkpoint):
        with pytest.raises(expected):
            method()
    assert progress_tuple(search.progress) == progress


def test_exhausted_advance_keeps_the_current_cell_and_prior_committed_progress(partial_checkpoint):
    ledger = ample_budget()
    search = restore_fixture(partial_checkpoint, budget=ledger)
    before = progress_tuple(search.progress)
    ledger._charge(ledger.remaining_steps)
    for _ in range(2):
        with pytest.raises(s.EvaluationBudgetExceeded):
            search.advance()
        assert progress_tuple(search.progress) == before
        assert ledger.remaining_steps == 0


def test_unresolved_singular_root_never_discards_a_pending_cell_or_grants_completeness():
    x = s.Symbol("resumable_singular_x")
    program = s.compile_residual_system([x*x], x)
    search = program.begin_root_isolation([(-1, 1)], max_leaves=2, budget=ample_budget())
    search.advance(1)
    before = progress_tuple(search.progress)
    assert before[0] == 1 and before[1] == 2
    with pytest.raises(ValueError, match="inconclusive|subdivision|complete|unresolved"):
        search.advance()
    assert progress_tuple(search.progress) == before
    with pytest.raises(ValueError):
        search.finish()
    restored = program.restore_root_isolation(
        search.checkpoint(), [(-1, 1)], max_leaves=2, budget=ample_budget(),
    )
    assert progress_tuple(restored.progress) == before
    with pytest.raises(ValueError):
        restored.finish()


def test_held_original_domain_holes_remain_obligations_after_checkpoint_restore():
    x, y = s.symbols("resumable_hole_x resumable_hole_y")
    hidden_pole = s.Mul(0, s.Pow(y, -1, evaluate=False), evaluate=False)
    program = s.compile_residual_system([1, hidden_pole], (x, y))
    search = program.begin_root_isolation([(-1, 1), (-1, 1)], max_leaves=4, budget=ample_budget())
    restored = program.restore_root_isolation(
        search.checkpoint(), [(-1, 1), (-1, 1)], max_leaves=4, budget=ample_budget(),
    )
    with pytest.raises(ValueError):
        restored.advance(restored.max_advance_cells)
    assert not restored.progress.ready_for_verification
    with pytest.raises(ValueError):
        restored.finish()


def test_cancel_after_native_acceptance_still_prevents_publication(monkeypatch):
    x = s.Symbol("resumable_publication_x")
    ledger = ample_budget()
    search = s.compile_residual_system([x], x).begin_root_isolation([(-1, 1)], budget=ledger)
    search.advance()
    module = importlib.import_module("sympy.utilities._root_isolation_search")
    original = module.RootIsolationReceipt

    def cancel_after_wrapping(receipt):
        result = original(receipt)
        ledger.cancel()
        return result

    monkeypatch.setattr(module, "RootIsolationReceipt", cancel_after_wrapping)
    with pytest.raises(s.EvaluationCancelled):
        search.finish()


@pytest.mark.parametrize("max_cells", [0, 2048, -1, 10**100, True, 1.0])
def test_advance_requires_an_exact_bounded_builtin_integer(max_cells):
    x = s.Symbol("resumable_advance_limit_x")
    search = s.compile_residual_system([x], x).begin_root_isolation([(-1, 1)])
    before = progress_tuple(search.progress)
    with pytest.raises(ValueError, match="max_cells"):
        search.advance(max_cells)
    assert progress_tuple(search.progress) == before


def test_checkpoint_input_admission_precedes_box_hooks_and_rejects_state_pickle(partial_checkpoint):
    _, program, data, _ = partial_checkpoint
    calls = []

    class BytesSubclass(bytes):
        pass

    class Box:
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("untrusted checkpoint admission must come first")

    for invalid in ("{}", bytearray(data), memoryview(data), BytesSubclass(data), object()):
        with pytest.raises(TypeError, match="bytes"):
            program.restore_root_isolation(invalid, Box())
    with pytest.raises(ValueError, match="1 MiB"):
        program.restore_root_isolation(b" "*(1024*1024+1), Box())
    with pytest.raises(ValueError):
        program.restore_root_isolation(b"\x80\x04N.", [(-2, 2)])
    assert calls == []


def test_begin_and_restore_preserve_exact_source_shape_and_assumption_guards(partial_checkpoint):
    x, program, data, _ = partial_checkpoint
    floating = s.compile_residual_system([s.Add(x, s.Float(0.5), evaluate=False)], x)
    for method, args in ((floating.begin_root_isolation, ([(-2, 2)],)),
                         (floating.restore_root_isolation, (data, [(-2, 2)]))):
        with pytest.raises(ValueError, match="exact source"):
            method(*args)
    for method, args in ((program.begin_root_isolation, ([(0, 0)],)),
                         (program.restore_root_isolation, (data, [(0, 0)]))):
        with pytest.raises(ValueError, match="positive width"):
            method(*args)
    for invalid in (True, 16.0, 7, 129):
        with pytest.raises(ValueError, match="precision"):
            program.begin_root_isolation([(-2, 2)], precision=invalid)
    for invalid in (True, 1.0, 0, 1025):
        with pytest.raises(ValueError, match="max_leaves"):
            program.restore_root_isolation(data, [(-2, 2)], max_leaves=invalid)
    rectangular = s.compile_residual_system([x, x*x], x)
    with pytest.raises(ValueError, match="square"):
        rectangular.begin_root_isolation([(-2, 2)])


def test_search_and_progress_bindings_are_immutable_exact_native_owners():
    from sympy.utilities import RootIsolationProgress, RootIsolationSearch

    assert RootIsolationSearch is s.RootIsolationSearch
    assert RootIsolationProgress is s.RootIsolationProgress
    x = s.Symbol("resumable_immutable_x")
    search = s.compile_residual_system([x], x).begin_root_isolation([(-1, 1)])
    progress = search.progress
    for value, field in ((search, "_search"), (progress, "_progress")):
        with pytest.raises(AttributeError):
            setattr(value, field, object())
        with pytest.raises(AttributeError):
            delattr(value, field)
        with pytest.raises(AttributeError):
            value.__init__(getattr(value, field))
    for constructor in (RootIsolationSearch, RootIsolationProgress):
        with pytest.raises(TypeError):
            constructor(object())
    for constructor in (_native.RootIsolationSearch, _native.RootIsolationProgress):
        with pytest.raises(TypeError):
            constructor()
    assert "candidate_root_count" in repr(progress)
    assert "RootIsolationSearch" in repr(search)


def test_native_source_and_compiled_entries_share_the_same_checked_continuation():
    x = s.Symbol("resumable_native_x")
    compiled = s.compile_residual_system([x], x).native_system
    source = _native.SourceResidualSystem([_native_expr(x)], [_symbol_binding(x)])
    box = [(_native_expr(-1), _native_expr(1))]
    search = source.begin_root_isolation(box, 16, 1)
    assert type(search) is _native.RootIsolationSearch
    ledger = search.budget
    assert type(ledger) is s.EvaluationBudget and search.budget is ledger
    with pytest.raises(TypeError):
        search.advance(True)
    data = search.checkpoint()
    before = remaining(ledger)
    assert search.advance().ready_for_verification
    assert search.budget is ledger and remaining(ledger) != before
    restored = compiled.restore_root_isolation(data, box, 16, 1, budget=ample_budget())
    assert type(restored.progress) is _native.RootIsolationProgress
    assert restored.advance().ready_for_verification
    receipt = restored.finish()
    assert type(receipt) is _native.RootIsolationReceipt and receipt.root_count == 1


def test_checkpoint_replays_in_a_fresh_python_process_with_explicit_typed_source():
    x = s.Symbol("resumable_fresh_x", real=True)
    program = s.compile_residual_system([x*x-1], x)
    search = program.begin_root_isolation(
        [(-2, 2)], precision=16, max_leaves=64, budget=ample_budget(),
    )
    search.advance(1)
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
x = s.Symbol('resumable_fresh_x', real=True)
program = s.compile_residual_system([x*x-1], x)
budget = s.EvaluationBudget(max_steps=1<<45, max_memory_bytes=1<<40, max_allocations=1<<30)
search = program.restore_root_isolation(sys.stdin.buffer.read(), [(-2, 2)],
                                       precision=16, max_leaves=64, budget=budget)
assert not search.progress.ready_for_verification
for _ in range(2047):
    if search.progress.ready_for_verification:
        break
    search.advance(4)
receipt = search.finish()
assert receipt.domain == ((-2, 2),)
print(receipt.root_count)
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, _native.__name__, _native.__file__, str(shell_root)],
        input=search.checkpoint(),
        capture_output=True, timeout=90,
    )
    assert process.returncode == 0, process.stdout.decode() + process.stderr.decode()
    assert process.stdout.strip() == b"2"
