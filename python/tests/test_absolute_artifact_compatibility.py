"""Historical numerical Abs aliases migrate only through two exact replays."""

import importlib
import json

import pytest
import sympy as s


OLD = b'"Function":["abs",'
NEW = b'"Function":["Abs",'


def budget(steps=1 << 36):
    return s.EvaluationBudget(max_steps=steps, max_memory_bytes=1 << 40, max_allocations=1 << 30)


def encode(document):
    return json.dumps(document, separators=(",", ":")).encode()


def make_source(kind, x):
    # The outer Add differentiated the held zero product to zero before Abs
    # derivatives existed. Its saved derivative proofs contain no new jets.
    smooth = x-2 if kind == "residual" else (x-2)**2
    return s.Add(smooth, s.Mul(0, s.Abs(x, evaluate=False), evaluate=False), evaluate=False)


def saved(kind, portable, x, source=None):
    source = make_source(kind, x) if source is None else source
    compile_ = s.compile_residual_system if kind == "residual" else s.compile_objective
    program = compile_(source, x)
    canonical = program.to_portable_bytes() if portable else program.to_bytes()
    assert NEW in canonical and b"_fsym_abs_d" not in canonical
    return source, canonical, canonical.replace(NEW, OLD)


def load(kind, portable, data, source, axes, shared=None):
    if kind == "residual":
        loader = s.load_portable_residual_system if portable else s.load_residual_system
        return loader(data, residuals=source, args=axes, budget=shared)
    loader = s.load_portable_objective if portable else s.load_objective
    return loader(data, objective=source, args=axes, budget=shared)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_historical_programs_replay_to_canonical_sources_and_receipts(kind, portable):
    x = s.Dummy("historical_absolute", real=True)
    source, canonical, legacy = saved(kind, portable, x)
    restored = load(kind, portable, legacy, source, x)
    assert restored.variables[0] is x
    assert restored.evaluate(3) == (([1], [[1]]) if kind == "residual" else (1, [2], [[2]]))
    exported = restored.to_portable_bytes() if portable else restored.to_bytes()
    assert exported == canonical
    if kind == "residual":
        receipt = restored.certify_root_box([(1, 3)], budget=budget())
        assert receipt.residuals == (source,)
        replayed = s.RootBoxReceipt.from_bytes(receipt.to_bytes(), residuals=source, args=x)
        assert replayed.residuals == (source,)
    else:
        receipt = restored.certify_stationary_point([(1, 3)], budget=budget())
        assert receipt.objective == source
    # Neither omission of a zero derivative nor migration erases the cusp.
    with pytest.raises(ValueError):
        restored.evaluate(0)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_held_constant_absolute_source_was_already_compilable(kind, portable):
    x = s.Symbol("historical_constant")
    source, canonical, legacy = saved(kind, portable, x, s.Abs(1, evaluate=False))
    restored = load(kind, portable, legacy, source, x)
    assert restored.evaluate(7) == (([1], [[0]]) if kind == "residual" else (1, [0], [[0]]))
    assert (restored.to_portable_bytes() if portable else restored.to_bytes()) == canonical


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_absolute_values_inside_nonzero_saved_derivatives_are_migrated(kind, portable):
    x = s.Symbol("historical_derivative_absolute")
    source = s.Mul(x if kind == "residual" else x*x, s.Abs(2, evaluate=False), evaluate=False)
    source, canonical, legacy = saved(kind, portable, x, source)
    restored = load(kind, portable, legacy, source, x)
    assert restored.evaluate(3) == (([6], [[2]]) if kind == "residual" else (18, [12], [[4]]))
    assert (restored.to_portable_bytes() if portable else restored.to_bytes()) == canonical


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_mixed_alias_sources_are_not_a_historical_lowering_profile(kind, portable):
    x = s.Symbol("historical_mixed")
    source = s.Add(x, s.Abs(1, evaluate=False), s.Abs(2, evaluate=False), evaluate=False)
    source, canonical, _ = saved(kind, portable, x, source)
    mixed = canonical.replace(NEW, OLD, 1)
    with pytest.raises(ValueError):
        load(kind, portable, mixed, source, x)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_held_original_poles_survive_historical_replay(kind, portable):
    x = s.Symbol("historical_pole")
    hidden = s.Mul(0, s.Abs(s.Pow(x, -1, evaluate=False), evaluate=False), evaluate=False)
    source = s.Add(x if kind == "residual" else x*x, hidden, evaluate=False)
    source, _, legacy = saved(kind, portable, x, source)
    restored = load(kind, portable, legacy, source, x)
    with pytest.raises(ValueError):
        restored.evaluate(0)
    with pytest.raises(ValueError):
        restored.enclose([(-1, 1)], budget=budget())


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_symbol_text_cannot_be_rewritten_as_a_function_token(kind, portable):
    x = s.Symbol('{"Function":["abs",')
    source, canonical, legacy = saved(kind, portable, x)
    restored = load(kind, portable, legacy, source, x)
    assert restored.variables[0] is x
    assert (restored.to_portable_bytes() if portable else restored.to_bytes()) == canonical


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_typed_identity_and_source_changes_cannot_enter_the_legacy_profile(kind, portable):
    x = s.Symbol("historical_typed", positive=True)
    other = s.Symbol("historical_typed", real=True)
    source, _, legacy = saved(kind, portable, x)
    with pytest.raises(ValueError):
        load(kind, portable, legacy, make_source(kind, other), other)
    with pytest.raises(ValueError):
        load(kind, portable, legacy, source+1, x)
    with pytest.raises(ValueError):
        load(kind, portable, legacy, x-2 if kind == "residual" else (x-2)**2, x)


@pytest.mark.parametrize("portable", (False, True))
def test_residual_order_and_variable_order_are_retained(portable):
    x, y = s.symbols("historical_order_x historical_order_y")
    sources = [make_source("residual", x), y-1]
    program = s.compile_residual_system(sources, (x, y))
    canonical = program.to_portable_bytes() if portable else program.to_bytes()
    legacy = canonical.replace(NEW, OLD)
    assert load("residual", portable, legacy, sources, (x, y)).evaluate(2, 1)[0] == [0, 0]
    for expected, axes in ((sources[::-1], (x, y)), (sources, (y, x))):
        with pytest.raises(ValueError):
            load("residual", portable, legacy, expected, axes)


@pytest.mark.parametrize("name", ("abs", "Abs"))
def test_opaque_function_name_collisions_do_not_select_builtin_migration(name, monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    core = importlib.import_module("sympy.core")
    monkeypatch.setattr(core, "_user_functions", core._user_functions.copy())
    x = s.Symbol("historical_opaque")
    _, _, legacy = saved("residual", False, x)
    opaque = type(name, (s.Function,), {})

    def forbidden(*args, **kwargs):
        raise AssertionError("closed source admission must precede migration")

    monkeypatch.setattr(transport, "_legacy_sources", forbidden)
    with pytest.raises(NotImplementedError):
        s.load_residual_system(legacy, residuals=opaque(x, evaluate=False), args=x)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_forged_retained_proofs_never_reach_canonical_rewriting(kind, portable, monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    x = s.Symbol("historical_forgery")
    source, _, legacy = saved(kind, portable, x)
    document = json.loads(legacy)
    candidate = document["candidate"] if portable else document
    if kind == "residual":
        jacobian = candidate["candidate"] if portable else candidate["jacobian"]
    else:
        jacobian = candidate["candidate"]["gradient"]
    jacobian["entries"][0]["derivation"]["steps"][0]["rule"]["DefinitionalReduction"]["rule_name"] = "forged_rule"

    def forbidden(*args, **kwargs):
        raise AssertionError("an unverified historical proof must not be rewritten")

    monkeypatch.setattr(transport, "_canonical_bytes", forbidden)
    with pytest.raises(ValueError):
        load(kind, portable, encode(document), source, x)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_canonical_second_replay_still_checks_the_retained_evidence(kind, portable, monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    x = s.Symbol("historical_second_replay")
    source, _, legacy = saved(kind, portable, x)
    original = transport._canonical_bytes

    def corrupted(data, shared):
        document = json.loads(original(data, shared))
        candidate = document["candidate"] if portable else document
        jacobian = ((candidate["candidate"] if portable else candidate["jacobian"])
                    if kind == "residual" else candidate["candidate"]["gradient"])
        jacobian["entries"][0]["derivation"]["steps"] = []
        return encode(document)

    monkeypatch.setattr(transport, "_canonical_bytes", corrupted)
    with pytest.raises(ValueError):
        load(kind, portable, legacy, source, x)


@pytest.mark.parametrize("kind", ("residual", "objective"))
def test_context_and_canonicality_failures_do_not_retry(kind, monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    x = s.Symbol("historical_context")
    source, _, legacy = saved(kind, True, x)
    document = json.loads(legacy)
    document["context_digest"][0] ^= 1

    def forbidden(*args, **kwargs):
        raise AssertionError("context and encoding failures must not select legacy admission")

    monkeypatch.setattr(transport, "_legacy_sources", forbidden)
    for malformed in (b" "+legacy, encode(document)):
        with pytest.raises(ValueError):
            load(kind, True, malformed, source, x)


def test_old_residual_json_spelling_and_raw_candidate_format_remain_supported():
    x = s.Symbol("historical_json")
    source, canonical, legacy = saved("residual", False, x)
    document = json.loads(legacy)
    raw = {"schema_version": 1, "kind": "residual", "candidate": document["jacobian"]}
    for data in (json.dumps(document, indent=2).encode(), encode(raw)):
        assert s.load_residual_system(data, residuals=source, args=x).to_bytes() == canonical
    with pytest.raises(ValueError, match="canonical"):
        s.load_residual_system(b" "+encode(raw), residuals=source, args=x)


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_historical_replay_retains_exact_cumulative_step_accounting(kind, portable):
    x = s.Symbol("historical_budget")
    source, _, legacy = saved(kind, portable, x)
    measured = budget()
    load(kind, portable, legacy, source, x, measured)
    consumed = (1 << 36)-measured.remaining_steps
    assert consumed > len(legacy)
    exact = budget(consumed)
    load(kind, portable, legacy, source, x, exact)
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        load(kind, portable, legacy, source, x, budget(consumed-1))


@pytest.mark.parametrize("kind", ("residual", "objective"))
@pytest.mark.parametrize("portable", (False, True))
def test_cancellation_between_old_and_canonical_admission_publishes_nothing(kind, portable, monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    x = s.Symbol("historical_cancel")
    source, _, legacy = saved(kind, portable, x)
    shared = budget()
    original = transport._canonical_bytes

    def cancelled(data, same_budget):
        assert same_budget is shared
        result = original(data, same_budget)
        shared.cancel()
        return result

    monkeypatch.setattr(transport, "_canonical_bytes", cancelled)
    with pytest.raises(s.EvaluationCancelled):
        load(kind, portable, legacy, source, x, shared)


@pytest.mark.parametrize("bound_program", (False, True))
def test_historical_root_proofs_replay_to_canonical_receipts(bound_program):
    x = s.Symbol("historical_root")
    source = make_source("residual", x)
    receipt = s.certify_root_box(source, x, [(1, 3)], budget=budget())
    canonical = receipt.to_bytes()
    legacy = canonical.replace(NEW, OLD)
    program = s.compile_residual_system(source, x)
    replay = (lambda data, shared: program.replay_root_certificate(data, budget=shared)
              if bound_program else
              s.RootBoxReceipt.from_bytes(data, residuals=source, args=x, budget=shared))
    loaded = replay(legacy, budget())
    assert loaded.residuals == (source,)
    assert loaded.root_enclosure == ((2, 2),)
    assert loaded.to_bytes() == canonical
    with pytest.raises(ValueError):
        s.RootBoxReceipt.from_bytes(legacy, residuals=source+1, args=x, budget=budget())


def test_historical_root_cancellation_between_replays_is_terminal(monkeypatch):
    transport = importlib.import_module("sympy.utilities._legacy_absolute_transport")
    x = s.Symbol("historical_root_cancel")
    source = make_source("residual", x)
    data = s.certify_root_box(source, x, [(1, 3)], budget=budget()).to_bytes().replace(NEW, OLD)
    shared = budget()
    original = transport._canonical_bytes

    def cancelled(encoded, same_budget):
        result = original(encoded, same_budget)
        shared.cancel()
        return result

    monkeypatch.setattr(transport, "_canonical_bytes", cancelled)
    with pytest.raises(s.EvaluationCancelled):
        s.RootBoxReceipt.from_bytes(data, residuals=source, args=x, budget=shared)
