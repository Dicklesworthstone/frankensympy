"""Python and native Sessions exchange the same source-bound program artifacts."""

import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import sympy as s

from sympy.core import _native


LIMIT = 1 << 40


def budget(**changes):
    limits = dict(max_steps=LIMIT, max_memory_bytes=LIMIT, max_allocations=LIMIT)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def remaining(ledger):
    return ledger.remaining_steps, ledger.remaining_memory_bytes, ledger.remaining_allocations


def compile_program(kind, source, args, **options):
    if kind == "residual":
        return s.compile_residual_system(source, args, **options)
    return s.compile_objective(source, args, **options)


def load_program(kind, data, source, args, **options):
    if kind == "residual":
        return s.load_portable_residual_system(data, residuals=source, args=args, **options)
    return s.load_portable_objective(data, objective=source, args=args, **options)


def fixture(kind):
    x, y = s.symbols("portable_program_x portable_program_y", real=True)
    if kind == "residual":
        source, axes = (x*x + y, s.sin(y), 7), (y, x)
    else:
        source, axes = x*x + x*y + 3*y*y, (x, y)
    return source, axes, compile_program(kind, source, axes)


def assert_evaluation(kind, program, *, nested=False, ledger=None):
    point = (0, 2) if kind == "residual" else (2, -1)
    arguments = (point,) if nested else point
    if kind == "residual":
        assert program.evaluate(*arguments, budget=ledger) == (
            [4, 0, 7], [[1, 4], [1, 0], [0, 0]],
        )
        assert program.jacobian_vector_product([3, -2], *arguments, budget=ledger) == [-5, 3, 0]
        assert program.jacobian_transpose_vector_product(
            [2, -3, 11], *arguments, budget=ledger,
        ) == [-1, 8]
    else:
        assert program.evaluate(*arguments, budget=ledger) == (5, [3, -4], [[2, 1], [1, 6]])
        assert program.hessian_vector_product([3, -2], *arguments, budget=ledger) == [4, -9]


def encode(document):
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_portable_round_trip_retains_legacy_bytes_and_cumulative_budget(kind):
    source, axes, original = fixture(kind)
    legacy = original.to_bytes()
    ledger = budget()
    data = original.to_portable_bytes(budget=ledger)
    after_export = remaining(ledger)
    assert type(data) is bytes
    assert all(0 < amount < LIMIT for amount in after_export)
    document = json.loads(data)
    assert list(document) == ["schema_version", "context_digest", "candidate"]
    assert document["schema_version"] == 1
    assert len(document["context_digest"]) == 32
    assert list(document["candidate"]) == ["schema_version", "kind", "candidate"]
    assert document["candidate"]["kind"] == kind
    assert "receipt" not in document["candidate"]["candidate"]
    assert "shared_evaluation" not in document["candidate"]["candidate"]

    restored = load_program(kind, data, source, [axes], budget=ledger)
    assert all(after < before for after, before in zip(remaining(ledger), after_export))
    assert all(actual is expected for actual, expected in zip(restored.variables, axes))
    assert_evaluation(kind, restored, nested=True, ledger=ledger)
    assert restored.to_portable_bytes(budget=ledger) == data
    assert restored.to_bytes() == original.to_bytes() == legacy
    old_loader = s.load_residual_system if kind == "residual" else s.load_objective
    expected = {"residuals" if kind == "residual" else "objective": source}
    assert_evaluation(kind, old_loader(legacy, args=axes, **expected))
    with pytest.raises(ValueError):
        load_program(kind, legacy, source, axes)
    with pytest.raises(ValueError):
        old_loader(data, args=axes, **expected)


def test_public_portable_loaders_are_exported_from_their_owning_modules():
    from sympy.utilities import load_portable_objective, load_portable_residual_system
    from sympy.utilities.lambdify import load_portable_residual_system as residual_loader
    from sympy.utilities.objective import load_portable_objective as objective_loader

    assert s.load_portable_residual_system is load_portable_residual_system is residual_loader
    assert s.load_portable_objective is load_portable_objective is objective_loader


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_receiving_source_order_and_typed_symbol_identity_are_required(kind):
    source, axes, original = fixture(kind)
    data = original.to_portable_bytes()
    changed = (source[0] + 1, *source[1:]) if kind == "residual" else source + 1
    for expected, order in ((changed, axes), (source, axes[::-1])):
        with pytest.raises(ValueError):
            load_program(kind, data, expected, order)

    left, right = s.Dummy("portable_same_name"), s.Dummy("portable_same_name")
    typed = compile_program(kind, left*left, left).to_portable_bytes()
    assert load_program(kind, typed, left*left, left).variables[0] is left
    with pytest.raises(ValueError):
        load_program(kind, typed, right*right, right)


@pytest.mark.parametrize("kind", ["residual", "objective"])
@pytest.mark.parametrize("mutation", ["context", "schema", "kind", "status", "spacing", "duplicate"])
def test_context_and_canonical_envelope_cannot_be_forged(kind, mutation):
    source, axes, original = fixture(kind)
    data = original.to_portable_bytes()
    document = json.loads(data)
    if mutation == "context":
        document["context_digest"][0] ^= 1
    elif mutation == "schema":
        document["schema_version"] += 1
    elif mutation == "kind":
        document["candidate"]["kind"] = "objective" if kind == "residual" else "residual"
    elif mutation == "status":
        document["accepted"] = True
    forged = encode(document)
    if mutation == "spacing":
        forged = b" " + data
    elif mutation == "duplicate":
        forged = b'{"schema_version":1,' + data[1:]
    with pytest.raises(ValueError):
        load_program(kind, forged, source, axes)
    assert original.to_portable_bytes() == data


@pytest.mark.parametrize("kind", ["residual", "objective"])
@pytest.mark.parametrize("mutation", ["proof", "bytecode", "missing_entry"])
def test_independent_replay_rejects_altered_derivative_evidence(kind, mutation):
    source, axes, original = fixture(kind)
    data = original.to_portable_bytes()
    document = json.loads(data)
    candidate = document["candidate"]["candidate"]
    stage = candidate if kind == "residual" else candidate["hessian"]
    if mutation == "proof":
        stage["entries"][0]["derivation"]["steps"] = []
    elif mutation == "bytecode":
        stage["entries"][0]["compiled_deriv"]["ops"] = [{"LoadConst": 999.0}]
    else:
        stage["entries"].pop(0)
    ledger = budget()
    with pytest.raises(ValueError):
        load_program(kind, encode(document), source, axes, budget=ledger)
    assert ledger.remaining_steps < LIMIT
    restored = load_program(kind, data, source, axes, budget=ledger)
    assert_evaluation(kind, restored, ledger=ledger)


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_original_held_domains_and_float_restrictions_survive_transport(kind):
    x = s.Symbol("portable_guarded_x")
    source = s.Add(x*x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    data = compile_program(kind, source, x).to_portable_bytes()
    restored = load_program(kind, data, source, x)
    with pytest.raises(ValueError):
        restored.evaluate(-1)
    assert restored.evaluate(2)[0] == ([4] if kind == "residual" else 4)
    with pytest.raises(ValueError):
        load_program(kind, data, x*x, x)

    floating = x*x + s.Float(0.5)
    encoded = compile_program(kind, floating, x).to_portable_bytes()
    approximate = load_program(kind, encoded, floating, x)
    with pytest.raises(ValueError):
        approximate.enclose([(1, 2)])


@pytest.mark.parametrize("kind", ["residual", "objective"])
@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_transport_budget_failures_are_typed_and_do_not_publish(kind, dimension):
    source, axes, original = fixture(kind)
    data = original.to_portable_bytes()
    with pytest.raises(s.EvaluationBudgetExceeded):
        original.to_portable_bytes(budget=budget(**{dimension: 0}))
    published = []
    with pytest.raises(s.EvaluationBudgetExceeded):
        published.append(load_program(kind, data, source, axes, budget=budget(**{dimension: 0})))
    assert published == []
    assert_evaluation(kind, original)


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_cancellation_precedes_source_capture_and_result_publication(kind, monkeypatch):
    source, axes, original = fixture(kind)
    data = original.to_portable_bytes()
    cancelled = budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        original.to_portable_bytes(budget=cancelled)
    with pytest.raises(s.EvaluationCancelled):
        load_program(kind, data, object(), object(), budget=cancelled)
    native = _native.VerifiedResidualSystem if kind == "residual" else _native.VerifiedObjective
    expected = {"residuals" if kind == "residual" else "source": object()}
    with pytest.raises(s.EvaluationCancelled):
        native.from_portable_bytes(object(), variables=object(), budget=cancelled, **expected)
    assert remaining(cancelled) == (LIMIT, LIMIT, LIMIT)

    module_name = "lambdify" if kind == "residual" else "objective"
    module = importlib.import_module("sympy.utilities." + module_name)
    class_name = "_ResidualCallable" if kind == "residual" else "_ObjectiveCallable"
    constructor = getattr(module, class_name)
    ledger = budget()

    def cancel_before_publication(*args, **kwargs):
        result = constructor(*args, **kwargs)
        ledger.cancel()
        return result

    monkeypatch.setattr(module, class_name, cancel_before_publication)
    with pytest.raises(s.EvaluationCancelled):
        load_program(kind, data, source, axes, budget=ledger)
    assert ledger.remaining_steps < LIMIT


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_transport_admits_only_bounded_immutable_bytes_before_source_capture(kind, monkeypatch):
    module_name = "lambdify" if kind == "residual" else "objective"
    module = importlib.import_module("sympy.utilities." + module_name)
    capture = "_residual_source" if kind == "residual" else "_objective_source"

    def forbidden_capture(*args, **kwargs):
        pytest.fail("wire admission must precede source inspection")

    class BytesSubclass(bytes):
        pass

    monkeypatch.setattr(module, capture, forbidden_capture)
    for data in ("{}", bytearray(b"{}"), memoryview(b"{}"), BytesSubclass(b"{}"), object()):
        with pytest.raises(TypeError):
            load_program(kind, data, object(), object())
    with pytest.raises(ValueError):
        load_program(kind, b" " * (1024 * 1024 + 1), object(), object())


def run_session(requests):
    """Execute the actual native CLI; a configured missing binary is a failure."""
    binary = os.environ.get("FSYM_NATIVE_CLI")
    if not binary:
        pytest.skip("set FSYM_NATIVE_CLI to the built frankensympy CLI for cross-runtime replay")
    assert Path(binary).is_file(), "FSYM_NATIVE_CLI must name the real native binary"
    frames = [
        {"id": str(index), "type": operation, "payload": {"schema_version": 1, **payload}}
        for index, (operation, payload) in enumerate(requests)
    ]
    process = subprocess.run(
        [binary, "--compute-steps", "1000000000", "--verifier-steps", "1000000000"],
        input="".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames),
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stdout + "\n" + process.stderr
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(responses) == len(frames), process.stdout + "\n" + process.stderr
    assert [response["id"] for response in responses] == [frame["id"] for frame in frames]
    return responses


def session_fixture(kind):
    # Parse the exact same native source grammar on each side. This checks
    # transport interoperability without assuming a shell reordering is equal.
    x, y = s.symbols("x y")
    if kind == "residual":
        sources, axes = ["x^2+y", "sin(y)", "7"], (y, x)
        source = [_native.Expr(value) for value in sources]
        register = ("RegisterResidualSystem", {
            "system": "system:portable", "residuals": sources, "variables": ["y", "x"],
        })
        owner, operation, point = {"system": "system:portable"}, "ResidualSystem", ["0", "2"]
        evaluation = "EvaluateResidualJacobian"
    else:
        text, axes = "x^2+x*y+3*y^2", (x, y)
        source = _native.Expr(text)
        register = ("RegisterObjective", {
            "objective": "objective:portable", "source": text, "variables": ["x", "y"],
        })
        owner, operation, point = {"objective": "objective:portable"}, "Objective", ["2", "-1"]
        evaluation = "EvaluateObjective"
    handle = "program:portable" if kind == "residual" else "objective_program:portable"
    evaluate = (evaluation, {"program": handle, "point": point})
    export = ("ExportCompiled" + operation, {"program": handle})
    return source, axes, register, owner, operation, handle, evaluate, export


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_python_and_fresh_native_sessions_replay_in_both_directions(kind):
    source, axes, register, owner, operation, handle, evaluate, export = session_fixture(kind)
    original = compile_program(kind, source, axes)
    data = original.to_portable_bytes()
    responses = run_session([
        register,
        ("ReplayCompiled" + operation, {**owner, "program": handle, "artifact": data.decode()}),
        evaluate,
        export,
    ])
    for response in responses:
        assert response["status"] == "success", response
    result = responses[2]["result"]
    assert result["outcome"] == "approximate"
    assert result["numeric_model"] == "binary64"
    assert "evidence" not in result and "claim" not in result
    if kind == "residual":
        assert result["residuals"] == [4, 0, 7]
        assert result["jacobian"] == [1, 4, 1]
    else:
        assert result["value"] == 5
        assert result["gradient"] == [3, -4]
        assert result["hessian"] == [2, 1, 1, 6]
    assert responses[3]["result"]["artifact"].encode() == data

    # A second, fresh Session generates a candidate itself. Python admits its
    # evidence against the expected sources after that process has exited.
    generated = run_session([
        register, ("Compile" + operation, {**owner, "program": handle}), export,
    ])
    for response in generated:
        assert response["status"] == "success", response
    received = generated[-1]["result"]["artifact"].encode()
    restored = load_program(kind, received, source, axes)
    assert_evaluation(kind, restored)
    assert restored.to_portable_bytes() == received == data


@pytest.mark.parametrize("kind", ["residual", "objective"])
def test_native_session_rejection_does_not_publish_a_program_handle(kind):
    source, axes, register, owner, operation, handle, evaluate, _ = session_fixture(kind)
    data = compile_program(kind, source, axes).to_portable_bytes()
    document = json.loads(data)
    candidate = document["candidate"]["candidate"]
    stage = candidate if kind == "residual" else candidate["hessian"]
    stage["entries"][0]["compiled_deriv"]["ops"] = [{"LoadConst": 999.0}]
    responses = run_session([
        register,
        ("ReplayCompiled" + operation, {
            **owner, "program": handle, "artifact": encode(document).decode(),
        }),
        evaluate,
    ])
    assert responses[0]["status"] == "success"
    assert responses[1]["status"] != "success"
    assert responses[2]["code"] == "unknown_object"
