"""Saved objectives replay both derivative stages against receiver-owned inputs."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def budget(**overrides):
    return s.EvaluationBudget(**{
        "max_steps": 1 << 45,
        "max_memory_bytes": 1 << 40,
        "max_allocations": 1 << 30,
        **overrides,
    })


def encode(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@pytest.fixture
def objective():
    x = s.Symbol("saved_objective_x", real=True)
    y = s.Symbol("saved_objective_y", real=True)
    source = (x - 1)**2 + 2*(y + 2)**2 + x*y
    program = s.compile_objective(source, (x, y))
    return x, y, source, program, program.to_bytes()


def test_round_trip_preserves_value_gradient_hessian_and_nested_layout(objective):
    x, y, source, original, data = objective
    restored = s.load_objective(data, objective=source, args=[(x, y)])
    assert type(data) is bytes
    assert restored.to_bytes() == data
    assert restored.variables[0] is x and restored.variables[1] is y
    assert restored.evaluate((2, 3)) == (57.0, [5.0, 22.0], [[2.0, 1.0], [1.0, 4.0]])
    assert restored.evaluate_sparse((2, 3)) == (57.0, [5.0, 22.0], [2.0, 1.0, 1.0, 4.0])
    assert restored.hessian_vector_product([3, -2], (2, 3)) == [4.0, -5.0]
    assert restored.hessian_pattern == original.hessian_pattern
    assert restored.native_objective.proofs_verified == original.native_objective.proofs_verified
    with pytest.raises(TypeError):
        restored.evaluate(2, 3)
    with pytest.raises(AttributeError):
        restored._native_objective = None


def test_public_and_native_loaders_share_exact_source_admission(objective):
    from sympy.utilities import load_objective
    from sympy.utilities.objective import load_objective as direct_loader

    assert s.load_objective is load_objective is direct_loader
    x, y, source, _, data = objective
    native = _native.VerifiedObjective.from_bytes(
        data, source=_native_expr(source),
        variables=[_symbol_binding(x), _symbol_binding(y)],
    )
    assert native.to_bytes() == data
    assert native.source == _native_expr(source)
    with pytest.raises(TypeError):
        s.load_objective(data, args=(x, y))
    with pytest.raises(TypeError):
        s.load_objective(data, objective=source)


def test_every_sparse_zero_and_mixed_partial_is_retained():
    x, y, z = s.symbols("saved_sparse_x saved_sparse_y saved_sparse_z")
    source = x*y + z**3
    original = s.compile_objective(source, (x, y, z))
    restored = s.load_objective(original.to_bytes(), objective=source, args=(x, y, z))
    assert restored.hessian_pattern == ((0, 1), (1, 0), (2, 2))
    assert restored.native_objective.zero_entries_checked == 6
    assert restored.hessian_vector_product([7, 11, 13], 2, 3, 4) == [11, 7, 312]
    for scalar, axes in ((7, ()), (7, (x, y)), (2*x-y+7, (x, y))):
        program = s.compile_objective(scalar, axes)
        loaded = s.load_objective(program.to_bytes(), objective=scalar, args=axes)
        point = [2] * len(axes)
        assert loaded.evaluate(*point) == program.evaluate(*point)
        assert loaded.hessian_pattern == ()


def test_source_typed_order_assumptions_and_dummy_identity_cannot_be_substituted(objective):
    x, y, source, _, data = objective
    for changed_source, axes in ((source+1, (x, y)), (source, (y, x))):
        with pytest.raises(ValueError):
            s.load_objective(data, objective=changed_source, args=axes)
    other = s.Symbol(x.name)
    with pytest.raises(ValueError):
        s.load_objective(data, objective=source.subs(x, other), args=(other, y))
    left, right = s.Dummy("same_name"), s.Dummy("same_name")
    dummy_data = s.compile_objective(left**2, left).to_bytes()
    assert s.load_objective(dummy_data, objective=left**2, args=left).variables[0] is left
    with pytest.raises(ValueError):
        s.load_objective(dummy_data, objective=right**2, args=right)


@pytest.mark.parametrize("mutation", [
    "objective_bytecode", "gradient_proof", "hessian_proof", "hessian_bytecode",
    "missing_mixed_partial", "gradient_source", "variable_order",
])
def test_altered_evidence_is_rejected_without_regeneration(objective, mutation):
    x, y, source, original, data = objective
    document = json.loads(data)
    candidate = document["candidate"]
    if mutation == "objective_bytecode":
        candidate["compiled_objective"]["ops"] = [{"LoadConst": 0.0}]
    elif mutation.endswith("proof"):
        stage = "gradient" if mutation.startswith("gradient") else "hessian"
        candidate[stage]["entries"][0]["derivation"]["steps"] = []
    elif mutation == "hessian_bytecode":
        candidate["hessian"]["entries"][0]["compiled_deriv"]["ops"] = [{"LoadConst": 0.0}]
    elif mutation == "missing_mixed_partial":
        candidate["hessian"]["entries"].pop(1)
    elif mutation == "gradient_source":
        candidate["hessian"]["residuals"].reverse()
    else:
        candidate["gradient"]["vars"].reverse()
    with pytest.raises(ValueError):
        s.load_objective(encode(document), objective=source, args=(x, y))
    assert original.evaluate(2, 3)[0] == 57
    assert original.to_bytes() == data


def test_loader_does_not_use_compilation_constructor(objective, monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    original_class = _native.VerifiedObjective
    x, y, source, _, data = objective
    calls = []

    class ReplayOnly:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("loading must not regenerate derivatives")

        @staticmethod
        def from_bytes(data, **kwargs):
            calls.append(True)
            return original_class.from_bytes(data, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("loading must not compile another candidate")

    monkeypatch.setattr(module._native, "VerifiedObjective", ReplayOnly)
    monkeypatch.setattr(module, "compile_objective", forbidden)
    assert module.load_objective(data, objective=source, args=(x, y)).evaluate(2, 3)[0] == 57
    assert calls == [True]


@pytest.mark.parametrize("change", ["schema", "kind", "status", "noncanonical", "duplicate"])
def test_format_identity_canonicality_and_status_forgery_refuse(objective, change):
    x, y, source, _, data = objective
    document = json.loads(data)
    assert list(document) == ["schema_version", "kind", "candidate"]
    assert document["kind"] == "objective"
    if change == "schema":
        document["schema_version"] += 1
    elif change == "kind":
        document["kind"] = "residual"
    elif change == "status":
        document["accepted"] = True
    mutated = encode(document)
    if change == "noncanonical":
        mutated = b" " + data
    elif change == "duplicate":
        mutated = b'{"schema_version":1,' + data[1:]
    with pytest.raises(ValueError):
        s.load_objective(mutated, objective=source, args=(x, y))


def test_invalid_bytes_refuse_before_source_traversal(objective, monkeypatch):
    module = importlib.import_module("sympy.utilities.objective")
    data = objective[-1]

    class BytesSubclass(bytes):
        pass

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted bytes must not trigger source capture")

    monkeypatch.setattr(module, "_objective_source", forbidden)
    for value in (bytearray(data), memoryview(data), BytesSubclass(data), data.decode(), object()):
        with pytest.raises(TypeError, match="built-in bytes"):
            module.load_objective(value, objective=object(), args=object())
        with pytest.raises(TypeError, match="built-in bytes"):
            _native.VerifiedObjective.from_bytes(value, source=object(), variables=object())
    with pytest.raises(ValueError, match="1 MiB"):
        module.load_objective(b" "*(1024*1024+1), objective=object(), args=object())
    with pytest.raises(ValueError, match="1 MiB"):
        _native.VerifiedObjective.from_bytes(b" "*(1024*1024+1), source=object(), variables=object())


@pytest.mark.parametrize("payload", [b"", b"\xff", b"{}", b"["*97+b"0"+b"]"*97])
def test_malformed_and_deep_artifacts_fail_closed(objective, payload):
    x, y, source, _, _ = objective
    with pytest.raises(ValueError):
        s.load_objective(payload, objective=source, args=(x, y))


@pytest.mark.parametrize("dimension", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_export_and_replay_share_cumulative_resources_and_preserve_failures(objective, dimension):
    x, y, source, program, data = objective
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.to_bytes(budget=budget(**{dimension: 0}))
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.load_objective(data, objective=source, args=(x, y), budget=budget(**{dimension: 0}))
    shared = budget()
    before = shared.remaining_steps
    program.to_bytes(budget=shared)
    after_export = shared.remaining_steps
    loaded = s.load_objective(data, objective=source, args=(x, y), budget=shared)
    after_load = shared.remaining_steps
    assert before > after_export > after_load
    loaded.evaluate(2, 3, budget=shared)
    assert shared.remaining_steps < after_load
    shared.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.to_bytes(budget=shared)
    with pytest.raises(s.EvaluationCancelled):
        s.load_objective(data, objective=source, args=(x, y), budget=shared)


def test_original_held_domains_and_float_origin_guards_survive_replay():
    x = s.Symbol("saved_guard_x")
    held = s.Add(x*x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    data = s.compile_objective(held, x).to_bytes()
    loaded = s.load_objective(data, objective=held, args=x)
    assert loaded.evaluate(2) == (4, [4], [[2]])
    with pytest.raises(ValueError):
        loaded.evaluate(-1)
    with pytest.raises(ValueError):
        loaded.enclose([(-2, -1)], budget=budget())
    with pytest.raises(ValueError):
        s.load_objective(data, objective=x*x, args=x)

    floating = s.Add(x*x, s.Float(0.5), evaluate=False)
    data = s.compile_objective(floating, x).to_bytes()
    loaded = s.load_objective(data, objective=floating, args=x)
    assert loaded.evaluate(1)[0] == 1.5
    with pytest.raises(ValueError, match="exact source literals"):
        loaded.enclose([(0, 1)], budget=budget())


def test_restored_program_can_generate_fresh_stationary_evidence_and_minimize():
    x = s.Symbol("saved_positive_x", positive=True)
    source = (x-2)**2
    data = s.compile_objective(source, x).to_bytes()
    loaded = s.load_objective(data, objective=source, args=x)
    with pytest.raises(ValueError, match="assumption"):
        loaded.enclose([(-2, -1)], budget=budget())
    receipt = loaded.certify_stationary_point([(1, 3)], budget=budget())
    assert receipt.kind == "strict_minimum"
    candidate = loaded.minimize(4, budget=budget())
    assert candidate.point == pytest.approx([2])
    assert loaded.to_bytes() == data


def test_fresh_process_replays_source_and_both_derivative_stages(tmp_path):
    x = s.Symbol("saved_process_x", real=True)
    source = (x-2)**2
    data = s.compile_objective(source, x).to_bytes()
    artifact = tmp_path / "objective.json"
    artifact.write_bytes(data)
    shell_root = Path(s.__file__).resolve().parents[1]
    native_path = Path(_native.__file__).resolve()
    script = textwrap.dedent("""
        import importlib.util
        import json
        from pathlib import Path
        import sys
        spec = importlib.util.spec_from_file_location("fsym_python", sys.argv[2])
        native = importlib.util.module_from_spec(spec)
        sys.modules["fsym_python"] = native
        spec.loader.exec_module(native)
        sys.path.insert(0, sys.argv[1])
        import sympy as s
        from sympy.core import _native
        assert _native is native
        assert Path(_native.__file__).resolve() == Path(sys.argv[2]).resolve()
        assert Path(s.__file__).resolve() == Path(sys.argv[1]) / "sympy" / "__init__.py"
        x = s.Symbol("saved_process_x", real=True)
        source = (x-2)**2
        data = Path(sys.argv[3]).read_bytes()
        program = s.load_objective(data, objective=source, args=x)
        assert program.variables[0] is x
        assert program.to_bytes() == data
        print(json.dumps(program.evaluate(3)))
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(shell_root), str(native_path), str(artifact)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert json.loads(result.stdout) == [1.0, [2.0], [[2.0]]]
