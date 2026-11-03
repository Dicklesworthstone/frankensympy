"""Public persistence replays retained proofs against explicitly expected source."""

from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def _encode(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _budget():
    # Interval checking reserves 64 MiB per native invocation, cumulatively.
    return s.EvaluationBudget(
        max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30,
    )


@pytest.fixture
def system():
    x = s.Symbol("transport_x", real=True)
    y = s.Symbol("transport_y", real=True)
    residuals = (x*x + y - 5, x - y - 1)
    program = s.compile_residual_system(residuals, (x, y))
    return x, y, residuals, program.to_bytes()


def test_round_trip_rebinds_layout_and_preserves_every_evaluation_entry_point(system):
    x, y, residuals, data = system
    restored = s.load_residual_system(data, residuals=residuals, args=[(x, y)])
    assert type(data) is bytes
    assert restored.to_bytes() == data
    assert restored.variables[0] is x
    assert restored.variables[1] is y
    assert restored.evaluate((2, 1)) == ([0, 0], [[4, 1], [1, -1]])
    assert restored.residuals((3, 2)) == [6, 0]
    assert restored.jacobian((3, 2)) == [[6, 1], [1, -1]]
    assert restored.evaluate_batch([[(2, 1)], [(3, 2)]]) == [
        ([0, 0], [[4, 1], [1, -1]]),
        ([6, 0], [[6, 1], [1, -1]]),
    ]
    with pytest.raises(TypeError):
        restored.evaluate(2, 1)
    with pytest.raises(AttributeError):
        restored._native_system = None


def test_public_exports_and_native_transport_apply_the_same_binding(system):
    from sympy.utilities import load_residual_system
    from sympy.utilities.lambdify import load_residual_system as direct_loader

    assert load_residual_system is direct_loader is s.load_residual_system
    x, y, residuals, data = system
    native = _native.VerifiedResidualSystem.from_bytes(
        data, residuals=[_native_expr(value) for value in residuals],
        variables=[_symbol_binding(x), _symbol_binding(y)],
    )
    assert native.to_bytes() == data
    assert native.evaluate([2, 1]) == ([0, 0], [[4, 1], [1, -1]])


def test_restored_program_uses_the_callers_existing_ledger(system):
    x, y, residuals, data = system
    restored = s.load_residual_system(data, residuals=residuals, args=(x, y))
    budget = _budget()
    assert restored.residuals(2, 1, budget=budget) == [0, 0]
    remaining = budget.remaining_steps
    assert restored.jacobian(3, 2, budget=budget) == [[6, 1], [1, -1]]
    assert budget.remaining_steps < remaining < 1 << 45
    with pytest.raises(s.EvaluationBudgetExceeded):
        restored.evaluate(2, 1, budget=s.EvaluationBudget(max_steps=0))
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        restored.evaluate_batch([[2, 1]], budget=budget)


def test_loader_requires_explicit_residuals_and_arguments(system):
    x, y, residuals, data = system
    with pytest.raises(TypeError):
        s.load_residual_system(data, args=(x, y))
    with pytest.raises(TypeError):
        s.load_residual_system(data, residuals=residuals)
    with pytest.raises(TypeError):
        s.load_residual_system(data, residuals, (x, y))


def test_valid_artifact_for_another_source_cannot_replace_the_requested_program(system):
    x, y, residuals, data = system
    with pytest.raises(ValueError, match="expected exact source"):
        s.load_residual_system(data, residuals=(x*x+y-6, residuals[1]), args=(x, y))
    with pytest.raises(ValueError, match="expected typed order"):
        s.load_residual_system(data, residuals=residuals, args=(y, x))


def test_same_printed_symbol_names_do_not_erase_assumption_identity(system):
    x, y, _, data = system
    plain_x = s.Symbol(x.name)
    assert str(plain_x) == str(x)
    with pytest.raises(ValueError, match="expected typed order"):
        s.load_residual_system(
            data, residuals=(plain_x*plain_x+y-5, plain_x-y-1), args=(plain_x, y),
        )


def test_dummy_symbols_are_explicitly_matched_without_automatic_remapping():
    x, other = s.Dummy("transport_dummy"), s.Dummy("transport_dummy")
    source = x - 1
    data = s.compile_residual_system(source, x).to_bytes()
    restored = s.load_residual_system(data, residuals=source, args=x)
    assert restored.variables[0] is x
    assert restored.residuals(2) == [1]
    with pytest.raises(ValueError, match="expected typed order"):
        s.load_residual_system(data, residuals=other-1, args=other)


def test_loading_does_not_call_the_compilation_constructor(system, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    native_class = _native.VerifiedResidualSystem
    calls = []

    class ImportOnly:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("loading must replay the retained bundle")

        @staticmethod
        def from_bytes(data, **expected):
            calls.append(True)
            return native_class.from_bytes(data, **expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("loading must not regenerate derivative proofs")

    monkeypatch.setattr(module._native, "VerifiedResidualSystem", ImportOnly)
    monkeypatch.setattr(module, "compile_residual_system", forbidden)
    x, y, residuals, data = system
    restored = module.load_residual_system(data, residuals=residuals, args=(x, y))
    assert restored.residuals(2, 1) == [0, 0]
    assert calls == [True]


@pytest.mark.parametrize("mutation", ["proof", "bytecode", "source", "sparsity", "variables"])
def test_corrupt_retained_evidence_refuses_instead_of_being_regenerated(system, mutation):
    x, y, residuals, data = system
    document = json.loads(data)
    jacobian = document["jacobian"]
    if mutation == "proof":
        jacobian["entries"][0]["derivation"]["steps"] = []
    elif mutation == "bytecode":
        jacobian["entries"][0]["compiled_deriv"]["ops"] = [{"LoadConst": 0.0}]
    elif mutation == "source":
        jacobian["residuals"][0] = jacobian["residuals"][1]
    elif mutation == "sparsity":
        jacobian["entries"] = []
        jacobian["pattern"]["nonzeros"] = []
        jacobian["pattern"]["nnz"] = 0
        jacobian["column_colors"] = [0, 0]
        jacobian["num_colors"] = 1
    else:
        jacobian["vars"].reverse()
    with pytest.raises(ValueError):
        s.load_residual_system(_encode(document), residuals=residuals, args=(x, y))
    # Failed admission cannot mutate a separately restored checked program.
    assert s.load_residual_system(data, residuals=residuals, args=(x, y)).residuals(2, 1) == [0, 0]


@pytest.mark.parametrize("field", ["receipt", "verified", "has_inexact_source", "compiled"])
def test_serialized_status_and_provenance_flags_are_rejected(system, field):
    x, y, residuals, data = system
    document = json.loads(data)
    assert set(document) == {"schema_version", "jacobian"}
    document[field] = False if field == "has_inexact_source" else True
    with pytest.raises(ValueError):
        s.load_residual_system(_encode(document), residuals=residuals, args=(x, y))


@pytest.mark.parametrize("data", [
    b"", b"\xff", b"{}",
    b'{"schema_version":0,"jacobian":null}',
    b'{"schema_version":4294967295,"jacobian":null}',
    b'{"schema_version":1,"schema_version":1,"jacobian":null}',
    b"[" * 97 + b"0" + b"]" * 97,
])
def test_malformed_incompatible_and_deep_bytes_are_refused(system, data):
    x, y, residuals, _ = system
    with pytest.raises(ValueError):
        s.load_residual_system(data, residuals=residuals, args=(x, y))


def test_mutable_byte_sources_and_conversion_hooks_are_refused_before_source_work(system, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x, y, residuals, data = system

    class BytesSubclass(bytes):
        def __bytes__(self):
            raise AssertionError("bytes subclasses must not invoke hooks")

    class BytesHook:
        def __bytes__(self):
            raise AssertionError("arbitrary __bytes__ must not run")

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted bytes must be refused before caller-source work")

    monkeypatch.setattr(module, "_residual_source", forbidden)
    for bad in (bytearray(data), memoryview(data), BytesSubclass(data), BytesHook(), data.decode()):
        with pytest.raises(TypeError, match="built-in bytes"):
            module.load_residual_system(bad, residuals=residuals, args=(x, y))
        with pytest.raises(TypeError, match="built-in bytes"):
            _native.VerifiedResidualSystem.from_bytes(bad, residuals=object(), variables=object())


def test_byte_limit_is_enforced_before_expected_source_conversion(system, monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized bytes must not cause source traversal")

    monkeypatch.setattr(module, "_residual_source", forbidden)
    data = b" " * (16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        module.load_residual_system(data, residuals=object(), args=object())
    with pytest.raises(ValueError, match="16 MiB"):
        _native.VerifiedResidualSystem.from_bytes(data, residuals=object(), variables=object())


def test_expected_source_keeps_the_existing_python_literal_and_axis_caps(system):
    x, y, residuals, data = system
    with pytest.raises(ValueError, match="literal"):
        s.load_residual_system(data, residuals=[1 << 4096], args=(x, y))
    too_many = [s.Symbol(f"transport_limit_{index}") for index in range(257)]
    with pytest.raises(ValueError, match="variable"):
        s.load_residual_system(data, residuals=residuals, args=too_many)
    with pytest.raises(ValueError, match="duplicate"):
        s.load_residual_system(data, residuals=residuals, args=(x, x))


def test_float_origin_guard_is_recaptured_from_the_explicit_expected_source():
    x = s.Symbol("transport_float_x")
    floating = s.Add(x, s.Float(0.5), evaluate=False)
    data = s.compile_residual_system(floating, x).to_bytes()
    restored = s.load_residual_system(data, residuals=floating, args=x)
    assert restored.residuals(1) == [1.5]
    with pytest.raises(ValueError, match="exact source literals"):
        restored.enclose([(0, 1)])
    with pytest.raises(ValueError, match="exact source literals"):
        restored.certify_root_box([(-1, 0)])
    # The caller can explicitly request the corresponding exact mathematical
    # source; this is a fresh declaration, never a trusted flag from the file.
    exact = s.Add(x, s.Rational(1, 2), evaluate=False)
    exact_program = s.load_residual_system(data, residuals=exact, args=x)
    assert exact_program.enclose([(0, 0)]).residual_values == ((s.Rational(1, 2), s.Rational(1, 2)),)


def test_exact_held_source_domain_guards_survive_save_and_load():
    x = s.Symbol("transport_guard_x")
    guarded = s.Add(x, s.Mul(0, s.log(x), evaluate=False), evaluate=False)
    data = s.compile_residual_system(guarded, x).to_bytes()
    restored = s.load_residual_system(data, residuals=guarded, args=x)
    assert restored.residuals(1) == [1]
    with pytest.raises(ValueError):
        restored.jacobian(-1)
    with pytest.raises(ValueError):
        restored.enclose([(-2, -1)])
    with pytest.raises(ValueError, match="expected exact source"):
        s.load_residual_system(data, residuals=x, args=x)


def test_saved_matrix_program_is_a_snapshot_not_a_live_surface_reference():
    x = s.Symbol("transport_snapshot_x")
    expected = [x-1, 2*x-2]
    residuals = s.Matrix([[expected[0]], [expected[1]]])
    program = s.compile_residual_system(residuals, x)
    residuals[0, 0] = x - 9
    data = program.to_bytes()
    restored = s.load_residual_system(data, residuals=expected, args=x)
    assert restored.residuals(1) == [0, 0]
    with pytest.raises(ValueError, match="expected exact source"):
        s.load_residual_system(data, residuals=residuals, args=x)


def test_restored_assumptions_and_root_receipt_remain_bound_to_the_expected_source():
    x = s.Symbol("transport_positive_x", positive=True)
    binding = _symbol_binding(x)
    source = x*x - 2
    data = s.compile_residual_system(source, x).to_bytes()
    restored = s.load_residual_system(data, residuals=source, args=x)
    assert restored.variables[0] is x
    with pytest.raises(ValueError, match="assumption"):
        restored.enclose([(-2, -1)])
    budget = _budget()
    receipt = restored.certify_root_box(
        [(Fraction(7, 5), Fraction(3, 2))], precision=16, budget=budget,
    )
    assert receipt.residuals == (source,)
    # Programs retain caller-owned Python objects. Accepted receipts expose
    # immutable native bindings, whose complete typed identity must match.
    assert len(receipt.variables) == 1
    variable = receipt.variables[0]
    assert type(variable) is _native.SymbolBinding
    assert (variable.name, variable.plain, variable.identity_hex) == (
        binding.name, binding.plain, binding.identity_hex,
    )
    lo, hi = receipt.root_enclosure[0]
    assert lo*lo <= 2 <= hi*hi
    assert 0 <= receipt.contraction_bound < 1
    assert budget.remaining_steps < 1 << 45
    # A root receipt belongs to this verification call, not to the saved data.
    assert restored.to_bytes() == data


def test_checked_bytes_load_in_a_fresh_process_with_explicit_typed_declarations(tmp_path):
    x = s.Symbol("transport_process_x", positive=True)
    source = x*x - 2
    data = s.compile_residual_system(source, x).to_bytes()
    artifact = tmp_path / "verified-program.json"
    artifact.write_bytes(data)
    shell_root = Path(s.__file__).resolve().parents[1]
    native_path = Path(_native.__file__).resolve()
    native_root = native_path.parent.parent if native_path.name == "__init__.py" else native_path.parent
    script = textwrap.dedent("""
        from fractions import Fraction
        import json
        from pathlib import Path
        import sys

        sys.path[:0] = [sys.argv[1], sys.argv[2]]
        import sympy as s
        from sympy.core import _native, _symbol_binding

        assert Path(s.__file__).resolve() == Path(sys.argv[1]) / "sympy" / "__init__.py"
        x = s.Symbol("transport_process_x", positive=True)
        binding = _symbol_binding(x)
        source = x*x - 2
        data = Path(sys.argv[3]).read_bytes()
        restored = s.load_residual_system(data, residuals=source, args=x)
        assert restored.variables[0] is x
        assert restored.residuals(2) == [2]
        assert restored.to_bytes() == data
        budget = s.EvaluationBudget(
            max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30,
        )
        receipt = restored.certify_root_box(
            [(Fraction(7, 5), Fraction(3, 2))], precision=16, budget=budget,
        )
        assert receipt.residuals == (source,)
        assert len(receipt.variables) == 1
        variable = receipt.variables[0]
        assert type(variable) is _native.SymbolBinding
        assert (variable.name, variable.plain, variable.identity_hex) == (
            binding.name, binding.plain, binding.identity_hex,
        )
        lo, hi = receipt.root_enclosure[0]
        assert lo*lo <= 2 <= hi*hi
        assert 0 <= receipt.contraction_bound < 1
        print(json.dumps({
            "value": restored.residuals(2),
            "typed_identity": _symbol_binding(x).identity_hex,
            "preserved": restored.to_bytes() == data,
        }))
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(shell_root), str(native_root), str(artifact)],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert json.loads(completed.stdout) == {
        "value": [2.0], "typed_identity": _symbol_binding(x).identity_hex, "preserved": True,
    }
