"""Portable root witnesses require exact receiver sources and independent proof."""

from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding


LIMITS = {
    "max_steps": 1 << 45,
    "max_memory_bytes": 1 << 40,
    "max_allocations": 1 << 30,
}
REMAINING = {
    "max_steps": "remaining_steps",
    "max_memory_bytes": "remaining_memory_bytes",
    "max_allocations": "remaining_allocations",
}


def budget(**limits):
    return s.EvaluationBudget(**{**LIMITS, **limits})


def encode(document):
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@pytest.fixture
def root():
    x = s.Symbol("portable_root_x", positive=True)
    source = x*x - 2
    program = s.compile_residual_system(source, x)
    receipt = program.certify_root_box(
        [(F(13, 10), F(3, 2))], center=[F(7, 5)],
        preconditioner=[[F(5, 14)]], precision=24, budget=budget(),
    )
    return x, source, program, receipt


def assert_same_claim(actual, expected):
    assert type(actual) is s.RootBoxReceipt
    assert actual.residuals == expected.residuals
    assert [(v.name, v.plain, v.identity_hex) for v in actual.variables] == [
        (v.name, v.plain, v.identity_hex) for v in expected.variables
    ]
    for field in (
        "domain", "center", "preconditioner", "precision_digits",
        "root_enclosure", "contraction_bound", "claim",
    ):
        assert getattr(actual, field) == getattr(expected, field)


def test_source_and_compiled_receivers_replay_without_serialized_acceptance(root):
    x, source, program, receipt = root
    data = receipt.to_bytes()
    assert type(data) is bytes
    document = json.loads(data)
    assert list(document) == ["schema_version", "context_digest", "residuals", "variables", "witness"]
    assert "root_enclosure" not in document and "accepted" not in document
    assert "contraction_bound" not in document
    assert encode(document) == data
    assert_same_claim(program.replay_root_certificate(data), receipt)
    replayed = s.RootBoxReceipt.from_bytes(data, residuals=source, args=x)
    assert_same_claim(replayed, receipt)
    assert replayed.to_bytes() == data
    lo, hi = replayed.root_enclosure[0]
    assert lo*lo <= 2 <= hi*hi
    assert replayed.contraction_bound == s.Rational(1, 14)
    for field in ("_receipt", "domain", "root_enclosure"):
        with pytest.raises(AttributeError):
            setattr(replayed, field, None)


@pytest.mark.parametrize("backend", ["frankennumpy", "frankenscipy", "frankenscipy_analytic", "sparse"])
def test_existing_certified_solvers_now_produce_independently_portable_proofs(root, backend):
    x, source, program, _ = root
    receipt = program.solve_certified(1, backend=backend, precision=24, budget=budget())
    encoded = receipt.to_bytes(budget=budget())
    restored_program = s.load_residual_system(program.to_bytes(), residuals=source, args=[x])
    assert_same_claim(restored_program.replay_root_certificate(encoded, budget=budget()), receipt)
    assert_same_claim(
        s.RootBoxReceipt.from_bytes(encoded, residuals=source, args=x, budget=budget()),
        receipt,
    )


def test_receiver_source_order_typed_identity_and_original_argument_order_are_exact(root):
    x, source, _, receipt = root
    data = receipt.to_bytes()
    for other in (s.Symbol(x.name), s.Symbol(x.name, negative=True), s.Dummy(x.name)):
        with pytest.raises(ValueError, match="sources or ordered symbol identities"):
            s.RootBoxReceipt.from_bytes(data, residuals=other*other-2, args=other)
    with pytest.raises(ValueError, match="sources or ordered symbol identities"):
        s.RootBoxReceipt.from_bytes(data, residuals=source*2, args=x)

    y = s.Symbol("portable_root_y")
    a = s.Symbol("portable_root_a")
    sources = [a+y-3, a-y-1]
    program = s.compile_residual_system(sources, [y, a])
    accepted = program.certify_root_box(
        [(F(9, 10), F(11, 10)), (F(19, 10), F(21, 10))],
        center=[1, 2], preconditioner=[[F(1, 2), F(-1, 2)], [F(1, 2), F(1, 2)]],
        budget=budget(),
    )
    encoded = accepted.to_bytes()
    assert_same_claim(s.RootBoxReceipt.from_bytes(encoded, residuals=sources, args=[[y, a]]), accepted)
    for residuals, args in ((sources[::-1], [y, a]), (sources, [a, y])):
        with pytest.raises(ValueError, match="sources or ordered symbol identities"):
            s.RootBoxReceipt.from_bytes(encoded, residuals=residuals, args=args)


def test_replay_uses_captured_sources_after_mutable_input_rebinding():
    x = s.Symbol("portable_mutable_x")
    source = x*x-2
    matrix = s.Matrix([source])
    arguments = [x]
    program = s.compile_residual_system(matrix, arguments)
    receipt = program.certify_root_box(
        [(F(13, 10), F(3, 2))], center=[F(7, 5)], preconditioner=[[F(5, 14)]],
        budget=budget(),
    )
    data = receipt.to_bytes()
    matrix[0, 0] = x*x-3
    arguments[0] = s.Symbol("rebound_coordinate")
    assert_same_claim(program.replay_root_certificate(data), receipt)
    with pytest.raises(ValueError):
        s.RootBoxReceipt.from_bytes(data, residuals=matrix, args=arguments)


def test_source_only_piecewise_proofs_need_no_symbolic_derivative_program():
    x = s.Symbol("portable_piecewise_x")
    source = s.Piecewise((s.log(x), x > 0), (999, True), evaluate=False)
    receipt = s.certify_root_box(
        source, x, [(F(9, 10), F(11, 10))], center=[1], preconditioner=[[1]],
        precision=24, budget=budget(),
    )
    replayed = s.RootBoxReceipt.from_bytes(receipt.to_bytes(), residuals=source, args=x)
    assert_same_claim(replayed, receipt)
    assert replayed.root_enclosure[0][0] <= 1 <= replayed.root_enclosure[0][1]


def test_held_source_poles_cannot_be_removed_or_crossed_by_a_mutated_witness():
    x = s.Symbol("portable_pole_x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    source = s.Add(x, -1, s.Mul(0, reciprocal, evaluate=False), evaluate=False)
    program = s.compile_residual_system(source, x)
    receipt = program.certify_root_box(
        [(F(9, 10), F(11, 10))], center=[1], preconditioner=[[1]], budget=budget(),
    )
    data = receipt.to_bytes()
    with pytest.raises(ValueError, match="sources or ordered symbol identities"):
        s.RootBoxReceipt.from_bytes(data, residuals=x-1, args=x)
    document = json.loads(data)
    document["witness"]["domain"][0]["radius"] = document["witness"]["center"][0]
    with pytest.raises(ValueError, match="zero|domain"):
        program.replay_root_certificate(encode(document))
    assert_same_claim(program.replay_root_certificate(data), receipt)


def test_native_valid_negative_root_cannot_bypass_captured_positive_assumptions(root):
    x, source, program, _ = root
    native_q = _native.py_rational
    negative = program.native_system.certify_root_box(
        [(native_q(-3, 2), native_q(-13, 10))], [native_q(-7, 5)],
        [[native_q(-5, 14)]], 24, budget=budget(),
    )
    # The core checks the literal mathematical source; the public adapter must
    # also enforce the declared facts encoded in the caller's typed symbols.
    data = negative.to_bytes(budget=budget())
    with pytest.raises(ValueError, match="assumption"):
        program.replay_root_certificate(data, budget=budget())
    with pytest.raises(ValueError, match="assumption"):
        s.RootBoxReceipt.from_bytes(data, residuals=source, args=x, budget=budget())


@pytest.mark.parametrize("change", ["inverse", "context", "schema", "accepted", "enclosure"])
def test_changed_claim_context_and_witnesses_refuse_without_poisoning_program(root, change):
    _, _, program, receipt = root
    data = receipt.to_bytes()
    document = json.loads(data)
    if change == "inverse":
        document["witness"]["preconditioner"][0][0] = document["witness"]["center"][0]
    elif change == "context":
        document["context_digest"][0] ^= 1
    elif change == "schema":
        document["schema_version"] += 1
    elif change == "accepted":
        document["accepted"] = True
    else:
        document["root_enclosure"] = []
    with pytest.raises(ValueError):
        program.replay_root_certificate(encode(document), budget=budget())
    assert_same_claim(program.replay_root_certificate(data), receipt)


def test_bytes_admission_and_canonical_depth_bounds_precede_python_input_hooks(root):
    x, source, program, receipt = root
    data = receipt.to_bytes()

    class BytesSubclass(bytes):
        def __len__(self):
            raise AssertionError("byte subclass hook must not run")

        def __bytes__(self):
            raise AssertionError("byte subclass hook must not run")

    for bad in (BytesSubclass(data), bytearray(data), memoryview(data), data.decode()):
        with pytest.raises(TypeError, match="exact built-in bytes"):
            program.replay_root_certificate(bad)
        with pytest.raises(TypeError, match="exact built-in bytes"):
            s.RootBoxReceipt.from_bytes(bad, residuals=object(), args=object())
    for bad in (b" "+data, b"["*97, b" "*((1 << 20)+1), b'{"schema_version":1,'+data[1:]):
        with pytest.raises(ValueError):
            s.RootBoxReceipt.from_bytes(bad, residuals=source, args=x)
    with pytest.raises(TypeError):
        s.RootBoxReceipt.from_bytes(data, residuals=source)


@pytest.mark.parametrize("operation", ["export", "replay"])
def test_every_resource_allowance_is_cumulative_through_final_publication(root, operation):
    _, _, program, receipt = root
    data = receipt.to_bytes()

    def run(ledger):
        if operation == "export":
            return receipt.to_bytes(budget=ledger)
        return program.replay_root_certificate(data, budget=ledger)

    probe = budget()
    run(probe)
    spent = {key: limit-getattr(probe, REMAINING[key]) for key, limit in LIMITS.items()}
    assert all(value > 0 for value in spent.values())
    if operation == "export":
        assert spent["max_memory_bytes"] >= 2*len(data)
    else:
        assert spent["max_memory_bytes"] >= 64*1024*1024
    for key, consumed in spent.items():
        bounded = budget(**{key: consumed-1})
        with pytest.raises(s.EvaluationBudgetExceeded):
            run(bounded)
        assert getattr(bounded, REMAINING[key]) < consumed-1
    exact = budget(**spent)
    run(exact)
    with pytest.raises(s.EvaluationBudgetExceeded):
        run(exact)
    cancelled = budget()
    cancelled.cancel()
    before = tuple(getattr(cancelled, field) for field in REMAINING.values())
    with pytest.raises(s.EvaluationCancelled):
        run(cancelled)
    assert tuple(getattr(cancelled, field) for field in REMAINING.values()) == before


def test_proof_replays_in_a_fresh_process_without_a_compiled_program(root):
    x, _, _, receipt = root
    data = receipt.to_bytes()
    shell_root = Path(s.__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.util
        import json
        from pathlib import Path
        import sys
        spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])
        native = importlib.util.module_from_spec(spec)
        sys.modules[sys.argv[1]] = native
        spec.loader.exec_module(native)
        sys.path.insert(0, sys.argv[3])
        import sympy as s
        from sympy.core import _native, _symbol_binding
        assert _native is native
        assert Path(s.__file__).resolve() == Path(sys.argv[3]) / 'sympy' / '__init__.py'
        x = s.Symbol('portable_root_x', positive=True)
        data = bytes.fromhex(sys.stdin.read())
        receipt = s.RootBoxReceipt.from_bytes(data, residuals=x*x-2, args=x)
        lo, hi = receipt.root_enclosure[0]
        assert lo*lo <= 2 <= hi*hi
        print(json.dumps({'preserved': receipt.to_bytes() == data,
                          'identity': _symbol_binding(x).identity_hex,
                          'type': type(receipt).__name__}))
    """)
    process = subprocess.run(
        [sys.executable, "-I", "-c", script, _native.__name__, _native.__file__, str(shell_root)],
        input=data.hex(), capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stdout+"\n"+process.stderr
    assert json.loads(process.stdout) == {
        "preserved": True, "identity": _symbol_binding(x).identity_hex, "type": "RootBoxReceipt",
    }
