"""Portable brackets prove root existence only after independent source replay."""

from fractions import Fraction as F
import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import sympy as s


LIMITS = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
REMAINING = dict(max_steps="remaining_steps", max_memory_bytes="remaining_memory_bytes",
                 max_allocations="remaining_allocations")


def budget(**changes):
    return s.EvaluationBudget(**{**LIMITS, **changes})


def encode(document):
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def replay(data, x, source, *, limits=None, tolerance=F(1, 10000), ledger=None):
    return s.ScalarRootBracketReceipt.from_bytes(
        data, expression=source, limits=(x, 1, 2) if limits is None else limits,
        tolerance=tolerance, budget=ledger,
    )


def native_replay(data, receipt, *, ledger=None):
    from sympy.core import _native

    native = receipt._receipt
    lower, upper = native.interval
    return _native.replay_scalar_root_bracket_native(
        data, native.expression, native.variable, lower, upper, native.absolute_tolerance,
        budget=ledger,
    )


def assert_same_claim(actual, expected):
    assert type(actual) is s.ScalarRootBracketReceipt
    assert actual.expression == expected.expression
    assert (actual.variable.name, actual.variable.plain, actual.variable.identity_hex) == (
        expected.variable.name, expected.variable.plain, expected.variable.identity_hex,
    )
    for field in (
        "interval", "bounds", "midpoint", "error_bound", "absolute_tolerance",
        "precision_digits", "endpoint_values", "claim",
    ):
        assert getattr(actual, field) == getattr(expected, field)


@pytest.fixture
def repeated_root():
    x = s.Symbol("portable_bracket_x", positive=True)
    source = (x-1)**3
    receipt = s.certified_root_bracket(source, (x, 1, 2), precision=24, budget=budget())
    return x, source, receipt


def test_public_and_native_receivers_share_canonical_existence_witnesses(repeated_root):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    assert type(data) is bytes and len(data) <= 1 << 20
    document = json.loads(data)
    assert document["schema_version"] == 1
    assert set(document) == {
        "schema_version", "context_digest", "residuals", "variables",
        "lower", "upper", "absolute_tolerance", "witness",
    }
    assert set(document["witness"]) == {"lower", "upper", "precision_digits"}
    assert encode(document) == data
    assert receipt._receipt.to_bytes() == data
    restored = replay(data, x, source)
    assert_same_claim(restored, receipt)
    assert restored.bounds == (1, 1)
    assert restored.to_bytes() == data
    native = native_replay(data, receipt)
    assert_same_claim(s.ScalarRootBracketReceipt(native), receipt)
    assert native.to_bytes() == data


@pytest.mark.parametrize("lower,upper", [(0, 1), (-1, 1), (0, 0)])
def test_native_generated_endpoint_and_singleton_receipts_replay_publicly(lower, upper):
    from sympy.core import _native, _symbol_binding

    x = s.Symbol("native_portable_bracket_x", real=True)
    native = _native.certified_scalar_root_bracket_native(
        x._value, _symbol_binding(x), _native.py_integer(lower), _native.py_integer(upper),
        _native.py_rational(1, 10000), 24, 128, budget=budget(),
    )
    data = native.to_bytes()
    restored = replay(data, x, x, limits=(x, lower, upper))
    assert restored.bounds == (0, 0)
    assert restored.interval == (lower, upper)
    assert restored.endpoint_values == ((0, 0), (0, 0))
    assert restored.to_bytes() == data


def test_nonsmooth_sources_retain_builtin_identity_despite_an_opaque_name():
    x = s.Symbol("portable_absolute_bracket_x", real=True)
    opaque = s.Function("abs")
    source = s.Abs(x, evaluate=False)
    receipt = s.certified_root_bracket(source, (x, 0, 1))
    data = receipt.to_bytes()
    assert b'"Function":["Abs",' in data
    restored = replay(data, x, source, limits=(x, 0, 1))
    assert_same_claim(restored, receipt)
    assert type(restored.expression) is s.Abs
    assert restored.bounds == (0, 0)
    assert restored.to_bytes() == data
    with pytest.raises(NotImplementedError):
        replay(data, x, opaque(x), limits=(x, 0, 1))


def test_a_multiple_root_interval_remains_existence_only_after_transport():
    x = s.Symbol("portable_multiple_bracket_x", real=True)
    source = x*(x-1)*(x+1)
    receipt = s.certified_root_bracket(source, (x, -2, 2), tolerance=4, max_bisections=0)
    restored = replay(receipt.to_bytes(), x, source, limits=(x, -2, 2), tolerance=4)
    assert_same_claim(restored, receipt)
    assert restored.bounds == (-2, 2)
    assert restored.claim == "at least one real root in the certified interval"
    assert not isinstance(restored, s.RootBoxReceipt)
    with pytest.raises(TypeError, match="native"):
        s.RootBoxReceipt(restored._receipt)


def test_changed_endpoint_evidence_refuses_without_regenerating(repeated_root, monkeypatch):
    from sympy.core import _native

    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    document = json.loads(data)
    # This statement has a root at one, but the supplied point two is not a
    # root. Candidate search would conceal this invalid transported witness.
    document["witness"]["lower"] = document["upper"]
    document["witness"]["upper"] = document["upper"]

    def forbidden(*args, **kwargs):
        raise AssertionError("bracket replay must not regenerate a root witness")

    monkeypatch.setattr(_native, "certified_scalar_root_bracket_native", forbidden)
    monkeypatch.setattr(s, "certified_root_bracket", forbidden)
    with pytest.raises(ValueError, match="opposite signs|zero value"):
        replay(encode(document), x, source)
    assert_same_claim(replay(data, x, source), receipt)


def test_sources_and_typed_axes_bind_before_decoding_foreign_witnesses(repeated_root):
    x, source, receipt = repeated_root
    document = json.loads(receipt.to_bytes())
    document["witness"] = {"not_a_bracket": True}
    data = encode(document)
    for other in (s.Symbol(x.name), s.Symbol(x.name, nonnegative=True), s.Dummy(x.name, positive=True)):
        with pytest.raises(ValueError, match="sources or ordered symbol identities"):
            replay(data, other, (other-1)**3)
    with pytest.raises(ValueError, match="sources or ordered symbol identities"):
        replay(data, x, s.Mul(1, source, evaluate=False))


@pytest.mark.parametrize("change", ["lower", "upper", "tolerance", "context"])
def test_complete_original_interval_tolerance_and_context_bind_before_witnesses(repeated_root, change):
    x, source, receipt = repeated_root
    document = json.loads(receipt.to_bytes())
    document["witness"] = {"not_a_bracket": True}
    limits = {"lower": (x, F(1, 2), 2), "upper": (x, 1, 3)}.get(change, (x, 1, 2))
    tolerance = F(1, 1000) if change == "tolerance" else F(1, 10000)
    expected = "bounds or tolerance"
    if change == "context":
        document["context_digest"][0] ^= 1
        expected = "assumption universe"
    with pytest.raises(ValueError, match=expected):
        replay(encode(document), x, source, limits=limits, tolerance=tolerance)


@pytest.mark.parametrize("hidden", [False, True])
def test_narrow_witnesses_cannot_hide_poles_elsewhere_in_the_original_interval(hidden):
    x = s.Symbol("portable_original_bracket_domain_x", real=True)
    pole = s.Pow(x, -1, evaluate=False)
    source = (s.Add(x-1, s.Mul(0, pole, evaluate=False), evaluate=False)
              if hidden else s.Mul(x-1, pole, evaluate=False))
    receipt = s.certified_root_bracket(source, (x, F(1, 2), 2), budget=budget())
    data = receipt.to_bytes()
    document = json.loads(data)
    endpoint = s.certified_root_bracket(x, (x, 0, 2)).to_bytes()
    document["lower"] = json.loads(endpoint)["lower"]
    # The source and small accepted bracket still match. Only the full original
    # domain changed, and its reciprocal pole must be evaluated independently.
    with pytest.raises(ValueError, match="zero|domain"):
        replay(encode(document), x, source, limits=(x, 0, 2))
    assert_same_claim(replay(data, x, source, limits=(x, F(1, 2), 2)), receipt)


def test_native_transport_cannot_bypass_public_declared_assumptions():
    from sympy.core import _native, _symbol_binding

    x = s.Symbol("portable_positive_bracket_x", positive=True)
    source = x+1
    native = _native.certified_scalar_root_bracket_native(
        source._value, _symbol_binding(x), _native.py_integer(-2), _native.py_integer(0),
        _native.py_rational(1, 10000), budget=budget(),
    )
    with pytest.raises(ValueError, match="positive|assumption"):
        replay(native.to_bytes(), x, source, limits=(x, -2, 0))


@pytest.mark.parametrize("change", ["schema", "accepted", "unique", "cached_values", "precision", "reverse"])
def test_unknown_or_unproved_claims_refuse_without_poisoning_valid_replay(repeated_root, change):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    document = json.loads(data)
    if change == "schema":
        document["schema_version"] += 1
    elif change in ("accepted", "unique"):
        document[change] = True
    elif change == "cached_values":
        document["witness"]["endpoint_values"] = [[0, 0], [0, 0]]
    elif change == "precision":
        document["witness"]["precision_digits"] = 129
    else:
        document["witness"]["lower"] = document["upper"]
        document["witness"]["upper"] = document["lower"]
    with pytest.raises(ValueError):
        replay(encode(document), x, source, ledger=budget())
    assert_same_claim(replay(data, x, source), receipt)


def test_byte_admission_precedes_source_and_conversion_hooks(repeated_root, monkeypatch):
    x, _, receipt = repeated_root
    data = receipt.to_bytes()
    certified = importlib.import_module("sympy.solvers.certified")

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected bytes reached mathematical preparation")

    class BytesSubclass(bytes):
        __len__ = forbidden
        __bytes__ = forbidden

    class Hostile:
        _sympy_ = forbidden
        __int__ = forbidden
        __float__ = forbidden

    monkeypatch.setattr(certified, "_numeric_source", forbidden)
    for bad in (BytesSubclass(data), bytearray(data), memoryview(data), data.decode()):
        with pytest.raises(TypeError, match="exact built-in bytes"):
            replay(bad, x, Hostile(), limits=(x, Hostile(), Hostile()))
        with pytest.raises(TypeError, match="exact built-in bytes"):
            native_replay(bad, receipt)
    oversized = b" "*((1 << 20)+1)
    with pytest.raises(ValueError, match="1 MiB|byte limit"):
        replay(oversized, x, Hostile(), limits=(x, Hostile(), Hostile()))
    with pytest.raises(ValueError, match="1 MiB|byte limit"):
        native_replay(oversized, receipt)
    with pytest.raises(TypeError):
        s.ScalarRootBracketReceipt.from_bytes(data)


def test_noncanonical_duplicate_and_excessively_nested_bytes_refuse(repeated_root):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    for bad in (b" "+data, data+b"\n", b"["*97, b'{"schema_version":1,'+data[1:], b"not json"):
        with pytest.raises(ValueError):
            replay(bad, x, source, ledger=budget())


@pytest.mark.parametrize("operation", ["export", "replay"])
def test_each_transport_resource_allowance_is_cumulative(repeated_root, operation):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()

    def run(ledger):
        if operation == "export":
            return receipt.to_bytes(budget=ledger)
        return replay(data, x, source, ledger=ledger)

    measured = budget()
    run(measured)
    spent = {name: limit-getattr(measured, REMAINING[name]) for name, limit in LIMITS.items()}
    assert all(value > 0 for value in spent.values())
    for name, consumed in spent.items():
        with pytest.raises(s.EvaluationBudgetExceeded):
            run(budget(**{name: consumed-1}))
    exact = budget(**spent)
    run(exact)
    assert all(getattr(exact, remaining) == 0 for remaining in REMAINING.values())
    with pytest.raises(s.EvaluationBudgetExceeded):
        run(exact)


def test_generation_export_and_replay_can_share_the_same_caller_ledger():
    x = s.Symbol("portable_bracket_shared_budget_x")
    ledger = budget()
    receipt = s.certified_root_bracket(
        x, (x, -1, 1), budget=ledger, verification_budget=ledger,
    )
    after_generation = tuple(getattr(ledger, field) for field in REMAINING.values())
    data = receipt.to_bytes(budget=ledger)
    after_export = tuple(getattr(ledger, field) for field in REMAINING.values())
    restored = replay(data, x, x, limits=(x, -1, 1), ledger=ledger)
    assert_same_claim(restored, receipt)
    after_replay = tuple(getattr(ledger, field) for field in REMAINING.values())
    assert all(end < middle < start for start, middle, end in zip(after_generation, after_export, after_replay))


def test_cancelled_transport_stops_before_source_preparation(repeated_root, monkeypatch):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    ledger = budget()
    ledger.cancel()
    before = tuple(getattr(ledger, field) for field in REMAINING.values())
    certified = importlib.import_module("sympy.solvers.certified")

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled bracket replay reached source preparation")

    monkeypatch.setattr(certified, "_numeric_source", forbidden)
    for call in (
        lambda: receipt.to_bytes(budget=ledger),
        lambda: replay(data, x, source, ledger=ledger),
        lambda: native_replay(data, receipt, ledger=ledger),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()
    assert tuple(getattr(ledger, field) for field in REMAINING.values()) == before


@pytest.mark.parametrize("operation", ["export", "replay"])
def test_cancellation_after_native_completion_prevents_publication(repeated_root, monkeypatch, operation):
    from sympy.core import _native

    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    ledger = budget()
    owner = _native.ScalarRootBracketReceipt if operation == "export" else _native
    name = "to_bytes" if operation == "export" else "replay_scalar_root_bracket_native"
    original = getattr(owner, name)
    finished = []

    def complete_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        finished.append(result)
        ledger.cancel()
        return result

    monkeypatch.setattr(owner, name, complete_then_cancel)
    with pytest.raises(s.EvaluationCancelled):
        if operation == "export":
            receipt.to_bytes(budget=ledger)
        else:
            replay(data, x, source, ledger=ledger)
    assert len(finished) == 1
    assert ledger.remaining_steps < LIMITS["max_steps"]


def test_replay_checks_cancellation_after_public_receipt_construction(repeated_root, monkeypatch):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    transport = importlib.import_module("sympy.solvers._bracket_certificate")
    real_receipt = transport.ScalarRootBracketReceipt
    ledger = budget()
    accepted = []

    def cancel_after_acceptance(native):
        restored = real_receipt(native)
        accepted.append(restored)
        ledger.cancel()
        return restored

    monkeypatch.setattr(transport, "ScalarRootBracketReceipt", cancel_after_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        replay(data, x, source, ledger=ledger)
    assert len(accepted) == 1 and accepted[0].bounds == (1, 1)


@pytest.mark.parametrize("operation", ["export", "replay"])
def test_transport_captures_one_omitted_default_for_the_complete_operation(
    repeated_root, monkeypatch, operation,
):
    from sympy.core import _native

    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    parametric = importlib.import_module("sympy.utilities._parametric_root")
    real_factory = parametric._fallback_budget
    created = []

    def capture():
        ledger = real_factory()
        created.append(ledger)
        return ledger

    monkeypatch.setattr(parametric, "_fallback_budget", capture)
    owner = _native.ScalarRootBracketReceipt if operation == "export" else _native
    name = "to_bytes" if operation == "export" else "replay_scalar_root_bracket_native"
    real_native = getattr(owner, name)
    seen = []

    def observe(*args, **kwargs):
        ledger = kwargs["budget"]
        seen.append(ledger)
        if operation == "replay":
            assert ledger.remaining_steps < 512 * 1024 * 1024
        return real_native(*args, **kwargs)

    monkeypatch.setattr(owner, name, observe)
    if operation == "export":
        assert receipt.to_bytes() == data
    else:
        assert_same_claim(replay(data, x, source), receipt)
    assert len(created) == 1 and seen == created
    assert created[0].remaining_steps < 512 * 1024 * 1024


def test_replayed_receipts_remain_immutable(repeated_root):
    x, source, receipt = repeated_root
    data = receipt.to_bytes()
    restored = replay(data, x, source)
    for name in ("_receipt", "expression", "interval", "bounds", "claim"):
        with pytest.raises(AttributeError):
            setattr(restored, name, None)
        with pytest.raises(AttributeError):
            delattr(restored, name)
    with pytest.raises(AttributeError):
        restored.__init__(receipt._receipt)
    with pytest.raises(AttributeError):
        restored.variable.name = "different_axis"
    assert_same_claim(restored, receipt)
    assert restored.to_bytes() == data


def test_portable_bracket_replays_in_a_fresh_process_without_a_generator(repeated_root):
    from sympy.core import _native, _symbol_binding

    x, _, receipt = repeated_root
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
        def forbidden(*args, **kwargs):
            raise AssertionError('portable bracket replay invoked the generator')
        native.certified_scalar_root_bracket_native = forbidden
        s.certified_root_bracket = forbidden
        x = s.Symbol('portable_bracket_x', positive=True)
        data = bytes.fromhex(sys.stdin.read())
        receipt = s.ScalarRootBracketReceipt.from_bytes(data, expression=(x-1)**3, limits=(x, 1, 2))
        assert receipt.bounds == (1, 1)
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
        "preserved": True, "identity": _symbol_binding(x).identity_hex,
        "type": "ScalarRootBracketReceipt",
    }
