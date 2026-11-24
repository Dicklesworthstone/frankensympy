"""Portable integral partitions grant evidence only after original-source replay."""

from fractions import Fraction as F
import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import sympy as s


pytestmark = pytest.mark.skipif(
    not hasattr(s.Matrix([[1]]), "_native_storage"),
    reason="source-bound native integral certificate extension",
)
LIMITS = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
REMAINING = dict(max_steps="remaining_steps", max_memory_bytes="remaining_memory_bytes",
                 max_allocations="remaining_allocations")


def budget(**changes):
    return s.EvaluationBudget(**{**LIMITS, **changes})


def encode(document):
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def replay(data, x, source, *, limits=None, tolerance=F(1, 10000), ledger=None):
    return s.IntegralEnclosureReceipt.from_bytes(
        data, integrand=source, limits=(x, 1, 2) if limits is None else limits,
        tolerance=tolerance, budget=ledger,
    )


def native_replay(data, receipt, *, ledger=None):
    from sympy.core import _native

    native = receipt._receipt
    lower, upper = native.interval
    return _native.replay_integral_certificate_native(
        data, native.integrand, native.variable, lower, upper, native.absolute_tolerance,
        budget=ledger,
    )


def assert_same_claim(actual, expected):
    assert type(actual) is s.IntegralEnclosureReceipt
    assert actual.integrand == expected.integrand
    assert (actual.variable.name, actual.variable.plain, actual.variable.identity_hex) == (
        expected.variable.name, expected.variable.plain, expected.variable.identity_hex,
    )
    for field in (
        "interval", "bounds", "midpoint", "error_bound", "absolute_tolerance",
        "precision_digits", "partition", "interval_count", "claim",
    ):
        assert getattr(actual, field) == getattr(expected, field)


@pytest.fixture
def polynomial():
    x = s.Symbol("portable_integral_x", positive=True)
    source = x*x
    receipt = s.certified_integral(source, (x, 1, 2), precision=24, budget=budget())
    return x, source, receipt


def test_public_and_native_receivers_share_canonical_untrusted_partitions(polynomial):
    x, source, receipt = polynomial
    data = receipt.to_bytes()
    assert type(data) is bytes and len(data) <= 1 << 20
    document = json.loads(data)
    assert document["schema_version"] == 1
    assert set(document) == {
        "schema_version", "context_digest", "residuals", "variables",
        "lower", "upper", "absolute_tolerance", "witness",
    }
    assert set(document["witness"]) == {"partition", "precision_digits"}
    assert encode(document) == data
    assert receipt._receipt.to_bytes() == data
    restored = replay(data, x, source)
    assert_same_claim(restored, receipt)
    assert restored.bounds == (s.Rational(7, 3),)*2
    assert restored.to_bytes() == data
    native = native_replay(data, receipt)
    assert_same_claim(s.IntegralEnclosureReceipt(native), receipt)
    assert native.to_bytes() == data


@pytest.mark.parametrize("lower,upper,expected", [(1, 2, F(7, 3)), (2, 1, F(-7, 3)), (2, 2, 0)])
def test_native_generated_receipts_replay_publicly_with_ordered_and_point_limits(
    lower, upper, expected,
):
    from sympy.core import _native, _symbol_binding

    x = s.Symbol("native_portable_integral_x", positive=True)
    source = x*x
    native = _native.certified_integral_native(
        source._value, _symbol_binding(x), _native.py_integer(lower), _native.py_integer(upper),
        _native.py_rational(1, 10000), 24, 512, budget=budget(),
    )
    data = native.to_bytes()
    restored = replay(data, x, source, limits=(x, lower, upper))
    expected = s.Rational(expected.numerator, expected.denominator)
    assert restored.bounds == (expected, expected)
    assert restored.interval == (lower, upper)
    assert restored.partition == ((lower,) if lower == upper else (min(lower, upper), max(lower, upper)))
    assert restored.to_bytes() == data


def test_absolute_corner_receipt_retains_builtin_identity_despite_opaque_name():
    x = s.Symbol("portable_absolute_integral_x", real=True)
    opaque = s.Function("abs")
    source = s.Abs(x, evaluate=False)
    tolerance = F(1, 10**100)
    receipt = s.certified_integral(
        source, (x, -1, 1), tolerance=tolerance, max_intervals=2, budget=budget(),
    )
    data = receipt.to_bytes()
    assert b'"Function":["Abs",' in data
    restored = replay(data, x, source, limits=(x, -1, 1), tolerance=tolerance)
    assert_same_claim(restored, receipt)
    assert type(restored.integrand) is s.Abs
    assert restored.bounds == (1, 1) and restored.partition == (-1, 0, 1)
    assert restored.to_bytes() == data
    with pytest.raises(NotImplementedError):
        replay(data, x, opaque(x), limits=(x, -1, 1), tolerance=tolerance)


def test_replay_verifies_the_supplied_partition_without_regenerating(monkeypatch):
    from sympy.core import _native

    x = s.Symbol("portable_partition_integral_x")
    source, tolerance = x**4, F(1, 1000)
    receipt = s.certified_integral(
        source, (x, 0, 1), tolerance=tolerance, precision=16, max_intervals=16, budget=budget(),
    )
    assert receipt.interval_count > 1
    data = receipt.to_bytes()
    document = json.loads(data)
    partition = document["witness"]["partition"]
    document["witness"]["partition"] = [partition[0], partition[-1]]

    def forbidden(*args, **kwargs):
        raise AssertionError("integral replay must not regenerate an adaptive partition")

    monkeypatch.setattr(_native, "certified_integral_native", forbidden)
    monkeypatch.setattr(s, "certified_integral", forbidden)
    with pytest.raises(ValueError, match="width"):
        replay(encode(document), x, source, limits=(x, 0, 1), tolerance=tolerance)
    restored = replay(data, x, source, limits=(x, 0, 1), tolerance=tolerance)
    assert_same_claim(restored, receipt)
    assert restored.bounds[0] <= s.Rational(1, 5) <= restored.bounds[1]


def test_sources_and_typed_axes_match_before_decoding_foreign_witnesses(polynomial):
    x, source, receipt = polynomial
    document = json.loads(receipt.to_bytes())
    document["witness"] = {"not_a_partition": True}
    data = encode(document)
    for other in (s.Symbol(x.name), s.Symbol(x.name, nonnegative=True), s.Dummy(x.name, positive=True)):
        with pytest.raises(ValueError, match="sources or ordered symbol identities"):
            replay(data, other, other*other)
    held_equivalent = s.Mul(1, source, evaluate=False)
    with pytest.raises(ValueError, match="sources or ordered symbol identities"):
        replay(data, x, held_equivalent)


@pytest.mark.parametrize("change", ["lower", "upper", "direction", "tolerance"])
def test_receiver_binds_the_exact_ordered_interval_and_requested_tolerance(polynomial, change):
    x, source, receipt = polynomial
    document = json.loads(receipt.to_bytes())
    document["witness"] = {"not_a_partition": True}
    limits = {"lower": (x, 2, 2), "upper": (x, 1, 3), "direction": (x, 2, 1)}.get(change, (x, 1, 2))
    tolerance = F(1, 1000) if change == "tolerance" else F(1, 10000)
    with pytest.raises(ValueError, match="bounds or absolute tolerance"):
        replay(encode(document), x, source, limits=limits, tolerance=tolerance)


def test_receiver_rechecks_held_endpoint_poles_even_for_a_zero_product():
    x = s.Symbol("portable_integral_pole_x")
    hidden = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    source = s.Add(1, hidden, evaluate=False)
    receipt = s.certified_integral(source, (x, 1, 2), budget=budget())
    data = receipt.to_bytes()
    with pytest.raises(ValueError, match="sources or ordered symbol identities"):
        replay(data, x, 1)

    # Use a real exported statement for canonical zero/one endpoints. Relabel
    # the original partition onto a new receiver-owned interval containing the
    # hidden pole; neither the zero factor nor serialized partition proves it.
    point = s.certified_integral(0, (x, 0, 1), budget=budget())
    endpoints = json.loads(point.to_bytes())
    document = json.loads(data)
    for field in ("lower", "upper", "witness"):
        document[field] = endpoints[field]
    with pytest.raises(ValueError, match="zero|domain"):
        replay(encode(document), x, source, limits=(x, 0, 1))
    assert_same_claim(replay(data, x, source), receipt)


def test_native_receipt_cannot_bypass_public_declared_assumptions():
    from sympy.core import _native, _symbol_binding

    x = s.Symbol("portable_positive_integral_x", positive=True)
    negative = _native.certified_integral_native(
        x._value, _symbol_binding(x), _native.py_integer(-2), _native.py_integer(-1),
        _native.py_rational(1, 10000), budget=budget(),
    )
    data = negative.to_bytes()
    with pytest.raises(ValueError, match="positive|assumption"):
        replay(data, x, x, limits=(x, -2, -1))


@pytest.mark.parametrize("change", ["schema", "context", "accepted", "bounds", "duplicate", "reverse", "too_many"])
def test_corrupt_or_unproved_claims_refuse_without_poisoning_valid_replay(polynomial, change):
    x, source, receipt = polynomial
    data = receipt.to_bytes()
    document = json.loads(data)
    if change == "schema":
        document["schema_version"] += 1
    elif change == "context":
        document["context_digest"][0] ^= 1
    elif change in ("accepted", "bounds"):
        document[change] = True if change == "accepted" else []
    else:
        partition = document["witness"]["partition"]
        if change == "duplicate":
            document["witness"]["partition"] = [partition[0], *partition]
        elif change == "reverse":
            document["witness"]["partition"] = partition[::-1]
        else:
            document["witness"]["partition"] = [partition[0]]*1026
    with pytest.raises(ValueError):
        replay(encode(document), x, source, ledger=budget())
    assert_same_claim(replay(data, x, source), receipt)


def test_byte_type_and_size_admission_precede_source_and_conversion_hooks(polynomial, monkeypatch):
    x, _, receipt = polynomial
    data = receipt.to_bytes()
    certified = importlib.import_module("sympy.integrals.certified")

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected bytes reached source preparation")

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
    # Required receiver statements cannot be inferred from the untrusted file.
    with pytest.raises(TypeError):
        s.IntegralEnclosureReceipt.from_bytes(data)


def test_noncanonical_duplicate_and_excessively_nested_bytes_refuse(polynomial):
    x, source, receipt = polynomial
    data = receipt.to_bytes()
    for bad in (b" "+data, data+b"\n", b"["*97, b'{"schema_version":1,'+data[1:], b"not json"):
        with pytest.raises(ValueError):
            replay(bad, x, source, ledger=budget())


@pytest.mark.parametrize("operation", ["export", "replay"])
def test_each_transport_resource_allowance_is_cumulative(polynomial, operation):
    x, source, receipt = polynomial
    data = receipt.to_bytes()

    def run(ledger):
        if operation == "export":
            return receipt.to_bytes(budget=ledger)
        return replay(data, x, source, ledger=ledger)

    measured = budget()
    run(measured)
    spent = {name: limit-getattr(measured, REMAINING[name]) for name, limit in LIMITS.items()}
    assert all(value > 0 for value in spent.values())
    if operation == "replay":
        assert spent["max_memory_bytes"] >= 72*1024*1024
    for name, consumed in spent.items():
        with pytest.raises(s.EvaluationBudgetExceeded):
            run(budget(**{name: consumed-1}))
    exact = budget(**spent)
    run(exact)
    assert all(getattr(exact, remaining) == 0 for remaining in REMAINING.values())
    with pytest.raises(s.EvaluationBudgetExceeded):
        run(exact)


def test_generation_export_and_replay_share_the_same_caller_ledger():
    x = s.Symbol("portable_integral_shared_budget_x")
    ledger = budget()
    receipt = s.certified_integral(x, (x, 1, 2), budget=ledger)
    after_generation = tuple(getattr(ledger, field) for field in REMAINING.values())
    data = receipt.to_bytes(budget=ledger)
    after_export = tuple(getattr(ledger, field) for field in REMAINING.values())
    restored = replay(data, x, x, ledger=ledger)
    assert_same_claim(restored, receipt)
    after_replay = tuple(getattr(ledger, field) for field in REMAINING.values())
    assert all(end < middle < start for start, middle, end in zip(after_generation, after_export, after_replay))


def test_cancelled_transport_stops_before_source_preparation(polynomial, monkeypatch):
    x, source, receipt = polynomial
    data = receipt.to_bytes()
    ledger = budget()
    ledger.cancel()
    before = tuple(getattr(ledger, field) for field in REMAINING.values())
    certified = importlib.import_module("sympy.integrals.certified")

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled replay reached source preparation")

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
def test_cancellation_after_native_completion_prevents_publication(polynomial, monkeypatch, operation):
    from sympy.core import _native

    x, source, receipt = polynomial
    data = receipt.to_bytes()
    ledger = budget()
    finished = []
    owner = _native.IntegralEnclosureReceipt if operation == "export" else _native
    name = "to_bytes" if operation == "export" else "replay_integral_certificate_native"
    original = getattr(owner, name)

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


def test_replayed_receipts_cannot_be_deleted_reinitialized_or_relabelled(polynomial):
    x, source, receipt = polynomial
    data = receipt.to_bytes()
    restored = replay(data, x, source)
    for name in ("_receipt", "interval", "bounds", "integrand"):
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


def test_integral_partition_replays_in_a_fresh_process_without_a_generator(polynomial):
    from sympy.core import _native, _symbol_binding

    x, _, receipt = polynomial
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
            raise AssertionError('portable integral replay invoked the generator')
        native.certified_integral_native = forbidden
        s.certified_integral = forbidden
        x = s.Symbol('portable_integral_x', positive=True)
        data = bytes.fromhex(sys.stdin.read())
        receipt = s.IntegralEnclosureReceipt.from_bytes(data, integrand=x*x, limits=(x, 1, 2))
        assert receipt.bounds == (s.Rational(7, 3), s.Rational(7, 3))
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
        "type": "IntegralEnclosureReceipt",
    }
