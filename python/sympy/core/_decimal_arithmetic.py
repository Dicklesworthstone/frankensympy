"""Arithmetic on retained decimal values, without binary64 intermediates.

The decimal payload is the value of a retained Float, not an instruction to
re-evaluate the constant that originally produced it. Exact native arithmetic
rounds each operation once at the greatest input Float precision. This bounded
lane does not replace the existing binary64 Float profile or claim to reproduce
mpmath's hidden binary significands. Ordinary arithmetic without a retained
decimal keeps its existing implementation. Float identity also retains
precision, and float-bearing expression identity preserves actual children;
those shared rules keep equality and hashing coherent across both owners.
"""

from fractions import Fraction
import math

from . import (
    Add, Basic, Expr, Float, Function, Integer, Mul, Pow, Rational, Zero, One, NegativeOne, Half,
    _ExactDecimalFloat, _EvalfSource, _UnsupportedEvalfSource, _native,
    _charge_nested_if_active, _format_decimal, _require_evalf_owners,
    _EVALF_OWNERS, _FLOAT_INTERN, _is_registered_function_class,
    _lookup_known_function, _symbol_from_binding, _undefined_function_semantic_identity,
    _ZERO, _ONE, _NEGATIVE_ONE, nan, zoo,
)
from ._opaque_arithmetic import _admit_source, _make, contains_opaque, stored_args
from ._function_capabilities import _owner_record, _owner_matches


UNHANDLED = object()
# Selection uses the existing opaque-surface limits so expressions without
# retained decimals keep their legacy admission behavior. Numeric evaluation
# has the stricter limits of its own source capture and native budget.
_MAX_NODES = 16384
_MAX_DEPTH = 128
_MAX_INPUT_BITS = 4096
_MAX_DIGITS = 1000
_NUMBERS = (Integer, Rational, Zero, One, NegativeOne, Half, Float, _ExactDecimalFloat)
_DPS = Float.__dict__["_dps"]
_VALUE = Basic.__dict__["_value"]
_MRO = type.__dict__["__mro__"]
_NAME = type.__dict__["__name__"]


def _is_exact(value, classes):
    return any(type(value) is owner for owner in classes)


def _is_basic(value):
    return any(owner is Basic for owner in type.__dict__["__mro__"].__get__(type(value)))


def contains_decimal(values):
    """Find retained owners without invoking public child or numeric hooks."""
    pending = [(value, 0) for value in values]
    seen = set()
    while pending:
        value, depth = pending.pop()
        if type(value) is _ExactDecimalFloat:
            return True
        if not _is_basic(value) or id(value) in seen:
            continue
        if len(seen) >= _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("retained decimal arithmetic exceeds its source limit")
        seen.add(id(value))
        children = stored_args(value) or ()
        if len(children) > _MAX_NODES - len(seen) - len(pending):
            raise ValueError("retained decimal arithmetic exceeds its argument limit")
        pending.extend((child, depth + 1) for child in children)
    return False


def _surface(value, capture):
    if _is_basic(value):
        return value
    if type(value) is int:
        bits = value.bit_length()
        if bits > _MAX_INPUT_BITS:
            raise ValueError("retained decimal integer exceeds its bit limit")
        capture.charge(bits + 1, (bits + 7) // 8 + 256, 4)
        return Integer(value)
    if type(value) is float:
        capture.charge(128, 1024, 4)
        return Float(value)
    raise TypeError("retained decimal arithmetic requires a symbolic or built-in numeric operand")


def _held(operation, values, capture):
    def make(kind, children):
        capture.charge(len(children) + 1, 256 * (len(children) + 1), len(children) + 2)
        if any(contains_opaque(child) for child in children):
            # Unknown classes remain actual surface owners. No constructor,
            # getter or native conversion on an opaque operand is invoked.
            return _make(kind, children, held=True)
        for child in children:
            native = capture.retain(child, Basic, "_value")
            if type(native) is _native.Expr:
                capture.copy(native, 3)
        return kind(*children, evaluate=False)

    if operation == "add":
        return make(Add, values)
    if operation == "mul":
        return make(Mul, values)
    if operation == "sub":
        return make(Add, (values[0], make(Mul, (_NEGATIVE_ONE, values[1]))))
    if operation == "div":
        return make(Mul, (values[0], make(Pow, (values[1], _NEGATIVE_ONE))))
    if operation == "pow":
        return make(Pow, values)
    raise ValueError("unknown retained decimal arithmetic operation")


def _build(operation, values, capture):
    # The two-level subtraction/division forms may copy a child four times
    # through constructors and as_expr. Reserve all those copies before any
    # constructor runs, on the same ledger as lowering and refinement.
    capture.charge(16 + len(values), 2048 + 128 * len(values), 16 + len(values))
    for value in values:
        capture.copy(value, 4)
    if operation == "add":
        return _native.Add(*values, evaluate=False).as_expr()
    if operation == "mul":
        return _native.Mul(*values, evaluate=False).as_expr()
    if operation == "sub":
        negative = _native.Mul(_native.py_integer(-1), values[1], evaluate=False).as_expr()
        return _native.Add(values[0], negative, evaluate=False).as_expr()
    if operation == "div":
        reciprocal = _native.Pow(values[1], _native.py_integer(-1), evaluate=False).as_expr()
        return _native.Mul(values[0], reciprocal, evaluate=False).as_expr()
    return _native.Pow(*values, evaluate=False).as_expr()


def _budget():
    # Ordinary operators have no budget keyword. Their numeric work still has
    # one finite ledger, shared by exact operations, rounding and publication.
    return _native.EvaluationBudget(
        max_steps=1_000_000_000,
        max_memory_bytes=512 * 1024 * 1024,
        max_allocations=4_000_000,
    )


def _finish(capture, result):
    _charge_nested_if_active(1)
    capture.unchanged()
    return result


def _capture(values, budget=None):
    capture = _EvalfSource(budget=budget)
    capture.charge(len(_EVALF_OWNERS) + 1, 128 * len(_EVALF_OWNERS), len(_EVALF_OWNERS))
    _require_evalf_owners()
    values = tuple(_surface(value, capture) for value in values)
    lowered = []
    for value in values:
        try:
            _original, native = capture.read(value)
        except _UnsupportedEvalfSource:
            capture.unchanged()
            return capture, None, values
        lowered.append(native)
    capture.unchanged()
    return capture, lowered, values


def _precision(values):
    precisions = [
        _DPS.__get__(value, Float) for value in values
        if type(value) is Float or type(value) is _ExactDecimalFloat
    ]
    if any(type(value) is not int or value < 1 for value in precisions):
        raise ValueError("retained decimal arithmetic requires positive exact precision")
    digits = max(precisions)
    if digits > _MAX_DIGITS:
        raise ValueError("retained decimal arithmetic precision must be in 1..=1000")
    return digits


def _numerator(value, capture):
    capture.copy(value)
    return value.exact_numerator()


def _output_reservation(capture, digits):
    # Includes PyO3 string/tuple publication and retained decimal formatting.
    capture.charge(4 * digits + 16, 24 * (digits + 64) + 2048, 20)


def from_exact_source(value, digits):
    """Round an admitted integer, rational or decimal once at the requested precision."""
    if type(value) is _ExactDecimalFloat:
        original_digits = _DPS.__get__(value, Float)
        if type(original_digits) is int and original_digits == digits:
            return value
    if digits > _MAX_DIGITS:
        raise ValueError("Float conversion precision must be in 1..=1000")
    budget = _budget()
    source = _ExactDecimalFloat(value, digits) if type(value) is str else value
    capture, lowered, _values = _capture((source,), budget)
    if lowered is None:
        raise ValueError("Float conversion requires a finite numerical atom")
    rational = lowered[0]
    if type(source) is not _ExactDecimalFloat:
        capture.copy(rational, 2)
        if max(rational.exact_numerator().bit_length(),
               rational.exact_denominator().bit_length()) > _MAX_INPUT_BITS:
            raise ValueError("Float exact input exceeds its bit limit")
    _output_reservation(capture, digits)
    real, imaginary = _native.evalf_complex_decimal_native(rational, digits, budget=budget)
    if imaginary != "0":
        raise RuntimeError("real Float input produced a non-real value")
    result = _ExactDecimalFloat(_format_decimal(real, digits), digits)
    return _finish(capture, result)


def truncate(value):
    """Truncate the retained rational value toward zero, including beyond binary64."""
    capture, lowered, _values = _capture((value,), _budget())
    if lowered is None:
        raise ValueError("integer conversion requires a finite retained decimal")
    rational = lowered[0]
    capture.copy(rational, 2)
    _nodes, size, _allocations = capture.cost(rational)
    capture.charge(64 * size * size + 1, 16 * size + 256, 8)
    numerator = rational.exact_numerator()
    magnitude = abs(numerator) // rational.exact_denominator()
    return _finish(capture, -magnitude if numerator < 0 else magnitude)


def arithmetic(operation, operands, *, operator_method=False):
    """Compute admitted scalar arithmetic, or retain its actual surface owners.

    UNHANDLED means there was no retained decimal anywhere in the original
    surface. Unsupported symbolic operands remain held, with their original
    identity and domains. Once scalar evaluation begins every failure is final.
    """
    if type(operands) is not tuple:
        raise ValueError("retained decimal arithmetic requires exact operand tuples")
    if not contains_decimal(operands):
        return UNHANDLED
    if operator_method and any(
        not _is_basic(value) and type(value) is not int and type(value) is not float
        for value in operands
    ):
        # Let Python perform foreign reflected dispatch (including matrices).
        # Constructors stay strict, and no admitted numeric failure is caught.
        return NotImplemented
    if len(operands) > 256:
        raise ValueError("retained decimal arithmetic requires bounded operand tuples")
    if operation not in ("add", "sub", "mul", "div", "pow"):
        raise ValueError("unknown retained decimal arithmetic operation")
    if operation in ("sub", "div", "pow") and len(operands) != 2:
        raise ValueError("binary retained decimal arithmetic requires two operands")
    budget = _budget()
    capture, lowered, values = _capture(operands, budget)
    # Coefficients must not lose their retained payload just because a symbol,
    # unsupported function or held non-atomic expression is another operand.
    if (lowered is None or any(not any(type(value) is kind for kind in _NUMBERS)
                               for value in values)
            or any(not (value.is_integer or value.is_rational) for value in lowered)):
        return _finish(capture, _held(operation, values, capture))
    digits = _precision(values)
    if operation == "div" and _numerator(lowered[1], capture) == 0:
        numerator_zero = _numerator(lowered[0], capture) == 0
        if _is_exact(values[0], (Float, _ExactDecimalFloat)) and _is_exact(values[1], (Float, _ExactDecimalFloat)):
            _finish(capture, None)
            raise ZeroDivisionError()
        return _finish(capture, nan if numerator_zero else zoo)
    if operation == "pow":
        capture.copy(lowered[1])
        if lowered[1].exact_denominator() != 1:
            return _finish(capture, _held(operation, values, capture))
        exponent = _numerator(lowered[1], capture)
        if exponent == 0:
            _output_reservation(capture, digits)
            one = ("1." + "0" * (digits - 1) if digits > 1 else "1") + "e0"
            result = (_ExactDecimalFloat(_format_decimal(one, digits), digits)
                      if _is_exact(values[1], (Float, _ExactDecimalFloat)) else _ONE)
            return _finish(capture, result)
        if exponent == 1 and not _is_exact(values[1], (Float, _ExactDecimalFloat)):
            return _finish(capture, values[0])
        if _numerator(lowered[0], capture) == 0:
            return _finish(capture, zoo if exponent < 0 else values[0])
        if abs(exponent) > 4096:
            raise ValueError("retained decimal integer power exceeds its exponent limit")
        capture.charge(16, 256, 2)
        lowered[1] = _native.py_integer(exponent)
    native = _build(operation, lowered, capture)
    _output_reservation(capture, digits)
    real, imaginary = _native.evalf_complex_decimal_native(native, digits, budget=budget)
    if imaginary != "0":
        raise RuntimeError("real retained decimal arithmetic produced a non-real value")
    result = _ZERO if real == "0" else _ExactDecimalFloat(_format_decimal(real, digits), digits)
    return _finish(capture, result)


def compare(left, right):
    """Exact sign of an admitted retained-decimal difference, or UNHANDLED."""
    if type(left) is not _ExactDecimalFloat and type(right) is not _ExactDecimalFloat:
        return UNHANDLED
    if not all(_is_exact(value, (*_NUMBERS, int, float)) for value in (left, right)):
        return UNHANDLED
    budget = _budget()
    capture, lowered, _values = _capture((left, right), budget)
    if lowered is None:
        return UNHANDLED
    # Exact rational point enclosures can never round a nonzero difference to
    # zero. One significant digit suffices to establish the exact sign, even
    # when both binary64 previews coincide or overflow.
    native = _build("sub", lowered, capture)
    _output_reservation(capture, 1)
    real, imaginary = _native.evalf_complex_decimal_native(native, 1, budget=budget)
    if imaginary != "0":
        raise TypeError("non-real retained decimal values are not ordered")
    sign = 0 if real == "0" else -1 if real.startswith("-") else 1
    return _finish(capture, sign)


def equal(left, right):
    """Float structural equality: stored value and precision must both agree."""
    if type(left) is not _ExactDecimalFloat and type(right) is not _ExactDecimalFloat:
        return UNHANDLED
    if not all(_is_exact(value, (Float, _ExactDecimalFloat, float)) for value in (left, right)):
        return False if all(_is_exact(value, _NUMBERS) for value in (left, right)) else NotImplemented
    capture, lowered, (left, right) = _capture((left, right), _budget())
    if lowered is None:
        return _finish(capture, False)
    result = (_DPS.__get__(left, Float) == _DPS.__get__(right, Float)
              and capture.same(lowered[0], lowered[1]))
    return _finish(capture, result)


def value_hash(value):
    """Profile-style Float hashing, including finite values beyond binary64."""
    capture, lowered, _values = _capture((value,), _budget())
    if lowered is None:
        raise ValueError("retained decimal hash requires a finite original payload")
    original = capture.retain(value, Basic, "_value")
    preview = capture.floating(original)
    if math.isfinite(preview):
        result = hash(preview)
    else:
        rational = lowered[0]
        capture.copy(rational, 2)
        _nodes, size, _allocations = capture.cost(rational)
        capture.charge(64 * size * size + 1, 16 * size + 256, 8)
        result = hash(("retained Float", _DPS.__get__(value, Float),
                       Fraction(rational.exact_numerator(), rational.exact_denominator())))
    return _finish(capture, result)


def _has_float_source(value):
    """Select float-bearing identity graphs without public child getters."""
    if not _is_basic(value):
        return False
    pending, seen = [(value, 0)], set()
    while pending:
        node, depth = pending.pop()
        if _is_exact(node, (Float, _ExactDecimalFloat)):
            return True
        if not _is_basic(node) or id(node) in seen:
            continue
        if len(seen) >= _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Float structural identity exceeds its source limit")
        seen.add(id(node))
        children = stored_args(node)
        if children is not None:
            if len(children) > _MAX_NODES - len(seen) - len(pending):
                raise ValueError("Float structural identity exceeds its argument limit")
            pending.extend((child, depth + 1) for child in children)
        else:
            try:
                native = _VALUE.__get__(node, Basic)
            except AttributeError:
                continue
            if (type(native) is _native.Expr and not native.is_number
                    and not native.is_symbol and native.has_function_bounded(_FLOAT_INTERN)):
                return True
    return False


def _structural_key(source, capture, owners, *, for_hash):
    """Actual child identity, with finite Float values and precision retained.

    Native-only children have their existing default 15-digit Float ownership.
    Hashes deliberately permit collisions between unequal Float values and
    precisions, as Float atom hashes do. Equality never uses those hash keys.
    """
    # The existing callback-free admission walk has a fixed surface envelope.
    # Reserve that envelope before it inspects opaque implementation metadata.
    capture.charge(_MAX_NODES, 64 * _MAX_NODES, 8)
    _admit_source((source,), check_identity=True)
    memo, active, records, interned = {}, set(), [], {}
    work = 0

    def touch(depth):
        nonlocal work
        work += 1
        if work > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Float structural identity exceeds its work limit")
        capture.charge(1, 192, 3)

    def label(cls):
        declaration = _undefined_function_semantic_identity(cls)
        if declaration is not None:
            return "undefined", declaration
        return "class", _NAME.__get__(cls) if for_hash else id(cls)

    def intern(record):
        existing = interned.get(record)
        if existing is not None:
            return existing
        index = len(records)
        records.append(record)
        interned[record] = index
        return index

    def float_record(original, exact, precision):
        preview = capture.floating(original)
        if for_hash:
            if math.isfinite(preview):
                return intern(("Float", hash(preview)))
            if exact is None:
                # Nonfinite legacy payloads do not participate in retained
                # scalar arithmetic; their structural hash is still stable.
                capture.copy(original)
                return intern(("Float nonfinite", original))
            capture.copy(exact, 2)
            _nodes, size, _allocations = capture.cost(exact)
            capture.charge(64 * size * size + 1, 16 * size + 256, 8)
            number = Fraction(exact.exact_numerator(), exact.exact_denominator())
            return intern(("Float", hash(("retained Float", precision, number))))
        if exact is None:
            capture.copy(original)
            return intern(("Float nonfinite", original))
        capture.copy(exact)
        return intern(("Float", precision, exact))

    def native_key(native, depth):
        touch(depth)
        capture.copy(native)
        if native.is_integer or native.is_rational:
            capture.copy(native, 2)
            return intern(("exact", native.exact_numerator(), native.exact_denominator()))
        if native.is_symbol:
            cls = type(_symbol_from_binding(native.symbol_binding))
            return intern(("leaf", label(cls), native))
        tag = native.func_name
        if tag == _FLOAT_INTERN:
            preview = capture.floating(native)
            exact = None
            if math.isfinite(preview):
                capture.charge(1074 * 1074, 4096, 8)
                exact = _native.py_rational(*preview.as_integer_ratio())
            return float_record(native, exact, 15)
        cls = Add if native.is_add else Mul if native.is_mul else Pow if native.is_pow else None
        if cls is None:
            name = "Abs" if tag == "abs" else "log" if tag == "ln" else tag
            cls = _lookup_known_function(name)
            if cls is not None and not _is_registered_function_class(cls):
                cls = None
        if cls is None:
            return intern(("leaf", label(Expr), native))
        children = capture.arguments(native)
        return intern(("surface", label(cls), tuple(native_key(child, depth + 1) for child in children)))

    def visit(node, depth):
        touch(depth)
        identity = id(node)
        if identity in active:
            raise ValueError("cyclic retained Float structural graph")
        if identity in memo:
            return memo[identity]
        kind = type(node)
        capture.classes.append((node, kind))
        if _is_exact(node, (Float, _ExactDecimalFloat)):
            original = capture.retain(node, Basic, "_value")
            precision = capture.retain(node, Float, "_dps")
            if type(precision) is not int or precision < 1:
                raise ValueError("Float structural precision must be a positive exact integer")
            try:
                _original, exact = capture.read(node)
            except _UnsupportedEvalfSource:
                exact = None
            result = float_record(original, exact, precision)
        else:
            for owner in _MRO.__get__(kind):
                if id(owner) not in owners:
                    capture.charge(4096, 16_384, 16)
                    owners[id(owner)] = (owner, _MRO.__get__(owner), _owner_record(owner))
            native = capture.retain(node, Basic, "_value")
            for name in ("_struct_args", "_surface_args"):
                capture.retain(node, Basic, name)
            if _is_exact(node, (Add, Mul, Pow)):
                capture.retain(node, kind, "_args")
            elif any(owner is Function for owner in _MRO.__get__(kind)):
                capture.retain(node, Function, "_args")
            children = stored_args(node)
            if children is not None:
                active.add(identity)
                result = intern(("surface", label(kind), tuple(visit(child, depth + 1) for child in children)))
                active.remove(identity)
            elif type(native) is _native.Expr:
                capture.cost(native)
                capture.closed_functions(native)
                capture.natives.append(native)
                if _is_exact(node, (Add, Mul, Pow)) or _is_registered_function_class(kind):
                    children = capture.arguments(native)
                    result = intern(("surface", label(kind), tuple(native_key(child, depth + 1) for child in children)))
                elif native.is_integer or native.is_rational or native.is_symbol or kind is Expr:
                    result = native_key(native, depth)
                else:
                    capture.copy(native)
                    result = intern(("leaf", label(kind), native))
            else:
                raise NotImplementedError("Float structural identity requires retained children")
        memo[identity] = result
        return result

    root = visit(source, 0)
    return tuple(records), root


def _structural_finish(capture, owners, result):
    capture.charge(4096 * len(owners) + 1, 16_384 * len(owners), 16 * len(owners))
    _charge_nested_if_active(1)
    capture.unchanged()
    for owner, expected_mro, record in owners.values():
        current = _MRO.__get__(owner)
        if (len(current) != len(expected_mro)
                or any(left is not right for left, right in zip(current, expected_mro))
                or not _owner_matches(owner, record)):
            raise RuntimeError("Float structural implementation changed during comparison")
    return result


def structural_equal(left, right):
    if not _has_float_source(left) and not _has_float_source(right):
        return UNHANDLED
    if not _is_basic(left) or not _is_basic(right):
        return False
    capture, _lowered, _values = _capture((), _budget())
    owners = {}
    left_key = _structural_key(left, capture, owners, for_hash=False)
    right_key = _structural_key(right, capture, owners, for_hash=False)
    return _structural_finish(capture, owners, left_key == right_key)


def structural_hash(value):
    if not _has_float_source(value):
        return UNHANDLED
    capture, _lowered, _values = _capture((), _budget())
    owners = {}
    key = _structural_key(value, capture, owners, for_hash=True)
    return _structural_finish(capture, owners, hash(key))
