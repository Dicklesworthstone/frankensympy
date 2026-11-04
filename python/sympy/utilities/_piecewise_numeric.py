"""Closed, stored-source admission for numerical Piecewise conditions.

These are typed syntax views, not boolean evaluation or Python class recovery.
No public args getter, comparison hook, printer or user callback is invoked.
The native True/False atoms are reserved only in a condition position.
"""

from ..core import (
    Basic, Equality, Function, GreaterThan, LessThan, Relational,
    StrictGreaterThan, StrictLessThan, Unequality, _native,
    _require_closed_function_class,
)
from ..core._function_capabilities import FunctionCapabilities, _implementation
from ..functions.elementary.piecewise import Piecewise
from ..logic.boolalg import And, BooleanFalse, BooleanFunction, BooleanTrue, Not, Or

_RELATIONS = (
    (Equality, "Equality"), (Unequality, "Unequality"),
    (StrictLessThan, "StrictLessThan"), (LessThan, "LessThan"),
    (StrictGreaterThan, "StrictGreaterThan"), (GreaterThan, "GreaterThan"),
)
_LOGICAL = ((And, "And"), (Or, "Or"), (Not, "Not"))
_CLASSES = (BooleanTrue, BooleanFalse, *(cls for cls, _ in _RELATIONS),
            *(cls for cls, _ in _LOGICAL))
_VALUE = Basic.__dict__["_value"]
_RELATIONAL_SIDES = Relational.__dict__["_relational_sides"]
_BOOLEAN_ARGS = BooleanFunction.__dict__["_args"]
_BRANCHES = Piecewise.__dict__["_piecewise_branches"]
_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_MISSING = object()
# Captured during package import, before user extensions can change semantics.
_CAPABILITIES = FunctionCapabilities(_CLASSES, _native.Expr, _VALUE, Function)
# These condition-specific hooks supplement the shared Function roster.
_EXTRA_NAMES = ("__bool__", "rel_op", "lhs", "rhs", "_decide", "_relational_sides")
_EXTRA = tuple(
    (cls, tuple((owner, tuple(_implementation(_DICT.__get__(owner).get(name, _MISSING))
                             for name in _EXTRA_NAMES))
                for owner in _MRO.__get__(cls)))
    for cls in _CLASSES
)
_TRUE = _native.Expr("True")
_FALSE = _native.Expr("False")
_PIECEWISE_HOOKS = tuple(
    (name, _implementation(_DICT.__get__(Piecewise).get(name, _MISSING)))
    for name in ("_build", "_from_native", "_piecewise_branches")
)


def _require_piecewise():
    _require_closed_function_class(Piecewise)
    for name, recorded in _PIECEWISE_HOOKS:
        current = _implementation(_DICT.__get__(Piecewise).get(name, _MISSING))
        if len(current) != len(recorded) or any(a is not b for a, b in zip(current, recorded)):
            raise NotImplementedError("modified Piecewise implementations have no closed numerical lowering")


def _require_condition(cls):
    _CAPABILITIES.require_class(cls)
    for recorded_cls, owners in _EXTRA:
        if cls is recorded_cls:
            for owner, recorded in owners:
                current = tuple(_implementation(_DICT.__get__(owner).get(name, _MISSING))
                                for name in _EXTRA_NAMES)
                if any(len(left) != len(right) or any(a is not b for a, b in zip(left, right))
                       for left, right in zip(current, recorded)):
                    raise NotImplementedError(
                        "modified boolean or relational implementations have no closed numerical lowering"
                    )
            return
    raise NotImplementedError("custom boolean conditions have no closed numerical lowering")


def branches(value, limit):
    """Return exact stored (value, condition) pairs in their source order."""
    if type(value) is Piecewise:
        _require_piecewise()
        try:
            pairs = _BRANCHES.__get__(value, Piecewise)
        except AttributeError:
            pairs = None
        if pairs is not None:
            if type(pairs) is not tuple or not pairs or len(pairs) > limit:
                raise ValueError("Piecewise requires a bounded nonempty tuple of branches")
            if any(type(pair) is not tuple or len(pair) != 2 for pair in pairs):
                raise ValueError("Piecewise branches require exact (value, condition) pairs")
            return pairs
        value = _VALUE.__get__(value, Basic)
    if type(value) is not _native.Expr:
        raise NotImplementedError("custom Piecewise classes have no closed numerical lowering")
    _native.CompiledFunction._validate_source(value)
    _require_piecewise()
    if value.func_name != "Piecewise":
        raise ValueError("expected a native Piecewise source")
    native_pairs = value.args
    if not native_pairs or len(native_pairs) > limit:
        raise ValueError("Piecewise requires a bounded nonempty tuple of branches")
    pairs = []
    for pair in native_pairs:
        if pair.func_name != "Tuple":
            raise ValueError("Piecewise branches require Tuple(value, condition)")
        children = pair.args
        if len(children) != 2:
            raise ValueError("Piecewise branches require Tuple(value, condition)")
        pairs.append(children)
    return tuple(pairs)


def condition(value, limit):
    """Return a fixed condition tag and typed operands; never evaluate it."""
    cls = type(value)
    if cls is bool:
        return ("True" if value else "False"), ()
    if cls is BooleanTrue or cls is BooleanFalse:
        _require_condition(cls)
        return ("True" if cls is BooleanTrue else "False"), ()
    if cls is _native.Expr:
        _native.CompiledFunction._validate_source(value)
        if value == _TRUE or value == _FALSE:
            _require_condition(BooleanTrue if value == _TRUE else BooleanFalse)
            return ("True" if value == _TRUE else "False"), ()
        tag = value.func_name
        matched = next((builtin for builtin, name in (*_RELATIONS, *_LOGICAL)
                        if tag == name), None)
        if matched is None:
            raise NotImplementedError("unsupported native Piecewise condition")
        _require_condition(matched)
        operands = value.args
    else:
        tagged = next(((builtin, tag) for builtin, tag in (*_RELATIONS, *_LOGICAL)
                       if cls is builtin), None)
        if tagged is None:
            raise NotImplementedError("custom or opaque Piecewise conditions require a callback contract")
        _require_condition(cls)
        _, tag = tagged
        descriptor = _RELATIONAL_SIDES if any(cls is item[0] for item in _RELATIONS) else _BOOLEAN_ARGS
        try:
            operands = descriptor.__get__(value, cls)
        except AttributeError:
            native = _VALUE.__get__(value, Basic)
            if type(native) is not _native.Expr:
                raise ValueError("condition has no stored typed source")
            _native.CompiledFunction._validate_source(native)
            if native.func_name != tag:
                raise ValueError("condition source tag disagrees with its exact class")
            operands = native.args
    if type(operands) is not tuple or len(operands) > limit:
        raise ValueError("Piecewise condition requires bounded exact stored arguments")
    if tag == "Not" and len(operands) != 1:
        raise ValueError("Not requires exactly one condition")
    if any(tag == name for _, name in _RELATIONS) and len(operands) != 2:
        raise ValueError("numerical comparison requires exactly two operands")
    return tag, operands
