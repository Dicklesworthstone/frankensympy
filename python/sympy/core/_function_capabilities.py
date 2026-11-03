"""Process-local admission for the implementations of registered functions.

This records actual classes, MRO owners and descriptor implementations. It is
neither a portable identity nor a sandbox for arbitrary mutation of Python
module globals. Checking a record never invokes a user descriptor, hash,
equality method, printer, or callback.
"""

from types import FunctionType

_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_NAME = type.__dict__["__name__"]
_CODE = FunctionType.__dict__["__code__"]
_DEFAULTS = FunctionType.__dict__["__defaults__"]
_KWDEFAULTS = FunctionType.__dict__["__kwdefaults__"]
_CLOSURE = FunctionType.__dict__["__closure__"]
_CLASS_FUNCTION = classmethod.__dict__["__func__"]
_STATIC_FUNCTION = staticmethod.__dict__["__func__"]
_PROPERTY_GET = property.__dict__["fget"]
_PROPERTY_SET = property.__dict__["fset"]
_PROPERTY_DEL = property.__dict__["fdel"]
_EMPTY = object()
_MAX_MEMBERS = 4096
_SEMANTIC_NAMES = frozenset({
    "eval", "fdiff", "_imp_", "_hashable_content", "nargs", "args", "func", "_args",
    "_value", "_surface_args", "_struct_args",
    "__new__", "__init__", "__getattribute__", "__getattr__", "__setattr__",
    "__delattr__", "__eq__", "__hash__", "__call__", "__instancecheck__",
    "__subclasscheck__", "_prop_handler",
})


def _semantic(name):
    return (type(name) is str and
            (name in _SEMANTIC_NAMES or name.startswith("_eval_") or name.startswith("is_")))


def _implementation(value, depth=0, active=None, work=None):
    """Identity-only tokens for a descriptor and its implementation bodies."""
    if active is None:
        active = set()
        work = [0]
    work[0] += 1
    if work[0] > _MAX_MEMBERS or depth > 32 or id(value) in active:
        raise ValueError("function implementation metadata exceeds admission limits")
    active.add(id(value))
    tokens = [value]
    kind = type(value)
    if kind is FunctionType:
        tokens.append(_CODE.__get__(value))
        defaults = _DEFAULTS.__get__(value)
        tokens.append(defaults)
        if defaults is not None:
            if type(defaults) is not tuple or len(defaults) > _MAX_MEMBERS:
                raise ValueError("function implementation defaults exceed admission limits")
            tokens.extend(defaults)
        defaults = _KWDEFAULTS.__get__(value)
        tokens.append(defaults)
        if defaults is not None:
            if (type(defaults) is not dict or len(defaults) > _MAX_MEMBERS
                    or any(type(key) is not str for key in defaults)):
                raise ValueError("function implementation defaults exceed admission limits")
            for key in sorted(defaults):
                tokens.extend((key, defaults[key]))
        closure = _CLOSURE.__get__(value)
        tokens.append(closure)
        if closure is not None:
            for cell in closure:
                try:
                    tokens.append(cell.cell_contents)
                except ValueError:
                    tokens.append(_EMPTY)
    elif kind is classmethod:
        tokens.extend(_implementation(_CLASS_FUNCTION.__get__(value), depth+1, active, work))
    elif kind is staticmethod:
        tokens.extend(_implementation(_STATIC_FUNCTION.__get__(value), depth+1, active, work))
    elif kind is property:
        for descriptor in (_PROPERTY_GET, _PROPERTY_SET, _PROPERTY_DEL):
            tokens.extend(_implementation(descriptor.__get__(value), depth+1, active, work))
    elif kind is dict:
        # Assumption handlers are an exact built-in mapping in the shell.
        # A replacement with an arbitrary mapping is compared by identity.
        if len(value) > _MAX_MEMBERS or any(type(key) is not str for key in value):
            raise ValueError("function implementation handlers exceed admission limits")
        for key in sorted(value):
            tokens.append(key)
            tokens.extend(_implementation(value[key], depth+1, active, work))
    if len(tokens) > _MAX_MEMBERS:
        raise ValueError("function implementation exceeds admission limits")
    active.remove(id(value))
    return tuple(tokens)


def _owner_record(owner):
    namespace = _DICT.__get__(owner)
    name = _NAME.__get__(owner)
    if len(namespace) > _MAX_MEMBERS:
        raise ValueError("function implementation namespace exceeds admission limits")
    if type(name) is not str or any(type(key) is not str for key in namespace):
        raise ValueError("function implementation metadata requires exact string names")
    names = tuple(sorted(name for name in namespace if _semantic(name)))
    return (name, tuple((key, _implementation(namespace[key])) for key in names))


def _owner_matches(owner, recorded):
    try:
        current = _owner_record(owner)
    except ValueError:
        return False
    if current[0] != recorded[0] or len(current[1]) != len(recorded[1]):
        return False
    for (name, tokens), (expected_name, expected_tokens) in zip(current[1], recorded[1]):
        if (name != expected_name or len(tokens) != len(expected_tokens)
                or not all(left is right for left, right in zip(tokens, expected_tokens))):
            return False
    return True


class FunctionCapabilities:
    """An immutable roster; current implementations are checked on admission."""

    __slots__ = ("_rows", "_native_names", "_native_has_function", "_native_is_number", "_native_is_symbol", "_value_slot")

    def __init__(self, classes, native_type, value_slot):
        owners = {}
        rows = []
        native_names = []
        for cls in classes:
            mro = _MRO.__get__(cls)
            metaclass = type(cls)
            observed = []
            for owner in (*mro, *_MRO.__get__(metaclass)):
                key = id(owner)
                if key not in owners:
                    owners[key] = (owner, _owner_record(owner))
                if not any(owner is existing[0] for existing in observed):
                    observed.append(owners[key])
            row = (cls, metaclass, mro, _MRO.__get__(metaclass), tuple(observed))
            rows.append(row)
            name = _NAME.__get__(cls)
            native_names.append((name, row))
            if name == "log":
                native_names.append(("ln", row))
            elif name == "Abs":
                native_names.append(("abs", row))
        object.__setattr__(self, "_rows", tuple(rows))
        object.__setattr__(self, "_native_names", tuple(native_names))
        object.__setattr__(self, "_native_has_function", native_type.has_function)
        object.__setattr__(self, "_native_is_number", native_type.__dict__["is_number"])
        object.__setattr__(self, "_native_is_symbol", native_type.__dict__["is_symbol"])
        object.__setattr__(self, "_value_slot", value_slot)

    def __setattr__(self, name, value):
        raise AttributeError("function capability records are immutable")

    def __delattr__(self, name):
        raise AttributeError("function capability records are immutable")

    def is_registered(self, cls):
        return any(cls is row[0] for row in self._rows)

    def native_value(self, source):
        return self._value_slot.__get__(source, type(source))

    @staticmethod
    def _matches(row, checked):
        cls, metaclass, recorded_mro, recorded_meta_mro, owners = row
        current_mro = _MRO.__get__(cls)
        current_meta_mro = _MRO.__get__(type(cls))
        if (type(cls) is not metaclass or len(current_mro) != len(recorded_mro)
                or not all(a is b for a, b in zip(current_mro, recorded_mro))
                or len(current_meta_mro) != len(recorded_meta_mro)
                or not all(a is b for a, b in zip(current_meta_mro, recorded_meta_mro))):
            return False
        for owner, record in owners:
            key = id(owner)
            if key not in checked:
                checked[key] = _owner_matches(owner, record)
            if not checked[key]:
                return False
        return True

    def is_current(self, cls):
        row = next((row for row in self._rows if cls is row[0]), None)
        return row is not None and self._matches(row, {})

    def require_class(self, cls):
        if not self.is_current(cls):
            raise NotImplementedError(
                "custom or opaque function implementations have no closed native lowering"
            )

    def require_native_tag(self, name):
        row = next((row for tag, row in self._native_names if name == tag), None)
        if row is not None and not self._matches(row, {}):
            raise NotImplementedError(
                "modified built-in function implementations have no closed native lowering"
            )

    def require_native(self, source):
        # These are borrowed native queries. No arbitrary source tree is
        # cloned or lifted into Python classes to discover hidden functions.
        if (self._native_is_number.__get__(source)
                or self._native_is_symbol.__get__(source)):
            return
        checked = {}
        for name, row in self._native_names:
            if self._native_has_function(source, name) and not self._matches(row, checked):
                raise NotImplementedError(
                    "modified built-in function implementations have no closed native lowering"
                )
