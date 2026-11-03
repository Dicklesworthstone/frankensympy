"""Arithmetic over retained Python applications and closed native regions.

The slots used while normalizing an operation exist only in that operation.
They are replaced by the original Python objects before publication and never
become native semantic identities.  An expression containing such an object
has no native value: closed algorithms must explicitly refuse it.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cmp_to_key

from . import (
    Add, Basic, Derivative, Float, Function, Mul, Pow, Rational, Relational,
    Symbol, _exact_surface_types, _is_closed_function_class, _known_functions,
    _native, _wrap, sympify,
)

_VALUE = Basic.__dict__["_value"]
_SURFACE = Basic.__dict__["_surface_args"]
_STRUCT = Basic.__dict__["_struct_args"]
_FUNCTION_ARGS = Function.__dict__["_args"]
_RELATIONAL_SIDES = Relational.__dict__["_relational_sides"]
_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_NAME = type.__dict__["__name__"]
_MAX_NODES = 16384
_MAX_DEPTH = 128


def _subclass(cls, base):
    return any(parent is base for parent in _MRO.__get__(cls))


def _slot(descriptor, value):
    try:
        return descriptor.__get__(value, type(value))
    except AttributeError:
        return None


def stored_args(value):
    """Read retained children without invoking a user property or setter."""
    cls = type(value)
    if not _subclass(cls, Basic):
        return None
    children = _slot(_SURFACE, value)
    if children is None:
        if cls is Add or cls is Mul or cls is Pow:
            children = _slot(cls.__dict__["_args"], value)
        elif _subclass(cls, Function):
            children = _slot(_FUNCTION_ARGS, value)
        elif _subclass(cls, Relational):
            children = _slot(_RELATIONAL_SIDES, value)
        else:
            children = _slot(_STRUCT, value)
    if children is not None and type(children) is not tuple:
        raise ValueError("retained symbolic children must be an exact tuple")
    return children


def _custom_function(cls):
    return (_subclass(cls, Function)
            and not _is_closed_function_class(cls))


def contains_opaque(value):
    """Whether a surface graph requires Python-owned identity.

    The walk is bounded and cycle checked, including closed retained children.
    Exact native-only input is already in the native identity universe; this
    routine never recovers a Python class from a native function's name.
    """
    stack = [(value, False, 0)]
    active = set()
    seen = set()
    count = 0
    opaque = False
    while stack:
        node, leaving, depth = stack.pop()
        if not _subclass(type(node), Basic):
            continue
        identity = id(node)
        if leaving:
            active.remove(identity)
            seen.add(identity)
            continue
        if identity in active:
            raise ValueError("cyclic retained symbolic graph")
        if identity in seen:
            continue
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("retained symbolic graph exceeds the arithmetic limit")
        children = stored_args(node)
        cls = type(node)
        if (not any(cls is builtin for builtin in _exact_surface_types())
                and not _is_closed_function_class(cls)):
            opaque = True
        if _custom_function(cls) or _slot(_STRUCT, node) is not None:
            opaque = True
        if _slot(_VALUE, node) is None:
            opaque = True
        if children is not None:
            count += len(children)
            if count > _MAX_NODES:
                raise ValueError("retained symbolic graph exceeds the argument limit")
            active.add(identity)
            stack.append((node, True, depth))
            stack.extend((child, False, depth + 1) for child in reversed(children))
        else:
            seen.add(identity)
    return opaque


def _children(value):
    children = stored_args(value)
    if children is not None:
        return children
    native = _slot(_VALUE, value)
    if type(native) is not _native.Expr:
        raise NotImplementedError("symbolic source has no retained children or closed native value")
    _validate_native_region(native)
    return tuple(_wrap(child) for child in native.args)


def _validate_native_region(native):
    """Bound borrowed native input before cloning or Python class recovery."""
    from . import _FLOAT_INTERN, _lookup_known_function

    _native.CompiledFunction._validate_source(native)
    pending = [native]
    while pending:
        current = pending.pop()
        if current.is_symbol or current.is_integer or current.is_rational:
            continue
        if current.is_add or current.is_mul or current.is_pow:
            pending.extend(current.args)
            continue
        name = current.func_name
        if name == "Constant":
            if not any(current == _native.Expr(literal) for literal in ("pi", "E", "I", "oo", "-oo", "zoo", "nan")):
                raise NotImplementedError("native function cannot impersonate a mathematical constant")
            continue
        if name not in ("diff", "Derivative", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", _FLOAT_INTERN):
            cls = _lookup_known_function(name)
            if cls is None or not _is_closed_function_class(cls):
                raise NotImplementedError("native-only arithmetic cannot recover an opaque Python function class")
        pending.extend(current.args)


def _key(value, *, names=False):
    """Bounded structural equality/hash key with no surface callbacks."""
    from . import Expr, Half, Integer, NegativeOne, One, Zero, _FLOAT_INTERN

    memo = {}
    records = []
    interned = {}
    work = [0]

    def charge(depth):
        work[0] += 1
        if work[0] > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("opaque arithmetic structural work exceeds its limit")

    def label(cls):
        # Class addresses are local equality keys only. They never cross into
        # native storage, serialization, receipts, or stable term identities.
        return _NAME.__get__(cls) if names else id(cls)

    def intern(record):
        found = interned.get(record)
        if found is None:
            found = len(records)
            records.append(record)
            interned[record] = found
        return found

    def native_key(native, depth):
        charge(depth)
        tag = native.func_name
        if native.is_integer:
            number = native.exact_numerator()
            cls = Zero if number == 0 else One if number == 1 else NegativeOne if number == -1 else Integer
        elif native.is_rational:
            cls = Half if native.exact_numerator() == 1 and native.exact_denominator() == 2 else Rational
        elif native.is_symbol:
            # Dummy identities are represented by their already registered
            # shell object; lookup is typed and does not construct anything.
            from . import _symbol_from_binding
            cls = type(_symbol_from_binding(native.symbol_binding))
        elif tag == _FLOAT_INTERN:
            cls = Float
        else:
            cls = {"Add": Add, "Mul": Mul, "Pow": Pow, "diff": Derivative,
                   "Derivative": Derivative}.get(tag)
            if cls is None:
                known = _known_functions.get(tag)
                if known is None or not _is_closed_function_class(known):
                    # Unknown native functions remain native semantic leaves.
                    # Their full typed payload includes the function tag; a
                    # generic Expr class plus children would erase U vs V.
                    return intern(("leaf", label(Expr), native))
                cls = known
            children = native.args
            if children:
                return intern(("surface", label(cls), tuple(native_key(child, depth + 1) for child in children)))
        return intern(("leaf", label(cls), native))

    def visit(node, depth):
        charge(depth)
        identity = id(node)
        if identity in memo:
            return memo[identity]
        cls = type(node)
        if not _subclass(cls, Basic):
            raise NotImplementedError("opaque arithmetic requires symbolic surface arguments")
        children = stored_args(node)
        if children is not None:
            result = intern(("surface", label(cls), tuple(visit(child, depth + 1) for child in children)))
        else:
            native = _slot(_VALUE, node)
            if type(native) is not _native.Expr:
                raise NotImplementedError("opaque arithmetic requires symbolic surface arguments")
            if any(cls is builtin for builtin in (Add, Mul, Pow)) or _is_closed_function_class(cls):
                result = intern(("surface", label(cls), tuple(native_key(child, depth + 1) for child in native.args)))
            else:
                result = intern(("leaf", label(cls), native))
        memo[identity] = result
        return result

    root = visit(value, 0)
    # Integer child references make hashing and comparing a shared surface DAG
    # linear in the admitted graph, rather than expanding repeated subtrees.
    return tuple(records), root


def surface_equal(left, right):
    if type(left) is not type(right):
        return False
    contains_opaque(left)
    contains_opaque(right)
    _admit_source((left, right), check_identity=True)
    return _key(left) == _key(right)


def surface_hash(value):
    contains_opaque(value)
    _admit_source((value,), check_identity=True)
    # Distinct same-named classes may share a hash, as in the oracle;
    # equality still compares their exact class identities.
    return hash(_key(value, names=True))


def _order_parts(value):
    """One level of structural order, without user printers or hooks."""
    cls = type(value)
    native = _slot(_VALUE, value)
    if _subclass(cls, Rational):
        return (0, Fraction(native.exact_numerator(), native.exact_denominator())), ()
    if cls is Float:
        return (0, value._as_python_float()), ()
    if _subclass(cls, Symbol):
        binding = native.symbol_binding
        return (1, binding.name, binding.identity_hex), ()
    if cls is Pow or cls is Mul or cls is Add:
        rank = 2 if cls is Pow else 3 if cls is Mul else 4
        children = _children(value)
        return (rank, len(children)), children
    if _subclass(cls, Function):
        children = _children(value)
        return (5, _NAME.__get__(cls), len(children)), children
    children = _children(value)
    if not children and type(native) is _native.Expr:
        # This is the rendering of an already closed native atom, never a
        # user printer and never an equality or lowering identity.
        return (1, str(native), ""), ()
    return (6, _NAME.__get__(cls), len(children)), children


def _ordered(children):
    # Comparing memoized child pairs avoids expanding shared function DAGs.
    # A structural ordering tie keeps input order, including distinct classes
    # with the same visible name and equal argument ordering keys.
    parts, compared = {}, {}
    work = [0]

    def compare(left, right):
        if left is right:
            return 0
        pair = (id(left), id(right))
        if pair in compared:
            return compared[pair]
        work[0] += 1
        if work[0] > _MAX_NODES:
            raise ValueError("opaque arithmetic ordering exceeds its work limit")
        for node in (left, right):
            if id(node) not in parts:
                parts[id(node)] = _order_parts(node)
        left_key, left_children = parts[id(left)]
        right_key, right_children = parts[id(right)]
        result = (left_key > right_key) - (left_key < right_key)
        if not result:
            for left_child, right_child in zip(left_children, right_children):
                result = compare(left_child, right_child)
                if result:
                    break
        compared[pair] = result
        compared[(pair[1], pair[0])] = -result
        return result

    return sorted(children, key=cmp_to_key(compare))


def _make(cls, children, *, held=False):
    obj = object.__new__(cls)
    _VALUE.__set__(obj, None)
    if held:
        cls.__dict__["_args"].__set__(obj, tuple(children))
    else:
        if cls is Add or cls is Mul:
            children = _ordered(children)
        _SURFACE.__set__(obj, tuple(children))
    return obj


def _admit_application(value):
    # The inherited Function contract is commutative. An explicit override
    # needs its own shell algebra; native commutative normalization cannot
    # silently reinterpret it. Inspect definitions without executing them.
    for cls in _MRO.__get__(type(value)):
        namespace = _DICT.__get__(cls)
        if ("__eq__" in namespace and namespace["__eq__"] is not Basic.__dict__["__eq__"]
                or "__hash__" in namespace and namespace["__hash__"] is not Basic.__dict__["__hash__"]
                or "_hashable_content" in namespace):
            raise NotImplementedError(
                "opaque arithmetic with custom equality or hashing needs a surface algebra"
            )
        if "is_commutative" in namespace:
            declaration = namespace["is_commutative"]
            from . import Expr
            if declaration is not True and declaration is not Expr.__dict__["is_commutative"]:
                raise NotImplementedError("opaque arithmetic with custom commutativity needs a surface algebra")
        if cls is Function:
            break


def _admit_source(values, *, check_identity):
    pending, seen = list(values), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        cls = type(node)
        if not _subclass(cls, Basic):
            raise NotImplementedError("opaque arithmetic requires symbolic surface arguments")
        if _subclass(cls, Function):
            if check_identity:
                _admit_application(node)
        elif not any(cls is builtin for builtin in _exact_surface_types()):
            raise NotImplementedError(
                "custom symbolic subclasses require a supervised Python override lane; "
                "the native fast path accepts exact built-in classes only"
            )
        pending.extend(stored_args(node) or ())


def arithmetic(operator, values, *, evaluate=True, cls=None):
    """Normalize one mixed operation, or return None for a closed operation."""
    admitted = tuple(contains_opaque(value) for value in values)
    if not any(admitted):
        return None
    values = tuple(value if _subclass(type(value), Basic) else sympify(value) for value in values)
    _admit_source(values, check_identity=evaluate)
    target = cls or {"add": Add, "mul": Mul, "pow": Pow}[operator]
    if not evaluate:
        return _make(target, values, held=True)

    # The oracle processes already flattened arguments first and appends the
    # contents of nested associative operators to the pending queue. This
    # matters when distinct classes have equal visible ordering keys.
    if operator in ("add", "mul"):
        queue, flattened = list(values), []
        index = 0
        while index < len(queue):
            value = queue[index]
            index += 1
            if type(value) is target:
                queue.extend(_children(value))
            else:
                flattened.append(value)
            if len(queue) > _MAX_NODES:
                raise ValueError("opaque arithmetic exceeds the argument limit")
        values = tuple(flattened)

    constructor = {"add": _native.Add, "mul": _native.Mul, "pow": _native.Pow}[operator]
    return native_transform(values, lambda lowered: constructor(*lowered, evaluate=True).as_expr())


def native_transform(values, operation, *, multiple=False, atomize_functions=False):
    """Apply one closed native operation with owned, temporary application slots.

    Every input participates in admission and collision avoidance. The private
    caller supplies a native operation; no local application slot survives the
    returned surface graph or becomes a portable semantic identity. Polynomial
    callers may atomize every application and return an exact, bounded tuple
    of results; all outputs use this one slot environment.
    """
    values = tuple(values)
    for value in values:
        contains_opaque(value)
    _admit_source(values, check_identity=True)
    reserved = set()
    stack = list(values)
    seen = set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if len(seen) > _MAX_NODES:
            raise ValueError("opaque arithmetic exceeds the source limit")
        native = _slot(_VALUE, value)
        if type(native) is _native.Expr:
            _validate_native_region(native)
            reserved.update(binding.name for binding in native.free_symbol_bindings)
        children = stored_args(value)
        if children is not None:
            stack.extend(children)
    prefix = "__fsym_operation_slot_"
    slot_names = set()
    slot_index = [0]
    applications = []
    by_key = {}
    by_native = {}
    closed = {}
    composites = {}
    encoded = {}
    encode_work = [0]

    def encode(value):
        encode_work[0] += 1
        if encode_work[0] > _MAX_NODES:
            raise ValueError("opaque arithmetic normalization exceeds its work limit")
        if id(value) in encoded:
            return encoded[id(value)][1]
        kind = type(value)
        if atomize_functions and not (kind in (Add, Mul, Pow) or _subclass(kind, Function)):
            native = _slot(_VALUE, value)
            if type(native) is _native.Expr and native.args:
                # A generic Expr can own a closed native composite. Admission
                # above has already excluded unknown or modified function
                # tags, so lifting cannot recover a custom class by its name.
                recovered = _wrap(native)
                if type(recovered) is not kind:
                    result = encode(recovered)
                    encoded[id(value)] = (value, result)
                    return result
        if kind is Add or kind is Mul or kind is Pow:
            children = [encode(child) for child in _children(value)]
            if kind is Add:
                result = _native.Add(*children, evaluate=True).as_expr()
                # Native child traversal creates temporary shell wrappers.
                # Retain each owner for the entire operation so its address
                # cannot be recycled as the cache key of a different subtree.
                encoded[id(value)] = (value, result)
                if _slot(_SURFACE, value) is not None:
                    composites.setdefault(result, value)
                return result
            if kind is Mul:
                result = _native.Mul(*children, evaluate=True).as_expr()
                encoded[id(value)] = (value, result)
                if _slot(_SURFACE, value) is not None:
                    composites.setdefault(result, value)
                return result
            result = _native.Pow(*children, evaluate=True).as_expr()
            encoded[id(value)] = (value, result)
            if _slot(_SURFACE, value) is not None:
                composites.setdefault(result, value)
            return result
        if contains_opaque(value) or atomize_functions and _subclass(kind, Function):
            if not _subclass(kind, Function) and kind is not Derivative:
                raise NotImplementedError("opaque arithmetic supports retained Function applications and derivatives")
            if _subclass(kind, Function):
                _admit_application(value)
            key = _key(value)
            if key not in by_key:
                name = f"{prefix}{slot_index[0]:08d}"
                while name in reserved:
                    slot_index[0] += 1
                    name = f"{prefix}{slot_index[0]:08d}"
                slot_index[0] += 1
                slot_names.add(name)
                slot = _native.py_symbol(name)
                by_key[key] = slot
                by_native[slot] = value
                applications.append(value)
            result = by_key[key]
            encoded[id(value)] = (value, result)
            return result
        native = _slot(_VALUE, value)
        if type(native) is not _native.Expr:
            raise NotImplementedError("opaque arithmetic received an unsupported closed leaf")
        closed.setdefault(native, value)
        encoded[id(value)] = (value, native)
        return native

    lowered = [encode(value) for value in values]
    result = operation(lowered)
    if multiple:
        if type(result) is not tuple or len(result) > _MAX_NODES:
            raise TypeError("a multiple surface transformation must return a bounded exact tuple")
        results = result
    else:
        results = (result,)
    for value in results:
        if type(value) is not _native.Expr:
            raise TypeError("a closed surface transformation must return native expressions")
        _validate_native_region(value)

    def lift(value):
        encode_work[0] += 1
        if encode_work[0] > _MAX_NODES:
            raise ValueError("opaque arithmetic restoration exceeds its work limit")
        if value.is_symbol and value in by_native:
            return by_native[value]
        if value in closed:
            return closed[value]
        if value in composites:
            # Preserve already canonical surface subtrees. In particular,
            # distinct same-named applications can have tied ordering keys;
            # a native slot numbering must not reorder their callbacks.
            return composites[value]
        kind = Add if value.is_add else Mul if value.is_mul else Pow if value.is_pow else None
        if kind is not None:
            children = tuple(lift(child) for child in value.args)
            if any(contains_opaque(child) for child in children):
                return _make(kind, children)
        # Never publish a closed region containing an operation-local slot.
        if any(binding.name in slot_names for binding in value.free_symbol_bindings):
            raise RuntimeError("opaque arithmetic failed to restore a local application slot")
        return _wrap(value)

    restored = tuple(lift(value) for value in results)
    return restored if multiple else restored[0]
