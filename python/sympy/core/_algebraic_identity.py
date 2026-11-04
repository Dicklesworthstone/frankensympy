"""Bounded structural identity for expressions containing genuine bound roots.

A CRootOf's polynomial generator is bound. Its exact coefficient vector and
selected index establish its structural identity, as for the root class
itself. Enclosing ordered structure and ordinary typed native atoms remain
unchanged. This module does not simplify arithmetic, reorder commutative
children, compare approximate root values, or lower opaque Python functions.
Unadmitted native function tags refuse explicitly; this is not a general
semantic equality oracle for every symbolic constructor.
"""

from . import Basic, Function, _native
from . import _opaque_arithmetic as _opaque
from .. import core as _core

_VALUE = Basic.__dict__["_value"]
_NAME = type.__dict__["__name__"]
_MAX_NODES = 16384
_MAX_DEPTH = 128


def _limit(work, depth, amount=1):
    work[0] += amount
    if work[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise ValueError("bound-root identity exceeds its structural work limit")


def _root_name():
    root_type = _core._bound_algebraic_root_type
    return None if root_type is None else _NAME.__get__(root_type)


def _has_root(value):
    """Probe retained slots without running surface properties or callbacks."""
    root_type = _core._bound_algebraic_root_type
    if root_type is None or not _opaque._subclass(type(value), Basic):
        return False
    pending = [(value, False, 0)]
    active, seen, keepalive = set(), set(), {}
    work = [0]
    while pending:
        node, leaving, depth = pending.pop()
        if not _opaque._subclass(type(node), Basic):
            continue
        identity = id(node)
        if leaving:
            active.remove(identity)
            seen.add(identity)
            continue
        if identity in active:
            raise ValueError("cyclic retained expression in bound-root identity")
        if identity in seen:
            continue
        _limit(work, depth)
        keepalive[identity] = node
        if type(node) is root_type:
            return True
        children = _opaque.stored_args(node)
        if children is not None:
            _limit(work, depth, len(children))
            active.add(identity)
            pending.append((node, True, depth))
            pending.extend((child, False, depth + 1) for child in reversed(children))
            continue
        native = _opaque._slot(_VALUE, node)
        if type(native) is _native.Expr and native.has_function(_root_name()):
            # A candidate tag in a closed native region is fully preflighted
            # and checked before it can contribute a bound-root identity.
            return True
        seen.add(identity)
    return False


def _key(value):
    """Build a compact ordered DAG key; refusals never become inequality."""
    root_type = _core._bound_algebraic_root_type
    root_name = _root_name()
    records, interned, memo, active = [], {}, {}, set()
    work = [0]

    def intern(record):
        index = interned.get(record)
        if index is None:
            index = len(records)
            interned[record] = index
            records.append(record)
        return index

    def validate(native):
        if type(native) is not _native.Expr:
            raise NotImplementedError("bound-root identity requires retained children or a closed native value")
        # Borrowed native preflight bounds depth, fanout and literal payloads
        # before args clones or polynomial coefficient extraction.
        _native.CompiledFunction._validate_source(native)

    def count_payload(native, depth):
        _limit(work, depth)
        children = native.args
        _limit(work, depth, len(children))
        for child in children:
            count_payload(child, depth + 1)

    def root_record(native, depth):
        _core._require_closed_function_class(root_type)
        validate(native)
        children = native.args
        if native.func_name != root_name or len(children) != 2:
            raise ValueError("invalid bound-root identity payload")
        polynomial, index = children
        count_payload(polynomial, depth + 1)
        _limit(work, depth + 1)
        if not index.is_integer:
            raise ValueError("bound-root identity requires an exact integer index")
        selected = index.exact_numerator()
        bindings = polynomial.free_symbol_bindings
        if len(bindings) != 1:
            raise ValueError("bound-root identity requires a univariate defining polynomial")
        # Original-source admission precedes expansion in this existing
        # native routine. Every zero coefficient and the typed binding survive;
        # only that bound generator is absent from the resulting identity.
        coefficients = _native.real_root_coefficients_native(polynomial, bindings[0])
        if not 0 <= selected < len(coefficients) - 1:
            raise ValueError("bound-root identity index is outside the defining degree")
        exact = []
        for coefficient in coefficients:
            numerator = coefficient.exact_numerator()
            denominator = coefficient.exact_denominator()
            _limit(work, depth, 1 + (max(abs(numerator).bit_length(),
                                        denominator.bit_length()) + 63) // 64)
            exact.append((numerator, denominator))
        return ("bound-root", id(root_type), tuple(exact), selected)

    def condition_key(native, depth):
        # These tags belong only to the native Piecewise condition protocol.
        # Preserve their exact tag and order; never resolve a user class by it.
        tag = native.func_name
        if tag == "_FsymBooleanSymbol":
            _limit(work, depth)
            children = native.args
            if len(children) != 1:
                raise ValueError("invalid symbolic Piecewise condition")
            if children[0].is_symbol:
                return intern(("condition-symbol", native_key(children[0], depth + 1, checked=True)))
            # A native substitution may replace the guarded symbol with a
            # typed relational/Boolean condition; the wrapper then disappears
            # when lifted through Piecewise._from_native.
            if children[0].func_name in (
                    "Equality", "Unequality", "StrictLessThan", "LessThan",
                    "StrictGreaterThan", "GreaterThan", "And", "Or", "Not",
                    "Xor", "Implies", "Equivalent"):
                return condition_key(children[0], depth + 1)
            raise ValueError("invalid substituted Piecewise condition")
        relationals = ("Eq", "Ne", "Lt", "Le", "Gt", "Ge",
                       "Equality", "Unequality", "StrictLessThan", "LessThan",
                       "StrictGreaterThan", "GreaterThan")
        logical = ("And", "Or", "Not", "Xor", "Implies", "Equivalent")
        if tag not in relationals and tag not in logical:
            return native_key(native, depth, checked=True)
        _limit(work, depth)
        children = native.args
        _limit(work, depth, len(children))
        if ((tag in relationals and len(children) != 2)
                or (tag == "Not" and len(children) != 1)
                or (tag == "Implies" and len(children) != 2)):
            raise ValueError("invalid native Piecewise condition arity")
        child_key = native_key if tag in relationals else condition_key
        if tag in relationals:
            keys = tuple(child_key(child, depth + 1, checked=True) for child in children)
        else:
            keys = tuple(child_key(child, depth + 1) for child in children)
        return intern(("piecewise-condition", tag, keys))

    def piecewise_key(native, depth):
        from ..functions.elementary.piecewise import Piecewise

        # The owner is the actual registered built-in. Name lookup could be
        # captured by a same-named user Function and is inappropriate here.
        _core._require_closed_function_class(Piecewise)
        branches = native.args
        _limit(work, depth, len(branches))
        keys = []
        for branch in branches:
            _limit(work, depth + 1)
            pair = branch.args
            if branch.func_name != "Tuple" or len(pair) != 2:
                raise ValueError("invalid native Piecewise branch payload")
            _limit(work, depth + 1, len(pair))
            keys.append(intern(("piecewise-branch",
                                native_key(pair[0], depth + 2, checked=True),
                                condition_key(pair[1], depth + 2))))
        return intern(("node", id(Piecewise), "Piecewise", tuple(keys)))

    def native_key(native, depth, *, checked=False):
        _limit(work, depth)
        if not checked:
            validate(native)
        tag = native.func_name
        if tag == root_name:
            return intern(root_record(native, depth))
        if native.is_number or native.is_symbol:
            # Actual typed atoms, never a Function whose name impersonates
            # Integer, Symbol or Constant. SymbolBinding/Dummy payloads survive.
            return intern(("atom", native))
        if tag == _core._FLOAT_INTERN:
            children = native.args
            if (len(children) != 1 or not children[0].is_integer
                    or not 0 <= children[0].exact_numerator() < 1 << 64):
                raise ValueError("invalid native Float payload in bound-root identity")
            return intern(("atom", native))
        if tag == "Piecewise":
            return piecewise_key(native, depth)
        if native.is_add:
            cls = _core.Add
        elif native.is_mul:
            cls = _core.Mul
        elif native.is_pow:
            cls = _core.Pow
        elif tag in ("Add", "Mul", "Pow", "Integer", "Rational", "Symbol", "Constant"):
            raise NotImplementedError("a native function cannot impersonate a core expression node")
        else:
            cls = {"diff": _core.Derivative, "Derivative": _core.Derivative}.get(tag)
        if cls is None:
            cls = _core._lookup_known_function(tag)
            if cls is None or not _core._is_closed_function_class(cls):
                raise NotImplementedError(
                    "bound-root identity cannot recover an opaque function from a native name"
                )
        children = native.args
        _limit(work, depth, len(children))
        return intern(("node", id(cls), tag,
                       tuple(native_key(child, depth + 1, checked=True) for child in children)))

    def visit(node, depth):
        _limit(work, depth)
        identity = id(node)
        if identity in active:
            raise ValueError("cyclic retained expression in bound-root identity")
        if identity in memo:
            return memo[identity][1]
        cls = type(node)
        if not _opaque._subclass(cls, Basic):
            raise NotImplementedError("bound-root identity requires symbolic surface arguments")
        native = _opaque._slot(_VALUE, node)
        if cls is root_type:
            result = intern(root_record(native, depth))
        else:
            function = _opaque._subclass(cls, Function)
            if function:
                # Keep the existing refusal for custom identity or
                # commutativity hooks; never execute them while making a key.
                _opaque._admit_application(node)
            elif not any(cls is builtin for builtin in _core._exact_surface_types()):
                raise NotImplementedError("bound-root identity requires exact built-in arithmetic classes")
            children = _opaque.stored_args(node)
            if (children is not None and type(native) is _native.Expr
                    and cls in (_core.Add, _core.Mul, _core.Pow)):
                actual = (native.is_add if cls is _core.Add else
                          native.is_mul if cls is _core.Mul else native.is_pow)
                if not actual:
                    raise NotImplementedError("a native function cannot impersonate retained arithmetic")
            if children is None:
                if function and not _core._is_closed_function_class(cls):
                    raise NotImplementedError(
                        "bound-root identity requires retained arguments for an opaque function"
                    )
                result = native_key(native, depth)
            else:
                _limit(work, depth, len(children))
                active.add(identity)
                child_keys = tuple(visit(child, depth + 1) for child in children)
                active.remove(identity)
                if function and not _core._is_closed_function_class(cls):
                    tag = None
                elif type(native) is _native.Expr:
                    tag = native.func_name
                else:
                    tag = {_core.Add: "Add", _core.Mul: "Mul", _core.Pow: "Pow"}.get(cls)
                result = intern(("node", id(cls), tag, child_keys))
        # Retain the object alongside its ID: a mutable parent cannot make a
        # later object reuse a memoized address during this operation.
        memo[identity] = (node, result)
        return result

    root = visit(value, 0)
    return tuple(records), root


def hash_override(value):
    """Return a root-aware hash, or None to retain the ordinary fast path."""
    if not _has_root(value):
        return None
    return hash(_key(value))


def equality_override(left, right):
    """Return a root-aware decision, or None when neither side has a root.

    Unsupported identities and structural work limits raise explicitly. They
    never fall through to raw native equality or become a false decision.
    """
    if not _has_root(left) and not _has_root(right):
        return None
    if not _opaque._subclass(type(left), Basic) or not _opaque._subclass(type(right), Basic):
        return False
    return _key(left) == _key(right)

