"""Bounded common-subexpression elimination for closed scalar expressions.

The optimizer works on an operation-local DAG. Ordered children retain held
multiplicities; common Add/Mul arguments are extracted as multisets. Definitions
are emitted before their users, and temporary names cannot capture input names.
Opaque/custom, noncommutative and binding inputs are returned unchanged: moving
their computations requires a separate scope/effect contract. This surface
transformation does not produce a mathematical certificate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterator

from ..core import (
    Add, Application, AssocOp, Atom, AtomicExpr, Basic, ComplexInfinity, Dummy, Expr, Float, Function,
    FunctionClass, Half, Integer, Mul, NegativeOne, Number, One, Pow, Rational,
    Symbol, Zero, _ExactDecimalFloat, _FLOAT_INTERN, _NativeFunction, _charge_nested_if_active,
    _is_closed_function_class, _native, sympify,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from ..core.lambdify_subs import Subs
from ..functions.elementary.piecewise import Piecewise
from ..integrals.integrals import Integral
from ..series import Order

_MAX_WORK = 200_000
_MAX_DEPTH = 128
_MAX_INPUTS = 4096
_MAX_SYMBOL_ATTEMPTS = 4096
_NATIVE_EXPR = _native.Expr
_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_NAME = type.__dict__["__name__"]
_SCOPED = (Integral, Subs, Order, Piecewise)
_VALUE = Basic.__dict__["_value"]
_SURFACE = Basic.__dict__["_surface_args"]
_STRUCT = Basic.__dict__["_struct_args"]
_ASSUMPTIONS = Symbol.__dict__["_assumptions"]
_DUMMY_NAME = Dummy.__dict__["_dummy_name"]
_FLOAT_DPS = Float.__dict__["_dps"]
_DECIMAL = _ExactDecimalFloat.__dict__["_decimal"]
_HELD = ((Add, Add.__dict__["_args"]),
         (Mul, Mul.__dict__["_args"]),
         (Pow, Pow.__dict__["_args"]))
_ATOMS = (Basic, Atom, AtomicExpr, Expr, Number, Integer, Rational, Zero, One,
          NegativeOne, Half, Float, _ExactDecimalFloat, ComplexInfinity, Symbol, Dummy)
_OWNERS = tuple((owner, _owner_record(owner)) for owner in (
    Basic, Atom, AtomicExpr, Expr, Number, Integer, Rational, Zero, One,
    NegativeOne, Half, Float, _ExactDecimalFloat, ComplexInfinity, Symbol, Dummy,
    Add, Mul, Pow, AssocOp, Application, Function, FunctionClass, _NativeFunction,
))


_MISSING = object()
_OWNER_MROS = tuple((owner, _MRO.__get__(owner)) for owner, _ in _OWNERS)
# FunctionCapabilities tracks semantic hooks; these additional fields are read
# by scalar storage and construction. Capture absent overrides as well.
_EXTRA_NAMES = ("name", "__str__", "_assumptions", "_dps", "_decimal",
                "_dummy_name", "_dummy_number")
_EXTRA_RECORDS = tuple(
    (owner, name, _implementation(_DICT.__get__(owner).get(name, _MISSING)))
    for owner, _ in _OWNERS for name in _EXTRA_NAMES
)


def _require_original():
    if (any(not _owner_matches(owner, record) for owner, record in _OWNERS)
            or any(len(_MRO.__get__(owner)) != len(record)
                   or any(a is not b for a, b in zip(_MRO.__get__(owner), record))
                   for owner, record in _OWNER_MROS)):
        raise NotImplementedError("CSE requires the original built-in scalar implementations")
    for owner, name, expected in _EXTRA_RECORDS:
        current = _implementation(_DICT.__get__(owner).get(name, _MISSING))
        if len(current) != len(expected) or any(a is not b for a, b in zip(current, expected)):
            raise NotImplementedError("CSE requires the original built-in scalar metadata")


def _slot(descriptor, value):
    try:
        return descriptor.__get__(value, type(value))
    except AttributeError:
        return None


def numbered_symbols(prefix: str = "x", cls: Any = None, start: int = 0,
                     exclude: Any = (), *args: Any, **assumptions: Any) -> Iterator[Any]:
    """Infinite generator of symbols prefix0, prefix1, ... ."""
    cls = cls or Symbol
    exclude = set(exclude or ())
    i = start
    while True:
        symbol = cls("%s%d" % (prefix, i), *args, **assumptions)
        if symbol not in exclude:
            yield symbol
        i += 1


class _Unsupported(Exception):
    """An input needs a scope, noncommutative or Python effect contract."""


class _Graph:
    """Integer child references avoid expanding a shared expression DAG."""

    def __init__(self):
        self.remaining = _MAX_WORK
        self.nodes = []
        self.interned = {}
        self.sources = {}
        self.active = set()
        self.reserved_names = set()
        self.has_approximate = False
        self.functions = {}

    def charge(self, count=1, depth=0):
        if depth > _MAX_DEPTH or count < 0 or count > self.remaining:
            raise ValueError("CSE exceeds its work/depth limit")
        self.remaining -= count
        if count:
            _charge_nested_if_active(count)

    def intern(self, key, node):
        existing = self.interned.get(key)
        if existing is not None:
            return existing
        index = len(self.nodes)
        self.nodes.append(node)
        self.interned[key] = index
        return index

    def source(self, value, depth=0):
        self.charge(depth=depth)
        identity = id(value)
        if identity in self.active:
            raise ValueError("CSE source contains a cycle")
        if identity in self.sources:
            return self.sources[identity][1]
        cls = type(value)
        arithmetic = cls is Add or cls is Mul or cls is Pow
        function = _is_closed_function_class(cls)
        atom = any(cls is owner for owner in _ATOMS)
        if not (arithmetic or function or atom) or any(cls is owner for owner in _SCOPED):
            raise _Unsupported
        if function and any(
                any(name in _DICT.__get__(owner)
                    for name in ("bound_symbols", "variables", "limits", "variable_count"))
                for owner in _MRO.__get__(cls)):
            raise _Unsupported
        if _slot(_STRUCT, value) is not None:
            raise _Unsupported
        native = _slot(_VALUE, value)
        if type(native) is not _NATIVE_EXPR:
            raise _Unsupported
        if function:
            name = _NAME.__get__(cls)
            tag = native.func_name
            if tag != name and (name, tag) not in (("log", "ln"), ("Abs", "abs")):
                raise NotImplementedError("CSE function class and native source tag disagree")
            self.functions[id(cls)] = (cls, name)
        if atom:
            if cls is Symbol or cls is Dummy:
                facts = _slot(_ASSUMPTIONS, value)
                if (type(facts) is not dict or len(facts) > 64
                        or any(type(key) is not str or type(fact) is not bool
                               for key, fact in facts.items())):
                    raise ValueError("CSE requires literal scalar assumptions")
                if facts.get("commutative") is False:
                    raise _Unsupported
                name = (_slot(_DUMMY_NAME, value) if cls is Dummy else native.symbol_binding.name)
                if type(name) is not str:
                    raise ValueError("CSE requires literal symbol names")
                self.reserved_names.add(name)
                self.reserved_names.add(native.symbol_binding.name)
            elif native.is_symbol:
                # A generic Expr does not own the declared Symbol assumptions.
                raise _Unsupported
            elif cls is not Float and cls is not _ExactDecimalFloat and not native.is_number:
                # This getter only matches the native enum variant; asking for
                # args here would clone a generic Expr composite before refusal.
                raise _Unsupported
            if (cls is Float or cls is _ExactDecimalFloat) and native.func_name != _FLOAT_INTERN:
                raise ValueError("CSE Float has no canonical native payload tag")
            precision = ((_slot(_FLOAT_DPS, value), _slot(_DECIMAL, value))
                         if cls is _ExactDecimalFloat else
                         (_slot(_FLOAT_DPS, value),) if cls is Float else ())
            if precision:
                self.has_approximate = True
                if (type(precision[0]) is not int or precision[0] < 1
                        or len(precision) == 2 and type(precision[1]) is not str):
                    raise ValueError("CSE requires canonical Float precision metadata")
            key = ("atom", id(cls), native, precision)
            result = self.intern(key, (cls, (), value, False))
        else:
            retained = _slot(_SURFACE, value)
            held = False
            for owner, descriptor in _HELD:
                if cls is owner:
                    original = _slot(descriptor, value)
                    if original is not None:
                        retained, held = original, True
                    break
            if retained is None:
                retained = value.args
            if type(retained) is not tuple:
                raise ValueError("CSE requires immutable stored argument tuples")
            self.charge(len(retained), depth)
            self.active.add(identity)
            try:
                children = tuple(self.source(child, depth + 1) for child in retained)
            finally:
                self.active.remove(identity)
            result = self.intern(("node", id(cls), children),
                                 (cls, children, value, held))
        # Retain the owner while its process-local identity is in the memo.
        self.sources[identity] = (value, result)
        return result

    def make(self, cls, children, held=False):
        self.charge(len(children) + 1)
        children = tuple(children)
        return self.intern(("node", id(cls), children), (cls, children, None, held))

    def postorder(self, roots):
        ordered, seen = [], set()

        def visit(index, depth):
            self.charge(depth=depth)
            if index in seen:
                return
            seen.add(index)
            for child in self.nodes[index][1]:
                visit(child, depth + 1)
            ordered.append(index)

        for root in roots:
            visit(root, 0)
        return ordered

    def common_arguments(self, roots, blocked):
        if self.has_approximate:
            # Reassociation can change floating-point rounding. Exact repeated
            # subtrees may still be shared without factoring their operands.
            return roots
        for op in (Mul, Add):
            nodes = [index for index in self.postorder(roots)
                     if self.nodes[index][0] is op]
            counts = {index: Counter(self.nodes[index][1]) for index in nodes}
            groups = {}
            for position, left in enumerate(nodes):
                for right in nodes[position + 1:]:
                    self.charge(len(counts[left]) + len(counts[right]) + 1)
                    shared = counts[left] & counts[right]
                    size = sum(shared.values())
                    if (2 <= size < max(len(self.nodes[left][1]), len(self.nodes[right][1]))
                            and not any(blocked(child) for child in shared)):
                        # Integer references have a deterministic encounter order.
                        key = tuple(sorted(shared.elements()))
                        groups[key] = None
            for group in sorted(groups, key=lambda key: (-len(key), key)):
                # Grouping is structural: folding a held 0/x to zero would
                # erase the original undefined point even before substitution.
                common = self.make(op, group, held=True)
                required = Counter(group)
                memo = {}

                def rewrite(index, depth=0):
                    self.charge(depth=depth)
                    if index == common or index in memo:
                        return memo.get(index, index)
                    cls, children, _, held = self.nodes[index]
                    if not children:
                        return index
                    updated = tuple(rewrite(child, depth + 1) for child in children)
                    if cls is op:
                        available = Counter(updated)
                        self.charge(len(updated) + len(required))
                        if all(available[child] >= count for child, count in required.items()):
                            remaining = dict(required)
                            rest, inserted = [], False
                            for child in updated:
                                if remaining.get(child, 0):
                                    remaining[child] -= 1
                                    if not inserted:
                                        rest.append(common)
                                        inserted = True
                                else:
                                    rest.append(child)
                            updated = tuple(rest)
                            if len(updated) == 1:
                                memo[index] = updated[0]
                                return updated[0]
                    result = index if updated == children else self.make(cls, updated, held)
                    memo[index] = result
                    return result

                roots = [rewrite(root) for root in roots]
        return roots


def cse(exprs: Any, symbols: Any = None, optimizations: Any = None,
        postprocess: Any = None, order: str = "canonical", ignore: Any = (),
        list: bool = True) -> tuple:
    """Return topologically ordered definitions and reduced scalar expressions.

    Unsupported effectful/binding inputs are preserved without invoking their
    constructors or inspecting their public argument/hash/equality hooks. Held
    built-in arithmetic is retained when reconstruction is needed. Nonempty
    custom optimization pipelines require a separate supervised contract.
    """
    import builtins

    _require_original()
    if (optimizations is not None
            and not (type(optimizations) in (builtins.list, tuple) and not optimizations)):
        raise NotImplementedError("CSE custom optimization pipelines are not supported")
    if order not in ("canonical", "none"):
        raise ValueError("CSE order must be 'canonical' or 'none'")
    single = type(exprs) not in (builtins.list, tuple)
    if not single and len(exprs) > _MAX_INPUTS:
        raise ValueError("CSE exceeds its input limit")
    def prepare(value):
        if any(owner is Basic for owner in _MRO.__get__(type(value))):
            return value
        if type(value) in (int, float, bool, str):
            return sympify(value)
        raise TypeError("CSE inputs must be symbolic expressions or built-in scalar literals")

    items = [prepare(exprs)] if single else [prepare(value) for value in exprs]
    graph = _Graph()
    try:
        roots = [graph.source(value) for value in items]
    except _Unsupported:
        reduced = items
        replacements = []
    else:
        ignored = set()
        for value in ignore:
            graph.charge()
            if type(value) is not Symbol and type(value) is not Dummy:
                raise TypeError("CSE ignore entries must be built-in Symbols")
            try:
                ignored.add(graph.source(value))
            except _Unsupported:
                raise NotImplementedError("CSE ignore symbols require commuting scalar declarations") from None
        blocked_cache = {}

        def blocked(index):
            if index not in blocked_cache:
                graph.charge()
                blocked_cache[index] = (index in ignored or
                    any(blocked(child) for child in graph.nodes[index][1]))
            return blocked_cache[index]

        roots = graph.common_arguments(roots, blocked)
        seen, repeated = set(), set()

        def find(index, depth=0):
            graph.charge(depth=depth)
            children = graph.nodes[index][1]
            if not children:
                return
            if index in seen:
                if not blocked(index):
                    repeated.add(index)
                return
            seen.add(index)
            for child in children:
                find(child, depth + 1)

        for root in roots:
            find(root)
        ordered = [index for index in graph.postorder(roots) if index in repeated]
        stream = numbered_symbols() if symbols is None else iter(symbols)
        mapping = {}
        attempts = 0
        for index in ordered:
            while True:
                graph.charge()
                attempts += 1
                if attempts > _MAX_SYMBOL_ATTEMPTS:
                    raise ValueError("CSE exceeds its temporary-symbol attempt limit")
                try:
                    symbol = next(stream)
                except StopIteration:
                    raise ValueError("Symbols iterator ran out of symbols.") from None
                _require_original()
                if type(symbol) is not Symbol and type(symbol) is not Dummy:
                    raise TypeError("CSE temporaries must be built-in Symbols")
                facts = _slot(_ASSUMPTIONS, symbol)
                if (type(facts) is not dict or len(facts) > 64
                        or any(type(key) is not str or type(fact) is not bool
                               for key, fact in facts.items())):
                    raise ValueError("CSE temporaries require literal scalar assumptions")
                if facts.get("commutative") is False:
                    raise ValueError("CSE temporaries must be commutative")
                if any(key != "commutative" for key in facts):
                    raise ValueError("CSE temporaries require generic scalar assumptions")
                native = _slot(_VALUE, symbol)
                if type(native) is not _NATIVE_EXPR or not native.is_symbol:
                    raise ValueError("CSE temporary has no native Symbol identity")
                name = (_slot(_DUMMY_NAME, symbol) if type(symbol) is Dummy
                        else native.symbol_binding.name)
                if type(name) is not str:
                    raise ValueError("CSE temporaries require literal symbol names")
                if name not in graph.reserved_names:
                    graph.reserved_names.add(name)
                    mapping[index] = symbol
                    break
        _require_original()
        if any(_NAME.__get__(cls) != name or not _is_closed_function_class(cls)
               for cls, name in graph.functions.values()):
            raise NotImplementedError("CSE function implementation changed during optimization")
        memo = {}

        def emit(index, depth=0, definition=False):
            graph.charge(depth=depth)
            if not definition and index in mapping:
                return mapping[index]
            if index in memo:
                return memo[index]
            cls, children, original, held = graph.nodes[index]
            if not children:
                return original
            values = tuple(emit(child, depth + 1) for child in children)
            unchanged = (original is not None and
                         all(value is graph.nodes[child][2]
                             for child, value in zip(children, values)))
            if unchanged:
                result = original
            elif cls is Add or cls is Mul or cls is Pow:
                result = cls(*values, evaluate=not held)
            else:
                if not _is_closed_function_class(cls):
                    raise NotImplementedError("CSE function implementation changed during optimization")
                result = cls(*values, evaluate=False)
            memo[index] = result
            return result

        replacements = [(mapping[index], emit(index, definition=True)) for index in ordered]
        reduced = [emit(root) for root in roots]
    if postprocess is not None:
        return postprocess(replacements, reduced)
    if not list:
        reduced = reduced[0] if single else tuple(reduced) if type(exprs) is tuple else reduced
    return replacements, reduced


__all__ = ["cse", "numbered_symbols"]
