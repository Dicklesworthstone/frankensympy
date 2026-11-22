"""Piecewise-defined expressions (upstream sympy.functions.elementary.piecewise).

Piecewise is a native-lowered function application over
``Tuple(expr, cond)`` branches; ExprCondPair is the structural view of one
branch. Conditions that evaluate to ``false`` are dropped and a leading
``true`` condition returns its expression.
"""

from __future__ import annotations

from typing import Any

from ...core import Expr, Function


def _as_boolean(c: Any) -> Any:
    from ...core import sympify
    from ...logic.boolalg import false, true

    if c is True:
        return true
    if c is False:
        return false
    return sympify(c)


class ExprCondPair(Expr):
    """One ``(expr, cond)`` branch of a Piecewise."""

    __slots__ = ()

    def __new__(cls, expr: Any, cond: Any):
        from ...core import sympify

        obj = object.__new__(cls)
        obj._struct_args = (sympify(expr), _as_boolean(cond))
        return obj

    @property
    def expr(self):
        return self._struct_args[0]

    @property
    def cond(self):
        return self._struct_args[1]

    def __iter__(self):
        return iter(self._struct_args)

    def __reduce__(self):
        from ...logic.boolalg import BooleanFalse, BooleanTrue

        expression, condition = self._struct_args
        # Exact Boolean atoms reconstruct through their zero-argument
        # singleton constructors. A plain Symbol("True") stays a Symbol.
        if type(condition) is BooleanTrue or type(condition) is BooleanFalse:
            condition = type(condition) is BooleanTrue
        return type(self), (expression, condition)

    def __str__(self) -> str:
        return f"({self.expr}, {self.cond})"

    __repr__ = __str__


def _cond_native(c: Any) -> Any:
    """Lower closed condition syntax, or retain its original Python surface.

    Ordinary Symbol/Dummy conditions are valid symbolic Piecewise inputs, but
    are not the BooleanTrue/BooleanFalse atoms in the native condition protocol.
    A structural guard preserves their typed identity through native arithmetic;
    custom Boolean classes retain their Python surface instead.
    """
    from ...core import Dummy, Symbol, _native, _native_expr
    from ...logic.boolalg import (
        And, BooleanFalse, BooleanFunction, BooleanTrue, Equivalent, Implies, Not, Or, Xor,
    )

    if type(c) is BooleanTrue or c is True:
        return _native.Expr("True")
    if type(c) is BooleanFalse or c is False:
        return _native.Expr("False")
    if type(c) is Symbol or type(c) is Dummy:
        # This is an internal structural marker, never a recovered Python
        # function class. In particular Symbol("True") stays a symbolic guard.
        return _native.py_function("_FsymBooleanSymbol", _native_expr(c))
    logical = ((And, "And"), (Or, "Or"), (Not, "Not"), (Xor, "Xor"),
               (Implies, "Implies"), (Equivalent, "Equivalent"))
    name = next((name for cls, name in logical if type(c) is cls), None)
    if name is not None:
        children = tuple(_cond_native(a) for a in BooleanFunction.__dict__["_args"].__get__(c, type(c)))
        if any(child is None for child in children):
            return None
        return _native.py_function(name, *children)
    if any(type(c) is cls for cls in _relationals().values()):
        try:
            return _native_expr(c)
        except NotImplementedError:
            return None
    return None


def _cond_lift(v: Any) -> Any:
    from ...core import _native, _wrap
    from ...logic import boolalg

    name = v.func_name
    if name == "_FsymBooleanSymbol":
        children = v.args
        if len(children) != 1:
            raise ValueError("invalid symbolic Piecewise condition")
        if children[0].is_symbol:
            return _wrap(children[0])
        # Native substitution may replace a symbolic guard by a relational
        # or Boolean tree. Recover that typed condition, without interpreting
        # an original Symbol("True") as a Boolean literal.
        if (children[0].func_name in _relationals()
                or children[0].func_name in ("And", "Or", "Not", "Xor", "Implies", "Equivalent")):
            return _cond_lift(children[0])
        raise ValueError("invalid substituted Piecewise condition")
    # Typed same-name symbols are not the legacy plain Boolean protocol atoms.
    if v == _native.Expr("True"):
        return boolalg.true
    if v == _native.Expr("False"):
        return boolalg.false
    if name in ("And", "Or", "Not", "Xor", "Implies", "Equivalent"):
        return getattr(boolalg, name)(*[_cond_lift(a) for a in v.args])
    rel = _RELATIONALS.get(name)
    if rel is not None and len(v.args) == 2:
        # Through the constructor, so substituted numbers decide it.
        return rel(_wrap(v.args[0]), _wrap(v.args[1]))
    return _wrap(v)


def _relationals() -> dict:
    from ...core import Eq, Ge, Gt, Le, Lt, Ne

    return {
        "Equality": Eq, "Unequality": Ne, "StrictLessThan": Lt,
        "LessThan": Le, "StrictGreaterThan": Gt, "GreaterThan": Ge,
    }


class _LazyRelationals(dict):
    def get(self, key, default=None):  # type: ignore[override]
        if not self:
            self.update(_relationals())
        return super().get(key, default)


_RELATIONALS = _LazyRelationals()


def _canonical_branches(pairs: tuple, evaluate: bool) -> list:
    from ...logic.boolalg import BooleanFalse, BooleanTrue

    branches = []
    for p in pairs:
        expr, cond = (p.expr, p.cond) if isinstance(p, ExprCondPair) else tuple(p)
        pair = ExprCondPair(expr, cond)
        if evaluate and isinstance(pair.cond, BooleanFalse):
            continue
        if evaluate and any(pair.cond == b.cond and pair.expr == b.expr for b in branches):
            continue
        branches.append(pair)
        if evaluate and isinstance(pair.cond, BooleanTrue):
            break
    return branches


class Piecewise(Function):
    """``Piecewise((expr1, cond1), (expr2, cond2), ...)``.

    Lowered natively as ``Piecewise(Tuple(e1, c1), ...)`` so it takes part
    in ordinary arithmetic, substitution and differentiation; branches
    whose condition is false drop and a leading true condition returns its
    expression, both at construction and whenever a native result is
    lifted back (e.g. after ``subs``).
    """

    __slots__ = ("_piecewise_branches",)

    def __new__(cls, *pairs: Any, evaluate: bool = True):
        branches = _canonical_branches(pairs, evaluate)
        if not branches:
            from ...core import nan

            return nan
        from ...logic.boolalg import BooleanTrue

        if evaluate and isinstance(branches[0].cond, BooleanTrue):
            return branches[0].expr
        return cls._build(branches)

    @classmethod
    def _build(cls, branches: list) -> "Piecewise":
        from ...core import _native, _native_expr
        from ...core._opaque_arithmetic import contains_opaque

        conditions = tuple(_cond_native(branch.cond) for branch in branches)
        obj = object.__new__(cls)
        # Retain opaque branch expressions as well as symbolic conditions.
        # Neither a custom Function class nor its Python-owned children can
        # be recovered from a native name. Closed numerical paths still
        # refuse this surface; the supervised hybrid path captures it directly.
        if (any(condition is None for condition in conditions)
                or any(contains_opaque(branch.expr) for branch in branches)):
            obj._value = None
            Function.__dict__["_args"].__set__(obj, tuple(branches))
        else:
            native = [_native.py_function("Tuple", _native_expr(branch.expr), condition)
                      for branch, condition in zip(branches, conditions)]
            obj._value = _native.py_function("Piecewise", *native)
        # Numerical admission reads this slot without reconstructing conditions
        # or calling their public properties. Preserve held relationals and the
        # exact typed Symbol/Dummy objects chosen by the caller.
        Piecewise.__dict__["_piecewise_branches"].__set__(
            obj, tuple((branch.expr, branch.cond) for branch in branches)
        )
        return obj

    @classmethod
    def _from_native(cls, value: Any) -> Any:
        """Lift a native ``Piecewise`` and re-evaluate its branches."""
        from ...core import _wrap

        pairs = []
        for t in value.args:
            if t.func_name != "Tuple" or len(t.args) != 2:
                obj = object.__new__(cls)
                obj._value = value
                return obj
            pairs.append((_wrap(t.args[0]), _cond_lift(t.args[1])))
        return cls(*pairs)

    @property
    def args(self) -> tuple:
        from ...core import _wrap

        try:
            pairs = Piecewise.__dict__["_piecewise_branches"].__get__(self, Piecewise)
        except AttributeError:
            pairs = None
        if pairs is not None:
            return tuple(ExprCondPair(expr, cond) for expr, cond in pairs)
        out = []
        for t in self._value.args:
            out.append(ExprCondPair(_wrap(t.args[0]), _cond_lift(t.args[1])))
        return tuple(out)

    @property
    def free_symbols(self) -> set:
        out: set = set()
        for b in self.args:
            out |= set(getattr(b.expr, "free_symbols", set()))
            out |= set(getattr(b.cond, "free_symbols", set()))
        return out

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        return Piecewise(
            *[(b.expr.subs(*args, **kwargs), b.cond.subs(*args, **kwargs)
               if hasattr(b.cond, "subs") else b.cond) for b in self.args]
        )

    def diff(self, *symbols: Any) -> Any:
        return Piecewise(*[(b.expr.diff(*symbols), b.cond) for b in self.args])

    def doit(self, **hints: Any) -> Any:
        return Piecewise(*[(b.expr.doit(**hints) if hasattr(b.expr, "doit") else b.expr, b.cond)
                           for b in self.args])


_MAX_FOLD_NODES = 16_384
_MAX_FOLD_DEPTH = 128
_MAX_FOLD_BRANCHES = 256


def _fold_owner_records():
    from ...core import _exact_surface_types
    from ...core._function_capabilities import _owner_record
    from ...logic.boolalg import BooleanFalse, BooleanTrue, _BOOLEAN_FUNCTIONS

    mro = type.__dict__["__mro__"]
    owners = []
    for cls in (Piecewise, ExprCondPair, BooleanTrue, BooleanFalse,
                *_BOOLEAN_FUNCTIONS, *_exact_surface_types(), *_relationals().values()):
        for owner in (*mro.__get__(cls), *mro.__get__(type(cls))):
            if not any(owner is row[0] for row in owners):
                owners.append((owner, mro.__get__(owner), _owner_record(owner)))
    return tuple(owners)


def _fold_method_records():
    from ...core._function_capabilities import _implementation

    # The shared capability roster covers core identity/constructor hooks.
    # Folding additionally consults these decomposition and ordering methods.
    names = ("_build", "_from_native", "canonical", "negated", "reversed",
             "lhs", "rhs", "free_symbols", "sort_key", "__str__", "__repr__",
             "as_coeff_Add", "as_coeff_Mul")
    namespace = type.__dict__["__dict__"]
    return tuple((owner, name, _implementation(namespace.__get__(owner)[name]))
                 for owner, _, _ in _FOLD_OWNERS
                 for name in names if name in namespace.__get__(owner))


# Capture before callers can replace an exact builtin's args, func, equality,
# constructor or relational implementation and acquire a closed folding lane.
_FOLD_OWNERS = _fold_owner_records()
_FOLD_METHODS = _fold_method_records()


def _require_fold_owners():
    from ...core._function_capabilities import _owner_matches

    mro = type.__dict__["__mro__"]
    for owner, original_mro, record in _FOLD_OWNERS:
        current = mro.__get__(owner)
        if (len(current) != len(original_mro)
                or any(a is not b for a, b in zip(current, original_mro))
                or not _owner_matches(owner, record)):
            raise NotImplementedError("piecewise_fold requires original built-in owners")
    current = _fold_method_records()
    if len(current) != len(_FOLD_METHODS) or any(
        owner is not original_owner or name != original_name or len(tokens) != len(original)
        or any(a is not b for a, b in zip(tokens, original))
        for (owner, name, tokens), (original_owner, original_name, original)
        in zip(current, _FOLD_METHODS)
    ):
        raise NotImplementedError("piecewise_fold requires original built-in owners")


class _PiecewiseFolder:
    """One bounded surface transformation; no result is published mid-fold."""

    def __init__(self, effects):
        from ...core import _active_callback_session

        self.effects = effects
        self.session = _active_callback_session()
        self.remaining = _MAX_FOLD_NODES
        self.records = {}
        self.active = set()

    def charge(self, amount=1, depth=0):
        from ...core import _charge_nested_if_active

        if depth > _MAX_FOLD_DEPTH or amount > self.remaining:
            raise ValueError("piecewise_fold exceeds its work/depth limit")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def product_size(self, lengths):
        size = 1
        for length in lengths:
            if length and size > _MAX_FOLD_BRANCHES // length:
                raise ValueError("piecewise_fold exceeds its branch limit")
            size *= length
        return size

    @staticmethod
    def raw_children(node):
        """Count retained callback input/output without calling user properties."""
        from ...core import (
            Basic, _FLOAT_INTERN, _is_closed_function_class, _lookup_known_function,
            _native, _wrap,
        )
        from ...core._opaque_arithmetic import _subclass, stored_args
        from ...logic.boolalg import BooleanAtom, BooleanFunction

        cls = type(node)
        if cls is Piecewise:
            pairs = Piecewise.__dict__["_piecewise_branches"].__get__(node, cls)
            if (type(pairs) is not tuple or len(pairs) > _MAX_FOLD_BRANCHES
                    or any(type(pair) is not tuple or len(pair) != 2 for pair in pairs)):
                raise ValueError("piecewise_fold requires bounded stored branch pairs")
            return tuple(child for pair in pairs for child in pair)
        if _subclass(cls, BooleanAtom):
            return ()
        if _subclass(cls, BooleanFunction):
            return BooleanFunction.__dict__["_args"].__get__(node, cls)
        if _subclass(cls, Basic):
            retained = stored_args(node)
            if retained is not None:
                return retained
            native = Basic.__dict__["_value"].__get__(node, cls)
            # The arithmetic transform admission deliberately excludes the
            # Piecewise Tuple/Boolean protocol. Admit that existing protocol
            # here, while retaining its refusal of native-only user classes.
            _native.CompiledFunction._validate_source(native)
            pending = [native]
            structural = {
                "Add", "Mul", "Pow", "Constant", "Piecewise", "Tuple",
                "_FsymBooleanSymbol", "And", "Or", "Not", "Xor", "Implies",
                "Equivalent", "Derivative", "diff", _FLOAT_INTERN,
                *_relationals(), "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
            }
            while pending:
                current = pending.pop()
                if current.is_symbol or current.is_integer or current.is_rational:
                    continue
                name = current.func_name
                if name not in structural:
                    owner = _lookup_known_function(name)
                    if owner is None or not _is_closed_function_class(owner):
                        raise NotImplementedError(
                            "piecewise_fold cannot recover a native-only Python function class")
                pending.extend(current.args)
            return tuple(_wrap(child) for child in native.args)
        return node if type(node) is tuple or type(node) is list else ()

    @classmethod
    def node_count(cls, value):
        pending = [(value, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > _MAX_FOLD_NODES or depth > _MAX_FOLD_DEPTH:
                raise ValueError("piecewise_fold callback graph exceeds its limit")
            children = cls.raw_children(node)
            if type(children) is not tuple and type(children) is not list:
                raise TypeError("piecewise_fold requires finite stored arguments")
            if len(children) > _MAX_FOLD_NODES - count - len(pending):
                raise ValueError("piecewise_fold callback graph exceeds its limit")
            pending.extend((child, depth + 1) for child in children)
        return count

    def effect(self, kind, callback, args):
        from ...core import supervised_callback_session

        if self.session is None:
            self.session = self.effects.enter_context(supervised_callback_session())

        def selected(*values):
            result = callback(*values)
            # A callback can mutate a builtin. Refuse inside its owning
            # invocation before another closed operation or result publication.
            _require_fold_owners()
            return result

        return self.session.invoke_fresh(
            "piecewise_fold." + kind, "piecewise-fold", self.node_count(args),
            selected, args, self.node_count,
        )

    def capture(self, node, depth=0):
        """Snapshot children and detect cycles before any branch reconstruction."""
        from ...core import Basic, _exact_surface_types, _is_closed_function_class
        from ...core._opaque_arithmetic import _subclass
        from ...logic.boolalg import BooleanFalse, BooleanTrue, _BOOLEAN_FUNCTIONS

        self.charge(depth=depth)
        identity = id(node)
        if identity in self.active:
            raise ValueError("piecewise_fold encountered a cyclic symbolic graph")
        if identity in self.records:
            return self.records[identity]
        if not _subclass(type(node), Basic):
            return (node, (), False, False)
        closed = (any(type(node) is cls for cls in (
            Piecewise, ExprCondPair, BooleanTrue, BooleanFalse,
            *_BOOLEAN_FUNCTIONS, *_exact_surface_types(),
        )) or _is_closed_function_class(type(node)))
        children = (self.raw_children(node) if closed
                    else self.effect("args", lambda value: value.args, (node,)))
        if type(children) is not tuple:
            raise TypeError("piecewise_fold requires an exact tuple of symbolic arguments")
        if len(children) > self.remaining:
            raise ValueError("piecewise_fold exceeds its argument limit")
        self.active.add(identity)
        records = tuple(self.capture(child, depth + 1) for child in children)
        self.active.remove(identity)
        result = (node, children, type(node) is Piecewise or any(r[2] for r in records),
                  not closed or any(r[3] for r in records))
        self.records[identity] = result
        return result

    def call(self, kind, callback, args, *, unknown=False):
        self.charge()
        if unknown or any(self.capture(arg)[3] for arg in args):
            result = self.effect(kind, callback, args)
        else:
            result = callback(*args)
        self.capture(result)
        return result

    def equal(self, left, right):
        if left is right:
            return True
        return self.call("equal", lambda a, b: bool(a == b), (left, right))

    def pairs(self, node):
        children = self.capture(node)[1]
        return tuple(zip(children[::2], children[1::2]))

    def groups(self, node, children):
        """Align identical ordered guard sequences before taking a product."""
        from ...core import Add, Mul

        if type(node) is not Add and not (
            type(node) is Mul and self.call(
                "commutative", lambda value: value.is_commutative is True, (node,))
        ):
            return tuple((child,) for child in children)
        ordinary, grouped = [], []
        for child in children:
            if type(child) is not Piecewise:
                ordinary.append((child,))
                continue
            conditions = tuple(condition for _, condition in self.pairs(child))
            for previous, members in grouped:
                if len(previous) == len(conditions) and all(
                    self.equal(a, b) for a, b in zip(previous, conditions)
                ):
                    members.append(child)
                    break
            else:
                grouped.append((conditions, [child]))
        # Piecewise operands follow ordinary operands, as in the profile's
        # Add/Mul grouping. Sorting uses typed conditions, never printed names.
        from ...logic.boolalg import _bool_sort_key

        grouped = self.call(
            "guard_order", lambda: sorted(grouped, key=lambda item: (
                sum(self.node_count(c) for c in item[0]),
                tuple(_bool_sort_key(c) for c in item[0]),
            )), (), unknown=any(self.capture(c)[3] for row, _ in grouped for c in row),
        )
        return tuple(ordinary) + tuple(tuple(members) for _, members in grouped)

    def plan(self, node, depth=0):
        """Conservative branch counts, before constructors can run user code."""
        self.charge(depth=depth)
        record = self.capture(node)
        if not record[2]:
            return 1
        if type(node) is Piecewise:
            sizes = []
            for value, condition in self.pairs(node):
                # Guards can themselves contain arithmetic Piecewise products
                # and Boolean ITE expansion. Plan them before any branch eval.
                self.plan(condition, depth + 1)
                self.nnf_size(condition, depth + 1)
                sizes.append(self.plan(value, depth + 1))
            size = sum(sizes)
        else:
            sizes = []
            for group in self.groups(node, record[1]):
                if len(group) == 1:
                    sizes.append(self.plan(group[0], depth + 1))
                else:
                    aligned = zip(*(self.pairs(item) for item in group))
                    sizes.append(sum(self.product_size(
                        self.plan(value, depth + 1) for value, _ in row
                    ) for row in aligned))
            size = self.product_size(sizes)
        if size > _MAX_FOLD_BRANCHES:
            raise ValueError("piecewise_fold exceeds its branch limit")
        return size

    def rebuild(self, node, values):
        return self.call("rebuild", lambda source, *args: source.func(*args),
                         (node, *values))

    def conjunction(self, *conditions):
        from ...logic.boolalg import And

        return self.call("and", And, conditions)

    def canonical_condition(self, condition):
        """The profile's rational coefficient normalization of relationals."""
        from ...core import Relational
        from ...core._opaque_arithmetic import _subclass
        from ...logic.boolalg import Boolean, BooleanFunction

        if _subclass(type(condition), Relational):
            def canonical(rel):
                rel = rel.canonical
                if isinstance(rel.rhs, Boolean):
                    return rel
                offset, term = rel.lhs.as_coeff_Add(rational=True)
                coefficient, lhs = term.as_coeff_Mul(rational=True)
                if coefficient == 0:
                    raise NotImplementedError(
                        "piecewise_fold cannot normalize a held constant relational")
                rhs = (rel.rhs - offset) / coefficient
                constructor = rel.reversed.func if coefficient < 0 else rel.func
                return constructor(lhs, rhs)

            return self.call("relational", canonical, (condition,))
        if _subclass(type(condition), BooleanFunction):
            children = self.capture(condition)[1]
            mapped = tuple(self.canonical_condition(child) for child in children)
            if any(a is not b for a, b in zip(children, mapped)):
                return self.rebuild(condition, mapped)
        return condition

    def cnf(self, condition, depth=0):
        """Bound distribution before creating the clauses of a merged guard."""
        from itertools import product
        from ...logic.boolalg import And, Or

        self.charge(depth=depth)
        if type(condition) is And:
            return self.conjunction(*(self.cnf(c, depth + 1) for c in self.capture(condition)[1]))
        if type(condition) is not Or:
            return condition
        children = tuple(self.cnf(c, depth + 1) for c in self.capture(condition)[1])
        clauses = tuple(self.capture(c)[1] if type(c) is And else (c,) for c in children)
        size = self.product_size(map(len, clauses))
        self.charge(size)
        return self.conjunction(*(self.call("or", Or, row) for row in product(*clauses)))

    def nnf_size(self, condition, depth=0):
        """Upper bound for ITE/Boolean expansion, including duplicated inputs."""
        from ...logic.boolalg import Equivalent, ITE, Xor

        self.charge(depth=depth)
        children = self.capture(condition)[1]
        costs = [self.nnf_size(child, depth + 1) for child in children]
        size = 1 + sum(costs)
        if type(condition) is ITE:
            size = 6 + 2 * costs[0] + costs[1] + costs[2]
        elif type(condition) is Piecewise and costs:
            size = costs[-2]
            for offset in range(len(costs) - 4, -1, -2):
                size = 6 + 2 * costs[offset + 1] + costs[offset] + size
        elif type(condition) is Equivalent:
            size = 1 + 2 * sum(costs) + 3 * len(costs)
        elif type(condition) is Xor:
            if len(costs) > 8:
                raise ValueError("piecewise_fold exceeds its Boolean expansion limit")
            size = 1 + 2 ** len(costs) * (1 + sum(costs) + len(costs))
        if size > _MAX_FOLD_NODES:
            raise ValueError("piecewise_fold exceeds its Boolean expansion limit")
        return size

    def condition(self, condition, depth):
        from ...core import Relational, Symbol
        from ...logic.boolalg import Boolean, ITE, Or, true, simplify_logic, to_nnf

        if self.capture(condition)[2]:
            condition = self.fold(condition, True, depth + 1)
            if type(condition) is Piecewise:
                pairs = self.pairs(condition)
                cover = self.call("or", Or, tuple(c for _, c in pairs))
                if cover is not true:
                    raise ValueError("Piecewise conditions must cover all cases")
                if not all(isinstance(value, (Boolean, Relational, Symbol)) for value, _ in pairs):
                    raise TypeError("Piecewise condition branches must be Boolean")
                result = pairs[-1][0]
                for value, guard in reversed(pairs[:-1]):
                    result = self.call("ite", ITE, (guard, value, result))
                condition = result
        if type(condition) is ITE:
            self.charge(self.nnf_size(condition))
            condition = self.call("nnf", to_nnf, (condition,))
            condition = self.call(
                "logic", lambda value: simplify_logic(value, form="cnf"), (condition,))
        return condition

    def finish(self, pairs, evaluate, depth=0):
        from ...core import Ge, Gt, Le, Lt, Relational, nan
        from ...logic.boolalg import And, Or, false, true

        self.charge(depth=depth)
        if len(pairs) > _MAX_FOLD_BRANCHES:
            raise ValueError("piecewise_fold exceeds its branch limit")
        # The constructor's unconditional false/true handling applies even
        # to raw folds. Full normalization remains local to this operation.
        normalized, seen = [], []
        for value, condition in pairs:
            self.charge()
            if condition is false:
                continue
            if evaluate:
                condition = self.canonical_condition(condition)
                factors = self.capture(condition)[1] if type(condition) is And else ()
                if any(self.equal(part, old) for part in (condition, *factors) for old in seen):
                    continue
                if type(condition) is And:
                    kept = []
                    for factor in factors:
                        if isinstance(factor, Relational):
                            opposite = self.call("complement", lambda c: c.negated.canonical, (factor,))
                            if any(self.equal(opposite, old) for old in seen):
                                continue
                            if type(factor) is Lt or type(factor) is Gt:
                                weak = self.call("weak", Le if type(factor) is Lt else Ge,
                                                 self.capture(factor)[1])
                                if any(self.equal(weak, old) for old in seen):
                                    kept = [false]
                                    break
                        kept.append(factor)
                    condition = self.conjunction(*kept)
                elif isinstance(condition, Relational):
                    opposite = self.call("complement", lambda c: c.negated.canonical, (condition,))
                    if any(self.equal(opposite, old) for old in seen):
                        condition = true
                if condition is false:
                    continue
            if evaluate is None or evaluate:
                if any(self.equal(condition, old) for old in seen):
                    continue
                seen.append(condition)
            if evaluate and normalized and self.equal(normalized[-1][0], value):
                merged = self.cnf(self.call("or", Or, (normalized[-1][1], condition)))
                normalized[-1] = (value, merged)
            else:
                normalized.append((value, condition))
            if condition is true or normalized[-1][1] is true:
                break
        if not normalized:
            return nan if evaluate else self.call("piecewise", Piecewise._build, ((),))
        if len(normalized) == 1 and normalized[0][1] is true:
            return normalized[0][0]
        if evaluate and (len(normalized) != len(pairs) or any(
            not self.equal(old_value, value) or not self.equal(old_condition, condition)
            for (old_value, old_condition), (value, condition) in zip(pairs, normalized)
        )):
            # Merging values may create a guard which shadows a later branch.
            # Match the constructor's repeated normalization, under the same
            # work/depth budget, before publishing the final branch graph.
            return self.finish(normalized, True, depth + 1)
        # Avoid the general constructor's different normalization policy.
        branches = tuple(self.call("branch", ExprCondPair, (value, condition))
                         for value, condition in normalized)
        return self.call("piecewise", lambda *args: Piecewise._build(args), branches)

    def fold(self, node, evaluate=True, depth=0):
        from itertools import product
        from ...logic.boolalg import true

        self.charge(depth=depth)
        record = self.capture(node)
        if not record[2]:
            return node
        output = []
        if type(node) is Piecewise:
            pairs = self.pairs(node)
            if not pairs:
                raise TypeError("At least one (expr, cond) pair expected.")
            for value, condition in pairs:
                condition = self.condition(condition, depth)
                folded = value if type(value) is Piecewise else self.fold(value, True, depth + 1)
                if type(folded) is Piecewise:
                    inner = self.pairs(folded)
                    if len(output) + len(inner) > _MAX_FOLD_BRANCHES:
                        raise ValueError("piecewise_fold exceeds its branch limit")
                    for branch, guard in inner:
                        output.append((self.fold(branch, True, depth + 1),
                                       self.conjunction(guard, condition)))
                else:
                    output.append((folded, condition))
        else:
            folded = []
            for group in self.groups(node, record[1]):
                if len(group) == 1:
                    value = group[0]
                else:
                    rows = zip(*(self.pairs(item) for item in group))
                    value = self.finish([
                        (self.rebuild(node, tuple(value for value, _ in row)), row[0][1])
                        for row in rows
                    ], False)
                folded.append(self.fold(value, True, depth + 1))
            alternatives = tuple(self.pairs(value) if type(value) is Piecewise
                                 else ((value, true),) for value in folded)
            self.charge(self.product_size(map(len, alternatives)))
            for row in product(*alternatives):
                output.append((self.rebuild(node, tuple(value for value, _ in row)),
                               self.conjunction(*(condition for _, condition in row))))
        result = self.finish(output, evaluate)
        if type(result) is Piecewise and any(self.capture(value)[2] for value, _ in self.pairs(result)):
            return self.fold(result, True, depth + 1)
        return result


def piecewise_fold(expr: Any, evaluate: bool | None = True) -> Any:
    """Lift nested Piecewise arithmetic, functions and predicates into branches.

    ``evaluate=False`` retains raw branch conditions; ``evaluate=None`` only
    removes duplicate/false conditions and branches after an unconditional one.
    The default also normalizes ordered relational guards and merges adjacent
    equal values. Expressions without Piecewise are returned unchanged.

    Source depth, total work and Cartesian branch growth are bounded. A refused
    or failed fold raises without publishing a partially transformed result.
    """
    from contextlib import ExitStack
    from ...core import Basic
    from ...core._opaque_arithmetic import _subclass

    if type(evaluate) is not bool and evaluate is not None:
        raise TypeError("piecewise_fold evaluate must be True, False, or None")
    if not _subclass(type(expr), Basic):
        return expr
    with ExitStack() as effects:
        _require_fold_owners()
        folder = _PiecewiseFolder(effects)
        folder.capture(expr)
        folder.plan(expr)
        return folder.fold(expr, evaluate)


__all__ = ["ExprCondPair", "Piecewise", "piecewise_fold"]
