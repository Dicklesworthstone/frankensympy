"""Boolean algebra and propositional logic expressions for FrankenSymPy."""

from __future__ import annotations

from typing import Any

from ..core import Expr, Symbol, _native


class Boolean(Expr):
    """A boolean expression."""

    __slots__ = ()

    def __and__(self, other: Any) -> "Boolean":
        return And(self, other)

    def __rand__(self, other: Any) -> "Boolean":
        return And(other, self)

    def __or__(self, other: Any) -> "Boolean":
        return Or(self, other)

    def __ror__(self, other: Any) -> "Boolean":
        return Or(other, self)

    def __invert__(self) -> "Boolean":
        return Not(self)

    def __rshift__(self, other: Any) -> "Boolean":
        return Implies(self, other)

    def __rrshift__(self, other: Any) -> "Boolean":
        return Implies(other, self)

    def __xor__(self, other: Any) -> "Boolean":
        return Xor(self, other)

    def __rxor__(self, other: Any) -> "Boolean":
        return Xor(other, self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return False
        if type(self) is not type(other):
            return False
        return self.args == getattr(other, "args", None)

    def __hash__(self) -> int:
        return hash((type(self), self.args))


class BooleanAtom(Boolean):
    """A boolean atom (true, false)."""

    __slots__ = ()


class BooleanTrue(BooleanAtom):
    """The singleton true."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if other is True:
            return True
        if other is False:
            return False
        from ..core import Number
        if isinstance(other, (Number, int, float)):
            return False
        return type(other) is type(self)

    def __hash__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "True"

    def __str__(self) -> str:
        return "True"


class BooleanFalse(BooleanAtom):
    """The singleton false."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        if other is False:
            return True
        if other is True:
            return False
        from ..core import Number
        if isinstance(other, (Number, int, float)):
            return False
        return type(other) is type(self)

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "False"

    def __str__(self) -> str:
        return "False"


true = BooleanTrue()
false = BooleanFalse()


# Upstream str precedence for boolean operators (printing/precedence.py):
# Xor/Implies/Equivalent 10, Or 20, And 30, Not 100; the infix joins use
# the Bitwise* levels so relational operands (35) are parenthesized.
_BOOL_PRECEDENCE = {
    "Xor": 10, "Implies": 10, "Equivalent": 10, "Or": 20, "And": 30, "Not": 100,
}
_BITWISE_OR, _BITWISE_XOR, _BITWISE_AND = 36, 37, 38


def _as_boolean(a: Any) -> Any:
    if isinstance(a, bool):
        return true if a else false
    return a


def _bool_precedence(a: Any) -> int:
    if isinstance(a, BooleanFunction):
        return _BOOL_PRECEDENCE.get(type(a).__name__, 1000)
    from ..printing.str import precedence

    return precedence(a)


def _bool_parenthesize(a: Any, level: int) -> str:
    if _bool_precedence(a) <= level:
        return f"({a})"
    return str(a)


def _bool_sort_key(a: Any) -> tuple:
    """Upstream ``default_sort_key`` for boolean trees (``Basic.sort_key``)."""
    from fractions import Fraction

    from ..core import Relational
    from ..printing.str import _ONE_KEY, sort_key

    if isinstance(a, (BooleanFunction, Relational, BooleanAtom)):
        args = tuple(a.args) if not isinstance(a, BooleanAtom) else ()
        return (
            (5, 0, type(a).__name__),
            (len(args), tuple(_bool_sort_key(x) for x in args)),
            _ONE_KEY,
            Fraction(1),
        )
    return sort_key(a)


def _node_count(a: Any) -> int:
    args = getattr(a, "args", ())
    if not isinstance(args, tuple):
        return 1
    return 1 + sum(_node_count(b) for b in args)


def _ordered(args: Any) -> list:
    """Upstream ``ordered``: node count first, then ``default_sort_key``."""
    return sorted(args, key=lambda a: (_node_count(a), _bool_sort_key(a)))


def _is_relational(a: Any) -> bool:
    from ..core import Relational

    return isinstance(a, Relational)


class BooleanFunction(Boolean):
    """A composite boolean function."""

    __slots__ = ("_args",)

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def args(self) -> tuple[Any, ...]:
        return getattr(self, "_args", ())

    @property
    def func(self) -> type:
        return type(self)

    @property
    def free_symbols(self) -> set:
        out: set = set()
        for a in self.args:
            out |= getattr(a, "free_symbols", set())
        return out

    def atoms(self, *types: Any) -> set:
        out: set = set()
        for a in self.args:
            if types and isinstance(a, types):
                out.add(a)
            if isinstance(a, BooleanFunction):
                out |= a.atoms(*types)
            elif hasattr(a, "atoms"):
                out |= a.atoms(*types) if types else a.atoms()
        return out

    def xreplace(self, rule: dict) -> Any:
        if self in rule:
            return rule[self]
        new = []
        for a in self.args:
            if a in rule:
                new.append(rule[a])
            elif isinstance(a, BooleanFunction):
                new.append(a.xreplace(rule))
            elif hasattr(a, "xreplace"):
                new.append(a.xreplace(rule))
            else:
                new.append(a)
        return type(self)(*new)

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1:
            pairs = args[0].items() if isinstance(args[0], dict) else args[0]
        else:
            pairs = [args]
        rule = {}
        for old, new in pairs:
            rule[_as_boolean(old)] = _as_boolean(new)
        scalar_rule = [(k, v) for k, v in rule.items() if not isinstance(v, Boolean)]
        out = []
        for a in self.args:
            if a in rule:
                out.append(rule[a])
            elif isinstance(a, BooleanFunction):
                out.append(a.subs(list(rule.items())))
            elif hasattr(a, "subs") and scalar_rule:
                out.append(a.subs(scalar_rule))
            else:
                out.append(a)
        return type(self)(*out)

    def to_nnf(self, simplify: bool = True) -> Any:
        return to_nnf(self, simplify)

    def simplify(self, **kwargs: Any) -> Any:
        return simplify_logic(self)

    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(str(a) for a in self.args)})"

    def __repr__(self) -> str:
        return self.__str__()


def _lattice_args(cls: type, args: Any, zero: Any, identity: Any) -> list | Any:
    flat: list[Any] = []
    for a in args:
        a = _as_boolean(a)
        if a is zero:
            return zero
        if a is identity:
            continue
        if isinstance(a, cls):
            flat.extend(a.args)
        elif a not in flat:
            flat.append(a)
    out: list[Any] = []
    for a in flat:
        if a not in out:
            out.append(a)
    return out


def _canonical_rel(r: Any) -> Any:
    return getattr(r, "canonical", r)


class And(BooleanFunction):
    """Logical AND."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        res = _lattice_args(cls, args, false, true)
        if res is false:
            return false
        rel: list = []
        for a in res:
            if _is_relational(a):
                c = _canonical_rel(a)
                if any(_canonical_rel(c.negated) == r for r in rel):
                    return false
                rel.append(c)
        if not res:
            return true
        if len(res) == 1:
            return res[0]
        obj = object.__new__(cls)
        obj._args = tuple(_ordered(res))
        return obj

    def __str__(self) -> str:
        return " & ".join(_bool_parenthesize(a, _BITWISE_AND) for a in self.args)

    __repr__ = __str__


class Or(BooleanFunction):
    """Logical OR."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        res = _lattice_args(cls, args, true, false)
        if res is true:
            return true
        rel: list = []
        for a in res:
            if _is_relational(a):
                c = _canonical_rel(a)
                if any(_canonical_rel(c.negated) == r for r in rel):
                    return true
                rel.append(c)
        if not res:
            return false
        if len(res) == 1:
            return res[0]
        obj = object.__new__(cls)
        obj._args = tuple(_ordered(res))
        return obj

    def __str__(self) -> str:
        return " | ".join(_bool_parenthesize(a, _BITWISE_OR) for a in self.args)

    __repr__ = __str__


class Not(BooleanFunction):
    """Logical NOT."""

    __slots__ = ()

    def __new__(cls, arg: Any, **kwargs: Any):
        arg = _as_boolean(arg)
        if arg is true:
            return false
        if arg is false:
            return true
        if isinstance(arg, Not):
            return arg.args[0]
        if _is_relational(arg):
            return arg.negated
        obj = object.__new__(cls)
        obj._args = (arg,)
        return obj

    def __str__(self) -> str:
        return "~" + _bool_parenthesize(self.args[0], 100)

    __repr__ = __str__


class Xor(BooleanFunction):
    """Logical XOR (upstream pairwise-cancelling argument set)."""

    __slots__ = ()

    def __new__(cls, *args: Any, remove_true: bool = True, **kwargs: Any):
        argset: list[Any] = []

        def toggle(a: Any) -> None:
            if a in argset:
                argset.remove(a)
            else:
                argset.append(a)

        for a in args:
            a = _as_boolean(a)
            if a is false:
                continue
            if isinstance(a, Xor):
                for b in a.args:
                    toggle(b)
            else:
                toggle(a)
        rels = [a for a in argset if _is_relational(a)]
        odd = False
        for i, r in enumerate(rels):
            for s in rels[i + 1:]:
                if r in argset and s in argset:
                    if _canonical_rel(r) == _canonical_rel(s):
                        argset.remove(r)
                        argset.remove(s)
                    elif _canonical_rel(r.negated) == _canonical_rel(s):
                        argset.remove(r)
                        argset.remove(s)
                        odd = not odd
        if odd:
            toggle(true)
        if not argset:
            return false
        if len(argset) == 1:
            return argset[0]
        if true in argset and remove_true:
            argset.remove(true)
            return Not(Xor(*argset))
        obj = object.__new__(cls)
        obj._args = tuple(_ordered(argset))
        return obj

    def __str__(self) -> str:
        return " ^ ".join(_bool_parenthesize(a, _BITWISE_XOR) for a in self.args)

    __repr__ = __str__


class Implies(BooleanFunction):
    """Logical Implication."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        if len(args) != 2:
            raise ValueError(
                "%d operand(s) used for an Implies (pairs are required): %s"
                % (len(args), str(args))
            )
        a, b = (_as_boolean(v) for v in args)
        if a in (true, false) or b in (true, false):
            return Or(Not(a), b)
        if a == b:
            return true
        if _is_relational(a) and _is_relational(b):
            if _canonical_rel(a) == _canonical_rel(b):
                return true
            if _canonical_rel(a.negated) == _canonical_rel(b):
                return b
        obj = object.__new__(cls)
        obj._args = (a, b)
        return obj


class Equivalent(BooleanFunction):
    """Logical Equivalence."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        argset: list[Any] = []
        for a in args:
            a = _as_boolean(a)
            if a not in argset:
                argset.append(a)
        rels = [a for a in argset if _is_relational(a)]
        for i, r in enumerate(rels):
            for s in rels[i + 1:]:
                if _canonical_rel(r.negated) == _canonical_rel(s):
                    return false
        if len(argset) <= 1:
            return true
        if true in argset:
            argset.remove(true)
            return And(*argset)
        if false in argset:
            argset.remove(false)
            return And(*[Not(a) for a in argset])
        obj = object.__new__(cls)
        obj._args = tuple(_ordered(argset))
        return obj


class Nand(BooleanFunction):
    """Logical NAND: upstream evaluates ``Nand(*a)`` to ``Not(And(*a))``."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        return Not(And(*args))


class Nor(BooleanFunction):
    """Logical NOR: upstream evaluates ``Nor(*a)`` to ``Not(Or(*a))``."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        return Not(Or(*args))


class Xnor(BooleanFunction):
    """Logical XNOR: upstream evaluates ``Xnor(*a)`` to ``Not(Xor(*a))``."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        return Not(Xor(*args))


class ITE(BooleanFunction):
    """If-Then-Else conditional operator on boolean expressions."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        if len(args) != 3:
            raise ValueError("expecting exactly 3 args")
        c, a, b = (_as_boolean(v) for v in args)
        if c is true:
            return a
        if c is false:
            return b
        if a == b:
            return a
        if a is true and b is false:
            return c
        if a is false and b is true:
            return Not(c)
        obj = object.__new__(cls)
        obj._args = (c, a, b)
        return obj


NAND = Nand
NOR = Nor
XNOR = Xnor


def _native_bool(expr: Any) -> _native.BoolExpr:
    """Converts a surface expression to a native BoolExpr."""
    if isinstance(expr, bool):
        return _native.BoolExpr.bool_const(expr)
    if isinstance(expr, BooleanTrue) or expr is true:
        return _native.BoolExpr.bool_const(True)
    if isinstance(expr, BooleanFalse) or expr is false:
        return _native.BoolExpr.bool_const(False)
    if isinstance(expr, Not):
        return _native.BoolExpr.bool_not(_native_bool(expr.args[0]))
    if isinstance(expr, And):
        return _native.BoolExpr.bool_and([_native_bool(a) for a in expr.args])
    if isinstance(expr, Or):
        return _native.BoolExpr.bool_or([_native_bool(a) for a in expr.args])
    if isinstance(expr, Implies):
        return _native.BoolExpr.bool_implies(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Equivalent):
        return _native.BoolExpr.bool_equivalent(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Xor):
        return _native.BoolExpr.bool_xor(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Nand):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_and([_native_bool(a) for a in expr.args]))
    if isinstance(expr, Nor):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_or([_native_bool(a) for a in expr.args]))
    if isinstance(expr, Xnor):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_xor(_native_bool(expr.args[0]), _native_bool(expr.args[1])))
    if isinstance(expr, ITE):
        c, a, b = [_native_bool(x) for x in expr.args]
        return _native.BoolExpr.bool_or([
            _native.BoolExpr.bool_and([c, a]),
            _native.BoolExpr.bool_and([_native.BoolExpr.bool_not(c), b]),
        ])
    if isinstance(expr, Symbol):
        return _native.BoolExpr.bool_var(expr.name)
    if isinstance(expr, str):
        return _native.BoolExpr.bool_var(expr)
    raise TypeError(f"cannot convert {type(expr).__name__} to BoolExpr")


def _from_native_bool(nb: _native.BoolExpr) -> Any:
    """Converts a native BoolExpr to a surface expression."""
    kind = nb.kind()
    if kind == "Const":
        return true if nb.const_value() else false
    if kind == "Var":
        return Symbol(nb.var_name())
    if kind == "Not":
        args = nb.args()
        return Not(_from_native_bool(args[0]))
    if kind == "And":
        args = nb.args()
        return And(*[_from_native_bool(a) for a in args])
    if kind == "Or":
        args = nb.args()
        return Or(*[_from_native_bool(a) for a in args])
    if kind == "Implies":
        args = nb.args()
        return Implies(_from_native_bool(args[0]), _from_native_bool(args[1]))
    if kind == "Equivalent":
        args = nb.args()
        return Equivalent(*[_from_native_bool(a) for a in args])
    raise ValueError(f"unknown BoolExpr kind: {kind}")


def is_literal(expr: Any) -> bool:
    """Upstream ``is_literal``: an atom, a non-boolean predicate, or its negation."""
    expr = _as_boolean(expr)
    if isinstance(expr, Not):
        return is_literal(expr.args[0])
    if isinstance(expr, BooleanFunction):
        return False
    return True


def _is_form(expr: Any, outer: type, inner: type) -> bool:
    vals = expr.args if isinstance(expr, outer) else [expr]
    for lit in vals:
        if isinstance(lit, inner):
            if not all(is_literal(a) for a in lit.args):
                return False
        elif not is_literal(lit):
            return False
    return True


def is_cnf(expr: Any) -> bool:
    return _is_form(_as_boolean(expr), And, Or)


def is_dnf(expr: Any) -> bool:
    return _is_form(_as_boolean(expr), Or, And)


def _distribute(e: Any, outer: type, inner: type) -> Any:
    """Upstream ``_distribute``: distribute ``outer`` over ``inner``."""
    if isinstance(e, inner):
        for conj in e.args:
            if isinstance(conj, outer):
                break
        else:
            return e
        rest = inner(*[a for a in e.args if a is not conj])
        return outer(*[_distribute(inner(c, rest), outer, inner) for c in conj.args])
    if isinstance(e, outer):
        return outer(*[_distribute(x, outer, inner) for x in e.args])
    return e


def distribute_and_over_or(expr: Any) -> Any:
    return _distribute(expr, And, Or)


def distribute_or_over_and(expr: Any) -> Any:
    return _distribute(expr, Or, And)


def eliminate_implications(expr: Any) -> Any:
    return to_nnf(expr, simplify=False)


def _find_predicates(expr: Any) -> set:
    if not isinstance(expr, BooleanFunction):
        return {expr}
    out: set = set()
    for a in expr.args:
        out |= _find_predicates(a)
    return out


def to_cnf(expr: Any, simplify: bool = False, force: bool = False) -> Any:
    """Converts a boolean expression to Conjunctive Normal Form (CNF)."""
    expr = _as_boolean(expr)
    if not isinstance(expr, BooleanFunction):
        return expr
    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(
                "To simplify a logical expression with more than 8 variables "
                "may take a long time and requires the use of `force=True`."
            )
        return simplify_logic(expr, "cnf", True, force=force)
    if is_cnf(expr):
        return expr
    return distribute_and_over_or(eliminate_implications(expr))


def to_dnf(expr: Any, simplify: bool = False, force: bool = False) -> Any:
    """Converts a boolean expression to Disjunctive Normal Form (DNF)."""
    expr = _as_boolean(expr)
    if not isinstance(expr, BooleanFunction):
        return expr
    if simplify:
        if not force and len(_find_predicates(expr)) > 8:
            raise ValueError(
                "To simplify a logical expression with more than 8 variables "
                "may take a long time and requires the use of `force=True`."
            )
        return simplify_logic(expr, "dnf", True, force=force)
    if is_dnf(expr):
        return expr
    return distribute_or_over_and(eliminate_implications(expr))


# --- Quine-McCluskey (port of upstream boolalg helpers) --------------------


def _check_pair(m1: list, m2: list) -> int:
    index = -1
    for k, v in enumerate(m1):
        if v != m2[k]:
            if index == -1:
                index = k
            else:
                return -1
    return index


def _simplified_pairs(terms: list) -> list:
    if not terms:
        return []
    simplified: list = []
    todo: list = list(range(len(terms)))
    by_ones: dict = {}
    for n, term in enumerate(terms):
        by_ones.setdefault(sum(1 for t in term if t == 1), []).append(n)
    for k in range(len(terms[0])):
        for i in by_ones.get(k, []):
            for j in by_ones.get(k + 1, []):
                index = _check_pair(terms[i], terms[j])
                if index != -1:
                    todo[i] = todo[j] = None
                    newterm = terms[i][:]
                    newterm[index] = 3
                    if newterm not in simplified:
                        simplified.append(newterm)
    if simplified:
        simplified = _simplified_pairs(simplified)
    simplified.extend([terms[i] for i in todo if i is not None])
    return simplified


def _rem_redundancy(l1: list, terms: list) -> list:
    if not terms:
        return []
    nterms, nl1 = len(terms), len(l1)
    dom = [[0] * nl1 for _ in range(nterms)]
    colcount = [0] * nl1
    rowcount = [0] * nterms
    for pi_, prime in enumerate(l1):
        for ti, term in enumerate(terms):
            if all(t == 3 or t == mt for t, mt in zip(prime, term)):
                dom[ti][pi_] = 1
                colcount[pi_] += 1
                rowcount[ti] += 1
    changed = True
    while changed:
        changed = False
        for ri in range(nterms):
            if rowcount[ri]:
                row = dom[ri]
                for r2 in range(nterms):
                    if ri != r2 and rowcount[ri] and rowcount[ri] <= rowcount[r2]:
                        row2 = dom[r2]
                        if all(row2[n] >= row[n] for n in range(nl1)):
                            rowcount[r2] = 0
                            changed = True
                            for pi_, prime in enumerate(row2):
                                if prime:
                                    dom[r2][pi_] = 0
                                    colcount[pi_] -= 1
        colcache: dict = {}
        for ci in range(nl1):
            if colcount[ci]:
                if ci not in colcache:
                    colcache[ci] = [dom[i][ci] for i in range(nterms)]
                col = colcache[ci]
                for c2 in range(nl1):
                    if ci != c2 and colcount[c2] and colcount[ci] >= colcount[c2]:
                        if c2 not in colcache:
                            colcache[c2] = [dom[i][c2] for i in range(nterms)]
                        col2 = colcache[c2]
                        if all(col[n] >= col2[n] for n in range(nterms)):
                            colcount[c2] = 0
                            changed = True
                            for ti, term in enumerate(col2):
                                if term and dom[ti][c2]:
                                    dom[ti][c2] = 0
                                    rowcount[ti] -= 1
        if not changed:
            maxterms, best = 0, -1
            for ci in range(nl1):
                if colcount[ci] > maxterms:
                    best, maxterms = ci, colcount[ci]
            if best != -1 and maxterms > 1:
                for pi_ in range(nl1):
                    if pi_ != best:
                        for ti, term in enumerate(colcache[best]):
                            if term and dom[ti][pi_]:
                                dom[ti][pi_] = 0
                                changed = True
                                rowcount[ti] -= 1
                                colcount[pi_] -= 1
    return [l1[i] for i in range(nl1) if colcount[i]]


def _ibin(n: int, bits: int) -> list:
    return [(n >> (bits - 1 - i)) & 1 for i in range(bits)]


def _input_to_binlist(inputlist: Any, variables: Any) -> list:
    import itertools

    out: list = []
    bits = len(variables)
    for val in inputlist:
        if isinstance(val, int):
            out.append(_ibin(val, bits))
        elif isinstance(val, dict):
            rest = [v for v in variables if v not in val]
            for t in itertools.product((0, 1), repeat=len(rest)):
                d = dict(zip(rest, t))
                d.update(val)
                out.append([d[v] for v in variables])
        elif isinstance(val, (list, tuple)):
            if len(val) != bits:
                raise ValueError(
                    "Each term must contain {bits} bits as there are"
                    "\n{bits} variables (or be an integer).".format(bits=bits)
                )
            out.append(list(val))
        else:
            raise TypeError("A term list can only contain lists, ints or dicts.")
    return out


def _sop_form(variables: Any, minterms: list, dontcares: list) -> Any:
    new = _simplified_pairs(minterms + dontcares)
    essential = _rem_redundancy(new, minterms)
    return Or(*[
        And(*[variables[n] if v == 1 else Not(variables[n]) for n, v in enumerate(t) if v != 3])
        for t in essential
    ])


def _pos_form(variables: Any, minterms: list, dontcares: list) -> Any:
    import itertools

    maxterms = [
        list(t) for t in itertools.product((0, 1), repeat=len(variables))
        if list(t) not in minterms and list(t) not in dontcares
    ]
    new = _simplified_pairs(maxterms + dontcares)
    essential = _rem_redundancy(new, maxterms)
    return And(*[
        Or(*[variables[n] if v == 0 else Not(variables[n]) for n, v in enumerate(t) if v != 3])
        for t in essential
    ])


def SOPform(variables: Any, minterms: Any, dontcares: Any = None) -> Any:
    """Smallest sum-of-products form (upstream Quine-McCluskey port)."""
    if not minterms:
        return false
    variables = tuple(variables)
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist(dontcares or [], variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError("%s in minterms is also in dontcares" % d)
    return _sop_form(variables, minterms, dontcares)


def POSform(variables: Any, minterms: Any, dontcares: Any = None) -> Any:
    """Smallest product-of-sums form (upstream Quine-McCluskey port)."""
    if not minterms:
        return false
    variables = tuple(variables)
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist(dontcares or [], variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError("%s in minterms is also in dontcares" % d)
    return _pos_form(variables, minterms, dontcares)


def _bool_value(expr: Any, model: dict) -> bool:
    expr = _as_boolean(expr)
    if expr is true:
        return True
    if expr is false:
        return False
    if isinstance(expr, Not):
        return not _bool_value(expr.args[0], model)
    if isinstance(expr, And):
        return all(_bool_value(a, model) for a in expr.args)
    if isinstance(expr, Or):
        return any(_bool_value(a, model) for a in expr.args)
    if isinstance(expr, Xor):
        return sum(_bool_value(a, model) for a in expr.args) % 2 == 1
    if isinstance(expr, Implies):
        return (not _bool_value(expr.args[0], model)) or _bool_value(expr.args[1], model)
    if isinstance(expr, Equivalent):
        vals = {_bool_value(a, model) for a in expr.args}
        return len(vals) == 1
    if isinstance(expr, ITE):
        c, a, b = expr.args
        return _bool_value(a, model) if _bool_value(c, model) else _bool_value(b, model)
    return model[expr]


def simplify_logic(
    expr: Any, form: str | None = None, deep: bool = True, force: bool = False,
    dontcare: Any = None,
) -> Any:
    """Simplest SOP/POS form via Quine-McCluskey (upstream algorithm)."""
    import itertools

    if form not in (None, "cnf", "dnf"):
        raise ValueError("form can be cnf or dnf only")
    expr = _as_boolean(expr)
    if form:
        form_ok = is_cnf(expr) if form == "cnf" else is_dnf(expr)
        if form_ok and all(is_literal(a) for a in getattr(expr, "args", ())):
            return expr
    if not isinstance(expr, BooleanFunction):
        return expr
    # Relationals and their negations share one variable.
    from ..core import Dummy

    repl: dict = {}
    undo: dict = {}
    rels = [r for r in _find_predicates(expr) if _is_relational(r)]
    if dontcare is not None:
        dontcare = _as_boolean(dontcare)
        rels += [r for r in _find_predicates(dontcare) if _is_relational(r) and r not in rels]
    while rels:
        r = rels.pop()
        if r in repl:
            continue
        d = Dummy()
        undo[d] = r
        repl[r] = d
        if r.negated in rels:
            repl[r.negated] = Not(d)
            rels.remove(r.negated)
    if repl:
        expr = expr.xreplace(repl)
        if dontcare is not None and isinstance(dontcare, BooleanFunction):
            dontcare = dontcare.xreplace(repl)
    variables = _find_predicates(expr)
    if not force and len(variables) > 8:
        return expr.xreplace(undo) if undo else expr
    if dontcare is not None:
        variables |= _find_predicates(dontcare)
        if not force and len(variables) > 8:
            variables = _find_predicates(expr)
            dontcare = None
    v = [a for a in _ordered(variables) if a not in (true, false)]

    def table(e: Any) -> list:
        rows = []
        for bits in itertools.product((0, 1), repeat=len(v)):
            if _bool_value(e, dict(zip(v, (bool(b) for b in bits)))):
                rows.append(list(bits))
        return rows

    truthtable = table(expr)
    dctable = table(dontcare) if dontcare is not None else []
    truthtable = [t for t in truthtable if t not in dctable]
    big = len(truthtable) >= 2 ** (len(v) - 1)
    if form == "dnf" or (form is None and big):
        res = _sop_form(v, truthtable, dctable)
    else:
        res = POSform(v, truthtable, dctable)
    if undo and isinstance(res, BooleanFunction):
        res = res.xreplace(undo)
    elif res in undo:
        res = undo[res]
    return res


def is_nnf(expr: Any, simplified: bool = True) -> bool:
    """Return True if expr is in Negation Normal Form, False otherwise."""
    expr = _as_boolean(expr)
    if is_literal(expr):
        return True
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, (And, Or)):
            if simplified:
                for a in e.args:
                    if Not(a) in e.args:
                        return False
            stack.extend(e.args)
        elif not is_literal(e):
            return False
    return True


def _nnf_lattice(cls: type, args: Any, simplify: bool) -> Any:
    argset: list = []
    for a in args:
        if not is_literal(a):
            a = to_nnf(a, simplify)
        if simplify:
            for b in (a.args if isinstance(a, cls) else (a,)):
                if Not(b) in argset:
                    return false if cls is And else true
                if b not in argset:
                    argset.append(b)
        else:
            argset.append(a)
    return cls(*argset)


def to_nnf(expr: Any, simplify: bool = True) -> Any:
    """Converts a boolean expression to Negation Normal Form (NNF)."""
    expr = _as_boolean(expr)
    if is_nnf(expr, simplify):
        return expr
    if isinstance(expr, And):
        return _nnf_lattice(And, expr.args, simplify)
    if isinstance(expr, Or):
        return _nnf_lattice(Or, expr.args, simplify)
    if isinstance(expr, Implies):
        a, b = expr.args
        return _nnf_lattice(Or, (Not(a), b), simplify)
    if isinstance(expr, Equivalent):
        args = expr.args
        clauses = [Or(Not(a), b) for a, b in zip(args, args[1:])]
        clauses.append(Or(Not(args[-1]), args[0]))
        return _nnf_lattice(And, clauses, simplify)
    if isinstance(expr, Xor):
        import itertools

        args = expr.args
        clauses = []
        for neg in range(0, len(args) + 1, 2):
            for S in itertools.combinations(args, neg):
                clauses.append(Or(*[Not(s) if s in S else s for s in args]))
        return _nnf_lattice(And, clauses, simplify)
    if isinstance(expr, ITE):
        a, b, c = expr.args
        return _nnf_lattice(And, (Or(Not(a), b), Or(a, c)), simplify)
    if isinstance(expr, Not):
        arg = expr.args[0]
        if isinstance(arg, Not):
            return to_nnf(arg.args[0], simplify)
        if isinstance(arg, And):
            return _nnf_lattice(Or, [Not(a) for a in arg.args], simplify)
        if isinstance(arg, Or):
            return _nnf_lattice(And, [Not(a) for a in arg.args], simplify)
        if isinstance(arg, Implies):
            a, b = arg.args
            return _nnf_lattice(And, (a, Not(b)), simplify)
        if isinstance(arg, Equivalent):
            return _nnf_lattice(And, (Or(*arg.args), Or(*[Not(a) for a in arg.args])), simplify)
        if isinstance(arg, Xor):
            import itertools

            args = arg.args
            clauses = []
            for neg in range(1, len(args) + 1, 2):
                for S in itertools.combinations(args, neg):
                    clauses.append(Or(*[Not(s) if s in S else s for s in args]))
            return _nnf_lattice(And, clauses, simplify)
        if isinstance(arg, ITE):
            a, b, c = arg.args
            return _nnf_lattice(And, (Or(a, Not(c)), Or(Not(a), Not(b))), simplify)
    return expr


def truth_table(expr: Any, variables: list[Any]) -> Any:
    """Generate a truth table for the boolean expression.

    Yields:
        (combination, result): list of 0/1 for each variable, and bool result.
    """
    import itertools
    from .inference import pl_true

    variables = list(variables)
    n = len(variables)
    for combination in itertools.product([0, 1], repeat=n):
        model = {var: bool(val) for var, val in zip(variables, combination)}
        res = bool(pl_true(expr, model))
        yield list(combination), res


__all__ = [
    "And",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "Equivalent",
    "ITE",
    "Implies",
    "NAND",
    "NOR",
    "Nand",
    "Nor",
    "Not",
    "Or",
    "POSform",
    "SOPform",
    "XNOR",
    "Xnor",
    "Xor",
    "false",
    "is_nnf",
    "simplify_logic",
    "to_cnf",
    "to_dnf",
    "to_nnf",
    "true",
    "truth_table",
]
