"""Pattern matching, structural replacement and traversal (upstream
``Basic.match`` / ``replace`` / ``find`` / ``count``, ``Wild``,
``preorder_traversal``, ``postorder_traversal``).

``Wild`` symbols carry an interned native name (as ``Dummy`` does) so they
survive native arithmetic and lift back to the same ``Wild`` object.
Matching is structural with commutative backtracking for ``Add``/``Mul``:
wild-free pattern terms are taken out of the expression first (``x + a``
against ``x`` gives ``a = 0``), then each remaining pattern term claims
expression terms, and a bare ``Wild`` absorbs whatever is left (the
operation's identity when nothing is). Every returned binding is checked
by substituting it back into the pattern, so a match is never reported
for a pattern that does not reproduce the expression.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from . import (
    Add,
    Basic,
    Function,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    _native,
    _wrap,
    sympify,
)

_WILD_PREFIX = "__fsymWild_"
_wild_by_number: dict[int, "Wild"] = {}
_wild_by_key: dict[tuple, "Wild"] = {}


class Wild(Symbol):
    """A pattern symbol: ``Wild('a', exclude=[x], properties=[pred])``."""

    __slots__ = ("_wild_name", "_wild_number", "exclude", "properties")

    is_Wild = True

    def __new__(cls, name: str, exclude: Any = (), properties: Any = (), **assumptions: Any):
        exclude = tuple(sympify(e) for e in exclude)
        properties = tuple(properties)
        key = (cls, name, exclude, properties)
        try:
            cached = _wild_by_key.get(key)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        number = len(_wild_by_number) + 1
        obj._wild_name = name
        obj._wild_number = number
        obj.exclude = exclude
        obj.properties = properties
        obj._assumptions = {}
        obj._value = _native.py_symbol(f"{_WILD_PREFIX}{number}_{name}")
        _wild_by_number[number] = obj
        try:
            _wild_by_key[key] = obj
        except TypeError:
            pass
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return self._wild_name

    def __str__(self) -> str:
        return self._wild_name + "_"

    __repr__ = __str__

    def _srepr(self) -> str:
        return f"Wild({self._wild_name!r})"

    def matches(self, expr: Any, repl_dict: Any = None) -> Any:
        return _match_wild(self, sympify(expr), dict(repl_dict or {}))


def _wild_from_native(name: str) -> Wild | None:
    if not name.startswith(_WILD_PREFIX):
        return None
    number = name[len(_WILD_PREFIX):].split("_", 1)[0]
    try:
        return _wild_by_number.get(int(number))
    except ValueError:
        return None


class WildFunction(Function):
    """Matches any applied function (``WildFunction('F')``)."""

    _wild_function_registry: dict = {}

    def __new__(cls, name: str, nargs: Any = None, **kwargs: Any):
        key = (name, nargs)
        reg = WildFunction._wild_function_registry
        if key not in reg:
            obj = Wild("_F_" + name)
            obj._wild_function_nargs = nargs  # type: ignore[attr-defined]
            reg[key] = obj
        return reg[key]


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def _args(e: Any) -> tuple:
    return tuple(getattr(e, "args", ()) or ())


def preorder_traversal(node: Any, keys: Any = None) -> Iterator[Any]:
    """Depth-first, node before its arguments (upstream order)."""
    node = sympify(node) if not isinstance(node, (tuple, list, set)) else node
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        args = list(_args(cur)) if isinstance(cur, Basic) else (list(cur) if isinstance(cur, (tuple, list)) else [])
        if keys:
            from ..printing.str import sort_key as _sk

            args.sort(key=_sk)
        stack.extend(reversed(args))


def postorder_traversal(node: Any, keys: Any = None) -> Iterator[Any]:
    """Depth-first, arguments before the node."""
    node = sympify(node)
    args = list(_args(node))
    if keys:
        from ..printing.str import sort_key as _sk

        args.sort(key=_sk)
    for a in args:
        if isinstance(a, Basic):
            yield from postorder_traversal(a, keys)
        else:
            yield a
    yield node


def bottom_up(rv: Any, F: Callable, atoms: bool = False, nonbasic: bool = False) -> Any:
    """Apply ``F`` to every node from the leaves up."""
    args = _args(rv)
    if args:
        new = tuple(bottom_up(a, F, atoms, nonbasic) for a in args)
        if new != args:
            try:
                rv = rv.func(*new)
            except Exception:
                pass
        rv = F(rv)
    elif atoms:
        rv = F(rv)
    return rv


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _wilds_in(e: Any) -> set:
    out = set()
    for n in preorder_traversal(e):
        if isinstance(n, Wild):
            out.add(n)
    return out


def _has_wild(e: Any) -> bool:
    return any(isinstance(s, Wild) for s in getattr(e, "free_symbols", ()) or ())


def _contains(expr: Any, sub: Any) -> bool:
    if expr == sub:
        return True
    if isinstance(sub, Symbol) and sub in getattr(expr, "free_symbols", ()):
        return True
    return any(_contains(a, sub) for a in _args(expr) if isinstance(a, Basic))


def _match_wild(w: Wild, expr: Any, d: dict) -> dict | None:
    if w in d:
        return d if d[w] == expr else None
    if any(_contains(expr, x) for x in w.exclude):
        return None
    for prop in w.properties:
        try:
            if not prop(expr):
                return None
        except Exception:
            return None
    nd = dict(d)
    nd[w] = expr
    return nd


def _size(e: Any) -> int:
    return sum(1 for _ in preorder_traversal(e))


def _match(pat: Any, expr: Any, d: dict | None) -> dict | None:
    if d is None:
        return None
    if isinstance(pat, Wild):
        nargs = getattr(pat, "_wild_function_nargs", "unset")
        if nargs != "unset":
            if not isinstance(expr, Function):
                return None
        return _match_wild(pat, expr, d)
    if not _has_wild(pat):
        return d if pat == expr else None
    if d:
        bound = pat.xreplace({k: v for k, v in d.items()})
        if not _has_wild(bound):
            return d if bound == expr else None
    if isinstance(pat, Add):
        return _match_assoc(Add, pat, expr, d)
    if isinstance(pat, Mul):
        return _match_assoc(Mul, pat, expr, d)
    if isinstance(pat, Pow):
        b, e = pat.args
        if isinstance(expr, Pow):
            r = _match(e, expr.args[1], _match(b, expr.args[0], d))
            if r is not None:
                return r
        if type(expr).__name__ == "exp":
            from . import E

            r = _match(e, expr.args[0], _match(b, E, d))
            if r is not None:
                return r
        return _match(e, Integer(1), _match(b, expr, d))
    pargs, eargs = _args(pat), _args(expr)
    if type(pat) is not type(expr) or len(pargs) != len(eargs):
        return None
    for p, e in zip(pargs, eargs):
        d = _match(p, e, d)
        if d is None:
            return None
    return d


def _match_assoc(op: Any, pat: Any, expr: Any, d: dict) -> dict | None:
    identity = Integer(0) if op is Add else Integer(1)
    pargs = list(pat.args)
    exact = [p for p in pargs if not _has_wild(p)]
    wild_args = [p for p in pargs if _has_wild(p)]
    rem = expr
    if exact:
        e_part = op(*exact)
        rem = expr - e_part if op is Add else expr / e_part
    if rem == identity:
        terms: list = []
    elif isinstance(rem, op):
        terms = list(rem.args)
    else:
        terms = [rem]
    terms.sort(key=_size, reverse=True)
    return _distribute(op, wild_args, terms, d, identity)


def _distribute(op: Any, pats: list, terms: list, d: dict | None, identity: Any) -> dict | None:
    if d is None:
        return None
    if not pats:
        return d if not terms else None
    complex_pats = [p for p in pats if not isinstance(p, Wild)]
    bare = [p for p in pats if isinstance(p, Wild)]
    if complex_pats:
        p, rest = complex_pats[0], complex_pats[1:] + bare
        for i, t in enumerate(terms):
            r = _match(p, t, d)
            if r is not None:
                res = _distribute(op, rest, terms[:i] + terms[i + 1:], r, identity)
                if res is not None:
                    return res
        r = _match(p, identity, d)
        if r is not None:
            return _distribute(op, rest, terms, r, identity)
        return None
    w, rest = bare[0], bare[1:]
    if not rest:
        value = op(*terms) if terms else identity
        return _match(w, value, d)
    for i, t in enumerate(terms):
        res = _distribute(op, rest, terms[:i] + terms[i + 1:], _match(w, t, d), identity)
        if res is not None:
            return res
    return _distribute(op, rest, terms, _match(w, identity, d), identity)


def match(expr: Any, pattern: Any, old: bool = False) -> dict | None:
    """``expr.match(pattern)``: a dict binding every Wild of ``pattern``
    so that it reproduces ``expr``, or None."""
    expr = sympify(expr)
    pattern = sympify(pattern)
    if pattern == expr:
        return {}
    d = _match(pattern, expr, {})
    if d is None:
        return None
    # Wilds the structure never reached (x.match(x + a)) take the identity
    # they matched; verify the binding exactly.
    try:
        if pattern.xreplace(d) != expr:
            from ..simplify import simplify

            if simplify(pattern.xreplace(d) - expr) != 0:
                return None
    except Exception:
        pass
    return d


# ---------------------------------------------------------------------------
# replace / find / count
# ---------------------------------------------------------------------------


def _query_fn(query: Any) -> Callable[[Any], Any]:
    """A function returning a truthy match result for a node."""
    if isinstance(query, type):
        return lambda e: isinstance(e, query) and {}
    if callable(query) and not isinstance(query, Basic):
        return lambda e: {} if query(e) else None
    query = sympify(query)
    if _has_wild(query):
        return lambda e: match(e, query)
    return lambda e: {} if e == query else None


def replace(expr: Any, query: Any, value: Any, map: bool = False, simultaneous: bool = True, exact: Any = None) -> Any:
    """Upstream ``replace``: type -> type/callable, pattern -> expression or
    callable of the bindings, predicate -> callable; applied bottom-up."""
    found: dict = {}
    is_type = isinstance(query, type)

    if is_type and isinstance(value, type):
        build = lambda node, m: value(*node.args)  # noqa: E731
    elif is_type:
        build = lambda node, m: value(*node.args)  # noqa: E731
    elif callable(query) and not isinstance(query, Basic):
        build = lambda node, m: value(node)  # noqa: E731
    else:
        q = sympify(query)
        if _has_wild(q):
            if callable(value) and not isinstance(value, Basic):
                build = lambda node, m: value(**{str(k.name): v for k, v in m.items()})  # noqa: E731
            else:
                build = lambda node, m: sympify(value).xreplace(m)  # noqa: E731
        else:
            build = lambda node, m: sympify(value)  # noqa: E731
    test = _query_fn(query)

    def rec(node: Any) -> Any:
        args = _args(node)
        if args and isinstance(node, Basic):
            new_args = tuple(rec(a) if isinstance(a, Basic) else a for a in args)
            if new_args != args:
                try:
                    node = node.func(*new_args)
                except Exception:
                    pass
        m = test(node)
        if m is not None and m is not False:
            new = build(node, m)
            if new != node:
                found[node] = new
            return new
        return node

    out = rec(sympify(expr))
    return (out, found) if map else out


def find(expr: Any, query: Any, group: bool = False) -> Any:
    test = _query_fn(query)
    hits = [n for n in preorder_traversal(expr) if isinstance(n, Basic) and test(n) not in (None, False)]
    if group:
        counts: dict = {}
        for h in hits:
            counts[h] = counts.get(h, 0) + 1
        return counts
    return set(hits)


def count(expr: Any, query: Any) -> int:
    test = _query_fn(query)
    return sum(1 for n in preorder_traversal(expr) if isinstance(n, Basic) and test(n) not in (None, False))


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------


def as_independent(expr: Any, *deps: Any, as_Add: Any = None, strict: bool = True) -> tuple:
    expr = sympify(expr)
    want_add = isinstance(expr, Add) if as_Add is None else as_Add
    op = Add if want_add else Mul
    identity = Integer(0) if want_add else Integer(1)
    if expr == 0:
        return Integer(0), Integer(0)
    args = expr.args if isinstance(expr, op) else (expr,)

    def depends(t: Any) -> bool:
        return any(_contains(t, dep) for dep in deps)

    indep = [t for t in args if not depends(t)]
    dep = [t for t in args if depends(t)]
    if not isinstance(expr, op):
        return (expr, identity) if not dep else (identity, expr)
    return op(*indep) if indep else identity, op(*dep) if dep else identity


def as_coefficients_dict(expr: Any, *syms: Any) -> Any:
    from collections import defaultdict

    expr = sympify(expr)
    out: Any = defaultdict(int)
    for t in (expr.args if isinstance(expr, Add) else (expr,)):
        c, rest = t.as_coeff_Mul()
        out[rest] = out[rest] + c
    return out


def as_coeff_mul(expr: Any, *deps: Any) -> tuple:
    expr = sympify(expr)
    factors = list(expr.args) if isinstance(expr, Mul) else [expr]
    if factors and isinstance(factors[0], Rational):
        return factors[0], tuple(factors[1:])
    return Integer(1), tuple(factors)


def as_coeff_add(expr: Any, *deps: Any) -> tuple:
    expr = sympify(expr)
    terms = list(expr.args) if isinstance(expr, Add) else [expr]
    if terms and isinstance(terms[0], Rational):
        return terms[0], tuple(terms[1:])
    return Integer(0), tuple(terms)


def is_polynomial(expr: Any, *syms: Any) -> bool:
    expr = sympify(expr)
    syms = set(syms) if syms else set(expr.free_symbols)

    def rec(e: Any) -> bool:
        if not (getattr(e, "free_symbols", set()) & syms):
            return True
        if isinstance(e, Symbol):
            return True
        if isinstance(e, (Add, Mul)):
            return all(rec(a) for a in e.args)
        if isinstance(e, Pow):
            b, k = e.args
            if getattr(k, "free_symbols", set()) & syms:
                return False
            return isinstance(k, Integer) and k >= 0 and rec(b)
        return False

    return rec(expr)


def is_rational_function(expr: Any, *syms: Any) -> bool:
    expr = sympify(expr)
    syms = set(syms) if syms else set(expr.free_symbols)

    def rec(e: Any) -> bool:
        if not (getattr(e, "free_symbols", set()) & syms):
            return True
        if isinstance(e, Symbol):
            return True
        if isinstance(e, (Add, Mul)):
            return all(rec(a) for a in e.args)
        if isinstance(e, Pow):
            b, k = e.args
            if getattr(k, "free_symbols", set()) & syms:
                return False
            return isinstance(k, Integer) and rec(b)
        return False

    return rec(expr)


__all__ = [
    "Wild",
    "WildFunction",
    "bottom_up",
    "postorder_traversal",
    "preorder_traversal",
]
