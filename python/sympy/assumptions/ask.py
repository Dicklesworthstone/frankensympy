"""Query and predicate registry for FrankenSymPy assumptions (WS04)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..core import Basic, Expr, Symbol, _native, _native_expr, _wrap
from .assume import AppliedPredicate, Predicate


class _PredicateRegistry:
    """Registry of mathematical predicates available under `Q`."""

    # Numeric tower
    complex = Predicate("complex")
    real = Predicate("real")
    rational = Predicate("rational")
    integer = Predicate("integer")
    algebraic = Predicate("algebraic")
    transcendental = Predicate("transcendental")

    # Sign bands
    positive = Predicate("positive")
    negative = Predicate("negative")
    nonnegative = Predicate("nonnegative")
    nonpositive = Predicate("nonpositive")
    zero = Predicate("zero")
    nonzero = Predicate("nonzero")

    # Number-theoretic
    prime = Predicate("prime")
    even = Predicate("even")
    odd = Predicate("odd")

    # Boundedness
    finite = Predicate("finite")
    infinite = Predicate("infinite")

    def __getattr__(self, name: str) -> Predicate:
        return Predicate(name)

    def __repr__(self) -> str:
        return "<PredicateRegistry Q>"


Q = _PredicateRegistry()


class AssumptionsContext(set):
    """A context of mathematical assumptions for symbols, backed by the native assumptions engine."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if _native is not None and hasattr(_native, "AssumptionsContext"):
            self._native_ctx = _native.AssumptionsContext()
        else:
            self._native_ctx = None
        for arg in args:
            if isinstance(arg, (list, tuple, set, frozenset)):
                for item in arg:
                    self.add(item)
            else:
                self.add(arg)

    def add(self, item: Any) -> None:
        super().add(item)
        if self._native_ctx is not None:
            for sym, pred in _extract_facts(item):
                self._native_ctx.assume(sym, pred)

    def update(self, *others: Any) -> None:
        for other in others:
            for item in other:
                self.add(item)

    def clear(self) -> None:
        super().clear()
        if _native is not None and hasattr(_native, "AssumptionsContext"):
            self._native_ctx = _native.AssumptionsContext()

    def assume(self, symbol: Symbol | str, predicate: Predicate | str) -> None:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        super().add(Predicate(pred_name)(Symbol(sym_name)))
        if self._native_ctx is not None:
            self._native_ctx.assume(sym_name, pred_name)

    def assume_domain(self, symbol: Symbol | str, domain: str) -> None:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        if self._native_ctx is not None:
            self._native_ctx.assume_domain(sym_name, domain)

    def deductions(self, symbol: Symbol | str) -> list[str]:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        if self._native_ctx is not None:
            return self._native_ctx.deductions(sym_name)
        return []

    def is_true(self, expr: Any, predicate: Predicate | str) -> bool | None:
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        native_e = _native_expr(expr)
        if self._native_ctx is not None:
            return self._native_ctx.is_true(native_e, pred_name)
        return None

    def query(self, expr: Any, predicate: Predicate | str) -> str:
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        native_e = _native_expr(expr)
        if self._native_ctx is not None:
            return self._native_ctx.query(native_e, pred_name)
        return "unknown"


global_assumptions = AssumptionsContext()


from contextlib import contextmanager


@contextmanager
def assuming(*assumptions: Any):
    """Context manager for temporarily adding assumptions to global_assumptions."""
    old_assumptions = list(global_assumptions)
    for a in assumptions:
        if isinstance(a, (list, tuple, set, frozenset)):
            for item in a:
                global_assumptions.add(item)
        else:
            global_assumptions.add(a)
    try:
        yield
    finally:
        global_assumptions.clear()
        for item in old_assumptions:
            global_assumptions.add(item)


def _extract_facts(assumptions: Any) -> list[tuple[str, str]]:
    """Extract (symbol_name, predicate_name) facts from assumptions argument."""
    if assumptions is None or assumptions is True or assumptions is False:
        return []
    if isinstance(assumptions, AppliedPredicate):
        sym_name = assumptions.expr.name if isinstance(assumptions.expr, Symbol) else str(assumptions.expr)
        return [(sym_name, assumptions.predicate.name)]
    if isinstance(assumptions, (list, tuple, set, frozenset)):
        facts = []
        for item in assumptions:
            facts.extend(_extract_facts(item))
        return facts
    # Check if it has .args (e.g. And(Q.positive(x), Q.integer(x)))
    if hasattr(assumptions, "args"):
        facts = []
        for arg in assumptions.args:
            facts.extend(_extract_facts(arg))
        return facts
    return []


_SYMBOL_FACTS = (
    "positive", "negative", "nonnegative", "nonpositive", "zero", "nonzero",
    "real", "integer", "rational", "even", "odd", "prime", "finite", "infinite",
    "complex", "imaginary", "irrational", "composite", "extended_real",
)


def _fact_pairs(assumptions: Any) -> list | None:
    """(expr, predicate name) pairs of a conjunction of applied predicates;
    None when the assumptions are not a plain conjunction."""
    from ..logic.boolalg import And

    if assumptions is None or assumptions is True:
        return []
    if isinstance(assumptions, AppliedPredicate):
        return [(assumptions.expr, assumptions.predicate.name)]
    if isinstance(assumptions, (list, tuple, set, frozenset, AssumptionsContext)):
        out: list = []
        for item in assumptions:
            sub = _fact_pairs(item)
            if sub is None:
                return None
            out.extend(sub)
        return out
    if isinstance(assumptions, And):
        out = []
        for item in assumptions.args:
            sub = _fact_pairs(item)
            if sub is None:
                return None
            out.extend(sub)
        return out
    return None


def _ask_via_properties(expr: Any, pred: str, pairs: list) -> bool | None:
    """Answer through the shell's exact assumption properties, with each
    symbol that carries given facts replaced by a fresh symbol declaring
    them together with its own assumptions."""
    from ..core import Dummy, sympify

    expr = sympify(expr)
    attr = "is_" + pred
    by_symbol: dict = {}
    for target, name in pairs:
        if isinstance(target, Symbol) and name in _SYMBOL_FACTS:
            by_symbol.setdefault(target, []).append(name)
    if by_symbol:
        mapping = {}
        for sym, names in by_symbol.items():
            declared = {k: v for k, v in getattr(sym, "_assumptions", {}).items()}
            for n in names:
                if declared.get(n) is False:
                    raise ValueError("inconsistent assumptions %s" % pairs)
                declared[n] = True
            try:
                # Dummy carries no assumptions in this profile; a private
                # symbol name keeps the replacement distinct from user atoms.
                mapping[sym] = Symbol("_fsym_ask_" + sym.name, **declared)
            except Exception:
                return None
        try:
            expr = expr.subs(mapping)
        except Exception:
            return None
    value = getattr(expr, attr, None)
    if value is None or callable(value):
        return None
    return bool(value) if isinstance(value, bool) else None


def ask(query: Any, assumptions: Any = None) -> bool | None:
    """Evaluate an assumption query in multi-valued logic.

    Parameters
    ----------
    query : AppliedPredicate
        The predicate query, e.g. ``Q.positive(x)`` or ``Q.real(5)``.
    assumptions : AppliedPredicate, sequence, or AssumptionsContext, optional
        The facts under which to evaluate the query.

    Returns
    -------
    bool or None
        ``True`` if entailed, ``False`` if refuted/contradicted, ``None`` if unknown.

    Examples
    --------
    >>> from sympy import Symbol, ask, Q
    >>> x = Symbol('x')
    >>> ask(Q.positive(x), Q.positive(x))
    True
    >>> ask(Q.real(x), Q.positive(x))
    True
    >>> ask(Q.negative(x), Q.positive(x))
    False
    >>> ask(Q.integer(x), Q.positive(x)) is None
    True
    >>> ask(Q.positive(5))
    True
    >>> ask(Q.even(4))
    True
    >>> ask(Q.odd(4))
    False
    """
    if isinstance(query, AppliedPredicate):
        expr = query.expr
        pred_name = query.predicate.name
    else:
        raise TypeError(f"query must be an AppliedPredicate (e.g. Q.positive(x)), got {type(query)}")

    if isinstance(assumptions, AssumptionsContext):
        return assumptions.is_true(expr, pred_name)

    pairs = _fact_pairs(assumptions)
    global_pairs = _fact_pairs(global_assumptions) or []
    if pairs is not None:
        # A fact about the queried expression itself decides directly.
        for target, name in pairs + global_pairs:
            if target == expr and name == pred_name:
                return True
        answer = _ask_via_properties(expr, pred_name, pairs + global_pairs)
        if answer is not None:
            return answer

    facts = _extract_facts(global_assumptions)
    if assumptions is not None and assumptions is not True:
        facts.extend(_extract_facts(assumptions))

    # Also include the symbol's own internal assumptions if any and no conflicting facts given
    if isinstance(expr, Symbol) and hasattr(expr, "_assumptions") and expr._assumptions:
        for k, v in expr._assumptions.items():
            if v is True and not any(f[0] == expr.name for f in facts):
                facts.append((expr.name, k))

    if _native is not None and hasattr(_native, "ask_expr"):
        native_e = _native_expr(expr)
        return _native.ask_expr(native_e, pred_name, facts if facts else None)

    return None
