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


class AssumptionsContext:
    """A context of mathematical assumptions for symbols."""

    __slots__ = ("_native_ctx",)

    def __init__(self) -> None:
        if _native is not None and hasattr(_native, "AssumptionsContext"):
            self._native_ctx = _native.AssumptionsContext()
        else:
            raise NotImplementedError("fsym_python AssumptionsContext is required")

    def assume(self, symbol: Symbol | str, predicate: Predicate | str) -> None:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        self._native_ctx.assume(sym_name, pred_name)

    def assume_domain(self, symbol: Symbol | str, domain: str) -> None:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        self._native_ctx.assume_domain(sym_name, domain)

    def deductions(self, symbol: Symbol | str) -> list[str]:
        sym_name = symbol.name if isinstance(symbol, Symbol) else str(symbol)
        return self._native_ctx.deductions(sym_name)

    def is_true(self, expr: Any, predicate: Predicate | str) -> bool | None:
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        native_e = _native_expr(expr)
        return self._native_ctx.is_true(native_e, pred_name)

    def query(self, expr: Any, predicate: Predicate | str) -> str:
        pred_name = predicate.name if isinstance(predicate, Predicate) else str(predicate)
        native_e = _native_expr(expr)
        return self._native_ctx.query(native_e, pred_name)


def _extract_facts(assumptions: Any) -> list[tuple[str, str]]:
    """Extract (symbol_name, predicate_name) facts from assumptions argument."""
    if assumptions is None:
        return []
    if isinstance(assumptions, AppliedPredicate):
        sym_name = str(assumptions.expr)
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

    facts = _extract_facts(assumptions)

    # Also include the symbol's own internal assumptions if any and no conflicting facts given
    if isinstance(expr, Symbol) and hasattr(expr, "_assumptions") and expr._assumptions:
        for k, v in expr._assumptions.items():
            if v is True and not any(f[0] == expr.name for f in facts):
                facts.append((expr.name, k))

    if _native is not None and hasattr(_native, "ask_expr"):
        native_e = _native_expr(expr)
        return _native.ask_expr(native_e, pred_name, facts if facts else None)

    return None
