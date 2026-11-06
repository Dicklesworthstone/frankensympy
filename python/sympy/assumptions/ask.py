"""Query and predicate registry for FrankenSymPy assumptions (WS04)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ..core import Dummy, Symbol, sympify
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
    """Mutable shell facts with fresh native snapshots for each query.

    The visible set is authoritative. Even an explicit ``set.remove(ctx, fact)``
    cannot leave a second native fact store stale. Normal insertions validate a
    candidate before publication; domains are a separate, typed declaration
    list and ``clear()`` clears both.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._domains = ()
        proposed = set()
        domains = []
        for arg in args:
            if isinstance(arg, (list, tuple, set, frozenset)):
                proposed.update(arg)
                if isinstance(arg, AssumptionsContext):
                    domains.extend(arg._domains)
            else:
                proposed.add(arg)
        self._publish(proposed, tuple(domains))

    @staticmethod
    def _validate(items, domains):
        from ._query import prepare_problem

        # Unsupported formulas may be stored but cannot grant an answer. An
        # unsupported conjunct cannot hide contradictory supported conjuncts
        # or an invalid typed domain during an otherwise atomic insertion.
        prepare_problem(items, domains, allow_unsupported=True)

    def _publish(self, items, domains):
        self._validate(items, domains)
        set.clear(self)
        set.update(self, items)
        self._domains = domains

    def add(self, item: Any) -> None:
        proposed = set(self)
        proposed.add(item)
        self._publish(proposed, self._domains)

    def update(self, *others: Any) -> None:
        proposed = set(self)
        for other in others:
            proposed.update(other)
        self._publish(proposed, self._domains)

    def symmetric_difference_update(self, other: Any) -> None:
        proposed = set(self)
        proposed.symmetric_difference_update(other)
        self._publish(proposed, self._domains)

    def __ior__(self, other):
        self.update(other)
        return self

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self

    def clear(self) -> None:
        set.clear(self)
        self._domains = ()

    def copy(self):
        return AssumptionsContext(self)

    def assume(self, symbol: Symbol | str, predicate: Predicate | str) -> None:
        symbol = _context_symbol(symbol)
        self.add(Predicate(_predicate_name(predicate))(symbol))

    def assume_domain(self, symbol: Symbol | str, domain: str) -> None:
        symbol = _context_symbol(symbol)
        if type(domain) is not str:
            raise TypeError("assumptions domain must be a string")
        if len(self._domains) >= 128:
            raise ValueError("assumptions context exceeds its domain declaration limit")
        self._publish(set(self), self._domains + ((symbol, domain),))

    def deductions(self, symbol: Symbol | str) -> list[str]:
        from ._query import NATIVE_PREDICATES, UnsupportedAssumptions, prepare_problem

        symbol = _context_symbol(symbol)
        try:
            problem, branches = prepare_problem(tuple(self), self._domains)
            return sorted(
                name for name in NATIVE_PREDICATES
                if problem.answer(symbol, name, branches) is True
            )
        except UnsupportedAssumptions:
            return []

    def is_true(self, expr: Any, predicate: Predicate | str) -> bool | None:
        return ask(Predicate(_predicate_name(predicate))(sympify(expr)), context=self)

    def query(self, expr: Any, predicate: Predicate | str) -> str:
        result = self.is_true(expr, predicate)
        return "Unknown" if result is None else "True" if result else "False"


def _context_symbol(symbol):
    if type(symbol) is str:
        return Symbol(symbol)
    if type(symbol) is not Symbol and type(symbol) is not Dummy:
        raise TypeError("assumptions require an exact Symbol, Dummy, or symbol name")
    return symbol


def _predicate_name(predicate):
    if type(predicate) is str:
        return predicate
    if type(predicate) is Predicate:
        return Predicate.__dict__["_name"].__get__(predicate)
    raise TypeError("assumptions require a Predicate or predicate name")


global_assumptions = AssumptionsContext()


@contextmanager
def assuming(*assumptions: Any):
    """Context manager for temporarily adding assumptions to global_assumptions."""
    old_assumptions = set(global_assumptions)
    old_domains = global_assumptions._domains
    try:
        # Entry is inside the restoration boundary too: a bad second fact
        # cannot leak the first fact when __enter__ raises.
        additions = []
        for assumption in assumptions:
            if isinstance(assumption, (list, tuple, set, frozenset)):
                additions.extend(assumption)
            else:
                additions.append(assumption)
        global_assumptions.update(additions)
        yield
    finally:
        set.clear(global_assumptions)
        set.update(global_assumptions, old_assumptions)
        global_assumptions._domains = old_domains


def ask(query: Any, assumptions: Any = None, context: Any = None) -> bool | None:
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
    if type(query) is not AppliedPredicate:
        raise TypeError(f"query must be an AppliedPredicate (e.g. Q.positive(x)), got {type(query)}")
    from ._query import UnsupportedAssumptions, prepare_problem

    if isinstance(assumptions, AssumptionsContext):
        if context is not None:
            raise TypeError("specify an assumptions context only once")
        context, assumptions = assumptions, None
    if context is None:
        context = global_assumptions
    if not isinstance(context, (set, frozenset)):
        raise TypeError("context must be an assumptions set")
    domains = context._domains if isinstance(context, AssumptionsContext) else ()
    try:
        problem, branches = prepare_problem((*context, assumptions), domains)
        return problem.answer(query.expr, _predicate_name(query.predicate), branches)
    except UnsupportedAssumptions:
        return None
