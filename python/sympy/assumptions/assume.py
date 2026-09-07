"""Symbolic assumptions predicates and applied predicates for FrankenSymPy."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _wrap


class Predicate(Basic):
    """A mathematical predicate in the assumptions lattice."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, expr: Any) -> AppliedPredicate:
        return AppliedPredicate(self, expr)

    def closure(self) -> list[str]:
        """Return all predicate consequences entailed by this predicate."""
        if _native is not None and hasattr(_native, "predicate_closure"):
            return _native.predicate_closure(self._name)
        return [self._name]

    def contradictions(self) -> list[str]:
        """Return all predicates contradicted by this predicate."""
        if _native is not None and hasattr(_native, "predicate_contradictions"):
            return _native.predicate_contradictions(self._name)
        return []

    def __repr__(self) -> str:
        return f"Q.{self._name}"

    def __str__(self) -> str:
        return f"Q.{self._name}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Predicate):
            return self._name == other._name
        return False

    def __hash__(self) -> int:
        return hash(self._name)


class AppliedPredicate(Basic):
    """A predicate applied to an expression: Q.positive(x)."""

    __slots__ = ("_predicate", "_expr")

    def __init__(self, predicate: Predicate, expr: Any) -> None:
        self._predicate = predicate
        self._expr = _wrap(expr) if not isinstance(expr, Basic) else expr

    @property
    def predicate(self) -> Predicate:
        return self._predicate

    @property
    def expr(self) -> Any:
        return self._expr

    @property
    def args(self) -> tuple[Any, ...]:
        return (self._expr,)

    def __repr__(self) -> str:
        return f"{self._predicate!r}({self._expr!r})"

    def __str__(self) -> str:
        return f"{self._predicate}({self._expr})"

    def __bool__(self) -> bool:
        raise TypeError(
            f"Cannot evaluate predicate application {self} as a boolean. "
            f"Use ask({self}) to query its truth value."
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AppliedPredicate):
            return self._predicate == other._predicate and self._expr == other._expr
        return False

    def __hash__(self) -> int:
        return hash((self._predicate, self._expr))
