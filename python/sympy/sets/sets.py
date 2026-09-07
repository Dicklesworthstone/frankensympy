"""Symbolic set algebra for FrankenSymPy."""

from __future__ import annotations

from typing import Any, Iterator

from ..core import Basic, Expr, _native, _native_expr, _wrap, Integer, S


class Set(Basic):
    """Abstract base class for symbolic sets."""

    __slots__ = ("_native_set",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def union(self, other: Any) -> Set:
        """Return the union of self and other."""
        return _wrap_set(self._native_set.union(_native_set(other)))

    def __or__(self, other: Any) -> Set:
        return self.union(other)

    def __ror__(self, other: Any) -> Set:
        return _wrap_set(_native_set(other).union(self._native_set))

    def intersect(self, other: Any) -> Set:
        """Return the intersection of self and other."""
        return _wrap_set(self._native_set.intersection(_native_set(other)))

    def intersection(self, other: Any) -> Set:
        return self.intersect(other)

    def __and__(self, other: Any) -> Set:
        return self.intersect(other)

    def __rand__(self, other: Any) -> Set:
        return _wrap_set(_native_set(other).intersection(self._native_set))

    def difference(self, other: Any) -> Set:
        """Return the relative complement: self \\ other."""
        return _wrap_set(self._native_set.difference(_native_set(other)))

    def __sub__(self, other: Any) -> Set:
        return self.difference(other)

    def symmetric_difference(self, other: Any) -> Set:
        """Return the symmetric difference: (self \\ other) ∪ (other \\ self)."""
        return _wrap_set(self._native_set.symmetric_difference(_native_set(other)))

    def __xor__(self, other: Any) -> Set:
        return self.symmetric_difference(other)

    def complement(self, universe: Any = None) -> Set:
        """Return the complement of self with respect to universe (default UniversalSet)."""
        if universe is None:
            return _wrap_set(self._native_set.complement())
        return _wrap_set(_native_set(universe).difference(self._native_set))

    def contains(self, other: Any) -> bool | None:
        """Decide element membership: True, False, or None when undecidable."""
        return self._native_set.contains(_native_expr(other))

    def __contains__(self, other: Any) -> bool:
        res = self.contains(other)
        return bool(res) if res is not None else False

    def is_subset(self, other: Any) -> bool | None:
        """Check if self is a subset of other."""
        return self._native_set.is_subset(_native_set(other))

    def is_superset(self, other: Any) -> bool | None:
        """Check if self is a superset of other."""
        return self._native_set.is_superset(_native_set(other))

    def is_proper_subset(self, other: Any) -> bool | None:
        """Check if self is a proper subset of other."""
        sub = self.is_subset(other)
        if sub is False or sub is None:
            return sub
        return self != other

    def is_proper_superset(self, other: Any) -> bool | None:
        """Check if self is a proper superset of other."""
        sup = self.is_superset(other)
        if sup is False or sup is None:
            return sup
        return self != other

    def is_disjoint(self, other: Any) -> bool | None:
        """Check if self and other have empty intersection."""
        return self._native_set.is_disjoint(_native_set(other))

    @property
    def is_empty(self) -> bool | None:
        return self._native_set.is_empty_set()

    @property
    def is_open(self) -> bool | None:
        return self._native_set.is_open()

    @property
    def is_closed(self) -> bool | None:
        return self._native_set.is_closed()

    @property
    def is_compact(self) -> bool | None:
        return self._native_set.is_compact()

    @property
    def measure(self) -> Expr | None:
        m = self._native_set.measure()
        return _wrap(m) if m is not None else None

    @property
    def boundary(self) -> Set | None:
        b = self._native_set.boundary()
        return _wrap_set(b) if b is not None else None

    @property
    def closure(self) -> Set | None:
        c = self._native_set.closure()
        return _wrap_set(c) if c is not None else None

    @property
    def interior(self) -> Set | None:
        i = self._native_set.interior()
        return _wrap_set(i) if i is not None else None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Set):
            return self._native_set == other._native_set
        return False

    def __hash__(self) -> int:
        return hash(self._native_set)

    def __repr__(self) -> str:
        return str(self._native_set)

    def __str__(self) -> str:
        return str(self._native_set)


class EmptySet(Set):
    """The empty set singleton: ∅."""

    __slots__ = ()
    is_empty = True

    def __new__(cls) -> EmptySet:
        return _EMPTY_SET

    @property
    def measure(self) -> Expr:
        return Integer(0)

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


_EMPTY_SET = object.__new__(EmptySet)
_EMPTY_SET._native_set = _native.SymSet.empty()


class UniversalSet(Set):
    """The universal set singleton: 𝕌."""

    __slots__ = ()
    is_empty = False

    def __new__(cls) -> UniversalSet:
        return _UNIVERSAL_SET

    def __bool__(self) -> bool:
        return True


_UNIVERSAL_SET = object.__new__(UniversalSet)
_UNIVERSAL_SET._native_set = _native.SymSet.universal()


class Interval(Set):
    """A real continuous interval."""

    __slots__ = ()

    def __new__(
        cls,
        start: Any,
        end: Any,
        left_open: bool = False,
        right_open: bool = False,
    ) -> Set:
        n_start = _native_expr(start)
        n_end = _native_expr(end)
        native = _native.SymSet.interval(n_start, n_end, bool(left_open), bool(right_open))
        if native.kind == "EmptySet":
            return _EMPTY_SET
        obj = object.__new__(cls)
        obj._native_set = native
        return obj

    @property
    def start(self) -> Expr:
        return _wrap(self._native_set.start)

    @property
    def end(self) -> Expr:
        return _wrap(self._native_set.end)

    @property
    def left(self) -> Expr:
        return self.start

    @property
    def right(self) -> Expr:
        return self.end

    @property
    def left_open(self) -> bool:
        return bool(self._native_set.left_open)

    @property
    def right_open(self) -> bool:
        return bool(self._native_set.right_open)

    @property
    def args(self) -> tuple[Expr, Expr, Any, Any]:
        return (
            self.start,
            self.end,
            S.true if self.left_open else S.false,
            S.true if self.right_open else S.false,
        )

    @classmethod
    def open(cls, start: Any, end: Any) -> Set:
        """Construct an open interval (start, end)."""
        return cls(start, end, True, True)

    @classmethod
    def Lopen(cls, start: Any, end: Any) -> Set:
        """Construct a left-open interval (start, end]."""
        return cls(start, end, True, False)

    @classmethod
    def Ropen(cls, start: Any, end: Any) -> Set:
        """Construct a right-open interval [start, end)."""
        return cls(start, end, False, True)

    def as_relational(self, symbol: Any) -> Any:
        """Return the interval as a relational formula for the given symbol."""
        from ..core import Lt, Le
        from ..logic import And
        left_rel = Lt(self.start, symbol) if self.left_open else Le(self.start, symbol)
        right_rel = Lt(symbol, self.end) if self.right_open else Le(symbol, self.end)
        return And(left_rel, right_rel)


class FiniteSet(Set):
    """A discrete finite set of explicit elements."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, set, frozenset)):
            args = tuple(args[0])
        if not args:
            return _EMPTY_SET
        native_elems = [_native_expr(a) for a in args]
        native = _native.SymSet.finite(native_elems)
        if native.kind == "EmptySet":
            return _EMPTY_SET
        obj = object.__new__(cls)
        obj._native_set = native
        return obj

    def __iter__(self) -> Iterator[Expr]:
        elems = self._native_set.elements or []
        for e in elems:
            yield _wrap(e)

    def __len__(self) -> int:
        elems = self._native_set.elements
        return len(elems) if elems is not None else 0

    @property
    def args(self) -> tuple[Expr, ...]:
        elems = self._native_set.elements or []
        return tuple(_wrap(e) for e in elems)


class Union(Set):
    """A union of symbolic sets."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            args = tuple(args[0])
        if not args:
            return _EMPTY_SET
        native_sets = [_native_set(a) for a in args]
        native = _native.SymSet.union_many(native_sets)
        return _wrap_set(native)

    @property
    def args(self) -> tuple[Set, ...]:
        return tuple(_wrap_set(s) for s in self._native_set.args)


class Intersection(Set):
    """An intersection of symbolic sets."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            args = tuple(args[0])
        if not args:
            return _UNIVERSAL_SET
        native_sets = [_native_set(a) for a in args]
        native = _native.SymSet.intersection_many(native_sets)
        return _wrap_set(native)

    @property
    def args(self) -> tuple[Set, ...]:
        return tuple(_wrap_set(s) for s in self._native_set.args)


class Complement(Set):
    """Relative complement of sets: a \\ b."""

    __slots__ = ()

    def __new__(cls, a: Any, b: Any) -> Set:
        return _wrap_set(_native_set(a).difference(_native_set(b)))


def _wrap_set(native_symset: Any) -> Set:
    kind = native_symset.kind
    if kind == "EmptySet":
        return _EMPTY_SET
    if kind == "UniversalSet":
        return _UNIVERSAL_SET
    if kind == "Interval":
        obj = object.__new__(Interval)
        obj._native_set = native_symset
        return obj
    if kind == "FiniteSet":
        obj = object.__new__(FiniteSet)
        obj._native_set = native_symset
        return obj
    if kind == "Union":
        obj = object.__new__(Union)
        obj._native_set = native_symset
        return obj
    if kind == "Intersection":
        obj = object.__new__(Intersection)
        obj._native_set = native_symset
        return obj
    if kind == "Complement":
        obj = object.__new__(Complement)
        obj._native_set = native_symset
        return obj
    obj = object.__new__(Set)
    obj._native_set = native_symset
    return obj


def _native_set(val: Any) -> Any:
    if isinstance(val, Set):
        return val._native_set
    if isinstance(val, (list, tuple, set, frozenset)):
        return _native.SymSet.finite([_native_expr(x) for x in val])
    raise TypeError(f"cannot convert {type(val).__name__} to SymSet")


class SymmetricDifference(Set):
    """Symmetric difference of sets: (a \\ b) ∪ (b \\ a)."""

    __slots__ = ()

    def __new__(cls, a: Any, b: Any) -> Set:
        return _wrap_set(_native_set(a).symmetric_difference(_native_set(b)))


__all__ = [
    "Complement",
    "EmptySet",
    "FiniteSet",
    "Intersection",
    "Interval",
    "Set",
    "SymmetricDifference",
    "Union",
    "UniversalSet",
]
