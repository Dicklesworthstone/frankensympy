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
        return Union(self, other)

    def __or__(self, other: Any) -> Set:
        return self.union(other)

    def __ror__(self, other: Any) -> Set:
        return Union(other, self)

    def intersect(self, other: Any) -> Set:
        """Return the intersection of self and other."""
        return Intersection(self, other)

    def intersection(self, other: Any) -> Set:
        return self.intersect(other)

    def __and__(self, other: Any) -> Set:
        return self.intersect(other)

    def __rand__(self, other: Any) -> Set:
        return Intersection(other, self)

    def difference(self, other: Any) -> Set:
        """Return the relative complement: self \\ other."""
        return Complement(self, other)

    def __sub__(self, other: Any) -> Set:
        return self.difference(other)

    def symmetric_difference(self, other: Any) -> Set:
        """Return the symmetric difference: (self \\ other) ∪ (other \\ self)."""
        return SymmetricDifference(self, other)

    def __xor__(self, other: Any) -> Set:
        return self.symmetric_difference(other)

    def complement(self, universe: Any = None) -> Set:
        """Return the complement of self with respect to universe (default UniversalSet)."""
        if universe is None:
            universe = _UNIVERSAL_SET
        return Complement(universe, self)

    def __mul__(self, other: Any) -> Set:
        if not isinstance(other, Set):
            try:
                other = _wrap_set(_native_set(other))
            except Exception:
                return NotImplemented
        return ProductSet(self, other)

    def __rmul__(self, other: Any) -> Set:
        if not isinstance(other, Set):
            try:
                other = _wrap_set(_native_set(other))
            except Exception:
                return NotImplemented
        return ProductSet(other, self)

    def __pow__(self, exp: Any) -> Set:
        if isinstance(exp, int) and exp >= 0:
            return ProductSet(*(self for _ in range(exp)))
        return NotImplemented

    def contains(self, other: Any) -> bool | None:
        """Decide element membership: True, False, or None when undecidable."""
        if getattr(self, "_native_set", None) is not None:
            return self._native_set.contains(_native_expr(other))
        return None

    def __contains__(self, other: Any) -> bool:
        res = self.contains(other)
        return bool(res) if res is not None else False

    def is_subset(self, other: Any) -> bool | None:
        """Check if self is a subset of other."""
        if getattr(self, "_native_set", None) is not None and getattr(other, "_native_set", None) is not None:
            return self._native_set.is_subset(other._native_set)
        if getattr(self, "is_empty", None) is True:
            return True
        if getattr(other, "is_empty", None) is True:
            return self.is_empty
        if isinstance(other, UniversalSet):
            return True
        return None

    def is_superset(self, other: Any) -> bool | None:
        """Check if self is a superset of other."""
        if isinstance(other, Set):
            return other.is_subset(self)
        return None

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
        if getattr(self, "_native_set", None) is not None and getattr(other, "_native_set", None) is not None:
            return self._native_set.is_disjoint(other._native_set)
        if getattr(self, "is_empty", None) is True or (isinstance(other, Set) and getattr(other, "is_empty", None) is True):
            return True
        return None

    @property
    def is_empty(self) -> bool | None:
        if getattr(self, "_native_set", None) is not None:
            return self._native_set.is_empty_set()
        return None

    @property
    def is_open(self) -> bool | None:
        if getattr(self, "_native_set", None) is not None:
            return self._native_set.is_open()
        return None

    @property
    def is_closed(self) -> bool | None:
        if getattr(self, "_native_set", None) is not None:
            return self._native_set.is_closed()
        return None

    @property
    def is_compact(self) -> bool | None:
        if getattr(self, "_native_set", None) is not None:
            return self._native_set.is_compact()
        return None

    @property
    def measure(self) -> Expr | None:
        if getattr(self, "_native_set", None) is not None:
            m = self._native_set.measure()
            return _wrap(m) if m is not None else None
        return None

    @property
    def boundary(self) -> Set | None:
        if getattr(self, "_native_set", None) is not None:
            b = self._native_set.boundary()
            return _wrap_set(b) if b is not None else None
        return None

    @property
    def closure(self) -> Set | None:
        if getattr(self, "_native_set", None) is not None:
            c = self._native_set.closure()
            return _wrap_set(c) if c is not None else None
        return None

    @property
    def interior(self) -> Set | None:
        if getattr(self, "_native_set", None) is not None:
            i = self._native_set.interior()
            return _wrap_set(i) if i is not None else None
        return None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Set):
            s_nat = getattr(self, "_native_set", None)
            o_nat = getattr(other, "_native_set", None)
            if s_nat is not None and o_nat is not None:
                return s_nat == o_nat
            return False
        return False

    def __hash__(self) -> int:
        s_nat = getattr(self, "_native_set", None)
        if s_nat is not None:
            return hash(s_nat)
        return id(self)

    def __repr__(self) -> str:
        s_nat = getattr(self, "_native_set", None)
        if s_nat is not None:
            return str(s_nat)
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return repr(self)


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
        s_start = str(n_start)
        s_end = str(n_end)
        if s_start in ("-oo", "-Infinity"):
            left_open = True
        if s_end in ("oo", "Infinity", "+oo"):
            right_open = True
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


class ProductSet(Set):
    """Cartesian product of symbolic sets."""

    __slots__ = ("_sets",)

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        if not args:
            return FiniteSet(())
        sets: list[Set] = []
        for a in args:
            if not isinstance(a, Set):
                a = _wrap_set(_native_set(a))
            if a.is_empty is True:
                return _EMPTY_SET
            sets.append(a)
        obj = object.__new__(cls)
        obj._native_set = None
        obj._sets = tuple(sets)
        return obj

    @property
    def sets(self) -> tuple[Set, ...]:
        return self._sets

    @property
    def args(self) -> tuple[Set, ...]:
        return self._sets

    def contains(self, other: Any) -> bool | None:
        if not isinstance(other, (tuple, list)) and type(other).__name__ != "Tuple":
            return False
        if len(other) != len(self._sets):
            return False
        res = True
        for elem, s in zip(other, self._sets):
            c = s.contains(elem)
            if c is False:
                return False
            if c is None:
                res = None
        return res

    def __contains__(self, other: Any) -> bool:
        res = self.contains(other)
        return bool(res) if res is not None else False

    @property
    def is_empty(self) -> bool | None:
        for s in self._sets:
            if s.is_empty is True:
                return True
        for s in self._sets:
            if s.is_empty is None:
                return None
        return False

    @property
    def is_open(self) -> bool | None:
        for s in self._sets:
            if s.is_open is False:
                return False
        for s in self._sets:
            if s.is_open is None:
                return None
        return True

    @property
    def is_closed(self) -> bool | None:
        for s in self._sets:
            if s.is_closed is False:
                return False
        for s in self._sets:
            if s.is_closed is None:
                return None
        return True

    @property
    def is_compact(self) -> bool | None:
        for s in self._sets:
            if s.is_compact is False:
                return False
        for s in self._sets:
            if s.is_compact is None:
                return None
        return True

    @property
    def measure(self) -> Expr | None:
        prod = Integer(1)
        for s in self._sets:
            m = s.measure
            if m is None:
                return None
            prod = prod * m
        return prod

    def is_subset(self, other: Any) -> bool | None:
        if self.is_empty is True:
            return True
        if isinstance(other, ProductSet):
            if len(self._sets) != len(other._sets):
                return False
            res = True
            for s1, s2 in zip(self._sets, other._sets):
                sub = s1.is_subset(s2)
                if sub is False:
                    return False
                if sub is None:
                    res = None
            return res
        if isinstance(other, EmptySet):
            return self.is_empty
        if isinstance(other, UniversalSet):
            return True
        return None

    def is_disjoint(self, other: Any) -> bool | None:
        if self.is_empty is True or (isinstance(other, Set) and other.is_empty is True):
            return True
        if isinstance(other, ProductSet):
            if len(self._sets) != len(other._sets):
                return True
            res = False
            for s1, s2 in zip(self._sets, other._sets):
                disj = s1.is_disjoint(s2)
                if disj is True:
                    return True
                if disj is None:
                    res = None
            return res
        return None

    def intersect(self, other: Any) -> Set:
        if not isinstance(other, Set):
            other = _wrap_set(_native_set(other))
        if isinstance(other, EmptySet) or other.is_empty is True:
            return _EMPTY_SET
        if isinstance(other, UniversalSet):
            return self
        if isinstance(other, ProductSet):
            if len(self._sets) != len(other._sets):
                return _EMPTY_SET
            return ProductSet(*(s1.intersect(s2) for s1, s2 in zip(self._sets, other._sets)))
        return Intersection(self, other)

    def union(self, other: Any) -> Set:
        if not isinstance(other, Set):
            other = _wrap_set(_native_set(other))
        if isinstance(other, EmptySet) or other.is_empty is True:
            return self
        if isinstance(other, UniversalSet):
            return _UNIVERSAL_SET
        if self == other:
            return self
        return Union(self, other)

    def __len__(self) -> int:
        import math
        return math.prod(len(s) for s in self._sets)

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        import itertools
        return itertools.product(*self._sets)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ProductSet):
            return self._sets == other._sets
        return False

    def __hash__(self) -> int:
        return hash(("ProductSet", self._sets))

    def __repr__(self) -> str:
        return f"ProductSet({', '.join(repr(s) for s in self._sets)})"

    def __str__(self) -> str:
        return repr(self)


class _SingletonSetMeta(type):
    """Metaclass allowing singleton set classes to act as set containers and compare directly."""

    def __contains__(cls, item: Any) -> bool:
        return item in cls()

    def __repr__(cls) -> str:
        return cls.__name__

    def __str__(cls) -> str:
        return cls.__name__

    def __eq__(cls, other: Any) -> bool:
        return cls() == other

    def __hash__(cls) -> int:
        return hash(cls.__name__)


class Reals(Interval, metaclass=_SingletonSetMeta):
    """The set of all real numbers: (-∞, ∞)."""

    __slots__ = ()

    def __new__(cls) -> Reals:
        return _REALS

    def __call__(self) -> Reals:
        return self

    def __repr__(self) -> str:
        return "Reals"

    def __str__(self) -> str:
        return "Reals"


_REALS = object.__new__(Reals)
_REALS._native_set = _native.SymSet.interval(
    _native_expr(Expr("-oo")),
    _native_expr(Expr("oo")),
    True,
    True,
)


class Union(Set):
    """A union of symbolic sets."""

    __slots__ = ("_py_args",)

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            args = tuple(args[0])
        if not args:
            return _EMPTY_SET
        can_native = True
        wrapped_args = []
        for a in args:
            if not isinstance(a, Set):
                try:
                    a = _wrap_set(_native_set(a))
                except Exception:
                    can_native = False
                    wrapped_args.append(a)
                    continue
            wrapped_args.append(a)
            if getattr(a, "_native_set", None) is None:
                can_native = False

        if can_native:
            native_sets = [a._native_set for a in wrapped_args]
            native = _native.SymSet.union_many(native_sets)
            return _wrap_set(native)

        filtered: list[Set] = []
        for s in wrapped_args:
            if isinstance(s, UniversalSet):
                return _UNIVERSAL_SET
            if isinstance(s, EmptySet) or getattr(s, "is_empty", None) is True:
                continue
            if s not in filtered:
                filtered.append(s)
        if not filtered:
            return _EMPTY_SET
        if len(filtered) == 1:
            return filtered[0]
        obj = object.__new__(cls)
        obj._native_set = None
        obj._py_args = tuple(filtered)
        return obj

    @property
    def args(self) -> tuple[Set, ...]:
        if getattr(self, "_py_args", None) is not None:
            return self._py_args
        return tuple(_wrap_set(s) for s in self._native_set.args)

    def contains(self, other: Any) -> bool | None:
        if getattr(self, "_py_args", None) is not None:
            res = False
            for s in self._py_args:
                c = s.contains(other)
                if c is True:
                    return True
                if c is None:
                    res = None
            return res
        return self._native_set.contains(_native_expr(other))

    def __repr__(self) -> str:
        if getattr(self, "_py_args", None) is not None:
            return f"Union({', '.join(repr(a) for a in self._py_args)})"
        return str(self._native_set)


class Intersection(Set):
    """An intersection of symbolic sets."""

    __slots__ = ("_py_args",)

    def __new__(cls, *args: Any) -> Set:
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            args = tuple(args[0])
        if not args:
            return _UNIVERSAL_SET
        can_native = True
        wrapped_args = []
        for a in args:
            if not isinstance(a, Set):
                try:
                    a = _wrap_set(_native_set(a))
                except Exception:
                    can_native = False
                    wrapped_args.append(a)
                    continue
            wrapped_args.append(a)
            if getattr(a, "_native_set", None) is None:
                can_native = False

        if can_native:
            native_sets = [a._native_set for a in wrapped_args]
            native = _native.SymSet.intersection_many(native_sets)
            return _wrap_set(native)

        filtered: list[Set] = []
        for s in wrapped_args:
            if isinstance(s, EmptySet) or getattr(s, "is_empty", None) is True:
                return _EMPTY_SET
            if isinstance(s, UniversalSet):
                continue
            if s not in filtered:
                filtered.append(s)
        if not filtered:
            return _UNIVERSAL_SET
        if len(filtered) == 1:
            return filtered[0]
        if all(isinstance(s, ProductSet) for s in filtered):
            lengths = {len(s.sets) for s in filtered}
            if len(lengths) == 1:
                n = lengths.pop()
                factors = []
                for i in range(n):
                    factors.append(Intersection(*(s.sets[i] for s in filtered)))
                return ProductSet(*factors)
            return _EMPTY_SET

        obj = object.__new__(cls)
        obj._native_set = None
        obj._py_args = tuple(filtered)
        return obj

    @property
    def args(self) -> tuple[Set, ...]:
        if getattr(self, "_py_args", None) is not None:
            return self._py_args
        return tuple(_wrap_set(s) for s in self._native_set.args)

    def contains(self, other: Any) -> bool | None:
        if getattr(self, "_py_args", None) is not None:
            res = True
            for s in self._py_args:
                c = s.contains(other)
                if c is False:
                    return False
                if c is None:
                    res = None
            return res
        return self._native_set.contains(_native_expr(other))

    def __repr__(self) -> str:
        if getattr(self, "_py_args", None) is not None:
            return f"Intersection({', '.join(repr(a) for a in self._py_args)})"
        return str(self._native_set)


class Complement(Set):
    """Relative complement of sets: a \\ b."""

    __slots__ = ("_py_args",)

    def __new__(cls, a: Any, b: Any) -> Set:
        if not isinstance(a, Set):
            a = _wrap_set(_native_set(a))
        if not isinstance(b, Set):
            b = _wrap_set(_native_set(b))
        if getattr(a, "_native_set", None) is not None and getattr(b, "_native_set", None) is not None:
            return _wrap_set(a._native_set.difference(b._native_set))
        if getattr(b, "is_empty", None) is True or a == b:
            return _EMPTY_SET if a == b else a
        if getattr(a, "is_empty", None) is True:
            return _EMPTY_SET
        obj = object.__new__(cls)
        obj._native_set = None
        obj._py_args = (a, b)
        return obj

    @property
    def args(self) -> tuple[Set, Set]:
        if getattr(self, "_py_args", None) is not None:
            return self._py_args
        return (self,)


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
        if getattr(val, "_native_set", None) is not None:
            return val._native_set
        raise TypeError(f"cannot convert {type(val).__name__} to SymSet")
    if isinstance(val, (list, tuple, set, frozenset)):
        return _native.SymSet.finite([_native_expr(x) for x in val])
    raise TypeError(f"cannot convert {type(val).__name__} to SymSet")


class SymmetricDifference(Set):
    """Symmetric difference of sets: (a \\ b) ∪ (b \\ a)."""

    __slots__ = ()

    def __new__(cls, a: Any, b: Any) -> Set:
        if not isinstance(a, Set):
            a = _wrap_set(_native_set(a))
        if not isinstance(b, Set):
            b = _wrap_set(_native_set(b))
        return (a - b) | (b - a)


__all__ = [
    "Complement",
    "EmptySet",
    "FiniteSet",
    "Intersection",
    "Interval",
    "ProductSet",
    "Reals",
    "Set",
    "SymmetricDifference",
    "Union",
    "UniversalSet",
]
