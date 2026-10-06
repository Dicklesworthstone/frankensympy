"""Infinite number sets, ranges, image sets and condition sets.

Python-side structural sets (upstream ``sympy.sets.fancysets`` and
``conditionset`` surface): ``Integers``/``Naturals``/``Naturals0`` are
singletons, ``Range`` is an arithmetic progression, ``ImageSet`` is the
image of a set under a ``Lambda`` and ``ConditionSet`` is
``{x in base | condition}``. Membership is decided only when it is exact;
otherwise ``contains`` answers None.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..core import Integer, Rational, sympify
from .sets import Set, _EMPTY_SET


def _is_integer_value(v: Any) -> bool | None:
    v = sympify(v)
    if isinstance(v, Rational):
        return v.q == 1
    return getattr(v, "is_integer", None)


class _NumberSet(Set):
    """Base of the singleton infinite number sets."""

    __slots__ = ()
    _instance = None
    _name = ""

    def __new__(cls) -> "_NumberSet":
        if cls.__dict__.get("_instance") is None:
            obj = object.__new__(cls)
            obj._native_set = None
            cls._instance = obj
        return cls._instance

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def args(self) -> tuple:
        return ()

    def __repr__(self) -> str:
        return self._name

    __str__ = __repr__

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(self._name)

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("cannot iterate over an infinite set")


class Integers(_NumberSet):
    """The set of all integers."""

    __slots__ = ()
    _name = "Integers"

    def contains(self, other: Any) -> bool | None:
        return _is_integer_value(other)

    def __iter__(self) -> Iterator[Any]:
        yield Integer(0)
        k = 1
        while True:
            yield Integer(k)
            yield Integer(-k)
            k += 1

    def intersect(self, other: Any) -> Set:
        from .sets import Interval

        if isinstance(other, Interval):
            lo, hi = other.start, other.end
            from ..functions import ceiling, floor

            if lo.is_finite and hi.is_finite:
                a = ceiling(lo)
                if other.left_open and a == lo:
                    a = a + 1
                b = floor(hi)
                if other.right_open and b == hi:
                    b = b - 1
                if isinstance(a, Integer) and isinstance(b, Integer):
                    return Range(a, b + 1) if b >= a else _EMPTY_SET
        return super().intersect(other)


class Naturals(_NumberSet):
    """The positive integers 1, 2, 3, ..."""

    __slots__ = ()
    _name = "Naturals"

    def contains(self, other: Any) -> bool | None:
        integer = _is_integer_value(other)
        if integer is False:
            return False
        positive = getattr(sympify(other), "is_positive", None)
        if integer and positive is not None:
            return positive
        return None

    def __iter__(self) -> Iterator[Any]:
        k = 1
        while True:
            yield Integer(k)
            k += 1


class Naturals0(_NumberSet):
    """The nonnegative integers 0, 1, 2, ..."""

    __slots__ = ()
    _name = "Naturals0"

    def contains(self, other: Any) -> bool | None:
        integer = _is_integer_value(other)
        if integer is False:
            return False
        nonneg = getattr(sympify(other), "is_nonnegative", None)
        if integer and nonneg is not None:
            return nonneg
        return None


class Rationals(_NumberSet):
    __slots__ = ()
    _name = "Rationals"

    def contains(self, other: Any) -> bool | None:
        return getattr(sympify(other), "is_rational", None)


class Complexes(_NumberSet):
    __slots__ = ()
    _name = "Complexes"

    def contains(self, other: Any) -> bool | None:
        return getattr(sympify(other), "is_complex", None) or (
            True if getattr(sympify(other), "is_number", False) else None
        )


class Range(Set):
    """``Range(start, stop, step)``: start, start + step, ... before stop."""

    __slots__ = ("_range",)

    def __new__(cls, *args: Any) -> Set:
        vals = [int(sympify(a)) for a in args]
        if len(vals) == 1:
            start, stop, step = 0, vals[0], 1
        elif len(vals) == 2:
            start, stop, step = vals[0], vals[1], 1
        else:
            start, stop, step = vals
        if step == 0:
            raise ValueError("step cannot be 0")
        r = range(start, stop, step)
        if len(r) == 0:
            return _EMPTY_SET
        # Canonical form: stop is the first value past the last element.
        r = range(r.start, r[-1] + step, step)
        obj = object.__new__(cls)
        obj._native_set = None
        obj._range = r
        return obj

    @property
    def start(self) -> Integer:
        return Integer(self._range.start)

    @property
    def stop(self) -> Integer:
        return Integer(self._range.stop)

    @property
    def step(self) -> Integer:
        return Integer(self._range.step)

    @property
    def args(self) -> tuple:
        return (self.start, self.stop, self.step)

    @property
    def is_empty(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:
        for v in self._range:
            yield Integer(v)

    def __len__(self) -> int:
        return len(self._range)

    def __getitem__(self, i: Any) -> Any:
        return Integer(self._range[i])

    def contains(self, other: Any) -> bool | None:
        v = sympify(other)
        if isinstance(v, Rational):
            return v.q == 1 and int(v.p) in self._range
        return None

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Range) and self._range == other._range

    def __hash__(self) -> int:
        return hash(("Range", self._range.start, self._range.stop, self._range.step))

    def __repr__(self) -> str:
        return "Range(%d, %d, %d)" % (self._range.start, self._range.stop, self._range.step)

    __str__ = __repr__


class ImageSet(Set):
    """``ImageSet(Lambda(x, f(x)), S)``: the image of ``S`` under ``f``."""

    __slots__ = ("_lamda", "_base_sets")

    def __new__(cls, lamda: Any, *base_sets: Any) -> Set:
        obj = object.__new__(cls)
        obj._native_set = None
        obj._lamda = lamda
        obj._base_sets = tuple(base_sets)
        return obj

    @property
    def lamda(self) -> Any:
        return self._lamda

    @property
    def base_sets(self) -> tuple:
        return self._base_sets

    @property
    def base_set(self) -> Any:
        return self._base_sets[0]

    @property
    def args(self) -> tuple:
        return (self._lamda, *self._base_sets)

    @property
    def is_empty(self) -> bool | None:
        if all(getattr(b, "is_empty", None) is False for b in self._base_sets):
            return False
        return None

    def contains(self, other: Any) -> bool | None:
        base = self.base_set
        if isinstance(base, (Range,)) or type(base).__name__ == "FiniteSet":
            return any(self._lamda(v) == sympify(other) for v in base)
        return None

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ImageSet)
            and str(self._lamda) == str(other._lamda)
            and self._base_sets == other._base_sets
        )

    def __hash__(self) -> int:
        return hash(("ImageSet", str(self._lamda), self._base_sets))

    def __repr__(self) -> str:
        return "ImageSet(%s)" % ", ".join([str(self._lamda)] + [str(b) for b in self._base_sets])

    __str__ = __repr__


class ConditionSet(Set):
    """``ConditionSet(x, condition, base)``: elements of ``base`` satisfying
    ``condition``."""

    __slots__ = ("_sym", "_condition", "_base")

    def __new__(cls, sym: Any, condition: Any, base_set: Any = None) -> Set:
        from ..logic.boolalg import false, true
        from .sets import _UNIVERSAL_SET

        if base_set is None:
            base_set = _UNIVERSAL_SET
        if condition is false or condition is False:
            return _EMPTY_SET
        if condition is true or condition is True:
            return base_set
        obj = object.__new__(cls)
        obj._native_set = None
        obj._sym = sym
        obj._condition = condition
        obj._base = base_set
        return obj

    @property
    def sym(self) -> Any:
        return self._sym

    @property
    def condition(self) -> Any:
        return self._condition

    @property
    def base_set(self) -> Any:
        return self._base

    @property
    def args(self) -> tuple:
        return (self._sym, self._condition, self._base)

    def contains(self, other: Any) -> bool | None:
        inside = self._base.contains(other)
        if inside is False:
            return False
        try:
            value = self._condition.subs(self._sym, other)
        except Exception:
            return None
        from ..logic.boolalg import false, true

        if value is true or value is True:
            return inside
        if value is false or value is False:
            return False
        return None

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ConditionSet) and self.args == other.args

    def __hash__(self) -> int:
        return hash(("ConditionSet", str(self._condition)))

    def __repr__(self) -> str:
        return "ConditionSet(%s, %s, %s)" % (self._sym, self._condition, self._base)

    __str__ = __repr__


def imageset(*args: Any) -> Set:
    """Image of a set under a function (upstream ``imageset``).

    ``imageset(Lambda(x, f), S)`` or ``imageset(x, f, S)``. A polynomial
    image of a closed real interval is computed exactly from its critical
    points; other images stay unevaluated ``ImageSet``.
    """
    from ..core import Symbol, diff
    from ..core.lambdify_subs import Lambda
    from .sets import FiniteSet, Interval

    if len(args) == 3:
        lam = Lambda(args[0], args[1])
        base = args[2]
    else:
        lam, base = args
        if not isinstance(lam, Lambda) and callable(lam):
            d = Symbol("x")
            lam = Lambda(d, lam(d))
    if isinstance(base, FiniteSet):
        return FiniteSet(*[lam(v) for v in base])
    if isinstance(base, Interval) and len(lam.signature) == 1:
        x = lam.signature[0]
        f = lam.expr
        from ..polys.polytools import Poly

        try:
            Poly(f, x)
            polynomial = not (f.free_symbols - {x})
        except Exception:
            polynomial = False
        a, b = base.start, base.end
        if polynomial and a.is_finite and b.is_finite and not base.left_open and not base.right_open:
            from .. import solve

            points = [a, b]
            for r in solve(diff(f, x), x):
                if r.is_real and (r - a).is_nonnegative and (b - r).is_nonnegative:
                    points.append(r)
            values = [f.subs(x, p) for p in points]
            lo = min(values, key=float)
            hi = max(values, key=float)
            return Interval(lo, hi)
    return ImageSet(lam, base)


__all__ = [
    "Complexes",
    "ConditionSet",
    "ImageSet",
    "Integers",
    "Naturals",
    "Naturals0",
    "Range",
    "Rationals",
    "imageset",
]
