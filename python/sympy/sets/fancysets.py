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

from ..core import Basic, Function, Integer, Rational, Tuple, sympify
from ..core.lambdify_subs import Lambda
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .sets import Set, _EMPTY_SET, _membership_decision


_LAMBDA_OWNERS = tuple(
    (owner, _owner_record(owner))
    for owner in dict.fromkeys(
        owner for cls in (Lambda, Tuple)
        for owner in (*type.__dict__["__mro__"].__get__(cls),
                      *type.__dict__["__mro__"].__get__(type(cls)))
    )
)
_LAMBDA_PROPERTIES = tuple(
    (name, _implementation(Lambda.__dict__[name]))
    for name in ("signature", "expr")
)
_MAX_FINITE_IMAGE = 4096
_TUPLE_ARGUMENTS = Function.__dict__["_args"]
_TUPLE_VALUE = Basic.__dict__["_value"]


def _image_lambda_guard(lamda):
    """Bind an exact built-in Lambda without invoking changed descriptors."""
    def implementations():
        if any(not _owner_matches(owner, record) for owner, record in _LAMBDA_OWNERS):
            raise NotImplementedError("exact images require the original Lambda implementation")
        namespace = type.__dict__["__dict__"].__get__(Lambda)
        for name, expected in _LAMBDA_PROPERTIES:
            tokens = _implementation(namespace.get(name))
            if (len(tokens) != len(expected)
                    or any(value is not original for value, original in zip(tokens, expected))):
                raise NotImplementedError("exact images require the original Lambda implementation")

    implementations()
    original = object.__getattribute__(lamda, "_struct_args")
    if type(original) is not tuple or len(original) != 2 or type(original[0]) is not Tuple:
        raise NotImplementedError("exact images require a valid retained Lambda signature")
    signature = original[0]
    arguments = _TUPLE_ARGUMENTS.__get__(signature, Function)
    value = _TUPLE_VALUE.__get__(signature, Basic)
    if type(arguments) is not tuple:
        raise NotImplementedError("exact images require immutable Lambda signature arguments")

    def unchanged():
        implementations()
        if object.__getattribute__(lamda, "_struct_args") is not original:
            raise NotImplementedError("image Lambda changed during checking")
        if (_TUPLE_ARGUMENTS.__get__(signature, Function) is not arguments
                or _TUPLE_VALUE.__get__(signature, Basic) is not value):
            raise NotImplementedError("image Lambda signature changed during checking")

    return unchanged


def _finite_image(lamda, base_sets):
    """Materialize a bounded finite image using exact symbolic values."""
    from itertools import product
    from .sets import FiniteSet

    count = 1
    for base in base_sets:
        if type(base) not in (FiniteSet, Range):
            return None
        count *= len(base)
        if count > _MAX_FINITE_IMAGE:
            return None
    return FiniteSet(*(lamda(*point) for point in product(*base_sets)))


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
        from decimal import Decimal
        from math import isfinite
        from ..core import Expr, Float, _ExactDecimalFloat, nan, oo, zoo

        value = sympify(other)
        if type(value) is Float:
            return isfinite(value._as_python_float())
        if type(value) is _ExactDecimalFloat:
            # A certified decimal can be finite even when its binary64
            # arithmetic backing overflows. Inspect that exact value.
            decimal = object.__getattribute__(value, "_decimal")
            if type(decimal) is not str:
                raise ValueError("invalid retained decimal Float")
            return Decimal(decimal).is_finite()
        if value is nan or value is oo or value is zoo or type(value) is type(zoo):
            return False
        if (type(value) is Expr
                and any(value._value == atom._value for atom in (nan, oo, -oo, zoo))):
            return False
        if getattr(value, "is_finite", None) is False:
            return False
        return _membership_decision(getattr(value, "is_complex", None))


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
        if not isinstance(lamda, Lambda):
            raise ValueError("First argument must be a Lambda")
        if not base_sets or any(not isinstance(base, Set) for base in base_sets):
            raise TypeError("ImageSet base arguments must be sets")
        if len(lamda.signature) != len(base_sets):
            raise ValueError("Incompatible signature")
        if any(base.is_empty is True for base in base_sets):
            return _EMPTY_SET
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
        image = _finite_image(self._lamda, self._base_sets)
        if image is not None:
            # Structural inequality is not a proof of non-membership when
            # free symbolic values can coincide after specialization.
            return image.contains(other)
        return None

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and self._lamda == other._lamda
            and self._base_sets == other._base_sets
        )

    def __hash__(self) -> int:
        return hash((type(self), self._lamda, self._base_sets))

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

    ``imageset(Lambda(x, f), S)`` or ``imageset(x, f, S)``. Finite products
    are evaluated with simultaneous Lambda arguments. Exact rational images
    of real intervals use complete checked critical points and original
    domain exclusions; unsupported images remain formal ``ImageSet`` objects.
    """
    from ..core import Dummy, Symbol
    from .sets import EmptySet, Interval, Reals

    if len(args) < 2:
        raise TypeError("imageset requires a function and at least one base set")
    if isinstance(args[0], Lambda):
        lam, bases = args[0], args[1:]
    elif callable(args[0]):
        if len(args) != 2:
            raise TypeError("a callable imageset currently requires one base set")
        variable = Dummy("x")
        lam, bases = Lambda(variable, args[0](variable)), args[1:]
    elif len(args) >= 3:
        lam = Lambda(args[0], args[1])
        bases = args[2:]
    else:
        raise TypeError("imageset requires a Lambda, callable, or variable and expression")

    guard = _image_lambda_guard(lam) if type(lam) is Lambda else None
    formal = ImageSet(lam, *bases)
    if type(formal) is EmptySet:
        return formal
    finite = _finite_image(lam, bases)
    if finite is not None:
        if guard is not None:
            guard()
        return finite
    if (type(lam) is Lambda and len(bases) == 1
            and type(bases[0]) in (Interval, Reals)
            and type(lam.signature[0]) in (Symbol, Dummy)):
        from ..calculus._rational_range import STATIONARY_UNSUPPORTED, rational_image

        checked = rational_image(lam.expr, lam.signature[0], bases[0], guard)
        if checked is not STATIONARY_UNSUPPORTED:
            return checked
    return formal


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
