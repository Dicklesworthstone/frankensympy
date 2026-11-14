"""Bounded exact minimal polynomials of multiquadratic numbers over QQ.

The square classes of the rational radicands are checked for independence.
Consequently the square-root monomials form a basis of a field, rather than
of a possibly reducible quotient algebra. The faithful multiplication
operator then has the same minimal polynomial as the original element.
The existing native matrix checker proves annihilation and minimal degree.
No numerical root selection, factorization guess, or upstream fallback is
involved. General algebraic extensions and nested irrational radicals are
outside this lane.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt

from ..core import (
    Add, AssocOp, Atom, AtomicExpr, Basic, Dummy, Expr, Half, I, Integer,
    Mul, NegativeOne, Number, One, Pow, Rational, Symbol, Zero,
    _charge_nested_if_active, _lift_native_result, _native,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .polyerrors import PolynomialError
from .polytools import GroebnerBasis, Poly, PurePoly, _IdealQuotientSnapshot

_MAX_GENERATORS = 4
_MAX_NODES = 4096
_MAX_DEPTH = 64
_MAX_POWER = 1024
_MAX_BITS = 4096
_MAX_OUTPUT_BITS = 8192
_MAX_WORK = 200_000
_ABSENT = object()
_DICT = type.__dict__["__dict__"]
_MRO = type.__dict__["__mro__"]
_VALUE = Basic.__dict__["_value"]
_SURFACE = Basic.__dict__["_surface_args"]
_STRUCT = Basic.__dict__["_struct_args"]
_ASSUMPTIONS = Symbol.__dict__["_assumptions"]
_HELD = ((Add, Add.__dict__["_args"]), (Mul, Mul.__dict__["_args"]),
         (Pow, Pow.__dict__["_args"]))
_RATIONAL_TYPES = (Integer, Rational, Zero, One, NegativeOne, Half)
_NATIVE_EXPR = _native.Expr
_NATIVE_SOURCE_SIZE = _NATIVE_EXPR.multiquadratic_source_size
_NATIVE_MATRIX = _native.Matrix
_NATIVE_MINPOLY = _NATIVE_MATRIX.minimal_polynomial
_NATIVE_INTEGER = _native.py_integer
_NATIVE_RATIONAL = _native.py_rational
_NATIVE_ATOM_PREFLIGHT = _native.ideal_source_preflight_native
_I_VALUE = _VALUE.__get__(I, Expr)
_OWNERS = tuple((owner, _owner_record(owner), _MRO.__get__(owner)) for owner in (
    Basic, Atom, AtomicExpr, Expr, Number, Rational, Integer, Zero, One,
    NegativeOne, Half, Symbol, Dummy, AssocOp, Add, Mul, Pow, Poly, PurePoly,
))
_METADATA_NAMES = ("name", "p", "q", "_assumptions", "_dummy_name", "_dummy_number")
_METADATA = tuple(
    (owner, name, _implementation(_DICT.__get__(owner).get(name, _ABSENT)))
    for owner, _, _ in _OWNERS for name in _METADATA_NAMES
)


def _require_original():
    if (_native.Expr is not _NATIVE_EXPR or _native.Matrix is not _NATIVE_MATRIX
            or _native.py_integer is not _NATIVE_INTEGER
            or _native.py_rational is not _NATIVE_RATIONAL
            or _native.ideal_source_preflight_native is not _NATIVE_ATOM_PREFLIGHT):
        raise NotImplementedError("minimal_polynomial requires its original native bridge")
    for owner, record, mro in _OWNERS:
        current = _MRO.__get__(owner)
        if (not _owner_matches(owner, record) or len(current) != len(mro)
                or any(a is not b for a, b in zip(current, mro))):
            raise NotImplementedError("minimal_polynomial requires original built-in scalar classes")
    for owner, name, expected in _METADATA:
        actual = _implementation(_DICT.__get__(owner).get(name, _ABSENT))
        if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
            raise NotImplementedError("minimal_polynomial requires original scalar metadata")


def _slot(descriptor, value):
    try:
        return descriptor.__get__(value, type(value))
    except AttributeError:
        return _ABSENT


class _Work:
    def __init__(self):
        self.remaining = _MAX_WORK

    def charge(self, count=1):
        if count < 0 or count > self.remaining:
            raise ValueError("minimal_polynomial exceeds its cumulative work limit")
        self.remaining -= count
        _charge_nested_if_active(count)

    def number(self, value, maximum=_MAX_BITS):
        self.charge()
        if (value.numerator.bit_length() > maximum
                or value.denominator.bit_length() > maximum):
            raise ValueError("minimal_polynomial exceeds its rational coefficient height limit")
        return value


class _Field:
    """QQ basis e_mask = product(sqrt(q_i)), with independent square classes."""

    def __init__(self, work):
        self.work = work
        self.radicands = []
        self.squares = [Fraction(1)]

    def add(self, left, right):
        out = dict(left)
        for mask, coefficient in right.items():
            value = self.work.number(out.get(mask, Fraction(0)) + coefficient)
            if value:
                out[mask] = value
            else:
                out.pop(mask, None)
        return out

    def mul(self, left, right):
        out = {}
        self.work.charge(len(left) * len(right))
        for first, a in left.items():
            for second, b in right.items():
                # Repeated roots contribute q_i; the remaining roots retain
                # their selected principal embeddings, including negative q_i.
                coefficient = self.work.number(a * b)
                coefficient = self.work.number(coefficient * self.squares[first & second])
                mask = first ^ second
                value = self.work.number(out.get(mask, Fraction(0)) + coefficient)
                if value:
                    out[mask] = value
                else:
                    out.pop(mask, None)
        return out

    def root(self, value):
        if not value:
            return {}
        for mask, square in enumerate(self.squares):
            ratio = self.work.number(value / square)
            if ratio < 0:
                continue
            self.work.charge(2)
            numerator, denominator = isqrt(ratio.numerator), isqrt(ratio.denominator)
            if numerator * numerator == ratio.numerator and denominator * denominator == ratio.denominator:
                # A product of principal negative square roots contributes
                # i**k. Its sign differs from the principal root of their
                # product whenever floor(k/2) is odd.
                negatives = sum(q < 0 for index, q in enumerate(self.radicands)
                                if mask & (1 << index))
                coefficient = Fraction(numerator, denominator)
                if (negatives // 2) % 2:
                    coefficient = -coefficient
                return {mask: coefficient}
        if len(self.radicands) == _MAX_GENERATORS:
            raise ValueError("minimal_polynomial supports at most four independent square classes")
        index = len(self.squares)
        products = [self.work.number(square * value) for square in self.squares]
        self.radicands.append(value)
        self.squares.extend(products)
        return {index: Fraction(1)}

    def inverse(self, value):
        if not value:
            raise ZeroDivisionError("minimal_polynomial source divides by zero")
        highest = max(value)
        if highest == 0:
            return {0: self.work.number(1 / value[0])}
        bit = 1 << (highest.bit_length() - 1)
        conjugate = {mask: -coefficient if mask & bit else coefficient
                     for mask, coefficient in value.items()}
        norm = self.mul(value, conjugate)
        if any(mask & bit for mask in norm):
            raise ValueError("minimal_polynomial conjugate norm failed exact reduction")
        result = self.mul(conjugate, self.inverse(norm))
        if self.mul(value, result) != {0: Fraction(1)}:
            raise ValueError("minimal_polynomial inverse failed its exact identity")
        return result

    def power(self, value, exponent):
        if abs(exponent) > _MAX_POWER:
            raise ValueError("minimal_polynomial integer power exceeds 1024")
        if exponent < 0:
            value, exponent = self.inverse(value), -exponent
        result = {0: Fraction(1)}
        while exponent:
            if exponent & 1:
                result = self.mul(result, value)
            exponent >>= 1
            if exponent:
                value = self.mul(value, value)
        return result

    def apply(self, operation, values):
        if operation == "Add":
            result = {}
            for value in values:
                result = self.add(result, value)
            return result
        if operation == "Mul":
            result = {0: Fraction(1)}
            for value in values:
                result = self.mul(result, value)
            return result
        if len(values) != 2:
            raise ValueError("minimal_polynomial requires a two-argument power")
        base, exponent = values
        if any(mask != 0 for mask in exponent):
            raise NotImplementedError("minimal_polynomial requires rational exponents")
        exponent = exponent.get(0, Fraction(0))
        if exponent.denominator == 1:
            return self.power(base, exponent.numerator)
        if exponent.denominator != 2 or any(mask != 0 for mask in base):
            raise NotImplementedError("minimal_polynomial supports square roots of rational values")
        return self.power(self.root(base.get(0, Fraction(0))), exponent.numerator)

    def operator(self, value):
        dimension = len(self.squares)
        self.work.charge(dimension * dimension)
        columns = [self.mul(value, {column: Fraction(1)}) for column in range(dimension)]
        return [[columns[column].get(row, Fraction(0))
                 for column in range(dimension)] for row in range(dimension)]


class _Source:
    """Read exact retained operands before any field cancellation or shortcut."""

    def __init__(self, field):
        self.field = field
        self.work = field.work
        self.nodes = 0
        self.active = set()
        self.memo = {}
        self.slots = []

    def charge(self, depth):
        self.nodes += 1
        self.work.charge()
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("minimal_polynomial source exceeds its node or depth limit")

    def retain(self, descriptor, value):
        stored = _slot(descriptor, value)
        self.slots.append((descriptor, value, stored))
        return stored

    def rational(self, native):
        # Borrowed atom admission bounds native integers before exporting them.
        _NATIVE_ATOM_PREFLIGHT(native)
        number = self.work.number(Fraction(native.exact_numerator(), native.exact_denominator()))
        return {0: number} if number else {}

    def native(self, value, depth):
        self.charge(depth)
        if type(value) is not _NATIVE_EXPR:
            raise TypeError("minimal_polynomial requires exact native scalar storage")
        if value.is_integer or value.is_rational:
            return self.rational(value)
        if value == _I_VALUE:
            return self.field.root(Fraction(-1))
        operation = value.func_name
        if operation not in ("Add", "Mul", "Pow"):
            raise NotImplementedError("minimal_polynomial supports exact multiquadratic constants over QQ")
        children = value.args
        if len(children) > _MAX_NODES - self.nodes:
            raise ValueError("minimal_polynomial source exceeds its node limit")
        # Materialize all operands before applying multiplication, powers or
        # cancellation. A held undefined child cannot disappear under zero.
        values = [self.native(child, depth + 1) for child in children]
        return self.field.apply(operation, values)

    def read(self, value, depth=0):
        self.charge(depth)
        if type(value) is int:
            if value.bit_length() > _MAX_BITS:
                raise ValueError("minimal_polynomial integer exceeds 4096 bits")
            return {0: Fraction(value)} if value else {}
        identity = id(value)
        if identity in self.active:
            raise ValueError("minimal_polynomial source contains a cycle")
        if identity in self.memo:
            return self.memo[identity][1]
        cls = type(value)
        arithmetic = any(cls is owner for owner, _ in _HELD)
        rational = any(cls is owner for owner in _RATIONAL_TYPES)
        if not arithmetic and not rational and cls is not Expr:
            raise NotImplementedError("minimal_polynomial requires built-in exact multiquadratic constants")
        native = self.retain(_VALUE, value)
        surface = self.retain(_SURFACE, value)
        structural = self.retain(_STRUCT, value)
        if (type(native) is not _NATIVE_EXPR
                or (structural is not _ABSENT and structural is not None)):
            raise NotImplementedError("minimal_polynomial requires original native scalar storage")
        if rational or cls is Expr:
            if (surface is not _ABSENT and surface is not None
                    and (type(surface) is not tuple or surface)):
                raise ValueError("minimal_polynomial requires literal exact atoms")
            if rational:
                if not (native.is_integer or native.is_rational):
                    raise ValueError("minimal_polynomial rational class has a nonrational payload")
                result = self.rational(native)
            elif native == _I_VALUE:
                result = self.field.root(Fraction(-1))
            else:
                raise NotImplementedError("minimal_polynomial supports exact multiquadratic constants over QQ")
        else:
            operation = next(name for owner, name in ((Add, "Add"), (Mul, "Mul"), (Pow, "Pow"))
                             if cls is owner)
            if native.func_name != operation:
                raise ValueError("minimal_polynomial arithmetic class and native payload disagree")
            # This scan borrows the entire expanded native tree, including I,
            # before any .args call can clone children. Surface DAG sharing
            # cannot hide an oversized native mirror behind memoized owners.
            self.work.charge(_NATIVE_SOURCE_SIZE(native))
            descriptor = next(slot for owner, slot in _HELD if cls is owner)
            held = self.retain(descriptor, value)
            children = held if held is not _ABSENT else surface
            self.active.add(identity)
            try:
                if children is _ABSENT:
                    result = self.native(native, depth)
                else:
                    if type(children) is not tuple or len(children) > _MAX_NODES - self.nodes:
                        raise ValueError("minimal_polynomial requires bounded original argument tuples")
                    values = [self.read(child, depth + 1) for child in children]
                    raw = native.args
                    if (len(raw) != len(children)
                            or any(_slot(_VALUE, child) != expected
                                   for child, expected in zip(children, raw))):
                        raise ValueError("minimal_polynomial retained and native arguments disagree")
                    result = self.field.apply(operation, values)
            finally:
                self.active.remove(identity)
        self.memo[identity] = (value, result)
        return result

    def unchanged(self):
        _require_original()
        if any(_slot(descriptor, value) is not previous for descriptor, value, previous in self.slots):
            raise RuntimeError("minimal_polynomial source mutated during exact evaluation")


def _checked_operator_polynomial(rows, work):
    dimension = len(rows)
    work.charge(dimension * dimension)
    entries = [_NATIVE_RATIONAL(value.numerator, value.denominator)
               for row in rows for value in row]
    coefficients = _NATIVE_MINPOLY(_NATIVE_MATRIX(dimension, dimension, entries))
    if (type(coefficients) is not list or not 2 <= len(coefficients) <= dimension + 1
            or any(type(value) is not _NATIVE_EXPR or not (value.is_integer or value.is_rational)
                   for value in coefficients)):
        raise ValueError("minimal_polynomial received an invalid checked coefficient list")
    values = [work.number(Fraction(value.exact_numerator(), value.exact_denominator()),
                          _MAX_OUTPUT_BITS) for value in coefficients]
    if values[-1] != 1:
        raise ValueError("minimal_polynomial received a nonmonic checked polynomial")
    return values


def _primitive_coefficients(coefficients, work):
    denominator = 1
    for value in coefficients:
        work.charge()
        denominator *= value.denominator // gcd(denominator, value.denominator)
        if denominator.bit_length() > _MAX_OUTPUT_BITS:
            raise ValueError("minimal_polynomial output denominator exceeds its height limit")
    integers = [value.numerator * (denominator // value.denominator) for value in coefficients]
    content = 0
    for value in integers:
        work.charge()
        if value.bit_length() > _MAX_OUTPUT_BITS:
            raise ValueError("minimal_polynomial output exceeds its coefficient height limit")
        content = gcd(content, value)
    sign = 1 if integers[-1] > 0 else -1
    return [sign * value // content for value in integers]


def minimal_polynomial(ex, x=None, compose=True, polys=False, domain=None):
    """Return the primitive minimal polynomial over QQ of an exact number.

    This bounded lane accepts rational constants, I, rational square roots,
    their sums/products, and integer powers or reciprocals. At most four
    independent square classes (field dimension sixteen), 4096 source nodes,
    depth 64, powers through 1024, and 4096-bit field coefficients are admitted.
    Every original operand is checked, including held undefined denominators.

    The result is an expression unless ``polys=True``. An explicit Symbol or
    Dummy is retained; with no ``x``, a fresh Dummy is used and ``polys=True``
    returns a PurePoly. Both Boolean ``compose`` hints use the same checked
    field method. Only the existing QQ domain representation (None or 'QQ')
    is supported. Other domains or algebraic extensions refuse explicitly.
    """
    _require_original()
    if type(compose) is not bool or type(polys) is not bool:
        raise TypeError("minimal_polynomial compose and polys options must be Boolean")
    if domain is not None and (type(domain) is not str or domain != "QQ"):
        raise NotImplementedError("minimal_polynomial currently supports the rational domain QQ")
    default_generator = x is None
    if default_generator:
        x = Dummy("x")
    if type(x) is not Symbol and type(x) is not Dummy:
        raise TypeError("minimal_polynomial generator must be an exact Symbol or Dummy")
    assumptions = _slot(_ASSUMPTIONS, x)
    if (type(assumptions) is not dict or len(assumptions) > 128
            or any(type(key) is not str or len(key) > 4096 or type(value) is not bool
                   for key, value in assumptions.items())):
        raise ValueError("minimal_polynomial requires literal generator assumptions")
    if assumptions.get("commutative") is False:
        raise PolynomialError("non-commutative expressions are not supported")
    parameter = _IdealQuotientSnapshot(GroebnerBasis([], (x,), "QQ", "lex"))
    parameter.inputs()
    work = _Work()
    field = _Field(work)
    source = _Source(field)
    element = source.read(ex)
    source.unchanged()
    rows = field.operator(element)
    coefficients = _checked_operator_polynomial(rows, work)
    source.unchanged()
    parameter.unchanged()
    coefficients = _primitive_coefficients(coefficients, work)
    native_x = _slot(_VALUE, x)
    expression = _NATIVE_INTEGER(0)
    for degree, coefficient in enumerate(coefficients):
        work.charge()
        if coefficient:
            expression = expression + _NATIVE_INTEGER(coefficient) * (
                native_x ** _NATIVE_INTEGER(degree))
    result = _lift_native_result(expression, x)
    if polys:
        result = (PurePoly if default_generator else Poly)(result, x, domain="QQ")
    source.unchanged()
    parameter.unchanged()
    return result


def minpoly(ex, x=None, compose=True, polys=False, domain=None):
    """Alias entry point for :func:`minimal_polynomial`."""
    return minimal_polynomial(ex, x=x, compose=compose, polys=polys, domain=domain)


__all__ = ["minimal_polynomial", "minpoly"]
