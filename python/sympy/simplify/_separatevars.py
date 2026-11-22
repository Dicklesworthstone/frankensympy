"""Bounded variable separation over original closed scalar expressions.

The native factor generator is used only for small exact polynomials. Its
results pass the existing independent coefficient reconstruction before typed
lifting. This adapter does not mint a proof or cancel original source domains.
"""

from fractions import Fraction
from types import BuiltinFunctionType

from ..core import (
    Add, Basic, Dummy, Expr, Function, Half, Integer, Mul, NegativeOne, One,
    Pow, Rational, Symbol, Zero, _charge_nested_if_active, _keep_coeff,
    _lift_native_result, _native, _native_expr, _surface_binding_key,
    _symbol_binding, _wrap, sympify,
)
from ..core._opaque_arithmetic import _key, _ordered, _subclass, stored_args
from ..polys._evaluation_owners import require_original_owners
from ..polys._opaque import _Polynomial
from ..polys.polytools import _RationalCancelSource


_MAX_WORK = 32_768
_MAX_DEPTH = 64
_MAX_AXES = 8
_MAX_TERMS = 128
_MAX_DEGREE = 16
_MAX_BITS = 256
_MAX_IMAGE_DEGREE = 64
_MAX_FACTORIZATIONS = 16
_INTEGERS = (Integer, Zero, One, NegativeOne)
_RATIONALS = (*_INTEGERS, Rational, Half)
_FACTOR_INTERFACES = tuple((name, getattr(_native, name)) for name in (
    "poly_factor_multivariate_native", "poly_factor_list_native",
))


def _interfaces():
    require_original_owners()
    for name, original in _FACTOR_INTERFACES:
        current = vars(_native).get(name)
        if (current is not original or type(current) is not BuiltinFunctionType
                or current.__self__ is not _native or current.__name__ != name):
            raise NotImplementedError("separatevars requires the original native factor interfaces")


def _number(value):
    if not any(type(value) is cls for cls in _RATIONALS):
        return None
    native = Basic.__dict__["_value"].__get__(value, type(value))
    return Fraction(native.exact_numerator(), native.exact_denominator())


def _multiply(factors):
    if not factors:
        return Integer(1)
    if len(factors) == 1:
        return factors[0]
    return Mul(*factors)


class _Separation:
    def __init__(self, expression, axes, force):
        self.snapshot = _RationalCancelSource(expression, axes)
        self.remaining = _MAX_WORK
        self.factorizations = 0
        self.force = force
        self.records = {}
        self.active = set()
        self.capture(expression)
        for axis in axes:
            self.capture(axis)
        self.original_obligations = self.obligations(expression)

    def charge(self, amount=1, depth=0):
        if amount > self.remaining or depth > _MAX_DEPTH:
            raise ValueError("separatevars exceeds its work/depth limit")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def unchanged(self):
        _interfaces()
        self.snapshot.unchanged()

    def capture(self, value, depth=0):
        self.charge(depth=depth)
        identity = id(value)
        if identity in self.active:
            raise ValueError("separatevars source contains a cycle")
        if identity in self.records:
            return self.records[identity]
        children = stored_args(value)
        if children is None:
            children = tuple(_wrap(child) for child in _native_expr(value).args)
        if type(children) is not tuple or len(children) > self.remaining:
            raise ValueError("separatevars requires bounded original argument tuples")
        self.active.add(identity)
        records = tuple(self.capture(child, depth + 1) for child in children)
        self.active.remove(identity)
        if type(value) is Symbol or type(value) is Dummy:
            facts = Symbol.__dict__["_assumptions"].__get__(value, Symbol)
            if facts.get("commutative", True) is not True:
                raise NotImplementedError("separatevars supports commuting scalar symbols only")
        symbols = ({value} if type(value) is Symbol or type(value) is Dummy else
                   set().union(*(record[2] for record in records)))
        if len(symbols) > _MAX_AXES:
            raise ValueError("separatevars supports at most eight source symbols")
        if type(value) is Pow:
            exponent = _number(children[1])
            if exponent == 0:
                raise NotImplementedError("separatevars preserves original held zero-power domains")
            if exponent is not None and exponent < 0 and _number(children[0]) == 0:
                raise NotImplementedError("separatevars preserves original undefined powers")
        record = (value, children, frozenset(symbols))
        self.records[identity] = record
        return record

    def nonzero(self, value):
        """Split product nonzero conditions only in the finite polynomial grammar."""
        self.charge()
        children = self.capture(value)[1]
        number = _number(value)
        if number is not None and number != 0:
            return set()
        if type(value) is Mul and self.shape(value) is not None:
            return set().union(*(self.nonzero(child) for child in children))
        if type(value) is Pow:
            exponent = _number(children[1])
            if exponent is not None and exponent.denominator == 1 and exponent > 0:
                return self.nonzero(children[0])
        return {("nonzero", _key(value))}

    def obligations(self, value):
        """Retain possible poles and opaque function domains, without cancellation."""
        from ..functions import Abs, cos, cosh, exp, log, sin, sinh

        pending, found, seen = [value], set(), set()
        while pending:
            node = pending.pop()
            self.charge()
            if id(node) in seen:
                continue
            seen.add(id(node))
            _, children, _ = self.capture(node)
            if type(node) is Pow:
                exponent = _number(children[1])
                if (exponent is not None and exponent < 0
                        or exponent is None and children[1].is_positive is not True):
                    found.update(self.nonzero(children[0]))
            elif type(node) is log and len(children) == 1:
                found.update(self.nonzero(children[0]))
            elif _subclass(type(node), Function) and type(node) not in (Abs, cos, cosh, exp, sin, sinh):
                # Functions with unmodeled poles/branches remain original
                # factors. In particular a held zero product cannot erase one.
                found.add(("function", _key(node)))
            pending.extend(children)
        return found

    def shape(self, value, depth=0):
        """Bound polynomial expansion before the native generator is called."""
        self.charge(depth=depth)
        number = _number(value)
        if number is not None:
            bits = max(number.numerator.bit_length(), number.denominator.bit_length())
            return {}, 1, bits
        if type(value) is Symbol or type(value) is Dummy:
            return {value: 1}, 1, 1
        if type(value) not in (Add, Mul, Pow):
            return None
        children = self.capture(value)[1]
        if type(value) is Pow:
            exponent = _number(children[1])
            if exponent is None or exponent.denominator != 1 or exponent < 0:
                return None
            if exponent > _MAX_DEGREE:
                raise ValueError("separatevars polynomial exponent exceeds its limit")
            base = self.shape(children[0], depth + 1)
            if base is None:
                return None
            degrees, terms, bits = base
            n = exponent.numerator
            result = ({axis: degree*n for axis, degree in degrees.items()}, terms**n, bits*n)
        else:
            shapes = tuple(self.shape(child, depth + 1) for child in children)
            if any(shape is None for shape in shapes):
                return None
            degrees, terms, bits = {}, (0 if type(value) is Add else 1), 0
            for part, count, size in shapes:
                if type(value) is Add:
                    terms += count
                    bits = max(bits, size) + 1
                    for axis, degree in part.items():
                        degrees[axis] = max(degrees.get(axis, 0), degree)
                else:
                    terms *= count
                    bits += size + max(1, count.bit_length())
                    for axis, degree in part.items():
                        degrees[axis] = degrees.get(axis, 0) + degree
                if terms > _MAX_TERMS:
                    raise ValueError("separatevars polynomial expansion exceeds its term limit")
            result = degrees, terms, bits
        degrees, terms, bits = result
        if terms > _MAX_TERMS or sum(degrees.values()) > _MAX_DEGREE or bits > _MAX_BITS:
            raise ValueError("separatevars polynomial exceeds its degree/term/coefficient limit")
        return result

    def factor(self, value):
        shape = self.shape(value)
        if shape is None:
            return None
        degrees, _, _ = shape
        image = 1
        for degree in degrees.values():
            image *= degree + 1
            if image - 1 > _MAX_IMAGE_DEGREE:
                raise ValueError("separatevars factorization exceeds its image-degree limit")
        if self.factorizations >= _MAX_FACTORIZATIONS:
            raise ValueError("separatevars exceeds its factorization count limit")
        self.factorizations += 1
        axes = tuple(_ordered(tuple(self.capture(value)[2])))
        bindings = tuple(_symbol_binding(axis) for axis in axes)
        native = _native_expr(value)
        self.unchanged()
        ring = _Polynomial((native,), tuple(_native_expr(axis) for axis in axes))
        ring.remaining = min(ring.remaining, self.remaining)
        initial = ring.remaining
        coefficients = ring.read(native)
        self.charge(len(coefficients))
        if any(max(coefficient.numerator.bit_length(), coefficient.denominator.bit_length()) > _MAX_BITS
               for coefficient in coefficients.values()):
            raise ValueError("separatevars polynomial coefficients exceed their bit limit")
        scale, factors = ring.factors(coefficients)
        self.charge(initial - ring.remaining)
        self.unchanged()
        if tuple(map(_surface_binding_key, ring.bindings)) != tuple(map(_surface_binding_key, bindings)):
            raise RuntimeError("separatevars factorization changed its typed generator ring")
        coefficient = _lift_native_result(scale, value, *axes)
        values = []
        for factor, multiplicity in factors:
            lifted = _lift_native_result(factor, value, *axes)
            values.append(lifted if multiplicity == 1 else Pow(lifted, multiplicity))
        result = _keep_coeff(coefficient, _multiply(values))
        self.capture(result)
        self.unchanged()
        return result

    def common_factors(self, expression, depth):
        """Extract structural common factors around nonpolynomial values."""
        terms = self.capture(expression)[1]
        factors = [list(self.capture(term)[1]) if type(term) is Mul else [term] for term in terms]
        if not factors:
            return expression
        common = []
        for factor in tuple(factors[0]):
            self.charge(len(factors))
            if all(factor in row for row in factors):
                common.append(factor)
                for row in factors:
                    row.remove(factor)
        if not common:
            return expression
        remainder = Add(*(_multiply(row) for row in factors))
        separated = self.separate(remainder, depth + 1)
        return _multiply((*common, separated))

    def separate(self, expression, depth=0):
        from ..functions import Abs, exp, log

        self.charge(depth=depth)
        _, children, symbols = self.capture(expression)
        if len(symbols) < 2:
            return expression
        kind = type(expression)
        if kind is Mul:
            factors = tuple(self.separate(child, depth + 1) for child in children)
            return (expression if all(a == b for a, b in zip(children, factors))
                    else _multiply(factors))
        if kind is Add:
            factored = self.factor(expression)
            if factored is not None:
                return factored
            # Keep the existing all-reciprocal sum behavior, without allowing
            # a general rational cancellation route to erase held domains.
            if all(type(child) is Pow and type(self.capture(child)[1][0]) in (Symbol, Dummy)
                   and (power := _number(self.capture(child)[1][1])) is not None
                   and power.denominator == 1 and power < 0 for child in children):
                from .simplify import _maybe_fuse_rational_add

                return _maybe_fuse_rational_add(expression)
            return self.common_factors(expression, depth)
        if kind is Pow:
            base, exponent = children
            number = _number(exponent)
            if number is not None and number < 0 and len(self.capture(base)[2]) > 1:
                raise NotImplementedError("separatevars preserves coupled original denominator domains")
            separated = self.separate(base, depth + 1)
            if type(separated) is not Mul:
                return expression if separated is base else Pow(separated, exponent)
            factors = self.capture(separated)[1]
            integer = exponent.is_integer is True
            if integer or self.force:
                if self.force and not integer and any(
                    type(factor) not in (Symbol, Dummy) and not
                    ((coefficient := _number(factor)) is not None and coefficient > 0)
                    for factor in factors
                ):
                    raise NotImplementedError("separatevars force supports scalar symbol product powers")
                return _multiply(tuple(Pow(factor, exponent) for factor in factors))
            if number is not None and number > 0:
                nonnegative = [factor for factor in factors if factor.is_nonnegative is True]
                other = [factor for factor in factors if factor.is_nonnegative is not True]
                if nonnegative:
                    pieces = [Pow(factor, exponent) for factor in nonnegative]
                    if other:
                        pieces.append(Pow(_multiply(other), exponent))
                    return _multiply(pieces)
            return expression if separated is base else Pow(separated, exponent)
        if kind is exp and type(children[0]) is Add:
            return _multiply(tuple(exp(term) for term in self.capture(children[0])[1]))
        if kind is Abs:
            argument = self.separate(children[0], depth + 1)
            if type(argument) is Mul:
                return _multiply(tuple(Abs(factor) for factor in self.capture(argument)[1]))
        if kind is log and len(children) == 1 and type(children[0]) is Mul:
            factors = self.capture(children[0])[1]
            if self.force:
                if any(type(factor) not in (Symbol, Dummy) and not
                    ((coefficient := _number(factor)) is not None and coefficient > 0)
                    for factor in factors):
                    raise NotImplementedError("separatevars force supports positive or symbolic log factors")
                return Add(*(log(factor) for factor in factors))
            positive = [factor for factor in factors if factor.is_positive is True]
            other = [factor for factor in factors if factor.is_positive is not True]
            if positive:
                terms = [log(factor) for factor in positive]
                if other:
                    terms.append(log(_multiply(other)))
                return Add(*terms)
        return expression

    def dictionary(self, expression, axes):
        if axes is None:
            return {"coeff": expression}
        if not axes:
            axes = tuple(_ordered(tuple(self.capture(expression)[2])))
            if not axes:
                return None
        groups = {axis: [] for axis in axes}
        groups["coeff"] = []
        factors = self.capture(expression)[1] if type(expression) is Mul else (expression,)
        for factor in factors:
            dependent = [axis for axis in groups if type(axis) is not str and axis in self.capture(factor)[2]]
            if len(dependent) > 1:
                return None
            groups[dependent[0] if dependent else "coeff"].append(factor)
        return {axis: _multiply(factors) for axis, factors in groups.items()}


def separatevars(expression, symbols, as_dict, force):
    _interfaces()
    if (type(as_dict) is not bool and as_dict is not None
            or type(force) is not bool and force is not None):
        raise TypeError("separatevars dict and force modes must be bool or None")
    if type(expression) is str and len(expression) > 4096:
        raise ValueError("separatevars input text exceeds its limit")
    if not _subclass(type(expression), Basic) and type(expression) not in (int, float, str, Fraction):
        raise TypeError("separatevars requires closed scalar input")
    expression = (Rational(expression.numerator, expression.denominator)
                  if type(expression) is Fraction else sympify(expression))
    original_symbols = None
    if as_dict and symbols is not None:
        if type(symbols) not in (tuple, list, set, frozenset):
            raise TypeError("separatevars symbols must be a finite built-in container")
        if len(symbols) > _MAX_AXES:
            raise ValueError("separatevars supports at most eight requested axes")
        original_symbols = tuple(symbols)
        for axis in original_symbols:
            if not _subclass(type(axis), Basic):
                raise AttributeError("separatevars symbols must have an is_Atom attribute")
            if (type(axis) not in (Symbol, Dummy, *_RATIONALS)
                    and not (type(axis) is Expr and _native_expr(axis).is_number)):
                raise ValueError("symbols must be Atoms.")
    axes = original_symbols if original_symbols is not None else ()
    operation = _Separation(expression, axes, force is True)
    result = operation.separate(expression)
    if as_dict:
        result = operation.dictionary(result, None if symbols is None else axes)
    values = tuple(result.values()) if type(result) is dict else (() if result is None else (result,))
    if result is not None:
        retained = set().union(*(operation.obligations(value) for value in values))
        if not operation.original_obligations.issubset(retained):
            raise NotImplementedError("separatevars cannot discard original domain obligations")
    if original_symbols is not None and (
        len(symbols) != len(original_symbols) or any(a is not b for a, b in zip(symbols, original_symbols))
    ):
        raise RuntimeError("separatevars requested axes mutated during evaluation")
    operation.unchanged()
    return result
