"""Bounded shell transformations that retain actual Python applications.

Replacement rebuilds only changed parents. Arithmetic expansion uses native
Add/Mul/Pow normalization around operation-local opaque slots; it does not
lower custom applications into native functions or create proof receipts.
"""

from __future__ import annotations

from fractions import Fraction

from . import (
    Add, Atom, Basic, Derivative, Float, Function, Integer, Mul, Pow, Rational, Relational,
    Symbol, _FUNCTION_CLASS_RANK, _ONE, _ZERO, _charge_nested_if_active, _has_user_override,
    _invoke_supervised_callback, _native, sympify,
)
from ._opaque_arithmetic import (
    _NAME, _admit_application, _admit_source, _children, _key, _subclass, contains_opaque, native_transform,
    stored_args, surface_equal,
)

_MAX_WORK = 16_384
_MAX_DEPTH = 128
_MAX_TERMS = 4096
_MAX_POWER = 64
_VALUE = Basic.__dict__["_value"]
_HELD_ADD = Add.__dict__["_args"]
_MISSING = object()


class _Work:
    __slots__ = ("remaining",)

    def __init__(self):
        self.remaining = _MAX_WORK

    def charge(self, amount=1, depth=0):
        if depth > _MAX_DEPTH or amount < 0 or amount > self.remaining:
            raise ValueError("surface transformation exceeds its work/depth limit")
        self.remaining -= amount
        _charge_nested_if_active(amount)


def _validate(values):
    for value in values:
        if _subclass(type(value), Basic):
            contains_opaque(value)


def _same(left, right):
    if left is right:
        return True
    if _subclass(type(left), Basic) and _subclass(type(right), Basic):
        return surface_equal(left, right)
    if type(right) is int:
        number = _fraction(left)
        return number is not None and number.denominator == 1 and number.numerator == right
    if type(right) is float and type(left) is Float:
        return Float.__dict__["_as_python_float"](left) == right
    return False


def _args(node):
    if _subclass(type(node), Atom):
        return ()
    if type(node) is Derivative:
        return Derivative.__dict__["args"].__get__(node, Derivative)
    retained = stored_args(node)
    if retained is not None:
        return retained
    return _children(node)


def _rebuild(node, children):
    cls = type(node)
    if cls is Derivative:
        return Derivative(*children, evaluate=False)
    if _subclass(cls, Function):
        _admit_application(node)
    result = cls(*children)
    _validate((result,))
    return result


def xreplace(expression, rule):
    """Simultaneous exact replacement; inserted values are never revisited."""
    source = dict.items(rule) if isinstance(rule, dict) else rule
    items = []
    try:
        iterator = iter(source)
    except TypeError as error:
        raise TypeError("xreplace mapping must be a dict or iterable of pairs") from error
    for entry in iterator:
        if len(items) >= _MAX_TERMS:
            raise ValueError("surface replacement exceeds its rule limit")
        if type(entry) not in (tuple, list) or len(entry) != 2:
            raise TypeError("xreplace mapping entries must be two-element pairs")
        items.append((entry[0], entry[1]))
    items = tuple(items)
    _validate((expression, *(item for pair in items for item in pair)))
    work = _Work()

    def visit(node, depth):
        work.charge(depth=depth)
        for old, new in items:
            work.charge()
            if _same(node, old):
                # The exact-root API returns the caller's object unchanged;
                # a changed parent constructor performs any needed sympify.
                return new, True
        if not _subclass(type(node), Basic):
            return node, False
        children = _args(node)
        replaced, changed = [], False
        for child in children:
            result, hit = visit(child, depth + 1)
            replaced.append(result)
            changed = changed or hit
        return (_rebuild(node, replaced), True) if changed else (node, False)

    result, _ = visit(expression, 0)
    _validate((result,))
    return result


def _fraction(value):
    from . import Half, NegativeOne, One, Zero

    if type(value) not in (Rational, Integer, Half, NegativeOne, One, Zero):
        return None
    native = _VALUE.__get__(value, Basic)
    return Fraction(native.exact_numerator(), native.exact_denominator())


def _integer(value):
    ratio = _fraction(value)
    return ratio.numerator if ratio is not None and ratio.denominator == 1 else None


def _effect_key(value):
    """Complete typed callback cache key without any user string renderer."""
    records, root = _key(value)
    remaining = [_MAX_WORK]

    def native_record(native, depth=0):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > _MAX_DEPTH:
            raise ValueError("surface callback identity exceeds its work/depth limit")
        if native.is_symbol:
            binding = native.symbol_binding
            return ("symbol", binding.name, binding.identity_hex)
        if native.is_integer or native.is_rational:
            return ("number", native.exact_numerator(), native.exact_denominator())
        return (native.func_name, tuple(native_record(child, depth + 1) for child in native.args),
                str(native) if native.func_name == "Constant" else "")

    converted = []
    for record in records:
        if record[0] == "leaf":
            _native.CompiledFunction._validate_source(record[2])
            record = (record[0], record[1], native_record(record[2]))
        converted.append(record)
    return repr((converted, root))


def _product_parts(node):
    coefficient, powers = Fraction(1), {}
    for factor in (_args(node) if type(node) is Mul else (node,)):
        rational = _fraction(factor)
        if rational is not None:
            coefficient *= rational
            continue
        base, exponent = factor, 1
        if type(factor) is Pow:
            candidate, power = _args(factor)
            count = _integer(power)
            if count is not None and count > 0:
                base, exponent = candidate, count
        powers[base] = powers.get(base, 0) + exponent
    return coefficient, powers


def _algebraic_match(node, old, new, work):
    """Admitted exact sum subsets and integer power/product multiplicities."""
    if type(node) in (Add, Mul, Pow) and type(old) in (Add, Mul, Pow):
        _admit_source((node, old), check_identity=True)
    if type(node) is Pow and type(old) is Pow:
        base, exponent = _args(node)
        old_base, old_exponent = _args(old)
        count, old_count = _fraction(exponent), _fraction(old_exponent)
        if count is not None and old_count not in (None, 0) and _same(base, old_base):
            quotient = count / old_count
            # Integer outer powers preserve the original principal power.
            # A fractional outer power would need assumptions about the base
            # and exponent and is intentionally not inferred here.
            if quotient.denominator == 1:
                return Pow(new, Integer(quotient.numerator))
    if type(node) is Add and type(old) is Add:
        def split_sum(value):
            coefficient, terms = Fraction(0), []
            for term in _args(value):
                number = _fraction(term)
                if number is None:
                    terms.append(term)
                else:
                    coefficient += number
            return coefficient, terms

        coefficient, terms = split_sum(node)
        old_coefficient, old_terms = split_sum(old)
        if not old_terms:
            return _MISSING
        for sign in (1, -1):
            remaining = list(terms)
            for term in old_terms:
                wanted = term if sign == 1 else Mul(Integer(-1), term)
                for index, candidate in enumerate(remaining):
                    work.charge()
                    if _same(candidate, wanted):
                        remaining.pop(index)
                        break
                else:
                    break
            else:
                offset = coefficient - sign * old_coefficient
                return Add(Mul(Integer(sign), new),
                           Rational(offset.numerator, offset.denominator), *remaining)
    if type(node) is Mul and type(old) is Mul:
        coefficient, powers = _product_parts(node)
        old_coefficient, old_powers = _product_parts(old)
        if not old_powers or not old_coefficient or not coefficient:
            return _MISSING
        if old_coefficient < 0 < coefficient:
            # Substitution keeps a negative coefficient literal: -2*a is
            # not extracted from 2*a, even though a quotient can be formed.
            return _MISSING
        count = min(powers.get(base, 0) // exponent for base, exponent in old_powers.items())
        if count > _MAX_POWER:
            raise ValueError("surface product substitution exceeds its multiplicity limit")
        while count > 0:
            work.charge(len(old_powers) + 1)
            ratio = coefficient / old_coefficient**count
            divisor = abs(old_coefficient)**count
            if (coefficient.numerator % divisor.numerator == 0
                    and coefficient.denominator % divisor.denominator == 0):
                remaining = [Rational(ratio.numerator, ratio.denominator), Pow(new, Integer(count))]
                for base, exponent in powers.items():
                    rest = exponent - count * old_powers.get(base, 0)
                    if rest:
                        remaining.append(Pow(base, Integer(rest)))
                return Mul(*remaining)
            count -= 1
    return _MISSING


def substitute(expression, old, new):
    """Substitute through retained applications with supervised user hooks."""
    _validate((expression, old, new))
    work = _Work()

    def visit(node, depth=0):
        work.charge(depth=depth)
        if _same(node, old):
            return new
        cls = type(node)
        if not _subclass(cls, Basic):
            return node
        if _has_user_override(cls, "_eval_subs"):
            result = _invoke_supervised_callback(
                "_eval_subs", _effect_key(node), repr((_effect_key(old), _effect_key(new))),
                node._eval_subs, (old, new), input_exprs=(node, old, new),
            )
            if result is not None:
                result = sympify(result)
                _validate((result,))
                return result
        match = _algebraic_match(node, old, new, work)
        if match is not _MISSING:
            return match
        children = _args(node)
        if not children:
            return node
        if (cls not in (Add, Mul, Pow, Derivative)
                and not _subclass(cls, Function) and not _subclass(cls, Relational)):
            raise NotImplementedError("substitution through this retained binder or structural class is unsupported")
        replaced = tuple(visit(child, depth + 1) for child in children)
        if all(_same(child, result) for child, result in zip(children, replaced)):
            return node
        return _rebuild(node, replaced)

    result = visit(expression)
    _validate((result,))
    return result


def _held_add_order(children, work):
    """The held Add term ordering, using only admitted stored children."""
    _admit_source(children, check_identity=True)
    terms, generators = [], {}
    for term in children:
        coefficient, powers = Fraction(1), {}
        for factor in (_args(term) if type(term) is Mul else (term,)):
            work.charge()
            number = _fraction(factor)
            if number is not None:
                coefficient *= number
                continue
            base, exponent = factor, Fraction(1)
            if type(factor) is Pow:
                possible_base, possible_exponent = _args(factor)
                number = _fraction(possible_exponent)
                if number is not None:
                    base, exponent = possible_base, number
            powers[base] = powers.get(base, Fraction(0)) + exponent
            generators[base] = None
        terms.append((coefficient, powers, term))

    keys = {}

    def generator_key(node, depth=0):
        work.charge(depth=depth)
        identity = id(node)
        if identity in keys:
            return keys[identity][1]
        cls = type(node)
        name = _NAME.__get__(cls)
        if _subclass(cls, Symbol):
            native = _VALUE.__get__(node, Basic)
            result = (2, 0, "Symbol", ((1, (native.symbol_binding.name,)),))
        elif _subclass(cls, Function):
            result = (4, _FUNCTION_CLASS_RANK.get(name, 0), name,
                      tuple(generator_key(child, depth + 1) for child in _args(node)))
        elif cls is Pow:
            result = generator_key(_args(node)[0], depth + 1)
        else:
            result = (5, 0, name, ())
        keys[identity] = (node, result)
        return result

    ordered = sorted(generators, key=generator_key)
    return [term for coefficient, powers, term in sorted(
        terms, key=lambda item: (tuple(-item[1].get(gen, 0) for gen in ordered), item[0]))]


def expand(expression, *, deep=True, mul=True, multinomial=True, power_exp=True,
           power_base=True, basic=True, log=True, modulus=None, **hints):
    """Distribute bounded products and integer powers around retained cells."""
    if modulus is not None or any(hints.values()):
        raise NotImplementedError("this retained expansion supports arithmetic and exponential hints only")
    _validate((expression,))
    work = _Work()

    def distribute(factors):
        current = [_ONE]
        for factor in factors:
            terms = _args(factor) if type(factor) is Add else (factor,)
            size = len(current) * len(terms)
            if size > _MAX_TERMS:
                raise ValueError("surface expansion exceeds its term limit")
            work.charge(size)
            result = Add(*(Mul(left, right) for left in current for right in terms))
            current = list(_args(result)) if type(result) is Add else [result]
        return Add(*current)

    def visit(node, depth=0):
        work.charge(depth=depth)
        if not _subclass(type(node), Basic):
            return node
        children = _args(node)
        if not children:
            return node
        cls = type(node)
        if cls is Derivative:
            # Expanding a formal derivative never differentiates its source.
            return node
        if _subclass(cls, Function):
            _admit_application(node)
            if any(_has_user_override(cls, f"_eval_expand_{hint}") for hint in
                   ("basic", "mul", "multinomial", "power_base", "power_exp", "log")):
                raise NotImplementedError("custom expansion hooks require an explicit supervised transform lane")
        elif cls not in (Add, Mul, Pow):
            return node
        replaced = tuple(visit(child, depth + 1) for child in children) if deep else children
        if cls is Add:
            try:
                _HELD_ADD.__get__(node, Add)
            except AttributeError:
                return Add(*replaced)
            return Add(*_held_add_order(replaced, work), evaluate=False)
        if cls is Mul:
            return distribute(replaced) if mul else Mul(*replaced)
        if cls is Pow:
            base, exponent = replaced
            count = _integer(exponent)
            number = _fraction(base)
            if power_exp and type(exponent) is Add and number is not None and number > 0:
                return Mul(*(Pow(base, term) for term in _args(exponent)))
            if multinomial and type(base) is Add and count is not None and count >= 0:
                if count > _MAX_POWER:
                    raise ValueError("surface expansion exceeds its integer power envelope")
                return distribute([base] * count)
            if power_base and type(base) is Mul and count is not None:
                return Mul(*(Pow(child, exponent) for child in _args(base)))
            return Pow(base, exponent)
        from ..functions.elementary.exponential import exp as exp_class

        if cls is exp_class and power_exp and type(replaced[0]) is Add:
            return Mul(*(exp_class(term) for term in _args(replaced[0])))
        return (_rebuild(node, replaced) if any(not _same(a, b) for a, b in zip(children, replaced))
                else node)

    result = visit(expression)
    _validate((result,))
    return result


def simplify(expression):
    """Apply closed arithmetic simplification with opaque applications intact."""
    _validate((expression,))
    return native_transform((expression,), lambda values: _native.py_simplify_no_trig(values[0]))
