"""Bounded exact polynomial transforms around retained Python applications.

Applications are formal commutative generators for one operation only. Native
factorization/GCD never receives a name-erased custom function. A small exact
coefficient interpreter bounds expansion and checks reconstruction before the
operation-local generators are restored to the original Python objects. These
checks establish polynomial identities, not portable derivative evidence or a
new irreducibility certificate for arbitrary Python functions.
"""

from fractions import Fraction
from math import gcd, lcm

from ..core import Basic, Function, Symbol, _charge_nested_if_active, _native
from ..core._opaque_arithmetic import (
    _DICT, _MRO, _children, _key, _ordered, _subclass, contains_opaque, native_transform,
)
from .polyerrors import PolynomialError

_MAX_GENERATORS = 8
_MAX_DEGREE = 32
_MAX_TERMS = 4096
_MAX_BITS = 4096
_MAX_WORK = 262144
_MAX_DEPTH = 128


def applies(value):
    return _subclass(type(value), Basic) and contains_opaque(value)


def _integer(value):
    return _native.py_integer(value)


def _rational(value):
    return _native.py_rational(value.numerator, value.denominator)


def _add(values):
    return _native.Add(*values, evaluate=True).as_expr()


def _mul(values):
    return _native.Mul(*values, evaluate=True).as_expr()


def _pow(base, exponent):
    return _native.Pow(base, _integer(exponent), evaluate=True).as_expr()


def _binding(value):
    binding = value.symbol_binding
    return binding.name, binding.identity_hex


class _Polynomial:
    """An independent bounded coefficient view of one formal polynomial ring."""

    def __init__(self, sources, requested=(), formal=()):
        self.remaining = _MAX_WORK
        found = {}
        pending = list(sources)
        while pending:
            self.charge()
            node = pending.pop()
            if node.is_symbol:
                found.setdefault(_binding(node), node)
                if len(found) > _MAX_GENERATORS:
                    raise ValueError("retained polynomial transforms support at most eight generators")
            else:
                pending.extend(node.args)
        preferred = []
        for value in requested:
            if not value.is_symbol:
                raise PolynomialError("retained polynomial generators must be symbols or function applications")
            key = _binding(value)
            if key in preferred:
                raise PolynomialError("duplicated polynomial generators")
            preferred.append(key)
            found.setdefault(key, value)
        if len(found) > _MAX_GENERATORS:
            raise ValueError("retained polynomial transforms support at most eight generators")
        # Ordinary typed symbols precede formal applications, matching the
        # shell's polynomial generator convention without consulting a user
        # printer. Explicit generators retain the caller's order.
        from .polytools import _GENS_ORDER
        import re

        def symbol_order(key):
            name, index = re.fullmatch(r"(.*?)(\d*)", key[0], re.DOTALL).groups()
            return (_GENS_ORDER.get(name, 1000), name, int(index) if index else 0, key[1])

        formal_keys = []
        for value in formal:
            if value.is_symbol:
                key = _binding(value)
                if key in found and key not in formal_keys:
                    formal_keys.append(key)
        ordinary = sorted((key for key in found if key not in preferred and key not in formal_keys),
                          key=symbol_order)
        keys = preferred + ordinary + [key for key in formal_keys if key not in preferred]
        self.symbols = tuple(found[key] for key in keys)
        self.bindings = tuple(value.symbol_binding for value in self.symbols)
        self.index = {key: index for index, key in enumerate(keys)}
        self.zero = (0,) * len(keys)
        self.memo = {}

    def charge(self, amount=1, depth=0):
        if depth > _MAX_DEPTH or amount < 0 or amount > self.remaining:
            raise ValueError("retained polynomial transformation exceeds its work/depth limit")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def bounded(self, polynomial):
        if len(polynomial) > _MAX_TERMS:
            raise ValueError("retained polynomial transformation exceeds its term limit")
        for monomial, coefficient in polynomial.items():
            if sum(monomial) > _MAX_DEGREE:
                raise ValueError("retained polynomial transformation exceeds its degree limit")
            if max(coefficient.numerator.bit_length(), coefficient.denominator.bit_length()) > _MAX_BITS:
                raise ValueError("retained polynomial transformation exceeds its coefficient limit")
        return polynomial

    def plus(self, left, right):
        self.charge(len(left) + len(right))
        result = dict(left)
        for monomial, coefficient in right.items():
            value = result.get(monomial, Fraction(0)) + coefficient
            if value:
                result[monomial] = value
            else:
                result.pop(monomial, None)
        return self.bounded(result)

    def times(self, left, right):
        self.charge(len(left) * len(right))
        result = {}
        for a, x in left.items():
            for b, y in right.items():
                monomial = tuple(i + j for i, j in zip(a, b))
                if sum(monomial) > _MAX_DEGREE:
                    raise ValueError("retained polynomial transformation exceeds its degree limit")
                coefficient = result.get(monomial, Fraction(0)) + x * y
                if coefficient:
                    result[monomial] = coefficient
                else:
                    result.pop(monomial, None)
                if len(result) > _MAX_TERMS:
                    raise ValueError("retained polynomial transformation exceeds its term limit")
        return self.bounded(result)

    def power(self, polynomial, exponent):
        if not 0 <= exponent <= _MAX_DEGREE:
            raise ValueError("retained polynomial transformation exceeds its exponent limit")
        result = {self.zero: Fraction(1)}
        while exponent:
            if exponent & 1:
                result = self.times(result, polynomial)
            exponent >>= 1
            if exponent:
                polynomial = self.times(polynomial, polynomial)
        return result

    def read(self, source, depth=0):
        self.charge(depth=depth)
        if source in self.memo:
            return self.memo[source]
        if source.is_integer or source.is_rational:
            coefficient = Fraction(source.exact_numerator(), source.exact_denominator())
            result = {self.zero: coefficient} if coefficient else {}
        elif source.is_symbol:
            monomial = list(self.zero)
            monomial[self.index[_binding(source)]] = 1
            result = {tuple(monomial): Fraction(1)}
        elif source.is_add:
            result = {}
            for child in source.args:
                result = self.plus(result, self.read(child, depth + 1))
        elif source.is_mul:
            result = {self.zero: Fraction(1)}
            for child in source.args:
                result = self.times(result, self.read(child, depth + 1))
        elif source.is_pow:
            base, exponent = source.args
            if not exponent.is_integer:
                raise PolynomialError("retained polynomial transforms require integer powers")
            exponent = exponent.exact_numerator()
            if exponent < 0:
                raise PolynomialError("a polynomial cannot contain a negative generator power")
            result = self.power(self.read(base, depth + 1), exponent)
        else:
            raise PolynomialError("retained polynomial transforms require exact rational coefficients")
        result = self.bounded(result)
        self.memo[source] = result
        return result

    def expression(self, polynomial):
        self.bounded(polynomial)
        terms = []
        for monomial, coefficient in polynomial.items():
            self.charge(1 + len(monomial))
            terms.append(_mul([_rational(coefficient), *(
                symbol if exponent == 1 else _pow(symbol, exponent)
                for symbol, exponent in zip(self.symbols, monomial) if exponent
            )]))
        return _add(terms)

    def fraction(self, source, depth=0):
        """Exact numerator/denominator with expansion bounded before allocation."""
        self.charge(depth=depth)
        one = {self.zero: Fraction(1)}
        if source.is_add:
            numerator, denominator = {}, one
            for child in source.args:
                n, d = self.fraction(child, depth + 1)
                numerator = self.plus(self.times(numerator, d), self.times(n, denominator))
                denominator = self.times(denominator, d)
            return numerator, denominator
        if source.is_mul:
            numerator, denominator = one, one
            for child in source.args:
                n, d = self.fraction(child, depth + 1)
                numerator, denominator = self.times(numerator, n), self.times(denominator, d)
            return numerator, denominator
        if source.is_pow:
            base, exponent = source.args
            if not exponent.is_integer:
                raise PolynomialError("retained rational transforms require integer powers")
            power = exponent.exact_numerator()
            if abs(power) > _MAX_DEGREE:
                raise ValueError("retained polynomial transformation exceeds its exponent limit")
            n, d = self.fraction(base, depth + 1)
            if power < 0:
                n, d, power = d, n, -power
            if not d:
                raise ValueError("retained rational transformation has a zero denominator")
            return self.power(n, power), self.power(d, power)
        return self.read(source, depth + 1), one

    def rep(self, polynomial, variables, index=0):
        """Bounded dense-recursive ordering key, not a semantic identity."""
        self.charge()
        if index == len(variables):
            return polynomial.get(self.zero, Fraction(0))
        if not polynomial:
            return ()
        variable = variables[index]
        maximum = max(monomial[variable] for monomial in polynomial)
        groups = [{} for _ in range(maximum + 1)]
        for monomial, coefficient in polynomial.items():
            reduced = (*monomial[:variable], 0, *monomial[variable + 1:])
            groups[monomial[variable]][reduced] = coefficient
        return tuple(self.rep(group, variables, index + 1) for group in reversed(groups))

    def factors(self, polynomial):
        source = self.expression(polynomial)
        if not self.symbols or not any(any(monomial) for monomial in polynomial):
            return source, []
        if len(self.symbols) == 1:
            scale, factors = _native.poly_factor_list_native(source, self.bindings[0])
        else:
            scale, factors = _native.poly_factor_multivariate_native(source, list(self.bindings))
        if len(factors) > _MAX_DEGREE:
            raise ValueError("native factorization exceeds the factor limit")
        if not (scale.is_integer or scale.is_rational):
            raise ValueError("native factorization scale must be exact rational")
        reconstruction = self.read(scale)
        ordered = []
        for factor, multiplicity in factors:
            if type(multiplicity) is not int or not 1 <= multiplicity <= _MAX_DEGREE:
                raise ValueError("native factorization has an invalid multiplicity")
            coefficients = self.read(factor)
            reconstruction = self.times(reconstruction, self.power(coefficients, multiplicity))
            variables = tuple(index for index in range(len(self.symbols))
                              if any(monomial[index] for monomial in coefficients))
            if not variables:
                raise ValueError("native factorization returned a constant factor")
            rep = self.rep(coefficients, variables)
            ordered.append(((len(rep), len(variables), multiplicity, rep), factor, multiplicity))
        if reconstruction != polynomial:
            raise ValueError("native factorization failed exact reconstruction")
        ordered.sort(key=lambda item: item[0])
        return scale, [(factor, multiplicity) for _, factor, multiplicity in ordered]


def _inputs(expression, gens):
    """Seed formal function order without invoking Python printers or hooks."""
    if len(gens) == 1 and type(gens[0]) in (tuple, list):
        gens = tuple(gens[0])
    if len(gens) > _MAX_GENERATORS:
        raise ValueError("retained polynomial transforms support at most eight generators")
    if any(not _subclass(type(gen), Basic) for gen in gens):
        raise PolynomialError("polynomial generators must be symbolic objects")
    applications, seen, pending = [], {}, [expression]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        # Native traversal creates wrappers; owners must outlive identity keys.
        seen[id(node)] = node
        if _subclass(type(node), Function):
            applications.append(node)
        else:
            pending.extend(_children(node))
        if len(seen) > 16384:
            raise ValueError("retained polynomial source exceeds its node limit")
    requested_symbols = {
        _binding(Basic.__dict__["_value"].__get__(gen, Basic))
        for gen in gens if _subclass(type(gen), Symbol)
    }
    requested_applications = {_key(gen) for gen in gens if _subclass(type(gen), Function)}
    for application in applications:
        if (not requested_symbols and not requested_applications
                or _key(application) in requested_applications):
            continue
        pending, owners = list(_children(application)), {}
        while pending:
            child = pending.pop()
            if id(child) in owners:
                continue
            owners[id(child)] = child
            if len(owners) > 16384:
                raise ValueError("retained coefficient dependency exceeds its node limit")
            _charge_nested_if_active(1)
            if _subclass(type(child), Function) and _key(child) in requested_applications:
                raise PolynomialError("function coefficient depends on an explicit polynomial generator")
            if _subclass(type(child), Symbol):
                raw = Basic.__dict__["_value"].__get__(child, Basic)
                if _binding(raw) in requested_symbols:
                    raise PolynomialError("function coefficient depends on an explicit polynomial generator")
            else:
                pending.extend(_children(child))
    return (*gens, *_ordered(applications), expression), len(gens)


def factor(expression, gens=(), *, listing=False):
    values, requested_count = _inputs(expression, gens)
    if any(_subclass(type(value), Function)
           and any(_DICT.__get__(owner).get("_eval_factor") is not None
                   for owner in _MRO.__get__(type(value)))
           for value in values):
        raise NotImplementedError("custom factor hooks require a supervised transform lane")
    multiplicities = []

    def operation(lowered):
        raw = lowered[-1]
        requested = lowered[:requested_count]
        ring = _Polynomial((raw,), requested, lowered[requested_count:-1])
        if listing:
            coefficients = ring.read(raw)
            scale, factors = ring.factors(coefficients)
            selected = []
            for value, multiplicity in factors:
                if requested and not any(value.has(gen) for gen in requested):
                    scale = _mul([scale, _pow(value, multiplicity)])
                else:
                    selected.append(value)
                    multiplicities.append(multiplicity)
            return (scale, *selected)
        numerator, denominator = ring.fraction(raw)
        nscale, nfactors = ring.factors(numerator)
        dscale, dfactors = ring.factors(denominator)
        return _mul([nscale, _pow(dscale, -1),
                     *(_pow(value, count) for value, count in nfactors),
                     *(_pow(value, -count) for value, count in dfactors)])

    result = native_transform(values, operation, multiple=listing, atomize_functions=True)
    return (result[0], list(zip(result[1:], multiplicities))) if listing else result


def cancel(expression, gens=()):
    values, requested_count = _inputs(expression, gens)

    def operation(lowered):
        raw = lowered[-1]
        ring = _Polynomial((raw,), lowered[:requested_count], lowered[requested_count:-1])
        numerator, denominator = ring.fraction(raw)
        if not denominator:
            raise ValueError("retained rational transformation has a zero denominator")
        if not numerator:
            return _integer(0)
        n, d = ring.expression(numerator), ring.expression(denominator)
        if ring.symbols:
            divisor = _native.poly_multivariate_gcd_native(n, d, list(ring.bindings))
            divisor_coeffs = ring.read(divisor)
            quotients = []
            for source, coefficients in ((n, numerator), (d, denominator)):
                quotient, remainder = _native.poly_multivariate_div_rem_native(
                    source, divisor, list(ring.bindings),
                )
                q = ring.read(quotient)
                if ring.read(remainder) or ring.times(q, divisor_coeffs) != coefficients:
                    raise ValueError("native cancellation failed exact division reconstruction")
                quotients.append(q)
            numerator, denominator = quotients
        common_denominator, common_numerator = 1, 0
        for coefficient in (*numerator.values(), *denominator.values()):
            common_denominator = lcm(common_denominator, coefficient.denominator)
            if common_denominator.bit_length() > _MAX_BITS:
                raise ValueError("retained rational normalization exceeds its coefficient limit")
        for coefficient in (*numerator.values(), *denominator.values()):
            common_numerator = gcd(common_numerator, int(coefficient * common_denominator))
        content = Fraction(common_numerator, common_denominator)
        if denominator[max(denominator)] < 0:
            content = -content
        n = ring.expression({m: c / content for m, c in numerator.items()})
        d = ring.expression({m: c / content for m, c in denominator.items()})
        return _mul([n, _pow(d, -1)])

    return native_transform(values, operation, atomize_functions=True)


def together(expression, *, deep=False, fraction=True):
    if deep:
        raise NotImplementedError("deep together through retained function applications is unsupported")
    if fraction is not True:
        raise NotImplementedError("retained together currently requires fraction=True")
    values, _ = _inputs(expression, ())

    def operation(lowered):
        # Check the rational work envelope before structural denominator work.
        ring = _Polynomial((lowered[-1],), formal=lowered[:-1])
        ring.fraction(lowered[-1])

        def parse(term):
            coefficient, numerator, denominator = Fraction(1), {}, {}
            for part in (term.args if term.is_mul else (term,)):
                ring.charge()
                if part.is_integer or part.is_rational:
                    coefficient *= Fraction(part.exact_numerator(), part.exact_denominator())
                    continue
                if part.is_pow:
                    base, power = part.args
                    exponent = power.exact_numerator()
                else:
                    base, exponent = part, 1
                target = denominator if exponent < 0 else numerator
                target[base] = target.get(base, 0) + abs(exponent)
            return coefficient, numerator, denominator

        def common(left, right, maximum=False):
            ring.charge(len(left) + len(right))
            if maximum:
                return {base: max(left.get(base, 0), right.get(base, 0))
                        for base in left.keys() | right.keys()}
            return {base: min(power, right[base]) for base, power in left.items() if base in right}

        def subtract(left, right):
            ring.charge(len(left))
            return {base: power - right.get(base, 0) for base, power in left.items()
                    if power != right.get(base, 0)}

        def powers(mapping, sign=1):
            return [base if power == sign == 1 else _pow(base, sign * power)
                    for base, power in mapping.items()]

        def coefficient_gcd(left, right):
            return Fraction(gcd(left.numerator, right.numerator),
                            lcm(left.denominator, right.denominator))

        def keep_coefficient(coefficient, body):
            if coefficient == 1:
                return body
            if body.is_add:
                # Match the factorized public form without re-entering the
                # shell: operation-local symbols must never enter its registry.
                return _native.Mul(_rational(coefficient), body, evaluate=False).as_expr()
            return _mul([_rational(coefficient), body])

        def combine(terms):
            parsed = [parse(term) for term in terms]
            coefficient, numerator, denominator = parsed[0]
            coefficient = abs(coefficient)
            for c, n, d in parsed[1:]:
                coefficient = coefficient_gcd(coefficient, abs(c))
                numerator = common(numerator, n)
                denominator = common(denominator, d)
            if not coefficient:
                return _integer(0)
            reduced = [(c/coefficient, subtract(n, numerator), subtract(d, denominator))
                       for c, n, d in parsed]
            shared_denominator = {}
            for _, _, d in reduced:
                shared_denominator = common(shared_denominator, d, maximum=True)
            top = _add([_mul([_rational(c), *powers(n),
                              *powers(subtract(shared_denominator, d))])
                        for c, n, d in reduced])
            if top.is_add:
                content = Fraction(0)
                for term in top.args:
                    value, _, _ = parse(term)
                    content = coefficient_gcd(content, abs(value))
                if content and content != 1:
                    top = _add([_mul([_rational(1/content), term]) for term in top.args])
                    coefficient *= content
            body = _mul([*powers(numerator), top, *powers(denominator, -1),
                         *powers(shared_denominator, -1)])
            return keep_coefficient(coefficient, body)

        def visit(source, depth=0):
            ring.charge(depth=depth)
            if source.is_add:
                return combine([visit(child, depth + 1) for child in source.args])
            if source.is_mul:
                return _mul([visit(child, depth + 1) for child in source.args])
            if source.is_pow:
                base, exponent = source.args
                return _pow(visit(base, depth + 1), exponent.exact_numerator())
            return source

        return visit(lowered[-1])

    return native_transform(values, operation, atomize_functions=True)
