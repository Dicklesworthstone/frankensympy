"""Checked Groebner bases and normal forms over bounded prime fields.

The public source, typed generators and coefficient domain remain retained
through original-source admission, native generation/checking and result lift.
This module delegates all polynomial mathematics to the prime-field kernels.
"""

from __future__ import annotations

from ..core import (
    Add, Dummy, Integer, Symbol, _lift_native_result, _native,
    _surface_binding_key, _symbol_binding,
)
from . import _finitefield as field
from .polyerrors import DomainError, GeneratorsNeeded, UnificationFailed
from .polytools import GroebnerBasis, Poly, _rational_ideal_source, _sort_gens


def requested(sources, options):
    domain = options.get("domain")
    if (options.get("modulus") is not None
            or type(domain) is str and domain.startswith("GF(")):
        return True
    if field._exact(sources, (list, tuple)):
        # Construction will reject larger requests before mathematical work.
        return any(field.is_finite(value) for value in sources[:33])
    return False


def _controls(options, order):
    for name in options:
        if name not in ("modulus", "domain", "symmetric", "polys", "method"):
            raise NotImplementedError("prime-field Groebner does not support option %r" % name)
    if type(order) is not str or order not in ("lex", "grlex", "grevlex"):
        raise ValueError("prime-field monomial order must be lex, grlex, or grevlex")
    method = options.get("method")
    if method is not None and (type(method) is not str or method != "buchberger"):
        raise NotImplementedError("prime-field Groebner supports only the Buchberger method")
    polynomial_output = options.get("polys")
    if polynomial_output is not None and type(polynomial_output) is not bool:
        raise TypeError("polys must be a bool or None")
    symmetric = options.get("symmetric", True)
    if type(symmetric) is not bool:
        raise TypeError("symmetric must be a bool")
    return polynomial_output, symmetric


def _unpack(sources, maximum):
    if not field._exact(sources, (list, tuple)):
        raise TypeError("prime-field Groebner requires a built-in list or tuple of sources")
    if len(sources) > maximum:
        raise ValueError("prime-field Groebner supports at most 32 basis polynomials")
    expressions, states, source_gens = [], [], []
    for value in sources:
        if field._exact(value, field._poly_types()):
            state = field._PolyState(value, require_field=False)
            if not field.is_finite(value):
                _rational_ideal_source(value)
            states.append(state)
            expressions.append(state.expression)
            source_gens.append(state.gens)
        else:
            expressions.append(value)
            source_gens.append(None)
    return expressions, states, source_gens


def _retained(expressions, generators, states, containers=(), scalars=()):
    # Unused declared Poly generators are source too. Retain their declaration
    # dictionaries, even when the expression is zero or omits that coordinate.
    originals = [*expressions, *scalars]
    seen = {id(value) for value in originals}
    for state in states:
        for generator in state.gens:
            if id(generator) not in seen:
                originals.append(generator)
                seen.add(id(generator))
    return field._Retained(originals, generators, states, containers)


def _field_prime(states, options):
    prime = options.get("modulus")
    domain = options.get("domain")
    if domain is not None:
        declared = field._domain_prime(domain)
        if prime is not None and field._prime(prime) != declared:
            raise DomainError("modulus and domain specify different prime fields")
        prime = declared
    finite = [state for state in states if field.is_finite(state.polynomial)]
    if prime is None:
        if not finite:
            raise DomainError("prime-field Groebner requires an explicit modulus or retained field")
        prime = finite[0].prime
    prime = field._prime(prime)
    if any(state.prime != prime for state in finite):
        raise UnificationFailed("cannot combine polynomials over different prime fields")
    return prime


def _ring_states(states, generators, prime):
    bindings = {_surface_binding_key(_symbol_binding(gen)) for gen in generators}
    for state in states:
        if field.is_finite(state.polynomial) and state.prime != prime:
            raise UnificationFailed("cannot combine polynomials over different prime fields")
        if any(_surface_binding_key(_symbol_binding(gen)) not in bindings for gen in state.gens):
            raise UnificationFailed("Poly generators are outside the requested typed polynomial ring")


def _infer_generators(expressions, states, retained):
    declared = [gen for state in states for gen in state.gens]
    aggregate = Add(*expressions, *declared, evaluate=False)
    inference = field._Request([aggregate], (), states)
    raw = _native.prime_field_polynomial_generators_native(inference.raw[0])
    inference.unchanged()
    retained.unchanged()
    generators = [_lift_native_result(value, *inference.context) for value in raw]
    inference.unchanged()
    retained.unchanged()
    result = tuple(_sort_gens(generators))
    if states:
        first = states[0].gens
        first_keys = tuple(_surface_binding_key(_symbol_binding(gen)) for gen in first)
        if (set(first_keys) == {_surface_binding_key(_symbol_binding(gen)) for gen in result}
                and all(tuple(_surface_binding_key(_symbol_binding(gen)) for gen in state.gens)
                        == first_keys for state in states)):
            result = first
    if not result:
        raise GeneratorsNeeded("a constant or empty prime-field ideal requires explicit generators")
    retained.unchanged()
    return result


def _call(request, prime, operation, order, symmetric):
    try:
        result = _native.prime_field_groebner_native(
            request.raw, request.bindings, prime, operation, order, symmetric,
            request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        from .polyerrors import CoercionFailed
        raise CoercionFailed("prime-field Groebner coefficients must be integers") from error
    request.unchanged()
    return result


def _request(expressions, generators, states, source_gens):
    request = field._Request(expressions, generators, states)
    request.source_bindings = [
        None if declared is None else [_symbol_binding(gen) for gen in declared]
        for declared in source_gens
    ]
    request.unchanged()
    return request


def construct(sources, gens, order, options):
    polynomial_output, symmetric = _controls(options, order)
    expressions, states, source_gens = _unpack(sources, 32)
    generators = field._generator_tuple(gens, {})
    modulus = options.get("modulus")
    scalars = () if modulus is None else (modulus,)
    if modulus is not None and not field._exact(modulus, field._INTEGER_TYPES):
        from .polyerrors import CoercionFailed
        raise CoercionFailed("modulus must be an exact integer")
    retained = _retained(
        expressions, generators, states,
        (sources, *field._generator_containers(gens, {})), scalars,
    )
    prime = _field_prime(states, options)
    retained.unchanged()
    if not generators:
        generators = _infer_generators(expressions, states, retained)
    _ring_states(states, generators, prime)
    request = _request(expressions, generators, states, source_gens)
    retained.unchanged()
    raw = _call(request, prime, "basis", order, symmetric)
    retained.unchanged()
    values = []
    for value in raw:
        values.append(_lift_native_result(value, *request.context))
        request.unchanged()
        retained.unchanged()
    result = _PrimeFieldGroebnerBasis(
        values, generators, prime, order, symmetric,
        bool(states) if polynomial_output is None else polynomial_output,
    )
    request.unchanged()
    retained.unchanged()
    return result


class _BasisState:
    """Retain every public basis field before a native call or result lift."""

    def __init__(self, basis):
        if type(basis) is not _PrimeFieldGroebnerBasis:
            raise TypeError("prime-field operations require an exact native-backed GroebnerBasis")
        self.basis = basis
        self.namespace = object.__getattribute__(basis, "__dict__")
        if (type(self.namespace) is not dict or len(self.namespace) > 16
                or any(type(key) is not str or len(key) > 128 for key in self.namespace)):
            raise TypeError("prime-field basis requires bounded built-in source storage")
        self.items = tuple(self.namespace.items())
        self.expressions = self.namespace.get("exprs", field._ABSENT)
        self.gens = self.namespace.get("gens", field._ABSENT)
        self.prime = self.namespace.get("_modulus", field._ABSENT)
        self.symmetric = self.namespace.get("_symmetric", field._ABSENT)
        self.polynomial_output = self.namespace.get("_polys", field._ABSENT)
        self.order = self.namespace.get("order", field._ABSENT)
        if (not field._exact(self.expressions, (list, tuple)) or len(self.expressions) > 32
                or type(self.gens) is not tuple or not 1 <= len(self.gens) <= 8
                or any(not field._exact(gen, (Symbol, Dummy)) for gen in self.gens)):
            raise ValueError("prime-field basis requires bounded expressions and typed generators")
        if (type(self.prime) is not int or not 2 <= self.prime <= 65521
                or type(self.namespace.get("domain")) is not str
                or self.namespace["domain"] != "GF(%d)" % self.prime
                or type(self.symmetric) is not bool or type(self.polynomial_output) is not bool):
            raise DomainError("prime-field Groebner domain metadata is inconsistent")
        if type(self.order) is not str or self.order not in ("lex", "grlex", "grevlex"):
            raise ValueError("unknown prime-field monomial order")

    def unchanged(self):
        current = object.__getattribute__(self.basis, "__dict__")
        if (current is not self.namespace or len(current) != len(self.items)
                or any(current.get(key, field._ABSENT) is not value for key, value in self.items)):
            raise RuntimeError("prime-field Groebner basis mutated during exact evaluation")

    def request(self, extra=()):
        expressions, states, source_gens = _unpack(
            [*extra, *self.expressions], 33 if extra else 32)
        retained = _retained(expressions, self.gens, states, (self.expressions,))
        _ring_states(states, self.gens, self.prime)
        request = _request(expressions, self.gens, states, source_gens)
        self.unchanged()
        retained.unchanged()
        return request, retained


class _PrimeFieldGroebnerBasis(GroebnerBasis):
    """A complete reduced basis over GF(p), checked anew after public mutation."""

    def __init__(self, expressions, generators, prime, order, symmetric, polynomial_output):
        super().__init__(expressions, generators, "GF(%d)" % prime, order)
        self._modulus = prime
        self._symmetric = symmetric
        self._polys = polynomial_output

    @property
    def polys(self):
        state = _BasisState(self)
        request, retained = state.request()
        raw = _call(request, state.prime, "polynomials", state.order, state.symmetric)
        state.unchanged()
        retained.unchanged()
        values = []
        for value in raw:
            values.append(field._publish(Poly, value, request, state.prime, state.symmetric))
            state.unchanged()
            retained.unchanged()
        return values

    def __iter__(self):
        return iter(self.polys if self._polys else self.exprs)

    def __getitem__(self, index):
        return (self.polys if self._polys else self.exprs)[index]

    def __eq__(self, other):
        if type(other) is _PrimeFieldGroebnerBasis:
            return (self.exprs == other.exprs and self.gens == other.gens
                    and self.domain == other.domain and self.order == other.order)
        if isinstance(other, GroebnerBasis):
            return False
        if field._exact(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __hash__(self):
        return hash((tuple(self.exprs), self.gens, self.domain, self.order))

    @property
    def is_zero_dimensional(self):
        state = _BasisState(self)
        request, retained = state.request()
        try:
            result = _native.prime_field_groebner_zero_dimensional_native(
                request.raw, request.bindings, state.prime, state.order,
                request.source_bindings)
        except _native.PrimeFieldCoefficientError as error:
            from .polyerrors import CoercionFailed
            raise CoercionFailed("prime-field Groebner coefficients must be integers") from error
        request.unchanged()
        retained.unchanged()
        state.unchanged()
        return result

    def _reduction(self, expression, polynomial_output):
        state = _BasisState(self)
        request, retained = state.request((expression,))
        raw = _call(request, state.prime, "reduce", state.order, state.symmetric)
        state.unchanged()
        retained.unchanged()
        if len(raw) != len(state.expressions) + 1:
            raise ValueError("prime-field normal form returned an invalid quotient count")
        result = []
        for value in raw:
            if polynomial_output and state.polynomial_output:
                result.append(field._publish(Poly, value, request, state.prime, state.symmetric))
            else:
                result.append(_lift_native_result(value, *request.context))
            request.unchanged()
            retained.unchanged()
            state.unchanged()
        return result[:-1], result[-1]

    def reduce(self, expression, auto=True):
        if type(auto) is not bool:
            raise TypeError("auto must be a bool")
        return self._reduction(expression, True)

    def contains(self, expression):
        return self._reduction(expression, False)[1] == Integer(0)

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError(
            "this ideal operation is not implemented over prime fields; "
            "supported operations are construction, reduction and membership")

    # Advanced rational algorithms must never reinterpret a finite-field basis.
    _native_basis = _ideal_inputs = _unsupported
    fglm = eliminate = intersection = quotient = syzygies = _unsupported
    invert = quotient_basis = multiplication_matrix = _unsupported
    minimal_polynomial = minpoly = saturate = _unsupported
