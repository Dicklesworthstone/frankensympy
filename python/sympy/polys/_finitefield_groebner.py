"""Checked Groebner bases and normal forms over bounded prime fields.

The public source, typed generators and coefficient domain remain retained
through original-source admission, native generation/checking and result lift.
This module delegates all polynomial mathematics to the prime-field kernels.
"""

from __future__ import annotations

from ..core import (
    Add, Dummy, Integer, Symbol, _lift_native_result, _native, _native_expr,
    _surface_binding_key, _symbol_binding,
)
from . import _finitefield as field
from .polyerrors import DomainError, GeneratorsNeeded, NotInvertible, UnificationFailed
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
        declared = []
        for gen in state.gens:
            native = _native_expr(gen)
            if type(native) is not _native.Expr or not native.is_symbol:
                raise ValueError("Poly generator requires an immutable native symbol identity")
            _native.prime_field_source_preflight_native(native)
            key = _surface_binding_key(_symbol_binding(gen))
            if key != _surface_binding_key(native.symbol_binding):
                raise ValueError("Poly generator identity disagrees with its assumptions")
            declared.append(key)
        if any(key not in bindings for key in declared):
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
    result = GroebnerBasis(values, generators, "GF(%d)" % prime, order)
    result._modulus = prime
    result._symmetric = symmetric
    result._polys = bool(states) if polynomial_output is None else polynomial_output
    request.unchanged()
    retained.unchanged()
    return result


def _basis_namespace(basis):
    """Read bounded original storage without a subclass's attribute protocol."""
    from types import GetSetDescriptorType

    descriptor = type.__dict__["__dict__"].__get__(GroebnerBasis).get("__dict__")
    if (type(descriptor) is not GetSetDescriptorType
            or descriptor.__objclass__ is not GroebnerBasis
            or descriptor.__name__ != "__dict__"):
        raise TypeError("Groebner operations require original basis storage")
    namespace = descriptor.__get__(basis, GroebnerBasis)
    if (type(namespace) is not dict or len(namespace) > 16
            or any(type(key) is not str or len(key) > 128 for key in namespace)):
        raise TypeError("Groebner operations require bounded built-in basis storage")
    return namespace


def has_field_marker(basis):
    """Retained field metadata cannot be erased by changing the domain label."""
    namespace = _basis_namespace(basis)
    domain = namespace.get("domain", field._ABSENT)
    return (any(key in namespace for key in ("_modulus", "_symmetric", "_polys"))
            or (type(domain) is str and domain.startswith("GF(")))


class _BasisState:
    """Retain every public basis field before a native call or result lift."""

    def __init__(self, basis):
        if type(basis) is not GroebnerBasis:
            raise TypeError("prime-field operations require an exact native-backed GroebnerBasis")
        self.basis = basis
        self.namespace = _basis_namespace(basis)
        self.items = tuple(self.namespace.items())
        self.expressions = self.namespace.get("exprs", field._ABSENT)
        self.gens = self.namespace.get("gens", field._ABSENT)
        self.prime = self.namespace.get("_modulus", field._ABSENT)
        self.symmetric = self.namespace.get("_symmetric", field._ABSENT)
        self.polynomial_output = self.namespace.get("_polys", field._ABSENT)
        self.order = self.namespace.get("order", field._ABSENT)
        if (not field._exact(self.expressions, (list, tuple)) or len(self.expressions) > 32
                or type(self.gens) is not tuple or len(self.gens) > 8
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
        current = _basis_namespace(self.basis)
        if (current is not self.namespace or len(current) != len(self.items)
                or any(current.get(key, field._ABSENT) is not value for key, value in self.items)):
            raise RuntimeError("prime-field Groebner basis mutated during exact evaluation")

    def request(self, extra=(), controls=()):
        expressions, states, source_gens = _unpack(
            [*extra, *self.expressions], 33 if extra else 32)
        retained = _retained(expressions, self.gens, states, (self.expressions,), controls)
        _ring_states(states, self.gens, self.prime)
        request = _request(expressions, self.gens, states, source_gens)
        self.unchanged()
        retained.unchanged()
        return request, retained


def polys(basis):
    state = _BasisState(basis)
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


def iterate(basis):
    state = _BasisState(basis)
    values = polys(basis) if state.polynomial_output else state.expressions
    state.unchanged()
    return iter(values)


def length(basis):
    state = _BasisState(basis)
    return len(state.expressions)


def getitem(basis, index):
    state = _BasisState(basis)
    values = polys(basis) if state.polynomial_output else state.expressions
    state.unchanged()
    return values[index]


def equals(basis, other):
    state = _BasisState(basis)
    if type(other) is GroebnerBasis:
        if not has_field_marker(other):
            return False
        other_state = _BasisState(other)
        result = (state.expressions == other_state.expressions
                  and state.gens == other_state.gens and state.prime == other_state.prime
                  and state.order == other_state.order
                  and state.polynomial_output == other_state.polynomial_output
                  and state.symmetric == other_state.symmetric)
        state.unchanged()
        other_state.unchanged()
        return result
    if isinstance(other, GroebnerBasis):
        return False
    if field._exact(other, (list, tuple)):
        result = list(iterate(basis)) == list(other)
        state.unchanged()
        return result
    return NotImplemented


def basis_hash(basis):
    state = _BasisState(basis)
    result = hash((tuple(state.expressions), state.gens, "GF(%d)" % state.prime,
                   state.order, state.polynomial_output, state.symmetric))
    state.unchanged()
    return result


def is_zero_dimensional(basis):
    state = _BasisState(basis)
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


def _reduction(basis, expression, polynomial_output):
    state = _BasisState(basis)
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


def reduce(basis, expression, auto=True):
    if type(auto) is not bool:
        raise TypeError("auto must be a bool")
    return _reduction(basis, expression, True)


def contains(basis, expression):
    return _reduction(basis, expression, False)[1] == Integer(0)


def _unchanged(state, request, retained):
    state.unchanged()
    request.unchanged()
    retained.unchanged()


def fglm(basis, order):
    """Convert over the retained field without reinterpreting it over QQ."""
    state = _BasisState(basis)
    if type(order) is not str or order not in ("lex", "grlex", "grevlex"):
        raise ValueError("prime-field monomial order must be lex, grlex, or grevlex")
    request, retained = state.request()
    try:
        raw = _native.prime_field_groebner_fglm_native(
            request.raw, request.bindings, state.prime, state.order, order,
            state.symmetric, request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        from .polyerrors import CoercionFailed
        raise CoercionFailed("prime-field Groebner coefficients must be integers") from error
    _unchanged(state, request, retained)
    if order == state.order:
        # Upstream returns the original object in this case. Admission and
        # current-basis verification above remain mandatory, including for
        # positive-dimensional ideals and sources modified after construction.
        return basis
    values = []
    for value in raw:
        values.append(_lift_native_result(value, *request.context))
        _unchanged(state, request, retained)
    result = GroebnerBasis(values, state.gens, "GF(%d)" % state.prime, order)
    result._modulus = state.prime
    result._symmetric = state.symmetric
    result._polys = state.polynomial_output
    _unchanged(state, request, retained)
    return result


def _quotient_operator_data(basis, element):
    state = _BasisState(basis)
    request, retained = state.request((element,))
    try:
        monomials, rows = _native.prime_field_quotient_multiplication_native(
            request.raw, request.bindings, state.prime, state.order,
            state.symmetric, request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        from .polyerrors import CoercionFailed
        raise CoercionFailed("prime-field quotient coefficients must be integers") from error
    _unchanged(state, request, retained)
    dimension = len(monomials)
    if (dimension > 64 or len(rows) != dimension
            or any(len(row) != dimension for row in rows)):
        raise ValueError("checked prime-field quotient multiplication has an invalid dimension")
    return state, request, retained, monomials, rows


def quotient_basis(basis):
    """Return every standard monomial, retaining all nilpotent directions."""
    state, request, retained, monomials, _ = _quotient_operator_data(basis, Integer(0))
    result = []
    for value in monomials:
        result.append(_lift_native_result(value, *request.context))
        _unchanged(state, request, retained)
    return tuple(result)


def multiplication_matrix(basis, element):
    """Lift the verified operator's residues using the basis's signed policy."""
    from ..matrices import Matrix

    state, request, retained, monomials, rows = _quotient_operator_data(basis, element)
    entries = []
    for row in rows:
        for value in row:
            entries.append(_lift_native_result(value, *request.context))
            _unchanged(state, request, retained)
    result = Matrix(len(monomials), len(monomials), entries)
    _unchanged(state, request, retained)
    return result


def invert(basis, element):
    """Return a checked inverse, including in positive-dimensional quotients."""
    state = _BasisState(basis)
    if len(state.expressions) > 31:
        raise ValueError("prime-field quotient inverses support at most 31 ideal generators")
    request, retained = state.request((element,))
    try:
        raw = _native.prime_field_ideal_inverse_native(
            request.raw, request.bindings, state.prime, state.order,
            state.symmetric, request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        from .polyerrors import CoercionFailed
        raise CoercionFailed("prime-field quotient coefficients must be integers") from error
    _unchanged(state, request, retained)
    if raw is None:
        raise NotInvertible(
            "polynomial is not invertible modulo the ideal over GF(%d)" % state.prime)
    result = _lift_native_result(raw, *request.context)
    _unchanged(state, request, retained)
    return result


def minimal_polynomial(basis, element, gen=None):
    """Return the monic generator of the complete annihilator ideal."""
    state = _BasisState(basis)
    if len(state.gens) > 7 or len(state.expressions) > 31:
        raise ValueError(
            "prime-field quotient minimal polynomials support seven variables and 31 ideal generators")
    before_parameter = None
    if gen is None:
        # Fresh Dummy construction crosses the replaceable symbol bridge.
        # Capture original list contents and held children before that call,
        # then hand off to the full source-and-parameter snapshot below.
        expressions, states, _ = _unpack([element, *state.expressions], 32)
        before_parameter = _retained(expressions, state.gens, states, (state.expressions,))
    parameter = Dummy("t") if gen is None else gen
    state.unchanged()
    if before_parameter is not None:
        before_parameter.unchanged()
    if not field._exact(parameter, (Symbol, Dummy)):
        raise TypeError("quotient minimal polynomial requires an exact Symbol or Dummy")
    # The parameter is retained before source conversion, including when the
    # checked result will be constant or will prove transcendence.
    request, retained = state.request((element,), (parameter,))
    if before_parameter is not None:
        before_parameter.unchanged()
    raw_parameter = _native_expr(parameter)
    if type(raw_parameter) is not _native.Expr or not raw_parameter.is_symbol:
        raise ValueError("quotient minimal polynomial has an invalid output symbol")
    _native.prime_field_source_preflight_native(raw_parameter)
    parameter_binding = _symbol_binding(parameter)
    parameter_key = _surface_binding_key(parameter_binding)
    if _surface_binding_key(raw_parameter.symbol_binding) != parameter_key:
        raise ValueError("quotient minimal polynomial output symbol has inconsistent typed assumptions")
    if any(_surface_binding_key(binding) == parameter_key for binding in request.bindings):
        raise ValueError("quotient minimal polynomial requires a distinct typed output variable")
    _unchanged(state, request, retained)
    try:
        raw = _native.prime_field_quotient_minimal_polynomial_native(
            request.raw, request.bindings, state.prime, parameter_binding,
            state.symmetric, request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        from .polyerrors import CoercionFailed
        raise CoercionFailed("prime-field quotient coefficients must be integers") from error
    _unchanged(state, request, retained)
    if raw is None:
        return None
    result = _lift_native_result(raw, *request.context, parameter)
    _unchanged(state, request, retained)
    return result


def unsupported(basis, *args, **kwargs):
    raise NotImplementedError(
        "this ideal operation is not implemented over prime fields")
