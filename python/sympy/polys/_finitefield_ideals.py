"""Complete ideal operations over the retained prime field.

The Python shell retains original sources and exact typed coordinates through
native admission, checked ideal construction and publication. Polynomial
mathematics stays in the modular kernels; no rational-field retry is permitted.
"""

from __future__ import annotations

from ..core import _lift_native_result, _native, _surface_binding_key, _symbol_binding
from . import _finitefield as field
from ._finitefield_groebner import (
    _BasisState, _request, _retained, _ring_states, _unpack, has_field_marker,
)
from .polyerrors import CoercionFailed, UnificationFailed
from .polytools import GroebnerBasis


def _unchanged(states, requests, retained):
    for state in states:
        state.unchanged()
    for request in requests:
        request.unchanged()
    retained.unchanged()


def _publish(raw, generators, states, requests, retained):
    _unchanged(states, requests, retained)
    context = tuple(value for request in requests for value in request.context)
    values = []
    for value in raw:
        values.append(_lift_native_result(value, *context))
        _unchanged(states, requests, retained)
    first = states[0]
    result = GroebnerBasis(values, generators, "GF(%d)" % first.prime, "lex")
    result._modulus = first.prime
    result._symmetric = first.symmetric
    result._polys = first.polynomial_output
    _unchanged(states, requests, retained)
    return result


def eliminate(basis, generators):
    state = _BasisState(basis)
    eliminated = field._generator_tuple(generators, {})
    expressions, polys, declared = _unpack(state.expressions, 32)
    retained = _retained(
        expressions, state.gens, polys,
        (state.expressions, *field._generator_containers(generators, {})),
        eliminated,
    )
    _ring_states(polys, state.gens, state.prime)
    # Include the requested axes in the same source snapshot. This admits their
    # immutable native identity even if a same-printed foreign atom was supplied.
    request = _request([*expressions, *eliminated], state.gens, polys,
                       [*declared, *([None] * len(eliminated))])
    _unchanged((state,), (request,), retained)
    bindings = [_symbol_binding(variable) for variable in eliminated]
    keys = [_surface_binding_key(binding) for binding in bindings]
    ring_keys = [_surface_binding_key(binding) for binding in request.bindings]
    if len(set(keys)) != len(keys) or any(key not in ring_keys for key in keys):
        raise ValueError("elimination requires distinct generators from the original typed ring")
    remaining = tuple(variable for variable, key in zip(state.gens, ring_keys) if key not in keys)
    _unchanged((state,), (request,), retained)
    try:
        raw = _native.prime_field_ideal_eliminate_native(
            request.raw[:len(expressions)], request.bindings, state.prime,
            bindings, state.symmetric, request.source_bindings[:len(expressions)])
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field ideal coefficients must be integers") from error
    return _publish(raw, remaining, (state,), (request,), retained)


def _binary_inputs(basis, other, operation):
    """Retain both complete source lists before any native source preflight."""
    left = _BasisState(basis)
    if type(other) is not GroebnerBasis:
        raise TypeError("ideal %s requires an exact GroebnerBasis" % operation)
    if not has_field_marker(other):
        raise UnificationFailed("ideal %s requires the same prime field" % operation)
    right = _BasisState(other)
    if left.prime != right.prime:
        raise UnificationFailed("ideal %s requires the same prime field" % operation)
    if operation == "intersection" and len(left.expressions) + len(right.expressions) > 32:
        raise ValueError("prime-field intersection supports at most 32 combined source polynomials")
    left_sources, left_polys, left_declared = _unpack(left.expressions, 32)
    right_sources, right_polys, right_declared = _unpack(right.expressions, 32)
    # Retain both bases before either source preflight. Unused right-hand axes
    # remain source data even when their immutable binding matches a left axis.
    retained = _retained(
        [*left_sources, *right_sources], left.gens, [*left_polys, *right_polys],
        (left.expressions, right.expressions), right.gens,
    )
    _ring_states(left_polys, left.gens, left.prime)
    _ring_states(right_polys, right.gens, right.prime)
    left_request = _request(left_sources, left.gens, left_polys, left_declared)
    _unchanged((left, right), (left_request,), retained)
    right_request = _request(right_sources, right.gens, right_polys, right_declared)
    _unchanged((left, right), (left_request, right_request), retained)
    if ({_surface_binding_key(binding) for binding in left_request.bindings}
            != {_surface_binding_key(binding) for binding in right_request.bindings}):
        raise UnificationFailed("ideal %s requires the same typed polynomial ring" % operation)
    return (left, right), (left_request, right_request), retained


def intersection(basis, other):
    states, requests, retained = _binary_inputs(basis, other, "intersection")
    left, _ = states
    left_request, right_request = requests
    try:
        raw = _native.prime_field_ideal_intersection_native(
            left_request.raw, right_request.raw, left_request.bindings,
            left.prime, left.symmetric, left_request.source_bindings,
            right_request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field ideal coefficients must be integers") from error
    return _publish(raw, left.gens, states, requests, retained)


def quotient(basis, other):
    states, requests, retained = _binary_inputs(basis, other, "quotient")
    left, _ = states
    left_request, right_request = requests
    # Zero and nonzero divisor cases are distinguished only after native
    # admission of both original lists, including every held source child.
    try:
        raw = _native.prime_field_ideal_quotient_native(
            left_request.raw, right_request.raw, left_request.bindings,
            left.prime, left.symmetric, left_request.source_bindings,
            right_request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field ideal coefficients must be integers") from error
    return _publish(raw, left.gens, states, requests, retained)


def saturate(basis, polynomial):
    state = _BasisState(basis)
    if len(state.expressions) > 31:
        raise ValueError("prime-field saturation supports at most 31 source polynomials")
    request, retained = state.request((polynomial,))
    try:
        raw = _native.prime_field_ideal_saturation_native(
            request.raw, request.bindings, state.prime, state.symmetric,
            request.source_bindings)
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field ideal coefficients must be integers") from error
    return _publish(raw, state.gens, (state,), (request,), retained)
