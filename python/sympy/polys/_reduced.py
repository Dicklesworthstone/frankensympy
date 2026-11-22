"""Bounded ordered QQ/ZZ polynomial reduction through checked native division.

The shell owns public types, exact typed axes and retained source storage.
All coefficient arithmetic, domain admission, reduction and verification stay
in the native request. Arbitrary divisors are never treated as a Groebner basis.
"""

from types import BuiltinFunctionType

from ..core import Dummy, Symbol, _lift_native_result, _native, _symbol_binding
from ..core._function_capabilities import _owner_matches, _owner_record
from . import _finitefield
from ._evaluation_owners import require_original_owners
from .polyerrors import GeneratorsError, GeneratorsNeeded, PolynomialError, UnificationFailed


_NATIVE_REDUCED = getattr(_native, "poly_reduced_native", None)
_POLY_OWNERS = ()
_MRO = type.__dict__["__mro__"]


def register_poly_owners(*classes):
    """Record public constructors eagerly, as soon as polytools defines them."""
    global _POLY_OWNERS
    if _POLY_OWNERS:
        raise RuntimeError("polynomial reduction owners were already registered")
    _POLY_OWNERS = tuple((cls, _MRO.__get__(cls), _owner_record(cls)) for cls in classes)


def _interfaces():
    require_original_owners()
    for cls, parents, record in _POLY_OWNERS:
        current = _MRO.__get__(cls)
        if (len(current) != len(parents) or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(cls, record)):
            raise TypeError("reduced requires original built-in polynomial owners")
    native = vars(_native).get("poly_reduced_native")
    if (native is not _NATIVE_REDUCED or type(native) is not BuiltinFunctionType
            or native.__self__ is not _native or native.__name__ != "poly_reduced_native"):
        raise TypeError("reduced requires the original checked native entry point")
    return native


def _axes(values):
    if type(values) is not tuple or len(values) > 8:
        raise GeneratorsError("reduced supports at most eight exact generators")
    if any(type(value) is not Symbol and type(value) is not Dummy for value in values):
        raise GeneratorsError("reduced requires exact built-in Symbol or Dummy generators")
    if len(set(values)) != len(values):
        raise GeneratorsError("reduced requires distinct typed generators")
    return values


def _generators(positional, options):
    containers = []
    keyword = options.get("gens")
    if keyword is not None:
        if positional:
            raise GeneratorsError("generators were supplied twice")
        positional = (keyword,)
    if len(positional) == 1 and type(positional[0]) in (tuple, list):
        container = positional[0]
        if len(container) > 8:
            raise GeneratorsError("reduced supports at most eight generators")
        positional = tuple(container)
        containers.append((container, positional))
    return _axes(tuple(positional)), containers


def reduced_entry(source, divisors, generators, options):
    from .polytools import (
        GroebnerBasis, Poly, PurePoly, _IdealQuotientSnapshot,
        _elimination_generator_union, _rational_ideal_source, _sort_gens,
    )

    if type(divisors) is not tuple and type(divisors) is not list:
        raise TypeError("reduced requires a built-in divisor list or tuple")
    if len(divisors) > 32:
        raise ValueError("reduced supports at most 32 divisors")
    original_divisors = tuple(divisors)
    values = (source, *original_divisors)
    if (_finitefield.requested(None, options)
            or any(_finitefield.is_finite(value) for value in values)):
        return _finitefield.reduced_entry(source, divisors, generators, options)
    native_entry = _interfaces()

    for name in options:
        if name not in {"domain", "order", "auto", "polys", "field", "gens"}:
            raise NotImplementedError("reduced does not support option %r" % name)
    domain = options.get("domain")
    if domain is not None and (type(domain) is not str or domain not in ("ZZ", "QQ")):
        raise NotImplementedError("reduced supports only ZZ and QQ coefficient domains")
    for name in ("auto", "polys", "field"):
        if options.get(name) is not None and type(options[name]) is not bool:
            raise TypeError("%s must be a bool" % name)
    field = options.get("field")
    if field is not None and domain is not None:
        raise ValueError("field and domain cannot both select the coefficient domain")
    order = options.get("order")
    if order is None:
        order = "lex"
    if type(order) is not str or order not in ("lex", "grlex", "grevlex"):
        raise ValueError("reduced supports lex, grlex and grevlex monomial orders")
    requested_gens, containers = _generators(generators, options)
    containers.append((divisors, original_divisors))

    states, original_bases = [], []
    for value in values:
        if type(value) is not Poly and type(value) is not PurePoly:
            states.append(None)
            continue
        namespace = object.__getattribute__(value, "__dict__")
        if (type(namespace) is not dict or len(namespace) > 32
                or any(type(key) is not str or len(key) > 128 for key in namespace)):
            raise TypeError("reduced requires bounded built-in Poly storage")
        _rational_ideal_source(value)
        ring = _axes(namespace.get("_gens"))
        if not ring:
            raise GeneratorsError("reduced requires nonempty original Poly generators")
        extra = namespace.get("_extra_syms", frozenset())
        if type(extra) is not frozenset or extra:
            raise NotImplementedError("reduced supports numeric QQ/ZZ coefficient domains")
        if type(namespace.get("_domain_given")) is not bool:
            raise TypeError("reduced requires an exact Poly domain declaration")
        declared = namespace.get("_domain") if namespace["_domain_given"] else None
        states.append((ring, declared))
        original_bases.append(GroebnerBasis([], ring, "QQ", order))

    retained = _IdealQuotientSnapshot(
        GroebnerBasis(list(values), requested_gens, "QQ", order), *original_bases,
    )
    sources, _, _, _ = retained.inputs()[0]
    source_symbols = [expression.free_symbols for expression in sources]

    def unchanged():
        _interfaces()
        for container, original in containers:
            if (len(container) != len(original)
                    or any(current is not old for current, old in zip(container, original))):
                raise RuntimeError("reduced source container mutated during exact evaluation")
        retained.unchanged()

    unchanged()
    # Upstream preserves polynomial rings/domains only for its two-Poly
    # fast path. Empty/multiple divisor lists and mixed expressions are
    # re-polified together. In particular, their unused axes are not inferred.
    pair = len(values) == 2 and all(state is not None for state in states)
    positional = pair and type(source) is PurePoly and not requested_gens
    coordinates = [None] * len(values)
    if requested_gens:
        gens = requested_gens
    elif positional:
        gens = states[0][0]
        if len(gens) != len(states[1][0]):
            raise UnificationFailed("PurePoly reduction requires equal generator counts")
        coordinates[1] = [_symbol_binding(variable) for variable in states[1][0]]
    elif pair:
        gens = _elimination_generator_union(states[0][0], states[1][0])
    else:
        symbols = sorted(set().union(*source_symbols), key=lambda variable:
                         (variable.name, _symbol_binding(variable).identity_hex))
        gens = tuple(_sort_gens(symbols))
    gens = _axes(gens)
    if not gens:
        raise GeneratorsNeeded("reduced requires at least one polynomial generator")

    all_symbols = set(gens) | set().union(*source_symbols)
    for state in states:
        if state is not None:
            all_symbols.update(state[0])
    for variable in all_symbols:
        if type(variable) is not Symbol and type(variable) is not Dummy:
            raise GeneratorsError("reduced requires exact typed polynomial generators")
        assumptions = object.__getattribute__(variable, "_assumptions")
        if assumptions.get("commutative", True) is not True:
            raise PolynomialError("noncommutative polynomial reduction is not supported")
    unchanged()

    snapshot = _IdealQuotientSnapshot(
        GroebnerBasis(list(values), gens, "QQ", order), *original_bases,
    )
    sources, gens, equations, bindings = snapshot.inputs()[0]
    unchanged()
    requested_domain = domain
    if field is True:
        requested_domain = "QQ"
    elif requested_domain is None and pair and any(state[1] == "QQ" for state in states):
        requested_domain = "QQ"
    automatic = options.get("auto")
    if automatic is None:
        automatic = domain is None and field is None
    original_axes = [([_symbol_binding(variable) for variable in state[0]]
                      if state is not None else None) for state in states]
    # An inferred numeric Poly must still be polynomial in its ORIGINAL axes.
    # QQ admits either exact integer or rational coefficients without changing
    # the separately selected operation/output domain.
    original_domains = [(state[1] or "QQ") if state is not None else None for state in states]
    raw_quotients, raw_remainder, result_domain = native_entry(
        equations, bindings, original_axes, original_domains, coordinates,
        domain=requested_domain or "auto", auto=automatic, order=order,
    )
    snapshot.unchanged()
    unchanged()
    if (type(raw_quotients) is not list or len(raw_quotients) != len(original_divisors)
            or type(result_domain) is not str or result_domain not in ("ZZ", "QQ")):
        raise RuntimeError("native reduction returned an invalid complete result")
    want_polys = options.get("polys")
    if want_polys is None:
        want_polys = any(state is not None for state in states)
    context = (*sources, *gens)
    result = []
    for raw in (*raw_quotients, raw_remainder):
        expression = _lift_native_result(raw, *context)
        result.append(Poly(expression, *gens, domain=result_domain) if want_polys else expression)
        snapshot.unchanged()
        unchanged()
    return result[:-1], result[-1]
