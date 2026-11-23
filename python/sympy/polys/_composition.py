"""Source-preserving functional composition over bounded QQ/ZZ rings.

The first declared operation coordinate is substituted. The remaining typed
axes are retained as coefficient variables; Python performs no polynomial
arithmetic. Both originals and their declared rings survive native admission,
independent coefficient checking and publication of the complete result.
"""

from types import BuiltinFunctionType

from ..core import (
    Basic, Dummy, Symbol, _lift_native_result, _native, _surface_binding_key,
    _symbol_binding,
)
from ..core._function_capabilities import _owner_matches, _owner_record
from . import _finitefield
from ._evaluation_owners import require_original_owners
from .polyerrors import GeneratorsError, GeneratorsNeeded, PolynomialError, UnificationFailed


_NATIVE_COMPOSE = getattr(_native, "polynomial_compose_native", None)
_POLY_OWNERS = ()
_MRO = type.__dict__["__mro__"]


def register_poly_owners(*classes):
    global _POLY_OWNERS
    if _POLY_OWNERS:
        raise RuntimeError("polynomial composition owners were already registered")
    _POLY_OWNERS = tuple((cls, _MRO.__get__(cls), _owner_record(cls)) for cls in classes)


def _interfaces():
    require_original_owners()
    for cls, parents, record in _POLY_OWNERS:
        current = _MRO.__get__(cls)
        if (len(current) != len(parents) or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(cls, record)):
            raise TypeError("compose requires original built-in polynomial owners")
    native = vars(_native).get("polynomial_compose_native")
    if (native is not _NATIVE_COMPOSE or type(native) is not BuiltinFunctionType
            or native.__self__ is not _native or native.__name__ != "polynomial_compose_native"):
        raise TypeError("compose requires the original checked native entry point")
    return native


def _axes(values):
    if type(values) is not tuple or len(values) > 8:
        raise GeneratorsError("compose supports at most eight exact generators")
    if any(type(value) is not Symbol and type(value) is not Dummy for value in values):
        raise GeneratorsError("compose requires exact built-in Symbol or Dummy generators")
    if len(set(values)) != len(values):
        raise GeneratorsError("compose requires distinct typed generators")
    return values


def _snapshot_symbols(snapshot):
    """An older immutable native atom cannot acquire a changed declaration."""
    seen = set()
    for node, owner, name, _, native in snapshot.slots:
        if (owner is Basic and name == "_value" and type(node) in (Symbol, Dummy)
                and id(node) not in seen):
            seen.add(id(node))
            if (type(native) is not _native.Expr or not native.is_symbol
                    or _surface_binding_key(native.symbol_binding)
                    != _surface_binding_key(_symbol_binding(node))):
                raise ValueError("composition source symbol identity disagrees with its assumptions")
    snapshot.unchanged()


def compose_entry(outer, inner, generators=(), options=None, *, method=False):
    from .polytools import (
        GroebnerBasis, Poly, PurePoly, _IdealQuotientSnapshot,
        _elimination_generator_union, _rational_ideal_source, _sort_gens,
    )

    options = {} if options is None else options
    # Select the existing field lane before rational admission. No exception
    # from either checked lane is caught and retried in a different domain.
    if not method:
        modular = _finitefield.binary_entry(outer, inner, generators, options, "compose")
        if modular is not None:
            return modular
    native_entry = _interfaces()
    for name in options:
        if name not in {"domain", "polys", "field", "gens"}:
            raise NotImplementedError("compose does not support option %r" % name)
    domain = options.get("domain")
    if domain is not None and (type(domain) is not str or domain not in ("ZZ", "QQ")):
        raise NotImplementedError("compose supports only ZZ and QQ coefficient domains")
    for name in ("polys", "field"):
        if options.get(name) is not None and type(options[name]) is not bool:
            raise TypeError("%s must be a bool" % name)
    if options.get("field") is not None and domain is not None:
        raise ValueError("field and domain cannot both select the coefficient domain")
    supplied = options.get("gens")
    if supplied is not None:
        if generators:
            raise GeneratorsError("generators were supplied twice")
        generators = (supplied,)
    containers = []
    if len(generators) == 1 and type(generators[0]) in (tuple, list):
        container = generators[0]
        if len(container) > 8:
            raise GeneratorsError("compose supports at most eight generators")
        generators = tuple(container)
        containers.append((container, generators))
    requested = _axes(tuple(generators))

    originals = (outer, inner)
    states, bases = [], []
    for value in originals:
        if type(value) is not Poly and type(value) is not PurePoly:
            states.append(None)
            continue
        namespace = object.__getattribute__(value, "__dict__")
        if (type(namespace) is not dict or len(namespace) > 32
                or any(type(key) is not str or len(key) > 128 for key in namespace)):
            raise TypeError("compose requires bounded built-in Poly storage")
        _rational_ideal_source(value)
        ring = _axes(namespace.get("_gens"))
        if not ring:
            raise GeneratorsError("compose requires nonempty original Poly generators")
        extra = namespace.get("_extra_syms", frozenset())
        if type(extra) is not frozenset or extra:
            raise NotImplementedError("compose supports numeric QQ/ZZ coefficient domains")
        if type(namespace.get("_domain_given")) is not bool:
            raise TypeError("compose requires an exact Poly domain declaration")
        declared = namespace.get("_domain") if namespace["_domain_given"] else None
        states.append((ring, declared))
        bases.append(GroebnerBasis([], ring, "QQ", "lex"))
    if method and states[0] is None:
        raise TypeError("compose methods require an exact Poly or PurePoly owner")

    retained = _IdealQuotientSnapshot(
        GroebnerBasis(list(originals), requested, "QQ", "lex"), *bases,
    )
    _snapshot_symbols(retained)
    sources, _, _, _ = retained.inputs()[0]
    source_symbols = [expression.free_symbols for expression in sources]

    def unchanged():
        _interfaces()
        retained.unchanged()
        for container, original in containers:
            if (len(container) != len(original)
                    or any(current is not old for current, old in zip(container, original))):
                raise RuntimeError("composition generator container mutated during evaluation")

    unchanged()
    carry_ring = method or all(state is not None for state in states)
    positional = carry_ring and type(outer) is PurePoly and states[1] is not None and not requested
    coordinates = None
    if requested:
        gens = requested
    elif positional:
        gens = states[0][0]
        if len(gens) != len(states[1][0]):
            raise UnificationFailed("PurePoly composition requires equal generator counts")
        coordinates = [_symbol_binding(variable) for variable in states[1][0]]
    elif carry_ring:
        gens = states[0][0]
        if states[1] is not None:
            gens = _elimination_generator_union(gens, states[1][0])
    else:
        symbols = sorted(set().union(*source_symbols), key=lambda variable:
                         (variable.name, _symbol_binding(variable).identity_hex))
        gens = tuple(_sort_gens(symbols))
    gens = _axes(gens)
    if not gens:
        raise GeneratorsNeeded("compose requires at least one polynomial generator")

    all_symbols = set(gens) | set().union(*source_symbols)
    for state in states:
        if state is not None:
            all_symbols.update(state[0])
    for variable in all_symbols:
        if type(variable) is not Symbol and type(variable) is not Dummy:
            raise GeneratorsError("compose requires exact typed polynomial generators")
        assumptions = object.__getattribute__(variable, "_assumptions")
        if assumptions.get("commutative", True) is not True:
            raise PolynomialError("noncommutative polynomial composition is not supported")
    unchanged()
    snapshot = _IdealQuotientSnapshot(
        GroebnerBasis(list(originals), gens, "QQ", "lex"), *bases,
    )
    _snapshot_symbols(snapshot)
    sources, gens, equations, bindings = snapshot.inputs()[0]
    unchanged()
    requested_domain = domain
    if options.get("field") is True:
        requested_domain = "QQ"
    elif domain is None and carry_ring and any(state is not None and state[1] == "QQ" for state in states):
        requested_domain = "QQ"
    original_axes = [([_symbol_binding(variable) for variable in state[0]]
                      if state is not None else None) for state in states]
    # Even inferred Polys are checked in their original typed axes. QQ admits
    # their exact rational coefficients without deciding the result domain.
    original_domains = [(state[1] or "QQ") if state is not None else None for state in states]
    raw, result_domain = native_entry(
        equations[0], equations[1], bindings, original_axes[0], original_axes[1],
        original_domains[0], original_domains[1], requested_domain or "auto", coordinates,
    )
    snapshot.unchanged()
    unchanged()
    if type(result_domain) is not str or result_domain not in ("ZZ", "QQ"):
        raise RuntimeError("native composition returned an invalid coefficient domain")
    expression = _lift_native_result(raw, *sources, *gens)
    snapshot.unchanged()
    unchanged()
    want_polys = options.get("polys")
    if want_polys is None:
        want_polys = any(state is not None for state in states)
    want_polys = method or want_polys
    if want_polys:
        cls = type(outer) if carry_ring else Poly
        result = object.__new__(cls)
        result._expr = expression
        result._gens = tuple(gens)
        result._domain = result_domain
        result._domain_given = True
        result._extra_syms = frozenset()
    else:
        result = expression
    snapshot.unchanged()
    unchanged()
    return result
