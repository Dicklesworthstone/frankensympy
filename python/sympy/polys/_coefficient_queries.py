"""Exact QQ/ZZ coefficient queries over one retained typed polynomial ring.

The native bridge admits the entire original source before projecting its
sparse term map. Python chooses query coordinates, restores exact coefficients
and preserves the existing separately selected coefficient-parameter surface.
A refusal from the checked map never retries through that legacy surface.
"""

from types import BuiltinFunctionType

from ..core import (
    Add, Basic, Dummy, Integer, Mul, Pow, Symbol, _DUMMY_PREFIX,
    _active_callback_session, _exact_surface_types, _is_closed_function_class,
    _lift_native_result, _native, _native_expr,
    _surface_binding_key, _symbol_binding,
)
from ..core._function_capabilities import _owner_matches, _owner_record
from . import _finitefield as retained
from ._evaluation_owners import require_original_owners
from .polyerrors import CoercionFailed, DomainError, PolynomialError


_NATIVE_TERMS = getattr(_native, "polynomial_terms_native", None)
_NATIVE_GENERIC = getattr(_native, "polynomial_query_generic_source_native", None)
_POLY_OWNERS = ()
_MRO = type.__dict__["__mro__"]
_VALUE_SLOT = Basic.__dict__["_value"]
_SESSION_TYPE = _native.SupervisedCallbackSession
_SESSION_CHARGE = _SESSION_TYPE.charge_nested_steps_atomic
_SESSION_CANCEL = _SESSION_TYPE.cancel
_SESSION_USED = _SESSION_TYPE.consumed_nested_steps
_SESSION_LIMIT = _SESSION_TYPE.max_nested_steps
_SESSION_CANCELLED = _SESSION_TYPE.cancelled
_SESSION_TAINTED = _SESSION_TYPE.unknown_effect_tainted


def register_poly_owners(*classes):
    global _POLY_OWNERS
    if _POLY_OWNERS:
        raise RuntimeError("polynomial query owners were already registered")
    _POLY_OWNERS = tuple((cls, _MRO.__get__(cls), _owner_record(cls)) for cls in classes)


def _interfaces(*, native=True):
    require_original_owners()
    for cls, parents, record in _POLY_OWNERS:
        current = _MRO.__get__(cls)
        if (len(current) != len(parents) or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(cls, record)):
            raise TypeError("coefficient queries require original built-in polynomial owners")
    if native:
        for name, original in (("polynomial_terms_native", _NATIVE_TERMS),
                               ("polynomial_query_generic_source_native", _NATIVE_GENERIC)):
            entry = vars(_native).get(name)
            if (entry is not original or type(entry) is not BuiltinFunctionType
                    or entry.__self__ is not _native or entry.__name__ != name):
                raise TypeError("coefficient queries require the original checked native entry point")


def _current(session):
    if _active_callback_session() is not session:
        raise RuntimeError("coefficient-query callback session changed during evaluation")
    if session is not None and (
        _SESSION_CANCELLED.__get__(session) or _SESSION_TAINTED.__get__(session)
    ):
        raise RuntimeError("cancelled or tainted callback session cannot publish coefficients")


def _charge(session, steps=1):
    _current(session)
    if session is not None:
        _SESSION_CHARGE(session, steps)
        _current(session)


def _order(value):
    if value is None:
        return "lex"
    if type(value) is not str or value not in ("lex", "grlex", "grevlex"):
        raise ValueError("monomial order must be lex, grlex, or grevlex")
    return value


def _state(polynomial):
    _interfaces(native=False)
    state = retained._PolyState(polynomial, require_field=False)
    namespace = state.namespace
    # Poly construction stores an actual symbolic owner. In particular, a
    # later forged string/list must never be parsed or coerced by selection.
    kind = type(state.expression)
    if (not any(kind is allowed for allowed in _exact_surface_types())
            and not _is_closed_function_class(kind)):
        raise TypeError("coefficient queries require an original built-in symbolic source")
    try:
        state.native_source = _VALUE_SLOT.__get__(state.expression, Basic)
    except AttributeError as error:
        raise ValueError("coefficient query source has no original native storage") from error
    if type(state.native_source) is not _native.Expr:
        raise ValueError("coefficient query source has invalid original native storage")
    if not state.gens:
        raise PolynomialError("coefficient queries require a nonempty generator tuple")
    if type(namespace.get("_domain_given")) is not bool:
        raise DomainError("polynomial coefficient-domain declaration is invalid")
    extra = namespace.get("_extra_syms")
    if (type(extra) is not frozenset or len(extra) > 64
            or any(not retained._exact(value, (Symbol, Dummy)) for value in extra)):
        raise DomainError("polynomial coefficient-symbol metadata is invalid")
    domain = namespace.get("_domain")
    if type(domain) is not str or len(domain) > 65_536:
        raise NotImplementedError("coefficient queries require a supported exact coefficient domain")
    extension = ((domain.startswith("ZZ[") or domain.startswith("QQ[")) and domain.endswith("]")) or (
        (domain.startswith("ZZ(") or domain.startswith("QQ(")) and domain.endswith(")")
    )
    legacy_domain = (extension or domain in ("RR", "CC", "EX", "ZZ_I", "QQ_I")
                     or domain.startswith("QQ<") and domain.endswith(">"))
    if domain not in ("QQ", "ZZ") and not legacy_domain and not retained.is_finite(polynomial):
        raise NotImplementedError("checked coefficient queries support only ZZ and QQ")
    return state, domain, legacy_domain


def _state_unchanged(state):
    state.unchanged()
    if _VALUE_SLOT.__get__(state.expression, Basic) is not state.native_source:
        raise RuntimeError("polynomial source mutated during coefficient selection")


def _generator_keys(state):
    keys = set()
    declarations = []
    for variable in state.gens:
        native = _VALUE_SLOT.__get__(variable, Basic)
        assumptions = Symbol.__dict__["_assumptions"].__get__(variable, Symbol)
        if (type(native) is not _native.Expr or not native.is_symbol
                or type(assumptions) is not dict or len(assumptions) > 128
                or any(type(key) is not str or len(key) > 4096 or type(value) is not bool
                       for key, value in assumptions.items())):
            raise ValueError("coefficient query generator has invalid original storage")
        if assumptions.get("commutative") is False:
            raise PolynomialError("coefficient queries require commutative generators")
        declarations.append((variable, native, assumptions, tuple(assumptions.items())))
        _native.ideal_source_preflight_native(native)
        key = _surface_binding_key(_symbol_binding(variable))
        if _surface_binding_key(native.symbol_binding) != key:
            raise ValueError("coefficient query generator identity disagrees with its assumptions")
        if type(variable) is Dummy:
            _check_dummy(variable, native)
        keys.add(key)
    _interfaces(native=False)
    _state_unchanged(state)
    for variable, native, assumptions, facts in declarations:
        current = Symbol.__dict__["_assumptions"].__get__(variable, Symbol)
        if (_VALUE_SLOT.__get__(variable, Basic) is not native or current is not assumptions
                or len(current) != len(facts)
                or any(current.get(key) is not value for key, value in facts)):
            raise RuntimeError("coefficient query generator mutated during selection")
    return keys


def _legacy_selected(state, legacy_domain):
    # This selection precedes polynomial admission. Existing parameter and
    # inferred nonrational coefficient surfaces retain their established path.
    # An explicit QQ/ZZ declaration is always checked in its stated ring.
    if legacy_domain:
        return True
    if state.namespace["_domain_given"]:
        return False
    source = state.expression
    native = state.native_source
    # A borrowed syntax probe bounds the whole tree and recognizes existing
    # generic coefficient surfaces before QQ polynomial admission. It never
    # normalizes, expands, or treats an operational failure as a domain switch.
    _interfaces()
    generic = _NATIVE_GENERIC(native)
    if _native_expr(source) is not native:
        raise RuntimeError("polynomial source changed during coefficient selection")
    _interfaces(native=False)
    _state_unchanged(state)
    generators = _generator_keys(state)
    parameter = any(_surface_binding_key(binding) not in generators
                    for binding in native.free_symbol_bindings)
    _interfaces(native=False)
    _state_unchanged(state)
    return bool(state.namespace["_extra_syms"]) or parameter or generic


def _legacy_terms(expression, generators):
    """The pre-existing coefficient-parameter term extraction."""
    from ..core import expand

    gens = list(generators)
    expression = expand(expression)
    accumulated = {}
    for term in (expression.args if isinstance(expression, Add) else (expression,)):
        if term == 0:
            continue
        powers = [0] * len(gens)
        coefficient = Integer(1)
        for factor in (term.args if isinstance(term, Mul) else (term,)):
            base, exponent = factor.args if isinstance(factor, Pow) else (factor, Integer(1))
            if base in gens and isinstance(exponent, Integer) and exponent.p > 0:
                powers[gens.index(base)] += int(exponent.p)
            else:
                coefficient = coefficient * factor
        key = tuple(powers)
        accumulated[key] = accumulated.get(key, Integer(0)) + coefficient
    result = [(powers, coefficient) for powers, coefficient in accumulated.items()
              if coefficient != 0]
    result.sort(key=lambda term: term[0], reverse=True)
    return result or [((0,) * len(gens), Integer(0))]


def _check_dummy(node, native):
    public_name = Dummy.__dict__["_dummy_name"].__get__(node, Dummy)
    number = Dummy.__dict__["_dummy_number"].__get__(node, Dummy)
    if (type(public_name) is not str or len(public_name) > 65_536
            or (type(number) is not int and type(number) is not bool)
            or number.bit_length() > 4096):
        raise ValueError("coefficient query requires a bounded original Dummy identity")
    if native.symbol_binding.name != f"{_DUMMY_PREFIX}_{int(number)}_{public_name}":
        raise ValueError("coefficient query Dummy identity disagrees with its native source")


def _check_dummies(request):
    # Dummy public names and indices are separate slots from their immutable
    # native identity. Bind both, including unused axes and selector atoms.
    seen = set()
    for node, owner, name, _, native in request.snapshot.slots:
        if (owner is Basic and name == "_value" and type(node) is Dummy
                and id(node) not in seen):
            seen.add(id(node))
            _check_dummy(node, native)


def _project(source, bindings, domain, order, session):
    _interfaces()
    try:
        return _NATIVE_TERMS(source, bindings, domain, order, session=session)
    except _native.PolynomialQueryCoefficientError as error:
        raise CoercionFailed(str(error)) from error


def _coordinates(values, count):
    if len(values) != count:
        raise ValueError("exponent of each generator must be specified")
    if any(not retained._exact(value, retained._INTEGER_TYPES) for value in values):
        raise CoercionFailed("monomial exponents must be exact integers")
    powers = tuple(retained._integer(value, "monomial exponent") for value in values)
    if any(exponent < 0 for exponent in powers):
        raise IndexError("monomial exponents must be nonnegative")
    return powers


def _query_result(operation, rows, count, coordinates, session):
    if operation == "terms":
        return rows
    if operation == "coeffs":
        return [coefficient for _, coefficient in rows]
    if operation == "monoms":
        return [powers for powers, _ in rows]
    if operation == "LC":
        return rows[0][1]
    if operation == "EC":
        return rows[-1][1]
    if operation == "TC":
        coordinates = (0,) * count
    if operation == "all_coeffs":
        if count != 1:
            raise PolynomialError("multivariate polynomials not supported")
        degree = rows[0][0][0]
        _charge(session, degree + 1)
        result = [Integer(0)] * (degree + 1)
        for powers, coefficient in rows:
            result[degree - powers[0]] = coefficient
        return result
    return next((coefficient for powers, coefficient in rows if powers == coordinates), Integer(0))


def query(polynomial, operation, argument=None, order=None):
    """Read a coefficient view while retaining source and query controls."""
    session = _active_callback_session()
    if session is not None and type(session) is not _SESSION_TYPE:
        raise TypeError("coefficient queries require an original SupervisedCallbackSession")
    try:
        _current(session)
        return _query(polynomial, operation, argument, _order(order), session)
    except BaseException:
        if (session is not None
                and _SESSION_USED.__get__(session) >= _SESSION_LIMIT.__get__(session)):
            _SESSION_CANCEL(session)
        raise


def _query(polynomial, operation, argument, order, session):
    state, domain, legacy_domain = _state(polynomial)
    if operation == "all_coeffs" and len(state.gens) != 1:
        raise PolynomialError("multivariate polynomials not supported")
    selector = operation == "coeff_monomial"
    container = selector and retained._exact(argument, (tuple, list))
    coordinates = argument if operation == "nth" or container else None
    if coordinates is not None:
        if not retained._exact(coordinates, (tuple, list)) or len(coordinates) > 8:
            raise ValueError("exponent of each generator must be specified")
        if len(coordinates) != len(state.gens):
            raise ValueError("exponent of each generator must be specified")
        if any(not retained._exact(value, retained._INTEGER_TYPES) for value in coordinates):
            raise CoercionFailed("monomial exponents must be exact integers")
    elif selector and not isinstance(argument, Basic) and type(argument) is not int:
        raise ValueError("coefficient selector must be an exact monomial or exponent tuple")

    legacy = _legacy_selected(state, legacy_domain)
    if legacy:
        if order != "lex":
            raise NotImplementedError("parameter coefficient queries currently support lex order")
        # This is the existing unverified coefficient surface, selected from
        # the original declaration before attempting the checked QQ map.
        _charge(session)
        _state_unchanged(state)
        rows = _legacy_terms(state.expression, state.gens)
        if coordinates is not None:
            powers = _coordinates(coordinates, len(state.gens))
        elif selector:
            request = retained._Request([argument], state.gens, (state,))
            _check_dummies(request)
            raw = _project(request.raw[0], request.bindings, "QQ", "lex", session)
            request.unchanged()
            if len(raw) != 1:
                raise ValueError("expected a monomial with coefficient one")
            monomial, coefficient = raw[0]
            if not coefficient.is_integer or coefficient.exact_numerator() != 1:
                raise ValueError("expected a monomial with coefficient one")
            powers = tuple(monomial)
        else:
            powers = None
        result = _query_result(operation, rows, len(state.gens), powers, session)
        _charge(session, len(rows) + 1)
        _interfaces(native=selector)
        _current(session)
        _state_unchanged(state)
        return result

    _interfaces()
    _state_unchanged(state)
    controls = list(coordinates) if coordinates is not None else ([argument] if selector else [])
    containers = [coordinates] if coordinates is not None else []
    original = retained._Retained([state.expression, *controls], state.gens, (state,), containers)
    # Retain controls before the first request charge can reenter Python.
    _charge(session)
    original.unchanged()
    request = retained._Request(
        [state.expression, *([argument] if selector and not container else [])],
        state.gens, (state,),
    )
    _check_dummies(request)

    def unchanged():
        _charge(session, len(original.snapshot.slots) + len(request.snapshot.slots) + 1)
        _interfaces()
        _current(session)
        original.unchanged()
        request.unchanged()

    unchanged()
    powers = _coordinates(coordinates, len(state.gens)) if coordinates is not None else None
    unchanged()
    if selector and not container:
        monomial_rows = _project(request.raw[1], request.bindings, "QQ", "lex", session)
        unchanged()
        if len(monomial_rows) != 1:
            raise ValueError("expected a monomial with coefficient one")
        monomial, coefficient = monomial_rows[0]
        if not coefficient.is_integer or coefficient.exact_numerator() != 1:
            raise ValueError("expected a monomial with coefficient one")
        powers = tuple(monomial)
    if retained.is_finite(polynomial):
        # Existing field methods dispatch before this helper. The newly
        # exposed coeff_monomial uses their original finite-field projection
        # after checking its selector as a coefficient-one monomial over QQ.
        rows = retained.terms(polynomial, order)
    else:
        raw = _project(request.raw[0], request.bindings,
                       domain if state.namespace["_domain_given"] else "auto", order, session)
        unchanged()
        rows = [(tuple(monomial), _lift_native_result(coefficient, *request.context))
                for monomial, coefficient in raw]
    rows = rows or [((0,) * len(state.gens), Integer(0))]
    unchanged()
    result = _query_result(operation, rows, len(state.gens), powers, session)
    unchanged()
    return result

