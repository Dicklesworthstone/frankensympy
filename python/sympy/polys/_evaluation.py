"""Exact QQ/ZZ polynomial evaluation retaining the original typed source.

The native lane admits the whole polynomial in its declared ring and checks
every projected coefficient independently. Python only resolves coordinate
selectors, preserves the coefficient-domain policy and restores scalar owners.
Preselected nonrational scalar substitutions retain the earlier shell path;
they do not acquire the native coefficient-map receipt.
"""

from ..core import (
    Basic, Dummy, Half, Integer, NegativeOne, One, Rational, Symbol, Zero,
    _fold_floats, _lift_native_result, _native, _native_expr,
)
from . import _finitefield as retained
from ._evaluation_owners import require_original_owners
from .polyerrors import CoercionFailed, DomainError, GeneratorsError, UnificationFailed


_RATIONAL_TYPES = (int, Integer, Rational, Zero, One, NegativeOne, Half)
_NO_LEGACY_RESULT = object()


def _legacy_scalar(state, args, auto, *, inspect_parameters=False):
    """Keep the existing scalar shell route, selected before native admission.

    Symbolic and floating substitutions, and existing coefficient-parameter
    sources, previously returned the ordinary substitution result. Preserve
    that behavior without presenting it as a checked QQ polynomial map. A
    refusal from the checked lane never reaches this function as a fallback.
    """
    if len(args) not in (1, 2):
        return _NO_LEGACY_RESULT
    value = args[-1]
    if retained._exact(value, (tuple, list, dict)):
        return _NO_LEGACY_RESULT
    exact_value = retained._exact(value, _RATIONAL_TYPES)
    if not exact_value and type(value) is not float and not isinstance(value, Basic):
        return _NO_LEGACY_RESULT
    domain = state.namespace.get("_domain")
    explicit = state.namespace.get("_domain_given") is True
    explicit_extension = (explicit and type(domain) is str and domain not in ("ZZ", "QQ"))
    carried_extension = len(state.namespace["_extra_syms"]) != 0
    string_selector = len(args) == 2 and type(args[0]) is str
    if (exact_value and not inspect_parameters and not explicit_extension
            and not carried_extension and not string_selector):
        # Checked rational requests must retain every scalar/control before
        # even source conversion. Parameter detection occurs after that capture.
        return _NO_LEGACY_RESULT
    # The eager owner record protects builtin equality/hash and the source
    # protocols; _native_expr additionally refuses opaque/custom classes.
    _native_expr(state.expression)
    require_original_owners()
    state.unchanged()
    parameter_source = (inspect_parameters and not explicit
                        and bool(state.expression.free_symbols - set(state.gens)))
    state.unchanged()
    if (exact_value and not parameter_source and not carried_extension
            and not explicit_extension and not string_selector):
        return _NO_LEGACY_RESULT
    _native_expr(value)
    require_original_owners()
    state.unchanged()
    if not auto and not exact_value:
        raise DomainError("nonrational scalar evaluation requires coefficient-domain extension")
    selector = 0 if len(args) == 1 else args[0]
    if type(selector) is int:
        if not -len(state.gens) <= selector < len(state.gens):
            raise GeneratorsError("generator is not in this polynomial")
        variable = state.gens[selector]
    else:
        if type(selector) is str and len(selector) > 65_536:
            raise GeneratorsError("generator name exceeds the evaluation envelope")
        # A string denotes the ordinary Symbol with that name. Compare its
        # complete typed identity: a same-name assumed Symbol or Dummy is not
        # silently selected by spelling alone.
        variable = Symbol(selector) if type(selector) is str else selector
        if not retained._exact(variable, (Symbol, Dummy)) or variable not in state.gens:
            raise GeneratorsError("generator is not in this polynomial")
    state.unchanged()
    result = state.expression.subs(variable, value)
    require_original_owners()
    state.unchanged()
    # Native substitution represents Float atoms exactly but leaves numeric
    # powers unevaluated. Apply the ordinary shell's Float arithmetic without
    # approximating symbolic coefficients or exact-only parts of the result.
    result = _fold_floats(result)
    require_original_owners()
    state.unchanged()
    return result


def evaluate(polynomial, args, auto):
    """Resolve sequential public selectors, then contract once in the kernel."""
    if type(auto) is not bool:
        raise TypeError("auto must be a bool for checked polynomial evaluation")
    require_original_owners()
    state = retained._PolyState(polynomial, require_field=False)
    namespace = state.namespace
    if type(namespace.get("_domain_given")) is not bool:
        raise DomainError("polynomial coefficient-domain declaration is invalid")
    extra = namespace.get("_extra_syms")
    if (type(extra) is not frozenset or len(extra) > 64
            or any(not retained._exact(value, (Symbol, Dummy)) for value in extra)):
        raise DomainError("polynomial coefficient-symbol metadata is invalid")
    legacy = _legacy_scalar(state, args, auto)
    if legacy is not _NO_LEGACY_RESULT:
        return legacy
    domain = namespace.get("_domain")
    if type(domain) is not str or domain not in ("QQ", "ZZ"):
        raise NotImplementedError("checked polynomial evaluation supports only ZZ and QQ")
    if extra:
        raise NotImplementedError("checked polynomial evaluation requires rational coefficients")
    domain = domain if namespace["_domain_given"] else "auto"
    containers = [args]
    mapping = None
    if len(args) == 1:
        supplied = args[0]
        if type(supplied) is dict:
            if len(supplied) > len(state.gens):
                raise ValueError("too many evaluation values provided")
            mapping = supplied
            pairs = tuple(supplied.items())
        elif retained._exact(supplied, (tuple, list)):
            if len(supplied) > len(state.gens):
                raise ValueError("too many evaluation values provided")
            containers.append(supplied)
            pairs = tuple(zip(state.gens, supplied))
        else:
            pairs = ((0, supplied),)
    elif len(args) == 2:
        pairs = (args,)
    else:
        raise TypeError("eval takes one value or a generator and value")
    for variable, value in pairs:
        if not retained._exact(variable, (int, Symbol, Dummy)):
            raise GeneratorsError("generator is not in this polynomial")
        if not retained._exact(value, _RATIONAL_TYPES):
            raise UnificationFailed("checked polynomial evaluation requires an exact rational value")
    controls = [item for pair in pairs for item in pair]
    original = retained._Retained([state.expression, *controls], state.gens, (state,), containers)

    def unchanged():
        require_original_owners()
        original.unchanged()
        if mapping is not None:
            if len(mapping) != len(pairs):
                raise RuntimeError("polynomial evaluation mapping mutated during exact evaluation")
            current = tuple(mapping.items())
            if any(key is not old_key or value is not old_value
                   for (key, value), (old_key, old_value) in zip(current, pairs)):
                raise RuntimeError("polynomial evaluation mapping mutated during exact evaluation")

    unchanged()
    request = retained._Request([state.expression], state.gens, (state,))
    unchanged()
    legacy = _legacy_scalar(state, args, auto, inspect_parameters=True)
    if legacy is not _NO_LEGACY_RESULT:
        request.unchanged()
        unchanged()
        return legacy
    remaining = list(range(len(state.gens)))
    values = [None] * len(state.gens)
    for variable, value in pairs:
        if type(variable) is int:
            if not -len(remaining) <= variable < len(remaining):
                raise GeneratorsError("generator is not in this polynomial")
            index = remaining[variable]
        else:
            index = retained._check_generator(state, variable)
            if index not in remaining:
                raise GeneratorsError("generator is not in this polynomial")
        # Exact-class admission and the retained storage snapshot precede this
        # conversion. Native code bounds the borrowed scalar before copying it.
        values[index] = _native_expr(value)
        remaining.remove(index)
        unchanged()
    try:
        raw, result_domain = _native.polynomial_evaluate_native(
            request.raw[0], request.bindings, values, domain, auto)
    except _native.PolynomialEvaluationCoefficientError as error:
        unchanged()
        raise CoercionFailed(str(error)) from error
    except _native.PolynomialEvaluationDomainError as error:
        unchanged()
        raise DomainError(str(error)) from error
    request.unchanged()
    unchanged()
    if type(result_domain) is not str or result_domain not in ("ZZ", "QQ"):
        raise RuntimeError("checked polynomial evaluation returned an invalid coefficient domain")
    if not pairs:
        # Empty mappings/sequences retain public identity only after native
        # source admission and independent identity verification.
        return polynomial
    expression = _lift_native_result(raw, *request.context, *controls)
    request.unchanged()
    unchanged()
    if remaining:
        # Exact built-in result classes have no user constructor to invoke.
        # Preserve unused axes and the admitted domain, including explicit QQ
        # or a rational assignment that promoted an otherwise integral source.
        result = object.__new__(type(polynomial))
        result_namespace = object.__getattribute__(result, "__dict__")
        if type(result_namespace) is not dict:
            raise TypeError("polynomial evaluation requires original result storage")
        result_namespace.update({
            "_expr": expression,
            "_gens": tuple(state.gens[index] for index in remaining),
            "_domain": result_domain,
            "_domain_given": True,
            "_extra_syms": frozenset(),
        })
    else:
        result = expression
    request.unchanged()
    unchanged()
    return result
