"""Bounded prime-field Poly dispatch and source-preserving publication.

Python retains the public domain, typed generators and original sources. All
polynomial arithmetic, coefficient reduction and factorization is performed
by the checked native prime-field kernels; this module never retries over QQ.
"""

from __future__ import annotations

from ..core import (
    Basic, Dummy, Integer, NegativeOne, One, Symbol, Zero,
    _lift_native_result, _native, _native_expr, _surface_binding_key,
    _symbol_binding,
)
from .polyerrors import (
    CoercionFailed, DomainError, GeneratorsError, GeneratorsNeeded,
    NotInvertible, PolynomialError, UnificationFailed,
)


_ABSENT = object()
_INTEGER_TYPES = (int, Integer, Zero, One, NegativeOne)


def _poly_types():
    from .polytools import Poly, PurePoly
    return Poly, PurePoly


def _exact(value, classes):
    """Type identity never invokes a user-defined metaclass comparison."""
    return any(type(value) is cls for cls in classes)


def is_finite(value):
    """Recognize the field marker without calling a public Poly protocol."""
    if not _exact(value, _poly_types()):
        return False
    namespace = object.__getattribute__(value, "__dict__")
    if (type(namespace) is not dict or len(namespace) > 32
            or any(type(key) is not str or len(key) > 128 for key in namespace)):
        raise TypeError("prime-field Poly requires bounded built-in source storage")
    domain = namespace.get("_domain")
    return ("_modulus" in namespace or
            (type(domain) is str and domain.startswith("GF(")))


def requested(source, options):
    domain = options.get("domain")
    return (options.get("modulus") is not None or is_finite(source) or
            (type(domain) is str and domain.startswith("GF(")))


def _options(options, allowed):
    for name in options:
        if name not in allowed:
            raise NotImplementedError("prime-field Poly does not support option %r" % name)
    if "polys" in options and type(options["polys"]) is not bool:
        raise TypeError("polys must be a bool")


def _integer(value, label):
    if not _exact(value, _INTEGER_TYPES):
        raise CoercionFailed("%s must be an exact integer" % label)
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError("prime-field integer exceeds 4096 bits")
        return value
    # Read the retained immutable native integer, not an overridable public
    # __int__/p/q protocol. Callers retain the complete request before this
    # bounded native preflight or any coefficient materialization.
    native = _native_expr(value)
    _native.prime_field_source_preflight_native(native)
    if not native.is_integer:
        raise CoercionFailed("%s must have an exact integer representation" % label)
    return native.exact_numerator()


def _prime(value):
    result = _integer(value, "modulus")
    if not 2 <= result <= 65521:
        raise ValueError("prime-field Poly requires a prime in 2..65521")
    return result


def _domain_prime(domain):
    if type(domain) is not str or len(domain) > 32:
        raise DomainError("prime-field Poly requires domain='GF(p)'")
    import re
    match = re.fullmatch(r"GF\(([0-9]{1,5})\)", domain)
    if match is None:
        raise DomainError("prime-field Poly requires domain='GF(p)'")
    return _prime(int(match.group(1)))


class _PolyState:
    """Bounded direct storage snapshot, including the field and its lift policy."""

    def __init__(self, polynomial, require_field=True):
        if not _exact(polynomial, _poly_types()):
            raise TypeError("prime-field operations require exact Poly or PurePoly operands")
        self.polynomial = polynomial
        self.namespace = object.__getattribute__(polynomial, "__dict__")
        if type(self.namespace) is not dict or len(self.namespace) > 32:
            raise TypeError("prime-field Poly requires bounded built-in source storage")
        if any(type(key) is not str or len(key) > 128 for key in self.namespace):
            raise TypeError("prime-field Poly requires bounded built-in source storage")
        if "as_expr" in self.namespace:
            raise TypeError("prime-field operations do not invoke overridden Poly.as_expr")
        self.items = tuple(self.namespace.items())
        self.expression = self.namespace.get("_expr", _ABSENT)
        self.gens = self.namespace.get("_gens", _ABSENT)
        if type(self.gens) is not tuple or not 1 <= len(self.gens) <= 8:
            raise GeneratorsError("prime-field Poly supports one to eight generators")
        if any(not _exact(gen, (Symbol, Dummy)) for gen in self.gens):
            raise GeneratorsError("prime-field Poly requires exact typed Symbol generators")
        self.prime = self.namespace.get("_modulus", _ABSENT)
        self.symmetric = self.namespace.get("_symmetric", _ABSENT)
        if require_field or is_finite(polynomial):
            if type(self.prime) is not int or not 2 <= self.prime <= 65521:
                raise DomainError("prime-field Poly has an invalid modulus")
            if type(self.symmetric) is not bool:
                raise DomainError("prime-field Poly has an invalid symmetric lift policy")
            if (type(self.namespace.get("_domain")) is not str or
                    self.namespace["_domain"] != "GF(%d)" % self.prime or
                    self.namespace.get("_domain_given") is not True or
                    type(self.namespace.get("_extra_syms")) is not frozenset or
                    self.namespace["_extra_syms"]):
                raise DomainError("prime-field Poly domain metadata is inconsistent")

    def unchanged(self):
        current = object.__getattribute__(self.polynomial, "__dict__")
        if (current is not self.namespace or len(current) != len(self.items) or
                any(current.get(key, _ABSENT) is not value for key, value in self.items)):
            raise RuntimeError("prime-field Poly source mutated during exact evaluation")


class _Retained:
    """Retain all source storage before even a scalar coercion/preflight."""

    def __init__(self, expressions, gens, states=(), containers=()):
        from .polytools import GroebnerBasis, _IdealQuotientSnapshot
        if len(expressions) > 129:
            raise ValueError("prime-field request exceeds its bounded source count")
        self.states = tuple(states)
        self.containers = []
        for container in containers:
            if not _exact(container, (tuple, list)) or len(container) > 65:
                raise TypeError("prime-field request requires bounded built-in containers")
            self.containers.append((container, tuple(container)))
        bases = [GroebnerBasis(expressions[i:i + 64], gens, "QQ", "lex")
                 for i in range(0, max(1, len(expressions)), 64)]
        self.snapshot = _IdealQuotientSnapshot(*bases)

    def unchanged(self):
        for container, original in self.containers:
            if len(container) != len(original) or any(a is not b for a, b in zip(container, original)):
                raise RuntimeError("prime-field source container mutated during exact evaluation")
        for state in self.states:
            state.unchanged()
        self.snapshot.unchanged()


class _Request(_Retained):
    """Admit an already retained request and its exact typed source bindings."""

    def __init__(self, expressions, gens, states=()):
        super().__init__(expressions, gens, states)
        # A native symbol binds the assumptions digest, but the commutative
        # sparse ring cannot interpret that opaque digest. Admit the retained
        # declaration itself, including unused generators and held children.
        for _, owner, name, _, facts in self.snapshot.slots:
            if (owner is Symbol and name == "_assumptions"
                    and facts.get("commutative") is False):
                raise GeneratorsError("prime-field polynomials require commutative symbols")
        (data,) = self.snapshot.inputs()
        self.sources, self.gens, self.raw, self.bindings = data
        # A stale public assumption dictionary must not reinterpret an older
        # immutable native atom, even if it was changed before this call.
        seen = set()
        for node, owner, name, _, native in self.snapshot.slots:
            if (owner is Basic and name == "_value" and _exact(node, (Symbol, Dummy))
                    and id(node) not in seen):
                seen.add(id(node))
                if (type(native) is not _native.Expr or not native.is_symbol or
                        _surface_binding_key(native.symbol_binding) !=
                        _surface_binding_key(_symbol_binding(node))):
                    raise ValueError("prime-field source symbol identity disagrees with its assumptions")
        self.unchanged()

    @property
    def context(self):
        return (*self.sources, *self.gens)


def _call(request, prime, symmetric, operation, parameter=0,
          derivative_orders=None, order="lex"):
    try:
        if len(request.gens) > 1 or operation == "reduced":
            if operation not in ("normalize", "neg", "add", "sub", "mul", "monic",
                                  "pow", "diff", "divrem", "reduced"):
                raise NotImplementedError(
                    "%s is not implemented for multivariate prime-field Poly" % operation)
            result = _native.prime_field_multivariate_native(
                request.raw, request.bindings, prime, operation, parameter,
                derivative_orders, order, symmetric)
        else:
            result = _native.prime_field_polynomial_native(
                request.raw[0], request.raw[1] if len(request.raw) == 2 else None,
                request.bindings[0], prime, operation, parameter, symmetric)
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field Poly coefficients must be integers") from error
    request.unchanged()
    return result


def _publish(cls, raw, request, prime, symmetric):
    expression = _lift_native_result(raw, *request.context)
    request.unchanged()
    result = object.__new__(cls)
    result._expr = expression
    result._gens = tuple(request.gens)
    result._domain = "GF(%d)" % prime
    result._domain_given = True
    result._extra_syms = frozenset()
    result._modulus = prime
    result._symmetric = symmetric
    request.unchanged()
    return result


def _generator_tuple(gens, options):
    if gens and options.get("gens") is not None:
        raise GeneratorsError("generators were specified both positionally and by keyword")
    if not gens and options.get("gens") is not None:
        supplied = options["gens"]
        gens = supplied if _exact(supplied, (tuple, list)) else (supplied,)
    if len(gens) == 1 and _exact(gens[0], (tuple, list)):
        gens = gens[0]
    if len(gens) > 8:
        raise GeneratorsError("prime-field Poly supports at most eight generators")
    if any(not _exact(gen, (Symbol, Dummy)) for gen in gens):
        raise GeneratorsError("prime-field Poly requires exact typed Symbol generators")
    return tuple(gens)


def _infer_generators(request):
    """Infer bounded native atoms before the profile's public generator sort."""
    from .polytools import _sort_gens

    raw = _native.prime_field_polynomial_generators_native(
        request.raw[0], request.raw[1] if len(request.raw) == 2 else None)
    request.unchanged()
    generators = [_lift_native_result(value, *request.context) for value in raw]
    request.unchanged()
    result = tuple(_sort_gens(generators))
    request.unchanged()
    return result


def _generator_containers(gens, options):
    result = [gens]
    if len(gens) == 1 and _exact(gens[0], (tuple, list)):
        result.append(gens[0])
    supplied = options.get("gens")
    if _exact(supplied, (tuple, list)):
        result.append(supplied)
    return result


def construct(cls, source, gens, options):
    _options(options, {"modulus", "domain", "symmetric", "gens"})
    if not any(cls is admitted for admitted in _poly_types()):
        raise TypeError("prime-field construction requires exact Poly or PurePoly")
    state = _PolyState(source, require_field=False) if _exact(source, _poly_types()) else None
    field_source = state is not None and is_finite(source)
    prime = options.get("modulus")
    generators = _generator_tuple(gens, options)
    expression = state.expression if state is not None else source
    if not generators and state is not None:
        generators = state.gens
    scalars = [] if prime is None else [prime]
    if prime is not None and not _exact(prime, _INTEGER_TYPES):
        raise CoercionFailed("modulus must be an exact integer")
    retained = _Retained([expression, *scalars], generators, (state,) if state else (),
                         _generator_containers(gens, options))
    domain = options.get("domain")
    if domain is not None:
        domain_prime = _domain_prime(domain)
        if prime is not None and _prime(prime) != domain_prime:
            raise DomainError("modulus and domain specify different prime fields")
        prime = domain_prime
    if prime is None:
        if not field_source:
            raise DomainError("prime-field Poly requires an explicit modulus")
        prime = state.prime
    prime = _prime(prime)
    retained.unchanged()
    symmetric = options.get("symmetric", state.symmetric if field_source else True)
    if type(symmetric) is not bool:
        raise TypeError("symmetric must be a bool")
    initial = None
    if not generators:
        initial = _Request([expression], (), (state,) if state else ())
        generators = _infer_generators(initial)
        if not generators:
            raise GeneratorsNeeded("a constant prime-field Poly requires an explicit generator")
    request = _Request([expression], generators, (state,) if state else ())
    raw = _call(request, prime, symmetric, "normalize")
    result = _publish(cls, raw[0], request, prime, symmetric)
    if initial is not None:
        initial.unchanged()
    retained.unchanged()
    return result


def from_list(cls, coefficients, gens, options):
    """Build a retained held polynomial without dropping zero coefficients."""
    from ..core import Add, Mul, Pow
    if not _exact(coefficients, (list, tuple)):
        raise TypeError("prime-field from_list requires a built-in list or tuple")
    if len(coefficients) > 65:
        raise ValueError("prime-field coefficient list exceeds degree 64")
    original = tuple(coefficients)
    generators = _generator_tuple((), {"gens": gens})
    if not generators:
        raise GeneratorsNeeded("from_list requires an explicit generator")
    if len(generators) != 1:
        raise NotImplementedError("multivariate prime-field from_list is not implemented")
    generator = generators[0]
    if any(not _exact(coefficient, _INTEGER_TYPES) for coefficient in original):
        raise CoercionFailed("prime-field coefficients must be exact integers")
    scalars = [options["modulus"]] if options.get("modulus") is not None else []
    if scalars and not _exact(scalars[0], _INTEGER_TYPES):
        raise CoercionFailed("modulus must be an exact integer")
    retained = _Retained([*original, *scalars], generators,
                         containers=[coefficients, *_generator_containers((), {"gens": gens})])
    # Bound and admit each original coefficient before creating held products.
    for coefficient in original:
        _integer(coefficient, "coefficient")
    retained.unchanged()
    terms = [Mul(coefficient, Pow(generator, Integer(len(original) - i - 1), evaluate=False),
                 evaluate=False) for i, coefficient in enumerate(original)]
    source = Add(*terms, evaluate=False) if terms else Integer(0)
    result = construct(cls, source, generators, options)
    retained.unchanged()
    if (len(coefficients) != len(original) or
            any(left is not right for left, right in zip(coefficients, original))):
        raise RuntimeError("prime-field coefficient list mutated during exact evaluation")
    return result


def _operands(polynomial, other=_ABSENT):
    left = _PolyState(polynomial)
    states = [left]
    sources = [left.expression]
    if other is not _ABSENT:
        if _exact(other, _poly_types()):
            right = _PolyState(other, require_field=False)
            if not is_finite(other) or right.prime != left.prime:
                raise UnificationFailed("cannot unify polynomials over different coefficient domains")
            if right.gens != left.gens:
                raise UnificationFailed("prime-field operands must use the same typed generator")
            states.append(right)
            sources.append(right.expression)
        else:
            if not _exact(other, _INTEGER_TYPES):
                raise UnificationFailed("prime-field arithmetic requires a same-field Poly or exact integer")
            sources.append(other)
    return left, _Request(sources, left.gens, states)


def operation(polynomial, name, other=_ABSENT, parameter=0, reverse=False,
              derivative_orders=None):
    state, request = _operands(polynomial, other)
    if reverse:
        request.raw.reverse()
    raw = _call(request, state.prime, state.symmetric, name, parameter, derivative_orders)
    if raw is None:
        raise NotInvertible("zero divisor")
    result = tuple(_publish(type(polynomial), value, request, state.prime, state.symmetric)
                   for value in raw)
    request.unchanged()
    return result if name in ("divrem", "gcdex") else result[0]


def scalar_operation(polynomial, name, other=_ABSENT):
    state, request = _operands(polynomial, other)
    raw = _call(request, state.prime, state.symmetric, name)
    result = _lift_native_result(raw[0], *request.context)
    request.unchanged()
    return result


def expression(polynomial):
    state, request = _operands(polynomial)
    raw = _call(request, state.prime, state.symmetric, "normalize")
    result = _lift_native_result(raw[0], *request.context)
    request.unchanged()
    return result


def coefficients(polynomial):
    state, request = _operands(polynomial)
    if len(state.gens) != 1:
        raise PolynomialError("multivariate polynomials not supported")
    raw = _call(request, state.prime, state.symmetric, "normalize")
    # Native normalization has certified an integer lift with degree <= 64.
    # The existing coefficient projection now reads that bounded canonical
    # result; it is not an alternative polynomial arithmetic lane.
    values = _native.poly_coeffs_native(raw[0], request.bindings[0])
    result = [_lift_native_result(value, *request.context) for value in values]
    request.unchanged()
    return result or [Integer(0)]


def _check_generator(state, gen):
    if type(gen) is int:
        if not -len(state.gens) <= gen < len(state.gens):
            raise GeneratorsError("generator is not in this polynomial")
        return gen % len(state.gens)
    if not _exact(gen, (Symbol, Dummy)) or gen not in state.gens:
        raise GeneratorsError("generator is not in this polynomial")
    native = _native_expr(gen)
    _native.prime_field_source_preflight_native(native)
    if (not native.is_symbol or _surface_binding_key(native.symbol_binding) !=
            _surface_binding_key(_symbol_binding(gen))):
        raise ValueError("prime-field generator identity disagrees with its assumptions")
    return state.gens.index(gen)


def degree(polynomial, gen=0):
    state = _PolyState(polynomial)
    if not _exact(gen, (int, Symbol, Dummy)):
        raise GeneratorsError("generator is not in this polynomial")
    retained = _Retained([state.expression, gen], state.gens, (state,))
    index = _check_generator(state, gen)
    if len(state.gens) > 1:
        values = terms(polynomial)
        retained.unchanged()
        if len(values) == 1 and values[0][1] == 0:
            from ..core import S
            return S.NegativeInfinity
        return max(powers[index] for powers, _ in values)
    values = coefficients(polynomial)
    retained.unchanged()
    if len(values) == 1 and values[0] == 0:
        from ..core import S
        return S.NegativeInfinity
    return len(values) - 1


def total_degree(polynomial):
    state = _PolyState(polynomial)
    if len(state.gens) == 1:
        result = degree(polynomial)
        from ..core import S
        return 0 if result is S.NegativeInfinity else result
    return max(sum(powers) for powers, _ in terms(polynomial))


def degree_list(polynomial):
    state = _PolyState(polynomial)
    values = terms(polynomial)
    state.unchanged()
    if len(values) == 1 and values[0][1] == 0:
        from ..core import S
        return (S.NegativeInfinity,) * len(state.gens)
    return tuple(max(powers[index] for powers, _ in values)
                 for index in range(len(state.gens)))


def _monomial_order(order):
    if order is None:
        return "lex"
    if type(order) is not str or order not in ("lex", "grlex", "grevlex"):
        raise ValueError("prime-field monomial order must be lex, grlex, or grevlex")
    return order


def terms(polynomial, order=None):
    order = _monomial_order(order)
    state = _PolyState(polynomial)
    if len(state.gens) > 1:
        _, request = _operands(polynomial)
        try:
            raw = _native.prime_field_multivariate_terms_native(
                request.raw[0], request.bindings, state.prime, state.symmetric, order)
        except _native.PrimeFieldCoefficientError as error:
            raise CoercionFailed("prime-field Poly coefficients must be integers") from error
        request.unchanged()
        result = [(tuple(powers), _lift_native_result(value, *request.context))
                  for powers, value in raw]
        request.unchanged()
        return result or [((0,) * len(state.gens), Integer(0))]
    values = coefficients(polynomial)
    result = [((len(values) - i - 1,), value)
              for i, value in enumerate(values) if value != 0]
    return result or [((0,), Integer(0))]


def nth(polynomial, coordinates):
    state = _PolyState(polynomial)
    if len(coordinates) != len(state.gens):
        raise GeneratorsError("prime-field nth requires one exponent per generator")
    if any(not _exact(coordinate, _INTEGER_TYPES) for coordinate in coordinates):
        raise CoercionFailed("monomial exponent must be an exact integer")
    retained = _Retained([state.expression, *coordinates], state.gens, (state,), [coordinates])
    powers = tuple(_integer(coordinate, "monomial exponent") for coordinate in coordinates)
    retained.unchanged()
    if any(exponent < 0 for exponent in powers):
        raise IndexError("monomial exponent must be nonnegative")
    if len(state.gens) > 1:
        result = next((coefficient for monomial, coefficient in terms(polynomial)
                       if monomial == powers), Integer(0))
        retained.unchanged()
        return result
    exponent = powers[0]
    values = coefficients(polynomial)
    result = values[-exponent - 1] if exponent < len(values) else Integer(0)
    retained.unchanged()
    return result


def trailing_coefficient(polynomial):
    state = _PolyState(polynomial)
    return nth(polynomial, (0,) * len(state.gens))


def is_constant(polynomial, value):
    state = _PolyState(polynomial)
    result = terms(polynomial) == [((0,) * len(state.gens), Integer(value))]
    state.unchanged()
    return result


def derivative(polynomial, specs):
    state = _PolyState(polynomial)
    if len(specs) > 64:
        raise ValueError("prime-field derivative accepts at most 64 specifications")
    controls = []
    containers = [specs]
    for spec in specs:
        if _exact(spec, (tuple, list)):
            if len(spec) != 2:
                raise GeneratorsError("a derivative specification must be (generator, order)")
            if not _exact(spec[0], (int, Symbol, Dummy)):
                raise GeneratorsError("generator is not in this polynomial")
            if not _exact(spec[1], _INTEGER_TYPES):
                raise CoercionFailed("derivative order must be an exact integer")
            containers.append(spec)
            controls.extend(spec)
        else:
            if not _exact(spec, (int, Symbol, Dummy)):
                raise GeneratorsError("generator is not in this polynomial")
            controls.append(spec)
    retained = _Retained([state.expression, *controls], state.gens, (state,), containers)
    orders = [0] * len(state.gens)
    for spec in specs or (state.gens[0],):
        if _exact(spec, (tuple, list)):
            if len(spec) != 2:
                raise GeneratorsError("a derivative specification must be (generator, order)")
            variable, count = spec
        else:
            variable, count = spec, 1
        index = _check_generator(state, variable)
        count = _integer(count, "derivative order")
        if not 0 <= count <= 64 - sum(orders):
            raise ValueError("prime-field derivative order must be in 0..64")
        orders[index] += count
    retained.unchanged()
    result = operation(polynomial, "diff", parameter=orders[0],
                       derivative_orders=orders if len(orders) > 1 else None)
    retained.unchanged()
    return result


def power(polynomial, exponent):
    state = _PolyState(polynomial)
    if not _exact(exponent, _INTEGER_TYPES):
        raise CoercionFailed("polynomial exponent must be an exact integer")
    retained = _Retained([state.expression, exponent], state.gens, (state,))
    exponent = _integer(exponent, "polynomial exponent")
    retained.unchanged()
    if not 0 <= exponent < 2**64:
        raise ValueError("polynomial exponent must be a nonnegative 64-bit integer")
    result = operation(polynomial, "pow", parameter=exponent)
    retained.unchanged()
    return result


def evaluate(polynomial, args):
    state = _PolyState(polynomial)
    if len(state.gens) > 1:
        return unsupported(polynomial, "multivariate evaluation")
    variable = None
    if len(args) == 1:
        value = args[0]
    elif len(args) == 2:
        variable, value = args
        if not _exact(variable, (int, Symbol, Dummy)):
            raise GeneratorsError("generator is not in this polynomial")
    else:
        raise TypeError("eval takes one value or a generator and value")
    if not _exact(value, _INTEGER_TYPES):
        raise UnificationFailed("prime-field evaluation requires an exact integer")
    retained = _Retained([state.expression, *args], state.gens, (state,), [args])
    if variable is not None:
        _check_generator(state, variable)
    result = operation(polynomial, "compose", value)
    retained.unchanged()
    return object.__getattribute__(result, "__dict__")["_expr"]


def shift(polynomial, value):
    # Admit the original scalar before constructing the linear composition.
    # It contains no potentially simplifying symbolic source.
    state, request = _operands(polynomial, value)
    if len(state.gens) > 1:
        return unsupported(polynomial, "multivariate shift")
    offset = _integer(value, "shift")
    request.unchanged()
    from ..core import Add
    linear = Add(state.gens[0], Integer(offset), evaluate=False)
    request.unchanged()
    shifted = _Request([state.expression, linear], state.gens, (state,))
    raw = _call(shifted, state.prime, state.symmetric, "compose")
    result = _publish(type(polynomial), raw[0], shifted, state.prime, state.symmetric)
    request.unchanged()
    return result


def factors(polynomial, mode="factor", expression=False):
    state, request = _operands(polynomial)
    if len(state.gens) > 1:
        return unsupported(polynomial, "multivariate factorization")
    try:
        unit, factors = _native.prime_field_polynomial_factor_native(
            request.raw[0], request.bindings[0], state.prime, state.symmetric, mode)
    except _native.PrimeFieldCoefficientError as error:
        raise CoercionFailed("prime-field Poly coefficients must be integers") from error
    request.unchanged()
    scale = _lift_native_result(unit, *request.context)
    if expression:
        from ..core import Mul, Pow, _keep_coeff
        terms = []
        for value, multiplicity in factors:
            factor = _lift_native_result(value, *request.context)
            terms.append(factor if multiplicity == 1 else
                         Pow(factor, Integer(multiplicity), evaluate=False))
        result = _keep_coeff(scale, Mul(*terms))
        request.unchanged()
        return result
    result = [(_publish(type(polynomial), value, request, state.prime, state.symmetric), multiplicity)
              for value, multiplicity in factors]
    request.unchanged()
    if mode == "sqf_part":
        return result[0][0] if result else _publish(
            type(polynomial), unit, request, state.prime, state.symmetric)
    return scale, result


def factor_entry(polynomial, gens, prime=None, expression=False):
    state = _PolyState(polynomial)
    generators = _generator_tuple(gens, {})
    if prime is not None and not _exact(prime, _INTEGER_TYPES):
        raise CoercionFailed("modulus must be an exact integer")
    retained = _Retained([state.expression, *([] if prime is None else [prime]), *generators],
                         state.gens, (state,), _generator_containers(gens, {}))
    if prime is not None and _prime(prime) != state.prime:
        raise DomainError("factorization modulus differs from the polynomial's field")
    retained.unchanged()
    if generators:
        if generators != state.gens:
            raise GeneratorsError("factorization generators differ from the polynomial's ring")
        _check_generator(state, generators[0])
    result = factors(polynomial, expression=expression)
    retained.unchanged()
    return result


def modulus(polynomial):
    return _PolyState(polynomial).prime


def binary_entry(left, right, generators, options, name):
    return _entry((left, right), generators, options, name)


def unary_entry(source, generators, options, name):
    return _entry((source,), generators, options, name)


def _entry(values, generators, options, name, containers=()):
    """One checked call for module-level operations with explicit GF options.

    Return ``None`` only when this is not a prime-field request. Raw symbolic
    expressions require explicit domain selection; mixed domain Poly objects
    without that selection refuse instead of silently extending to QQ.
    """
    explicit = requested(None, options)
    if not explicit and not any(is_finite(value) for value in values):
        return None
    allowed = {"modulus", "domain", "symmetric", "gens", "polys"}
    if name == "reduced":
        allowed.add("order")
    _options(options, allowed)
    order = _monomial_order(options.get("order"))
    states = tuple(_PolyState(value, require_field=False)
                   for value in values if _exact(value, _poly_types()))
    inherited = next((state for state in states if is_finite(state.polynomial)), None)
    if not explicit:
        for value in values:
            if not _exact(value, _poly_types()) and not _exact(value, _INTEGER_TYPES):
                raise UnificationFailed("symbolic operands require explicit same-field Poly construction")
    prime = options.get("modulus")
    by_id = {id(state.polynomial): state.expression for state in states}
    sources = [by_id.get(id(value), value) for value in values]
    original_generators = generators
    generators = _generator_tuple(generators, options)
    if not generators and states:
        generators = states[0].gens
    scalars = [] if prime is None else [prime]
    if prime is not None and not _exact(prime, _INTEGER_TYPES):
        raise CoercionFailed("modulus must be an exact integer")
    retained = _Retained([*sources, *scalars], generators, states,
                         [*containers, *_generator_containers(original_generators, options)])
    domain = options.get("domain")
    if domain is not None:
        domain_prime = _domain_prime(domain)
        if prime is not None and _prime(prime) != domain_prime:
            raise DomainError("modulus and domain specify different prime fields")
        prime = domain_prime
    prime = _prime(prime if prime is not None else inherited.prime)
    retained.unchanged()
    symmetric = options.get("symmetric", inherited.symmetric if inherited else True)
    if type(symmetric) is not bool:
        raise TypeError("symmetric must be a bool")
    if not explicit:
        for state in states:
            if not is_finite(state.polynomial) or state.prime != prime:
                raise UnificationFailed("cannot unify polynomials over different coefficient domains")
    initial = None
    if not generators:
        if len(sources) > 2:
            raise GeneratorsNeeded("prime-field reduction requires explicit generators")
        initial = _Request(sources, (), states)
        generators = _infer_generators(initial)
        if not generators:
            raise GeneratorsNeeded("constant prime-field operations require an explicit generator")
    if not explicit and any(state.gens != generators for state in states):
        raise UnificationFailed("prime-field operands must use the same typed generator")
    request = _Request(sources, generators, states)
    raw = _call(request, prime, symmetric, name, order=order)
    if raw is None:
        raise NotInvertible("zero divisor")
    want_polys = name not in ("resultant", "discriminant") and (
        options.get("polys", bool(states)) if name == "reduced" else
        bool(states) or options.get("polys", False))
    cls = type(inherited.polynomial) if inherited else _poly_types()[0]
    if want_polys:
        result = tuple(_publish(cls, value, request, prime, symmetric) for value in raw)
    else:
        result = tuple(_lift_native_result(value, *request.context) for value in raw)
    request.unchanged()
    if initial is not None:
        initial.unchanged()
    retained.unchanged()
    if name == "reduced":
        return list(result[:-1]), result[-1]
    return result if name in ("divrem", "gcdex") else result[0]


def reduced_entry(source, divisors, generators, options):
    """Retain an entire ordered divisor list before any native/scalar work."""
    if not _exact(divisors, (tuple, list)):
        raise TypeError("prime-field reduction requires a built-in divisor list or tuple")
    if len(divisors) > 32:
        raise ValueError("prime-field reduction supports at most 32 divisors")
    original = tuple(divisors)
    values = (source, *original)
    if not requested(None, options) and not any(is_finite(value) for value in values):
        raise NotImplementedError("reduced currently requires a prime-field polynomial domain")
    result = _entry(values, generators, options, "reduced", (divisors,))
    if len(divisors) != len(original) or any(a is not b for a, b in zip(divisors, original)):
        raise RuntimeError("prime-field divisor list mutated during exact evaluation")
    return result


def unsupported(polynomial, operation):
    _PolyState(polynomial)
    raise NotImplementedError("%s is not implemented for prime-field Poly" % operation)


def equal(polynomial, other):
    from ..core import Add, Half, Mul, Pow, Rational
    from .polytools import PurePoly
    left = _PolyState(polynomial)
    if _exact(other, _poly_types()):
        if not is_finite(other):
            return False
        right = _PolyState(other)
        if left.prime != right.prime or len(left.gens) != len(right.gens):
            return False
        if (left.gens != right.gens and type(polynomial) is not PurePoly
                and type(other) is not PurePoly):
            return False
        retained = _Retained([left.expression, right.expression, *left.gens, *right.gens],
                             (), (left, right))
        # Signed and unsigned coefficient lifts denote the same field element.
        result = _residue_terms(polynomial, left.prime) == _residue_terms(other, left.prime)
        retained.unchanged()
        return result
    if not _exact(other, (*_INTEGER_TYPES, Rational, Half, Symbol, Dummy, Add, Mul, Pow)):
        return False
    request = _Request([left.expression, other], left.gens, (left,))
    candidate_generators = _native.prime_field_polynomial_generators_native(request.raw[1])
    request.unchanged()
    bindings = tuple(_surface_binding_key(binding) for binding in request.bindings)
    if any(_surface_binding_key(generator.symbol_binding) not in bindings
           for generator in candidate_generators):
        return False
    try:
        converted = construct(type(polynomial), other, left.gens,
                              {"modulus": left.prime, "symmetric": left.symmetric})
    except (CoercionFailed, GeneratorsError, TypeError):
        request.unchanged()
        return False
    result = equal(polynomial, converted)
    request.unchanged()
    return result


def identity_hash(polynomial):
    state = _PolyState(polynomial)
    retained = _Retained([state.expression], state.gens, (state,))
    values = _residue_terms(polynomial, state.prime)
    # PurePoly compares by coefficients even against Poly with another name.
    # Omitting names for both classes keeps equal public objects hash-equal.
    result = hash(("GFPoly", state.prime, len(state.gens), values))
    retained.unchanged()
    return result


def _residue_terms(polynomial, prime):
    return tuple((powers, _native_expr(coefficient).exact_numerator() % prime)
                 for powers, coefficient in terms(polynomial))
