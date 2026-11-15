"""Bounded Fourier series with lazy, exact or explicitly formal coefficients.

Rational polynomials and finite products of affine sine/cosine terms use
typed native primitives, checked by differentiation before endpoint use.
Sparse finite trigonometric series are recognized without a harmonic scan.
Other admitted sources retain their definite Integral coefficients. No
failed integration is interpreted as a zero Fourier coefficient.
"""

from fractions import Fraction

from ..core import (
    Add, Basic, Dummy, E, Expr, Half, I, Integer, Mul, NegativeOne, One,
    Pow, Rational, Symbol, Tuple, Zero,
    _lift_native_result, _native, _native_expr, _require_symbol,
    _symbol_binding, _typed_symbol_facts, expand, pi, sympify,
)
from ..functions import cos, sin


_MAX_DEGREE = 32
_MAX_MODES = 128
_MAX_HARMONIC = 4096
_MAX_TERMS = 256
_MAX_COEFFICIENTS = 512
_MAX_WORK = 100000


def _rational(value):
    native = _native_expr(value)
    if native.is_integer or native.is_rational:
        return Fraction(native.exact_numerator(), native.exact_denominator())
    return None


def _normal(value):
    """Preflight coefficient expansion, including parameter-only powers."""
    _native.CompiledFunction._validate_source(_native_expr(value))
    visited = 0

    def size(node, depth):
        nonlocal visited
        visited += 1
        if visited > 4096 or depth > 64:
            raise ValueError("Fourier coefficient exceeds its expression bound")
        children = node.args
        if len(children) > 256:
            raise ValueError("Fourier coefficient exceeds its arity bound")
        sizes = [size(child, depth+1) for child in children]
        kind = type(node)
        if kind is Add:
            result = sum(sizes)
        elif kind is Mul:
            result = 1
            for count in sizes:
                if count > 4096 // result:
                    raise ValueError("Fourier coefficient exceeds its expansion bound")
                result *= count
        elif kind is Pow:
            exponent = _rational(children[1])
            if exponent is not None and exponent.denominator == 1 and exponent > 0:
                if exponent > 2*_MAX_DEGREE:
                    raise ValueError("Fourier coefficient exceeds its exponent bound")
                result = 1
                for _ in range(exponent.numerator):
                    if sizes[0] > 4096 // result:
                        raise ValueError("Fourier coefficient exceeds its expansion bound")
                    result *= sizes[0]
            else:
                result = 1
        else:
            result = 1
        if result > 4096:
            raise ValueError("Fourier coefficient exceeds its expansion bound")
        return max(1, result)

    size(value, 0)
    return expand(value)


def _input(value):
    """Admit ordinary exact inputs without invoking conversion callbacks."""
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError("Fourier series coefficient exceeds 4096 bits")
        return Integer(value)
    if type(value) is Fraction:
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096:
            raise ValueError("Fourier series coefficient exceeds 4096 bits")
        return Rational(value.numerator, value.denominator)
    if type(value) is str and len(value) <= 16384:
        return sympify(value)
    if not any(base is Basic for base in type.__dict__["__mro__"].__get__(type(value))):
        raise TypeError("Fourier series requires symbolic or exact rational inputs")
    return value


def _index(value, maximum, name):
    if type(value) is int:
        result = value
    elif type(value) in (Integer, NegativeOne, One, Zero):
        result = _rational(value).numerator
    else:
        raise TypeError(f"Fourier series {name} must be an exact integer")
    if result < 0:
        raise ValueError(f"Fourier series {name} must be nonnegative")
    if result > maximum:
        raise ValueError(f"Fourier series {name} exceeds its bound of {maximum}")
    return result


def _endpoint(value):
    """A rational number or rational multiple of pi, with exact ordering."""
    rational = _rational(value)
    if rational is not None:
        return rational, 0
    rational = _rational(_normal(value / pi))
    if rational is not None:
        return rational, 1
    raise NotImplementedError("Fourier series endpoints require rational numbers or rational multiples of pi")


def _period(start, end):
    a, a_kind = _endpoint(start)
    b, b_kind = _endpoint(end)
    if a and b and a_kind != b_kind:
        raise NotImplementedError("Fourier series endpoints require a common rational or pi scale")
    if b <= a:
        raise ValueError("Fourier series requires a finite interval with start < end")
    return _normal(end - start)


def _capture(*sources):
    """Bind retained argument order to the original typed native payload."""
    from ..integrals.transforms import _FourierSource

    snapshot = _FourierSource(*sources)
    aliases = {"Abs": "abs", "log": "ln"}
    for native, operator, children in snapshot.records():
        if native.func_name not in (operator, aliases.get(operator, operator)):
            raise ValueError("Fourier series retained operator differs from its native source")
        if children is not None:
            original = native.args
            if len(original) != len(children) or any(a != b for a, b in zip(original, children)):
                raise ValueError("Fourier series retained arguments differ from their native source")
    snapshot.unchanged()
    return snapshot


class _Modes:
    """Finite polynomial/trigonometric algebra with explicit expansion caps.

    A mode is (polynomial coefficients, kind, slope, phase), where kind is
    zero, one, or two for a polynomial, sine, or cosine respectively.
    Every original operand is examined before zero products/zeroth powers
    can remove it. An unsupported original operand keeps the whole source
    in the formal-integral lane.
    """

    def __init__(self, variable):
        self.variable = variable
        self.work = 0
        self.nodes = 0

    def charge(self, amount=1):
        self.work += amount
        if self.work > _MAX_WORK:
            raise ValueError("Fourier series exceeds its algebraic work bound")

    def polynomial(self, coefficients):
        self.charge(len(coefficients))
        if len(coefficients) > _MAX_DEGREE + 1:
            raise ValueError("Fourier series polynomial degree exceeds 32")
        values = list(coefficients)
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        return tuple(values)

    def multiply_polynomials(self, left, right):
        if len(left) + len(right) - 1 > _MAX_DEGREE + 1:
            raise ValueError("Fourier series polynomial degree exceeds 32")
        self.charge(len(left) * len(right))
        out = [Integer(0)] * (len(left) + len(right) - 1)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                out[i + j] = _normal(out[i + j] + a*b)
        return self.polynomial(out)

    def mode(self, coefficients, kind=0, slope=None, phase=None):
        coefficients = self.polynomial(coefficients)
        if not kind:
            return coefficients, 0, Integer(0), Integer(0)
        slope, phase = _normal(slope), _normal(phase)
        if slope == 0:
            value = sin(phase) if kind == 1 else cos(phase)
            return self.mode(tuple(_normal(c*value) for c in coefficients))
        return coefficients, kind, slope, phase

    def collect(self, modes):
        if len(modes) > _MAX_MODES:
            raise ValueError("Fourier series exceeds its finite mode bound")
        collected = {}
        for coefficients, kind, slope, phase in modes:
            self.charge(len(coefficients))
            key = kind, slope, phase
            old = collected.get(key, (Integer(0),))
            size = max(len(old), len(coefficients))
            out = [Integer(0)] * size
            for i in range(size):
                out[i] = _normal((old[i] if i < len(old) else 0)
                                 + (coefficients[i] if i < len(coefficients) else 0))
            collected[key] = self.polynomial(out)
        return [(coefficients, *key) for key, coefficients in collected.items()
                if coefficients != (Integer(0),)]

    def multiply(self, left, right):
        if len(left) * len(right) > _MAX_MODES:
            raise ValueError("Fourier series exceeds its finite product bound")
        out = []
        for p, kind, a, b in left:
            for q, other, c, d in right:
                coefficients = self.multiply_polynomials(p, q)
                if not kind:
                    out.append(self.mode(coefficients, other, c, d))
                elif not other:
                    out.append(self.mode(coefficients, kind, a, b))
                else:
                    half = tuple(_normal(value/2) for value in coefficients)
                    negative = tuple(-value for value in half)
                    if kind == other == 1:
                        out.extend((self.mode(half, 2, a-c, b-d),
                                    self.mode(negative, 2, a+c, b+d)))
                    elif kind == other == 2:
                        out.extend((self.mode(half, 2, a-c, b-d),
                                    self.mode(half, 2, a+c, b+d)))
                    elif kind == 1:
                        out.extend((self.mode(half, 1, a+c, b+d),
                                    self.mode(half, 1, a-c, b-d)))
                    else:
                        out.extend((self.mode(half, 1, a+c, b+d),
                                    self.mode(half, 1, c-a, d-b)))
                    if len(out) > _MAX_MODES:
                        raise ValueError("Fourier series exceeds its finite mode bound")
        return self.collect(out)

    def parse(self, node, depth=0):
        self.nodes += 1
        if self.nodes > 4096 or depth > 64:
            raise ValueError("Fourier series source exceeds its node or depth bound")
        self.charge()
        kind = type(node)
        if kind is Symbol or kind is Dummy:
            return [self.mode((Integer(0), Integer(1)) if node == self.variable else (node,))]
        # Retained arithmetic must be visited even if its native handle has
        # a simpler value. It may still contain an original denominator hole.
        if kind not in (Add, Mul, Pow):
            rational = _rational(node)
            if rational is not None:
                return [self.mode((node,))]
            if node in (pi, E, I):
                return [self.mode((node,))]
        children = node.args
        if len(children) > 256:
            raise ValueError("Fourier series source exceeds its arity bound")
        if kind is Add or kind is Mul:
            parts = [self.parse(child, depth+1) for child in children]
            if any(part is None for part in parts):
                return None
            if kind is Add:
                if sum(map(len, parts)) > _MAX_MODES:
                    raise ValueError("Fourier series exceeds its finite sum bound")
                return self.collect([mode for part in parts for mode in part])
            result = [self.mode((Integer(1),))]
            for part in parts:
                result = self.multiply(result, part)
            return result
        if kind is Pow:
            base, power = children
            parsed = self.parse(base, depth+1)
            atoms = (Integer, Rational, Half, NegativeOne, One, Zero)
            if type(power) not in atoms:
                return None
            exponent = _rational(power)
            if exponent is None or exponent.denominator != 1:
                return None
            exponent = exponent.numerator
            if abs(exponent) > _MAX_DEGREE:
                raise ValueError("Fourier series exponent exceeds 32")
            if exponent < 0:
                # Only nonzero literal constants have an admitted reciprocal.
                scalar = _rational(base) if type(base) in atoms else None
                if ((scalar is not None and scalar != 0)
                        or type(base) is Expr and base in (pi, E, I)):
                    return [self.mode((node,))]
                return None
            if parsed is None:
                return None
            result = [self.mode((Integer(1),))]
            for _ in range(exponent):
                result = self.multiply(result, parsed)
            return result
        if kind is sin or kind is cos:
            parsed = self.parse(children[0], depth+1)
            if parsed is None or len(parsed) != 1:
                return None
            coefficients, other, _, _ = parsed[0]
            if other or len(coefficients) > 2:
                return None
            slope = coefficients[1] if len(coefficients) == 2 else Integer(0)
            # Unknown parameter-dependent slopes require a separate domain
            # analysis before antiderivative denominators are introduced.
            if _rational(slope) is None and _rational(_normal(slope/pi)) is None:
                return None
            return [self.mode((Integer(1),), 1 if kind is sin else 2,
                              slope, coefficients[0])]
        return None

    def expression(self, mode):
        coefficients, kind, slope, phase = mode
        value = Add(*(coefficient*self.variable**degree
                      for degree, coefficient in enumerate(coefficients)))
        if kind:
            value *= (sin if kind == 1 else cos)(slope*self.variable + phase)
        return value


class _CoefficientSequence:
    """Lazy continuation of the previous zero-based coefficient arrays."""

    def __init__(self, owner, sine):
        self.owner, self.sine = owner, sine

    def __getitem__(self, index):
        index = _index(index, _MAX_HARMONIC-1, "coefficient index")
        return self.owner._coefficients(index+1)[int(self.sine)]


class FourierSeries(Expr):
    """Fourier expansion on one finite period with lazy coefficients.

    ``truncate(n)`` retains the first n harmonic terms not known to vanish,
    including the mean. A harmonic combines its sine and cosine parts.
    Unresolved Integral terms are retained, without a nonzero assertion.
    Finite trigonometric support terminates exactly; an exhausted scan or
    algebraic allowance raises instead of returning an incomplete sum.
    """

    __slots__ = ("_snapshot", "_signature_snapshot", "_source_args", "_algebra",
                 "_modes", "_finite", "_cache")

    def __new__(cls, function, limits):
        function = _input(function)
        if type(limits) is Tuple:
            limits = limits.args
        if type(limits) not in (tuple, list) or len(limits) != 3:
            raise ValueError("Fourier series limits must be (variable, start, end)")
        variable = _require_symbol(limits[0])
        start, end = _input(limits[1]), _input(limits[2])
        for value in (function, variable, start, end):
            _native.CompiledFunction._validate_source(_native_expr(value))
        snapshot = _capture(function, variable, start, end)
        _period(start, end)
        obj = object.__new__(cls)
        obj._struct_args = (function, Tuple(variable, start, end))
        obj._source_args = obj._struct_args
        obj._snapshot = snapshot
        obj._signature_snapshot = _capture(obj._struct_args[1])
        obj._algebra = _Modes(variable)
        obj._modes = obj._algebra.parse(function)
        obj._cache = {}
        obj._finite = obj._finite_coefficients()
        obj._unchanged()
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def _unchanged(self):
        if self._struct_args is not self._source_args:
            raise RuntimeError("Fourier series source changed after construction")
        self._snapshot.unchanged()
        self._signature_snapshot.unchanged()

    @property
    def function(self):
        return self.args[0]

    f = function

    @property
    def x(self):
        return self.args[1].args[0]

    @property
    def start(self):
        return self.args[1].args[1]

    @property
    def end(self):
        return self.args[1].args[2]

    @property
    def period(self):
        return _normal(self.end-self.start)

    @property
    def L(self):
        return _normal(self.period/2)

    @property
    def _omega(self):
        return _normal(2*pi/self.period)

    @property
    def free_symbols(self):
        return self.function.free_symbols | self.start.free_symbols | self.end.free_symbols

    @property
    def a0(self):
        return self._coefficients(0)[0]

    @property
    def an(self):
        return _CoefficientSequence(self, False)

    @property
    def bn(self):
        return _CoefficientSequence(self, True)

    def _finite_coefficients(self):
        if self._modes is None:
            return None
        out = {}
        for coefficients, kind, slope, phase in self._modes:
            if len(coefficients) != 1:
                return None
            amplitude = coefficients[0]
            if not kind:
                index, cosine, sine = 0, amplitude, Integer(0)
            else:
                ratio = _rational(_normal(slope/self._omega))
                if ratio is None or ratio.denominator != 1:
                    return None
                index = abs(ratio.numerator)
                if index > _MAX_HARMONIC:
                    raise ValueError("Fourier series harmonic exceeds 4096")
                sign = 1 if ratio.numerator >= 0 else -1
                if kind == 1:
                    cosine, sine = amplitude*sin(phase), sign*amplitude*cos(phase)
                else:
                    cosine, sine = amplitude*cos(phase), -sign*amplitude*sin(phase)
            previous = out.get(index, (Integer(0), Integer(0)))
            out[index] = (_normal(previous[0]+cosine), _normal(previous[1]+sine))
        return out

    def _integral(self, modes):
        total = Integer(0)
        binding = _symbol_binding(self.x)
        for mode in modes:
            integrand = self._algebra.expression(mode)
            self._algebra.charge(1 + len(mode[0]))
            native_integrand = _native_expr(integrand)
            raw = _native.integrate_native(native_integrand, binding,
                                           _typed_symbol_facts(integrand, self.x))
            primitive = _lift_native_result(raw, self.function, self.x)
            # This checks an antiderivative identity only. Finite endpoint
            # use additionally relies on the entire polynomial/trig grammar.
            residual = (raw.diff(binding)-native_integrand).expand().simplify()
            if not residual.is_zero:
                raise ValueError("Fourier series native primitive failed its derivative check")
            self._unchanged()
            total += primitive.subs(self.x, self.end)-primitive.subs(self.x, self.start)
        return _normal(total)

    def _coefficients(self, index):
        self._unchanged()
        if self._finite is not None:
            return self._finite.get(index, (Integer(0), Integer(0)))
        if index in self._cache:
            return self._cache[index]
        if len(self._cache) >= _MAX_COEFFICIENTS:
            raise ValueError("Fourier series exceeds its coefficient cache bound")
        scale = (Integer(1) if index == 0 else Integer(2))/self.period
        if self._modes is None:
            from ..integrals.integrals import Integral

            limits = (self.x, self.start, self.end)
            if index == 0:
                values = (scale*Integral(self.function, limits), Integer(0))
            else:
                angle = index*self._omega*self.x
                values = tuple(scale*Integral(Mul(self.function, basis, evaluate=False), limits)
                               for basis in (cos(angle), sin(angle)))
        elif index == 0:
            values = (_normal(scale*self._integral(self._modes)), Integer(0))
        else:
            values = tuple(_normal(scale*self._integral(self._algebra.multiply(
                self._modes, [self._algebra.mode((Integer(1),), kind,
                                                index*self._omega, Integer(0))])))
                           for kind in (2, 1))
        self._unchanged()
        self._cache[index] = values
        return values

    def term(self, index):
        index = _index(index, _MAX_HARMONIC, "harmonic")
        cosine, sine = self._coefficients(index)
        if index == 0:
            return cosine
        angle = index*self._omega*self.x
        result = cosine*cos(angle) + sine*sin(angle)
        self._unchanged()
        return result

    def _terms(self):
        indices = sorted(self._finite) if self._finite is not None else range(_MAX_HARMONIC+1)
        for index in indices:
            term = self.term(index)
            if term != 0:
                yield term
        if self._finite is None:
            raise ValueError("Fourier series exhausted its harmonic scan bound")

    def truncate(self, n=3):
        self._unchanged()
        if n is None:
            return self._terms()
        n = _index(n, _MAX_TERMS, "term count")
        if n == 0:
            return Integer(0)
        terms = []
        for term in self._terms():
            terms.append(term)
            if len(terms) == n:
                break
        result = Add(*terms)
        self._unchanged()
        return result

    def __repr__(self):
        return f"FourierSeries({self.function}, ({self.x}, {self.start}, {self.end}))"

    __str__ = __repr__


def fourier_series(f, limits=None, finite=True):
    """Build a lazy Fourier expansion over one finite real interval.

    The default interval is (-pi, pi). Exact polynomial/trigonometric
    coefficients are computed as requested, without a six-harmonic cutoff.
    Unsupported coefficient integrals remain formal. ``finite=False`` is
    accepted for the usual public signature; exact finite support is still
    used to avoid an impossible scan for further nonzero harmonics.
    """
    if type(finite) is not bool:
        raise TypeError("Fourier series finite option must be an exact bool")
    f = _input(f)
    _native.CompiledFunction._validate_source(_native_expr(f))
    snapshot = _capture(f)
    if limits is None:
        limits = ()
    elif type(limits) is Tuple:
        limits = limits.args
    if type(limits) not in (tuple, list) or len(limits) > 3:
        raise ValueError("Fourier series limits must have at most three entries")
    if len(limits) in (0, 2):
        symbols = f.free_symbols
        if len(symbols) > 1:
            raise ValueError("Fourier series variable must be specified for multivariate expressions")
        variable = next(iter(symbols)) if symbols else Symbol("x")
        limits = (variable, *(limits if limits else (-pi, pi)))
    elif len(limits) == 1:
        limits = (limits[0], -pi, pi)
    result = FourierSeries(f, limits)
    snapshot.unchanged()
    if result._finite is not None and not any(index for index in result._finite):
        return result.a0
    return result


__all__ = ["FourierSeries", "fourier_series"]
