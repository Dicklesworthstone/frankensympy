"""Bounded complete polynomial solutions of rational linear recurrences."""

from fractions import Fraction
from math import comb

from ..core import Dummy, Integer, Rational, Symbol, _active_callback_session
from ..polys import _coefficient_queries as queries
from ..polys._finitefield import _Request

_MAX_ORDER = 32
_MAX_DEGREE = 64
_MAX_BITS = 4096
_MAX_WORK = 2_000_000


def _bounded(q):
    if max(q.numerator.bit_length(), q.denominator.bit_length()) > _MAX_BITS:
        raise NotImplementedError("rsolve_poly coefficient growth exceeds the exact bit limit")
    return q


class _Work:
    def __init__(self):
        self.session = _active_callback_session()
        self.remaining = _MAX_WORK

    def charge(self, count):
        self.remaining -= count
        if self.remaining < 0:
            raise NotImplementedError("rsolve_poly exceeded the polynomial work limit")
        queries._charge(self.session, count)


def _source_rows(request, work):
    polynomials = []
    for source in request.raw:
        terms = queries._project(source, request.bindings, "QQ", "lex", work.session)
        if any(powers[0] > _MAX_DEGREE for powers, _ in terms):
            raise NotImplementedError("rsolve_poly input degree exceeds 64")
        work.charge(len(terms) + 1)
        polynomial = {}
        for (degree,), scalar in terms:
            value = _bounded(Fraction(scalar.exact_numerator(), scalar.exact_denominator()))
            if value:
                polynomial[degree] = value
        polynomials.append(polynomial)
    return polynomials


def _moments(coefficients, work):
    # M[j][h] = sum_i i**j * [n**h]a_i. The coefficient of n**(m-j+h)
    # in L(n**m) receives binom(m, j)*M[j][h].
    moments = []
    for j in range(_MAX_DEGREE + 1):
        row = {}
        for i, polynomial in enumerate(coefficients):
            work.charge(len(polynomial) + 1)
            weight = i ** j
            for h, value in polynomial.items():
                row[h] = _bounded(row.get(h, 0) + weight * value)
        moments.append({h: a for h, a in row.items() if a})
    return moments


def _degree_bound(moments, degree, order, rhs, work):
    # On the first nonvanishing degree diagonal, L(n**m) has leading
    # coefficient H(m) in degree m+b. A nonzero leading row cannot have
    # all its moments 0 through order r (Vandermonde at 0, ..., r).
    for drop in range(order + 1):
        b = degree - drop
        weights = [moments[j].get(b + j, 0) for j in range(drop + 1)]
        if any(weights):
            break
    else:
        raise RuntimeError("nonzero recurrence has no leading degree diagonal")

    def indicial(m):
        work.charge(len(weights))
        return _bounded(sum(comb(m, j) * weight for j, weight in enumerate(weights)))

    # All polynomial kernel degrees are nonnegative integer roots of H.
    # Same-sign forward differences at 65 certify a sign-definite tail:
    # H(65+k) = sum_j binom(k,j) * Delta^j H(65) for integer k >= 0.
    tail = [indicial(_MAX_DEGREE + 1 + i) for i in range(len(weights))]
    sign = 1 if tail[0] > 0 else -1
    if not tail[0]:
        raise NotImplementedError("rsolve_poly has an indicial root above degree 64")
    while tail:
        if sign * tail[0] < 0:
            raise NotImplementedError("rsolve_poly cannot exclude indicial roots above degree 64")
        tail = [_bounded(v - u) for u, v in zip(tail, tail[1:])]
    bound = max((i for i in range(_MAX_DEGREE + 1) if not indicial(i)), default=-1)
    if rhs:
        bound = max(bound, max(rhs) - b)
    if bound > _MAX_DEGREE:
        raise NotImplementedError("rsolve_poly solution degree exceeds 64")
    return bound


def _subtract(target, scale, polynomial, work):
    work.charge(len(polynomial) + 1)
    for exponent, value in polynomial.items():
        result = _bounded(target.get(exponent, 0) - scale * value)
        if result:
            target[exponent] = result
        else:
            target.pop(exponent, None)


def _solve_columns(moments, bound, rhs, work):
    # Save the polynomial preimage of each independent column, using its
    # highest output exponent as pivot. A dependent column yields a kernel
    # polynomial monic in a new degree and independent of the earlier ones.
    pivots, basis = {}, []
    for m in range(bound + 1):
        image, original = {}, {m: Fraction(1)}
        for j in range(m + 1):
            weight = comb(m, j)
            work.charge(len(moments[j]) + 1)
            for h, value in moments[j].items():
                k = m - j + h
                image[k] = _bounded(image.get(k, 0) + weight * value)
        image = {k: v for k, v in image.items() if v}
        while image and (k := max(image)) in pivots:
            row, preimage = pivots[k]
            scale = _bounded(image[k] / row[k])
            _subtract(image, scale, row, work)
            _subtract(original, scale, preimage, work)
        if image:
            pivots[max(image)] = image, original
        else:
            basis.append(original)

    remainder, particular = dict(rhs), {}
    while remainder:
        k = max(remainder)
        if k not in pivots:
            return None
        row, preimage = pivots[k]
        scale = _bounded(remainder[k] / row[k])
        _subtract(remainder, scale, row, work)
        _subtract(particular, -scale, preimage, work)
    return particular, basis


def _evaluate(poly, n, work):
    work.charge(len(poly) + 1)
    return _bounded(sum(q * n**k for k, q in poly.items()))


def _verify(coefficients, rhs, result, work):
    # An independent exact evaluation checks the particular solution and
    # every homogeneous direction. The residual degree bounds the number
    # of distinct integer zeros needed for a polynomial identity.
    degree = max(max(a, default=-1) for a in coefficients)
    particular, basis = result
    for y, f in [(particular, rhs), *((v, {}) for v in basis)]:
        count = max(max(f, default=-1), degree + max(y, default=-1)) + 1
        samples = [_evaluate(y, n, work) for n in range(count + len(coefficients) - 1)]
        for n in range(count):
            value = sum(_evaluate(a, n, work) * samples[n+i] for i, a in enumerate(coefficients))
            if _bounded(value) != _evaluate(f, n, work):
                raise RuntimeError("rsolve_poly solution failed the original recurrence")


def _expression(poly, n):
    return sum((Rational(value.numerator, value.denominator) * n**k
                for k, value in poly.items()), Integer(0))


def rsolve_poly(coeffs, f, n, shift=0, **hints):
    """Find every polynomial y(n) satisfying sum_i coeffs[i]*y(n+i) = f.

    Coefficients and forcing must be polynomials over QQ in the named index.
    Free constants are named starting at C<shift>. With symbols=True the
    result is (expression, constants). None denotes a proven inconsistent
    polynomial system. The bounded lane admits orders through 32 and degrees
    through 64, and refuses when it cannot exclude higher solution degrees.
    """
    if type(coeffs) not in (tuple, list) or not 1 <= len(coeffs) <= _MAX_ORDER + 1:
        raise ValueError("rsolve_poly requires 1 through 33 ordered coefficients")
    if type(n) not in (Symbol, Dummy):
        raise TypeError("rsolve_poly index must be a Symbol or Dummy")
    if type(shift) is not int or abs(shift) > 65536:
        raise ValueError("rsolve_poly constant shift must be a bounded integer")
    if hints.keys() - {"symbols"} or type(hints.get("symbols", False)) is not bool:
        raise ValueError("rsolve_poly supports only a boolean symbols hint")

    original = tuple(coeffs)
    work = _Work()
    queries._interfaces()
    request = _Request([*original, f], (n,))
    queries._check_dummies(request)

    def unchanged():
        queries._current(work.session)
        queries._interfaces()
        request.unchanged()
        if len(coeffs) != len(original) or any(x is not y for x, y in zip(coeffs, original)):
            raise RuntimeError("rsolve_poly coefficient list changed during solving")

    *coefficients, rhs = _source_rows(request, work)
    unchanged()
    while coefficients and not coefficients[-1]:
        coefficients.pop()
    if not coefficients:
        if rhs:
            return None
        raise NotImplementedError("zero recurrence operator has unbounded polynomial freedom")

    moments = _moments(coefficients, work)
    degree = max(max(a, default=-1) for a in coefficients)
    bound = _degree_bound(moments, degree, len(coefficients) - 1, rhs, work)
    result = _solve_columns(moments, bound, rhs, work)
    if result is None:
        unchanged()
        return None
    _verify(coefficients, rhs, result, work)
    particular, basis = result
    constants = []
    index = shift
    expression = _expression(particular, n)
    for direction in basis:
        while "C%d" % index == n.name:
            index += 1
        constant = Symbol("C%d" % index)
        constants.append(constant)
        expression += constant * _expression(direction, n)
        index += 1
    unchanged()
    return (expression, constants) if hints.get("symbols") else expression
