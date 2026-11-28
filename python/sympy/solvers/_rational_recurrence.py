"""Abramov universal denominators for linear recurrences over QQ(n)."""

from fractions import Fraction

from ..polys.polytools import Poly
from ._polynomial_recurrence import _MAX_DEGREE, _bounded, _evaluate, _expression


class _Ring:
    """Reuse the exact univariate QQ operations with bounded public degrees."""

    def __init__(self, axis, work):
        self.axis = axis
        self.work = work
        self.one = Poly(1, axis, domain="QQ")

    def source(self, row):
        return Poly(_expression(row, self.axis), self.axis, domain="QQ")

    def row(self, poly):
        terms = poly.terms()
        self.work.charge(len(terms) + 1)
        result = {}
        for (k,), scalar in terms:
            if k > _MAX_DEGREE:
                raise NotImplementedError("rsolve_ratio polynomial degree exceeds 64")
            value = _bounded(Fraction(scalar.p, scalar.q))
            if value:
                result[k] = value
        return result

    def multiply(self, a, b):
        da, db = a.degree(), b.degree()
        if da is None:
            return a
        if db is None:
            return b
        if da + db > _MAX_DEGREE:
            raise NotImplementedError("rsolve_ratio cleared degree exceeds 64")
        self.work.charge(1)
        # Shift by zero reifies the native product as a compact expanded
        # polynomial, so repeated LCM products never grow a held input tree.
        product = (a * b).shift(0)
        self.row(product)
        return product

    def root_disk(self, poly):
        # Center at an integer close to the mean of all complex roots. For
        # |z| >= R the leading term strictly dominates all remaining terms,
        # giving an exact root disk without approximating any roots.
        row = self.row(poly)
        d = max(row)
        center = (-row.get(d-1, 0) / (d * row[d])).__floor__()
        if abs(center).bit_length() > 30:
            raise NotImplementedError("rsolve_ratio root-disk center exceeds its bound")
        centered = self.row(poly.shift(center))
        for radius in range(1, 66):
            self.work.charge(len(centered))
            if abs(centered[d]) * radius**d > sum(
                    abs(a) * radius**k for k, a in centered.items() if k < d):
                return center, radius
        raise NotImplementedError("rsolve_ratio could not bound the endpoint root dispersion")

    def universal_denominator(self, coeffs):
        # A(n)=a_r(n-r), B(n)=a_0(n). A denominator chain can run from
        # a factor of A to one of B shifted forward by an integer h >= 0.
        # The exact disks bound every such h, including nonreal root pairs.
        a = coeffs[-1].shift(1-len(coeffs))
        b = coeffs[0]
        if a.degree() == 0 or b.degree() == 0:
            return self.one
        ca, ra = self.root_disk(a)
        cb, rb = self.root_disk(b)
        last = cb - ca + ra + rb - 1
        if last > 128:
            raise NotImplementedError("rsolve_ratio endpoint dispersion exceeds 128 shifts")
        denominator = self.one
        for h in range(last, -1, -1):
            self.work.charge(1)
            common = a.gcd(b.shift(h)).monic()
            degree = int(common.degree())
            if degree:
                if int(denominator.degree()) + degree*(h+1) > _MAX_DEGREE:
                    raise NotImplementedError("rsolve_ratio universal denominator exceeds degree 64")
                a = a.exquo(common)
                b = b.exquo(common.shift(-h))
                for j in range(h+1):
                    denominator = self.multiply(denominator, common.shift(-j))
                if a.degree() == 0 or b.degree() == 0:
                    break
        return denominator


def clear_denominators(coefficients, forcing, n, work):
    ring = _Ring(n, work)
    first = next(i for i, a in enumerate(coefficients) if a)
    # Relabel n -> n-first before dropping initial zero shifts. The unknown
    # in the resulting equation is y(n), not a shifted answer.
    polys = [ring.source(a).shift(-first) for a in coefficients[first:]]
    f = ring.source(forcing).shift(-first)
    if len(polys) == 1:
        # a(n)*y(n)=f(n) has one rational solution even when a has roots.
        return [{0: Fraction(1)}], ring.row(f), ring.row(polys[0])
    denominator = ring.universal_denominator(polys)
    if denominator.degree() == 0:
        return [ring.row(p) for p in polys], ring.row(f), {0: Fraction(1)}
    common = ring.one
    reduced = []
    for i, a in enumerate(polys):
        di = denominator.shift(i)
        g = a.gcd(di)
        numerator, di = a.exquo(g), di.exquo(g)
        common = ring.multiply(common, di.exquo(common.gcd(di))).monic()
        reduced.append((numerator, di))
    coeffs = [ring.row(ring.multiply(num, common.exquo(di))) for num, di in reduced]
    return coeffs, ring.row(ring.multiply(f, common)), ring.row(denominator)


def verify_rational(coefficients, forcing, denominator, result, work):
    # At every admitted integer, replay the ORIGINAL shifts using exact
    # fractions. Over the common product of denominators the residual
    # numerator has the stated degree, so this many distinct zero values
    # establish a rational identity. Poles are skipped explicitly.
    d = max(denominator)
    r = len(coefficients) - 1
    for u, rhs in [(result[0], forcing), *((y, {}) for y in result[1])]:
        degree = max(max(rhs, default=-1)+(r+1)*d,
                     max(max(a, default=-1) for a in coefficients)+max(u, default=-1)+r*d)
        needed = degree + 1
        for t in range(needed+(r+1)*d):
            denoms = [_evaluate(denominator, t+i, work) for i in range(r+1)]
            if not all(denoms):
                continue
            residual = sum(_evaluate(a, t, work)*_evaluate(u, t+i, work)/denoms[i]
                           for i, a in enumerate(coefficients))
            if _bounded(residual) != _evaluate(rhs, t, work):
                raise RuntimeError("rsolve_ratio solution failed the original rational recurrence")
            needed -= 1
            if not needed:
                break
        if needed:
            raise RuntimeError("rsolve_ratio failed to obtain enough regular verification points")


def rsolve_ratio(coeffs, f, n, **hints):
    """All rational-function solutions over QQ(n) of a linear recurrence.

    Coefficients and forcing are exact polynomials in n. The returned family
    satisfies the recurrence as a rational identity on its common defined
    domain. Integer poles can occur. None means that no rational-function
    solution exists; excessive dispersion or degree raises NotImplementedError.
    """
    from ._polynomial_recurrence import _solve_recurrence
    shift = hints.pop("shift", 0)
    return _solve_recurrence(coeffs, f, n, shift, hints, rational=True)
