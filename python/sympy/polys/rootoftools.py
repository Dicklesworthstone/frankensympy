"""Indexed roots of univariate rational polynomials (upstream ``CRootOf``).

Real-root identity, order, multiplicity and decimal refinement use complete
native exact-rational isolation. A separate reference checker verifies the
original polynomial's complete real-root count and every supplied interval.
The factor association checks the full monic factor product and exact interval
membership; it does not infer identity from numerical proximity.

Real ``evalf`` supports up to 512 correctly rounded significant digits using
certified endpoint agreement. The existing shell decimal-Float representation
preserves those rendered digits; subsequent Float arithmetic still uses the
shell's binary64 backend. Nonreal root ordering/evaluation retains the bounded
binary64 lane, with an explicit refusal when its real-root classification
conflicts with the exact count. This is not certified complex-root isolation.
"""

from __future__ import annotations

import math
from typing import Any

from ..core import Function, Integer, NegativeOne, One, Rational, Symbol, Zero, sympify


def _poly_of(f: Any, x: Any = None) -> tuple:
    from .polytools import Poly

    if isinstance(f, Poly):
        p = f
    else:
        f = sympify(f)
        if x is None:
            syms = sorted(f.free_symbols, key=lambda s: s.name)
            if len(syms) != 1:
                raise ValueError("CRootOf needs a univariate polynomial")
            x = syms[0]
        p = Poly(f, x)
    if len(p.gens) != 1 or p._parametric:
        raise NotImplementedError("CRootOf supports rational-coefficient univariate polynomials")
    # Poly stores the original typed tree. Validate it before any degree,
    # coefficient extraction or factorization can exponentiate a held power.
    _native_coefficients(p)
    return p, p.gen


def _native_coefficients(p: Any) -> list:
    """Ascending typed literal coefficients; no printed expression transport."""
    from ..core import _native, _native_expr, _symbol_binding

    return _native.real_root_coefficients_native(
        _native_expr(p.as_expr()), _symbol_binding(p.gen)
    )


def _real_intervals(p: Any) -> list:
    from ..core import _native

    return _native.real_roots_isolate_native(_native_coefficients(p))


def _real_count(p: Any) -> int:
    return sum(multiplicity for _, _, multiplicity in _real_intervals(p))


def _factor_data(p: Any) -> tuple:
    from ..core import _native

    coefficients = _native_coefficients(p)
    _, factors = p.factor_list()
    # Native checking binds the entire factor list (including multiplicities)
    # to the exact original polynomial before any root identity is returned.
    real = _native.indexed_real_roots_native(
        coefficients,
        [_native_coefficients(fac) for fac, _ in factors],
        [int(mult) for _, mult in factors],
    )
    return factors, real


def _order(zs: list) -> list:
    """Legacy numerical order, used only in the explicitly nonreal lane."""
    scale = max([1.0] + [abs(z) for z in zs])
    real = sorted(z.real for z in zs if abs(z.imag) <= 1e-10 * scale)
    cplx = sorted(
        (z for z in zs if abs(z.imag) > 1e-10 * scale),
        key=lambda z: (round(z.real, 9), z.imag),
    )
    return [complex(r, 0) for r in real] + cplx


def _approximate_factor_roots(p: Any, real_count: int) -> list:
    from .. import _numeric_poly_roots

    coefficients = [complex(float(c)) for c in p.all_coeffs()]
    if not all(math.isfinite(c.real) and math.isfinite(c.imag) for c in coefficients):
        raise NotImplementedError("nonreal CRootOf coefficients exceed the binary64 lane")
    roots = _order(_numeric_poly_roots(coefficients))
    if len(roots) != p.degree() or not all(
        math.isfinite(z.real) and math.isfinite(z.imag) for z in roots
    ):
        raise NotImplementedError("nonreal CRootOf numerical root set is incomplete")
    if sum(z.imag == 0 for z in roots) != real_count:
        raise NotImplementedError(
            "nonreal CRootOf numerical classification conflicts with the exact real-root count"
        )
    return roots


def _indexed_roots(p: Any) -> list:
    """[(numeric value or None, factor Poly, local index)] in global order.

    None identifies an exactly isolated real root. It deliberately avoids
    computing a floating approximation merely to choose that root's identity.
    """
    factors, real = _factor_data(p)
    real_entries = []
    real_counts = [0] * len(factors)
    for factor, local, multiplicity in real:
        real_entries.extend([(None, factors[factor][0], local)] * multiplicity)
        real_counts[factor] += 1
    nonreal = []
    for factor, ((fac, mult), count) in enumerate(zip(factors, real_counts)):
        if count == fac.degree():
            continue
        roots = _approximate_factor_roots(fac, count)
        for local in range(count, len(roots)):
            nonreal.extend([(roots[local], fac, local)] * mult)
    nonreal.sort(key=lambda entry: (round(entry[0].real, 9), entry[0].imag))
    return real_entries + nonreal


def _numeric_roots(p: Any) -> list:
    """Approximate values, retaining exact identification of real roots."""
    from ..core import _native

    result = []
    for value, factor, index in _indexed_roots(p):
        if value is None:
            value = complex(float(_native.real_root_decimal_native(_native_coefficients(factor), index, 17)), 0)
        result.append(value)
    return result


def _radical_roots(fac: Any) -> list | None:
    """Numerical ordering of explicit nonreal quadratic/binomial roots."""
    from .polytools import _binomial_roots, _quadratic_roots

    cs = fac.all_coeffs()
    if len(cs) == 3:
        roots = _quadratic_roots(*cs)
    elif all(c == 0 for c in cs[1:-1]):
        from ..core import expand

        roots = [expand(r) for r in _binomial_roots(cs[0], cs[-1], len(cs) - 1)]
    else:
        return None

    def key(root: Any) -> tuple:
        value = complex(root)
        if abs(value.imag) < 1e-12:
            return (0, value.real, 0.0)
        return (1, round(value.real, 9), value.imag)

    return sorted(roots, key=key)


def _real_radical(fac: Any, index: int) -> Any:
    """Real quadratic/binomial formulas ordered by exact coefficient signs."""
    from fractions import Fraction
    from ..core import _exact_ratio

    cs = fac.all_coeffs()
    exact = [Fraction(*_exact_ratio(c)) for c in cs]
    if len(cs) == 3:
        discriminant = exact[1]**2 - 4 * exact[0] * exact[2]
        if discriminant <= 0:
            return None

        def rational(value):
            return Rational(value.numerator, value.denominator)

        # These coefficients are already exact rationals. Keep their
        # arithmetic in that domain instead of invoking generic cancellation
        # and expansion merely to normalize a rational radicand.
        if exact[2] == 0:
            return rational(sorted((Fraction(0), -exact[1] / exact[0]))[index])
        if exact[1] == 0:
            magnitude = rational(-exact[2] / exact[0]) ** Rational(1, 2)
            return -magnitude if index == 0 else magnitude
        denominator = 2 * exact[0]
        center = rational(-exact[1] / denominator)
        radius = rational(discriminant) ** Rational(1, 2) / rational(denominator)
        roots = [center - radius, center + radius]
        if exact[0] < 0:
            roots.reverse()
        return roots[index]
    if len(cs) > 2 and all(c == 0 for c in exact[1:-1]):
        degree = len(cs) - 1
        alpha = -exact[-1] / exact[0]
        magnitude = Rational(abs(alpha.numerator), alpha.denominator) ** Rational(1, degree)
        if degree % 2:
            return -magnitude if alpha < 0 else magnitude
        if alpha > 0:
            return -magnitude if index == 0 else magnitude
    return None


def _make_root(cls: type, fac: Any, gen: Any, local: int, radicals: bool, real: bool) -> Any:
    coefficients = fac.all_coeffs()
    if len(coefficients) == 2:
        return -coefficients[1] / coefficients[0]
    if radicals:
        if real:
            explicit = _real_radical(fac, local)
            if explicit is not None:
                return explicit
        else:
            explicit = _radical_roots(fac)
            if explicit is not None:
                return explicit[local]
    expr = fac.as_expr()
    # The generator is bound. The pinned profile retains its printed name
    # but discards assumptions; input validation above used its full identity.
    bound_gen = Symbol(gen.name)
    if fac.gen != bound_gen:
        expr = expr.subs(fac.gen, bound_gen)
    return Function.__new__(cls, expr, Integer(local), evaluate=False)


class CRootOf(Function):
    """The indexed root of a univariate polynomial; real indices are exact."""

    def __new__(cls, f: Any, x: Any = None, index: Any = None, radicals: bool = False, expand: bool = True):
        if index is None and x is not None and not isinstance(x, Symbol):
            x, index = None, x
        p, gen = _poly_of(f, x)
        degree = len(_native_coefficients(p)) - 1
        index = sympify(index)
        if type(index) not in (Integer, Zero, One, NegativeOne):
            raise ValueError("expected an integer root index")
        k = int(index)
        if not -degree <= k < degree:
            raise IndexError("root index out of [%d, %d] range, got %d" % (-degree, degree - 1, k))
        if k < 0:
            k += degree
        # Do not enter the approximate complex lane for a real index, even
        # when the same polynomial also has nonreal roots.
        factors, real = _factor_data(p)
        offset = 0
        for factor, local, multiplicity in real:
            if offset <= k < offset + multiplicity:
                return _make_root(cls, factors[factor][0], gen, local, radicals, True)
            offset += multiplicity
        value, fac, local = _indexed_roots(p)[k]
        return _make_root(cls, fac, gen, local, radicals, value is None)

    @classmethod
    def eval(cls, *args: Any) -> Any:
        return None

    @property
    def poly(self) -> Any:
        from .polytools import PurePoly

        return PurePoly(self.args[0])

    @property
    def index(self) -> int:
        return int(self.args[1])

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.index == other.index and self.poly == other.poly

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        # PurePoly already ignores generator names in equality and hashing.
        return hash((type(self), self.poly, self.index))

    @property
    def free_symbols(self) -> set:
        return set()

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def _root_value(self) -> complex:
        from ..core import _native

        p, _ = _poly_of(self.args[0])
        real_count = _real_count(p)
        if self.index < real_count:
            text = _native.real_root_decimal_native(_native_coefficients(p), self.index, 17)
            return complex(float(text), 0)
        return _approximate_factor_roots(p, real_count)[self.index]

    @property
    def is_real(self) -> bool:
        p, _ = _poly_of(self.args[0])
        return self.index < _real_count(p)

    def evalf(self, n: int = 15, **kwargs: Any) -> Any:
        from ..core import I, _ExactDecimalFloat, _format_decimal, _native
        from .. import _float_digits

        if type(n) is not int or not 1 <= n <= 512:
            raise ValueError("CRootOf precision must be between 1 and 512 decimal digits")
        p, _ = _poly_of(self.args[0])
        if self.index < _real_count(p):
            text = _native.real_root_decimal_native(_native_coefficients(p), self.index, n)
            return _ExactDecimalFloat(_format_decimal(text, n), n)
        if n > 15:
            raise NotImplementedError(
                "precision-honest nonreal CRootOf: binary64 roots support at most 15 digits; requested %d" % n
            )
        value = self._root_value()
        real = _float_digits(value.real, n) if abs(value.real) > 1e-14 else Integer(0)
        return real + _float_digits(value.imag, n) * I

    def __float__(self) -> float:
        if not self.is_real:
            raise TypeError("cannot convert a non-real root to float")
        return self._root_value().real

    def __complex__(self) -> complex:
        return self._root_value()


rootof = CRootOf
ComplexRootOf = CRootOf
RootOf = CRootOf


def all_roots(f: Any, multiple: bool = True, radicals: bool = True, extension: bool = False) -> list:
    # Rational coefficients are already in the admitted exact domain whether
    # extension is requested or not; algebraic coefficient extensions refuse.
    p, gen = _poly_of(f)
    result = []
    previous = None
    for value, fac, local in _indexed_roots(p):
        key = (fac.as_expr(), local)
        if not multiple and key == previous:
            root, multiplicity = result[-1]
            result[-1] = (root, multiplicity + 1)
            continue
        root = _make_root(CRootOf, fac, gen, local, radicals, value is None)
        result.append(root if multiple else (root, 1))
        previous = key
    return result


def real_roots(f: Any, multiple: bool = True, radicals: bool = True, extension: bool = False) -> list:
    p, gen = _poly_of(f)
    factors, real = _factor_data(p)
    result = []
    for factor, local, multiplicity in real:
        root = _make_root(CRootOf, factors[factor][0], gen, local, radicals, True)
        if multiple:
            result.extend([root] * multiplicity)
        else:
            result.append((root, multiplicity))
    return result


__all__ = ["CRootOf", "ComplexRootOf", "RootOf", "all_roots", "real_roots", "rootof"]
