"""Indexed roots of univariate polynomials (upstream ``CRootOf``).

``CRootOf(f, k)`` is the ``k``-th root of ``f`` in upstream's order: real
roots ascending, then the non-real ones by real part with each conjugate
pair negative-imaginary first. A root whose irreducible factor is
linear or quadratic comes back explicitly (as upstream's trivial-root
path does); otherwise the object stays symbolic and evaluates
numerically from the square-free factor's roots (binary64, at most 15
honest digits). The polynomial's generator is bound, not free.
"""

from __future__ import annotations

from typing import Any

from ..core import Function, Integer, Rational, Symbol, sympify


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
    if p._parametric or len(p.gens) != 1:
        raise NotImplementedError("CRootOf supports rational-coefficient univariate polynomials")
    return p, p.gen


def _numeric_roots(p: Any) -> list:
    """All roots of ``p`` as complex numbers in upstream order, with
    multiplicity, factor by factor."""
    from .. import _numeric_poly_roots

    _, factors = p.factor_list()
    out = []
    for fac, mult in factors:
        coeffs = [complex(float(c)) for c in fac.all_coeffs()]
        out.extend(_numeric_poly_roots(coeffs) * mult)
    return _order(out)


def _order(zs: list) -> list:
    scale = max([1.0] + [abs(z) for z in zs])
    real = sorted(z.real for z in zs if abs(z.imag) <= 1e-10 * scale)
    cplx = sorted((z for z in zs if abs(z.imag) > 1e-10 * scale), key=lambda z: (round(z.real, 9), z.imag))
    return [complex(r, 0) for r in real] + cplx


def _indexed_roots(p: Any) -> list:
    """[(value, factor Poly, local index)] in upstream global order, one
    entry per root with multiplicity, local indices into each
    irreducible factor's own order."""
    from .. import _numeric_poly_roots

    _, factors = p.factor_list()
    entries = []
    for fac, mult in factors:
        local = _order(_numeric_poly_roots([complex(float(c)) for c in fac.all_coeffs()]))
        for i, z in enumerate(local):
            entries.extend([(z, fac, i)] * mult)
    scale = max([1.0] + [abs(e[0]) for e in entries])

    def key(e: Any) -> tuple:
        z = e[0]
        if abs(z.imag) <= 1e-10 * scale:
            return (0, z.real, 0.0)
        return (1, round(z.real, 9), z.imag)

    return sorted(entries, key=key)


def _radical_roots(fac: Any) -> list | None:
    """Explicit roots of a quadratic or binomial factor in upstream order
    (the radicals=True trivial cases), else None."""
    from .polytools import _binomial_roots, _quadratic_roots

    cs = fac.all_coeffs()
    if len(cs) == 3:
        rs = _quadratic_roots(*cs)
    elif all(c == 0 for c in cs[1:-1]):
        from ..core import expand

        rs = [expand(r) for r in _binomial_roots(cs[0], cs[-1], len(cs) - 1)]
    else:
        return None

    def key(r: Any) -> tuple:
        v = complex(r)
        if abs(v.imag) < 1e-12:
            return (0, v.real, 0.0)
        return (1, round(v.real, 9), v.imag)

    return sorted(rs, key=key)


class CRootOf(Function):
    """The ``index``-th complex root of a univariate polynomial."""

    def __new__(cls, f: Any, x: Any = None, index: Any = None, radicals: bool = False, expand: bool = True):
        if index is None and x is not None and not isinstance(x, Symbol):
            x, index = None, x
        p, gen = _poly_of(f, x)
        deg = p.degree()
        k = int(sympify(index))
        if not -deg <= k < deg:
            raise IndexError("root index out of [%d, %d] range, got %d" % (-deg, deg - 1, k))
        if k < 0:
            k += deg
        z, fac, local = _indexed_roots(p)[k]
        cs = fac.all_coeffs()
        if len(cs) == 2:
            # Upstream: roots of linear factors come back explicitly.
            return -cs[1] / cs[0]
        if radicals:
            explicit = _radical_roots(fac)
            if explicit is not None:
                return explicit[local]
        expr = fac.as_expr()
        if fac.gen != gen:
            expr = expr.subs(fac.gen, gen)
        return Function.__new__(cls, expr, Integer(local), evaluate=False)

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

    @property
    def free_symbols(self) -> set:
        return set()

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def _root_value(self) -> complex:
        p, _ = _poly_of(self.args[0])
        return _numeric_roots(p)[self.index]

    @property
    def is_real(self) -> bool:
        return abs(self._root_value().imag) <= 1e-10 * max(1.0, abs(self._root_value()))

    def evalf(self, n: int = 15, **kwargs: Any) -> Any:
        from ..core import I
        from .. import _float_digits

        if n > 15:
            raise NotImplementedError(
                "precision-honest CRootOf: binary64 roots are honest to at most 15 digits; requested %d" % n
            )
        v = self._root_value()
        if self.is_real:
            return _float_digits(v.real, n)
        re_part = _float_digits(v.real, n) if abs(v.real) > 1e-14 else Integer(0)
        return re_part + _float_digits(v.imag, n) * I

    def __float__(self) -> float:
        v = self._root_value()
        if abs(v.imag) > 1e-10 * max(1.0, abs(v)):
            raise TypeError("cannot convert a non-real root to float")
        return v.real

    def __complex__(self) -> complex:
        return self._root_value()


rootof = CRootOf
ComplexRootOf = CRootOf
RootOf = CRootOf


def all_roots(f: Any, x: Any = None, radicals: bool = True) -> list:
    p, _ = _poly_of(f, x)
    return [CRootOf(p, None, i, radicals=radicals) for i in range(p.degree())]


def real_roots(f: Any, x: Any = None, radicals: bool = True, multiple: bool = True) -> list:
    p, _ = _poly_of(f, x)
    zs = _numeric_roots(p)
    count = sum(1 for z in zs if z.imag == 0)
    return [CRootOf(p, None, i, radicals=radicals) for i in range(count)]


__all__ = ["CRootOf", "ComplexRootOf", "RootOf", "all_roots", "real_roots", "rootof"]
