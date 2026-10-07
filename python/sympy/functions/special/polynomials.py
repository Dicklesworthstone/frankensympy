"""Classical orthogonal polynomials (upstream ``sympy.functions.special.polynomials``).

``legendre``, ``chebyshevt``, ``chebyshevu``, ``hermite``, ``hermite_prob``
and ``laguerre`` evaluate natively: explicit for integer degree (exact
three-term recurrences), closed forms at the special points for symbolic
degree, and native derivative rules. The parametric families
(``gegenbauer``, ``assoc_laguerre``, ``jacobi``, ``assoc_legendre``) are
expanded here for integer degree, collected in the variable as upstream
prints them.
"""

from __future__ import annotations

from typing import Any

from ...core import Dummy, Function, Integer, Rational, _NativeFunction, expand, sympify


def _gamma_forms(cls_name: str, n: Any, x: Any) -> Any:
    from ...core import pi, sqrt
    from ...functions import cos, gamma

    if x != 0 or isinstance(n, Integer):
        return None
    if cls_name == "legendre":
        return sqrt(pi) / (gamma(Rational(1, 2) - n / 2) * gamma(n / 2 + 1))
    if cls_name == "hermite":
        return 2**n * sqrt(pi) / gamma(Rational(1, 2) - n / 2)
    if cls_name in ("chebyshevt", "chebyshevu"):
        return cos(pi * n / 2)
    return None


class _ClassicalPoly(_NativeFunction):
    __slots__ = ()

    @classmethod
    def eval(cls, n: Any, x: Any) -> Any:
        special = _gamma_forms(cls.__name__, sympify(n), sympify(x))
        if special is not None:
            return special
        return super().eval(n, x)


class legendre(_ClassicalPoly):
    """Legendre polynomial P_n(x)."""

    __slots__ = ()


class chebyshevt(_ClassicalPoly):
    """Chebyshev polynomial of the first kind T_n(x)."""

    __slots__ = ()


class chebyshevu(_ClassicalPoly):
    """Chebyshev polynomial of the second kind U_n(x)."""

    __slots__ = ()


class hermite(_ClassicalPoly):
    """Physicists' Hermite polynomial H_n(x)."""

    __slots__ = ()


class hermite_prob(_ClassicalPoly):
    """Probabilists' Hermite polynomial He_n(x)."""

    __slots__ = ()


class laguerre(_ClassicalPoly):
    """Laguerre polynomial L_n(x)."""

    __slots__ = ()


def _collect_in(coeffs: list, x: Any) -> Any:
    """sum c_k*x**k with each coefficient expanded and kept as a factor
    (upstream: x*(-a - 2) + x**2/2 + ...)."""
    from ...core import Add, Mul

    terms = []
    for k, c in enumerate(coeffs):
        c = expand(c)
        if c == 0:
            continue
        if k == 0:
            terms.append(c)
        else:
            terms.append(Mul(c, x**k))
    return Add(*terms) if terms else Integer(0)


def _int_degree(n: Any) -> int | None:
    n = sympify(n)
    if isinstance(n, Integer) and n >= 0 and n <= 200:
        return int(n)
    return None


def _poly_add(p: list, q: list, cp: Any = 1, cq: Any = 1) -> list:
    out = []
    for i in range(max(len(p), len(q))):
        a = p[i] if i < len(p) else 0
        b = q[i] if i < len(q) else 0
        out.append(expand(cp * a + cq * b))
    return out


def _shift(p: list) -> list:
    return [Integer(0)] + list(p)


class gegenbauer(Function):
    """Gegenbauer polynomial C_n^(a)(x)."""

    @classmethod
    def eval(cls, n: Any, a: Any, x: Any) -> Any:
        k = _int_degree(n)
        if k is None:
            from ...functions import gamma

            if sympify(x) == 1:
                return gamma(2 * a + n) / (gamma(2 * a) * gamma(n + 1))
            return None
        prev, cur = [Integer(1)], [Integer(0), 2 * a]
        if k == 0:
            return Integer(1)
        for j in range(1, k):
            nxt = _poly_add(_shift(cur), prev, 2 * (j + a) / (j + 1), -(j + 2 * a - 1) / (j + 1))
            prev, cur = cur, nxt
        return _collect_in(cur, x)


class assoc_laguerre(Function):
    """Generalized Laguerre polynomial L_n^(a)(x)."""

    @classmethod
    def eval(cls, n: Any, a: Any, x: Any) -> Any:
        k = _int_degree(n)
        if k is None:
            return None
        if k == 0:
            return Integer(1)
        prev, cur = [Integer(1)], [1 + a, Integer(-1)]
        for j in range(1, k):
            # (j+1) L_{j+1} = (2j + 1 + a - x) L_j - (j + a) L_{j-1}
            a1 = _poly_add(cur, _shift(cur), (2 * j + 1 + a) / (j + 1), Rational(-1, j + 1))
            nxt = _poly_add(a1, prev, 1, -(j + a) / (j + 1))
            prev, cur = cur, nxt
        return _collect_in(cur, x)


class jacobi(Function):
    """Jacobi polynomial P_n^(a,b)(x)."""

    @classmethod
    def eval(cls, n: Any, a: Any, b: Any, x: Any) -> Any:
        k = _int_degree(n)
        if k is None:
            return None
        from ...core import Symbol
        from ...functions import factorial

        t = Symbol("_fsym_jacobi_t")

        def binom(top: Any, j: int) -> Any:
            out = Integer(1)
            for i in range(j):
                out = out * (top - i)
            return out / factorial(j)

        total = Integer(0)
        for s in range(k + 1):
            total = total + binom(k + a, k - s) * binom(k + b, s) * ((t - 1) / 2) ** s * ((t + 1) / 2) ** (k - s)
        total = expand(total)
        from ...polys.polytools import _x_coeffs

        cs = _x_coeffs(total, t) or {0: total}
        coeffs = [cs.get(i, Integer(0)) for i in range(max(cs) + 1)]
        return _collect_in(coeffs, x)


class assoc_legendre(Function):
    """Associated Legendre function P_n^m(x) (Condon-Shortley phase)."""

    @classmethod
    def eval(cls, n: Any, m: Any, x: Any) -> Any:
        k = _int_degree(n)
        mm = sympify(m)
        if k is None or not isinstance(mm, Integer):
            return None
        mi = int(mm)
        if abs(mi) > k:
            return Integer(0)
        if mi < 0:
            return None
        from ...core import Symbol, diff, sqrt

        t = Symbol("_fsym_legendre_t")
        p = legendre(k, t)
        d = diff(p, t, mi) if mi else p
        out = (-1) ** mi * expand(d).subs(t, x)
        if mi % 2 == 0:
            return expand(out * (1 - x**2) ** (mi // 2))
        return out * sqrt(1 - x**2) ** mi


def _poly_wrapper(cls: Any, nparams: int = 0):
    def f(n: Any, *args: Any, polys: bool = False) -> Any:
        params = list(args[:nparams])
        x = args[nparams] if len(args) > nparams else None
        if x is None:
            x = Dummy("x")
        e = cls(n, *params, x)
        if polys:
            from ...polys.polytools import Poly

            return Poly(e, x)
        return e

    f.__name__ = cls.__name__ + "_poly"
    return f


legendre_poly = _poly_wrapper(legendre)
chebyshevt_poly = _poly_wrapper(chebyshevt)
chebyshevu_poly = _poly_wrapper(chebyshevu)
hermite_poly = _poly_wrapper(hermite)
hermite_prob_poly = _poly_wrapper(hermite_prob)
laguerre_poly = _poly_wrapper(laguerre)
gegenbauer_poly = _poly_wrapper(gegenbauer, 1)
jacobi_poly = _poly_wrapper(jacobi, 2)


def chebyshevt_root(n: Any, k: Any) -> Any:
    """k-th root of T_n: cos((2k + 1)*pi/(2n))."""
    from ...core import pi
    from ...functions import cos

    return cos((2 * k + 1) * pi / (2 * n))


def chebyshevu_root(n: Any, k: Any) -> Any:
    """k-th root of U_n: cos((k + 1)*pi/(n + 1))."""
    from ...core import pi
    from ...functions import cos

    return cos((k + 1) * pi / (n + 1))


__all__ = [
    "assoc_laguerre",
    "assoc_legendre",
    "chebyshevt",
    "chebyshevt_poly",
    "chebyshevt_root",
    "chebyshevu",
    "chebyshevu_poly",
    "chebyshevu_root",
    "gegenbauer",
    "gegenbauer_poly",
    "hermite",
    "hermite_poly",
    "hermite_prob",
    "hermite_prob_poly",
    "jacobi",
    "jacobi_poly",
    "laguerre",
    "laguerre_poly",
    "legendre",
    "legendre_poly",
]
