"""Integer solutions of Diophantine equations (upstream ``diophantine``).

Supported, exactly, with upstream's parametrizations:

* linear equations in any number of variables (``base_solution_linear``
  and upstream's coefficient-gcd chain; parameters ``t_0, t_1, ...``);
* ``a*x*y = n``: every divisor pair;
* homogeneous binary quadratics that factor over QQ into lines;
* positive-definite binary quadratics (bounded): every lattice point;
* the Pythagorean equation ``x**2 + y**2 = z**2``: ``(2*p*q, p**2 - q**2,
  p**2 + q**2)``.

Other classes (Pell-type, general ternary) raise NotImplementedError.
"""

from __future__ import annotations

from math import gcd, isqrt
from typing import Any


def _igcdex(a: int, b: int) -> tuple:
    """(x, y, g) with x*a + y*b = g = gcd(a, b) (upstream igcdex)."""
    if not a and not b:
        return (0, 1, 0)
    if not a:
        return (0, b // abs(b), abs(b))
    if not b:
        return (a // abs(a), 0, abs(a))
    x_sign = -1 if a < 0 else 1
    y_sign = -1 if b < 0 else 1
    a, b = abs(a), abs(b)
    x, y, r, s = 1, 0, 0, 1
    while b:
        c, q = a % b, a // b
        a, b, r, s, x, y = b, c, x - q * r, y - q * s, r, s
    return (x * x_sign, y * y_sign, a)


def base_solution_linear(c: int, a: int, b: int, t: Any = None) -> tuple:
    g = gcd(gcd(a, b), c) if c else gcd(a, b)
    if g > 1:
        a, b, c = a // g, b // g, c // g
    if c == 0:
        if t is None:
            return (0, 0)
        if b < 0:
            t = -t
        return (b * t, -a * t)
    x0, y0, d = _igcdex(abs(a), abs(b))
    x0 *= -1 if a < 0 else 1
    y0 *= -1 if b < 0 else 1
    if c % d == 0:
        if t is not None:
            if b < 0:
                t = -t
            return (c * x0 + b * t, c * y0 - a * t)
        return (c * x0, c * y0)
    return (None, None)


def _params(n: int) -> list:
    from ..core import Symbol

    return [Symbol("t_%d" % i, integer=True) for i in range(n)]


def _diop_linear(coeff: dict, var: list, const: int) -> set:
    from ..core import Add, Integer, Mul

    A = [int(coeff[v]) for v in var]
    g = 0
    for a in A:
        g = gcd(g, a)
    if const % g:
        return set()
    A = [a // g for a in A]
    c = Integer(-const // g)
    params = _params(len(var))
    if len(var) == 1:
        if c % A[0]:
            return set()
        return {(Integer(int(c) // A[0]),)}
    B = []
    if len(var) > 2:
        B.append(gcd(A[-2], A[-1]))
        A[-2] //= B[0]
        A[-1] //= B[0]
        for i in range(len(A) - 3, 0, -1):
            gg = gcd(B[0], A[i])
            B[0] //= gg
            A[i] //= gg
            B.insert(0, gg)
    B.append(A[-1])
    solutions = []
    for Ai, Bi in zip(A, B):
        tot_x, tot_y = [], []
        for arg in Add.make_args(c):
            if isinstance(arg, Integer):
                k, p = int(arg), Integer(1)
                pnew = params[0]
            else:
                kk, p = arg.as_coeff_Mul()
                k = int(kk)
                pnew = params[params.index(p) + 1]
            sol_x, sol_y = base_solution_linear(k, Ai, Bi, pnew)
            if p == 1:
                if sol_x is None:
                    return set()
            else:
                sol_x = _scale_param(sol_x, p)
                sol_y = _scale_param(sol_y, p)
            tot_x.append(sol_x)
            tot_y.append(sol_y)
        solutions.append(Add(*tot_x))
        c = Add(*tot_y)
    solutions.append(c)
    return {tuple(solutions)}


def _scale_param(s: Any, p: Any) -> Any:
    """Scale the parameter-free part of ``s`` by the parameter ``p``
    (upstream: sol.args[0]*p + sol.args[1] with the constant first)."""
    from ..core import Add, Integer

    if isinstance(s, Add):
        const = Add(*[a for a in s.args if not a.free_symbols])
        return const * p + (s - const)
    return s if getattr(s, "free_symbols", None) else s * p


def _divisors(n: int) -> list:
    n = abs(n)
    out = []
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            out.append(d)
            if d != n // d:
                out.append(n // d)
    return sorted(out)


def _diop_quadratic(coeffs: dict, x: Any, y: Any) -> set:
    """a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0."""
    from ..core import Integer

    a, b, c, d, e, f = (int(coeffs.get(k, 0)) for k in ("xx", "xy", "yy", "x", "y", "1"))
    # x*y = n
    if a == 0 and c == 0 and d == 0 and e == 0 and b != 0:
        if f % b:
            return set()
        n = -f // b
        if n == 0:
            raise NotImplementedError("diophantine: x*y = 0 has infinitely many solutions")
        out = set()
        for dv in _divisors(n):
            for s in (1, -1):
                p = s * dv
                out.add((Integer(p), Integer(n // p)))
        return out
    disc = b * b - 4 * a * c
    # homogeneous: lines through the origin
    if d == 0 and e == 0 and f == 0:
        if disc < 0:
            return {(Integer(0), Integer(0))}
        r = isqrt(disc)
        if r * r != disc:
            return {(Integer(0), Integer(0))}
        t = _params(1)[0]
        out = set()
        if a == 0:
            # y*(b*x + c*y) = 0
            out.add((t, Integer(0)))
            for sol in _line(b, c, t):
                out.add(sol)
            return out
        # a*x**2 + b*x*y + c*y**2 = a*(x - r1*y)*(x - r2*y), r = (-b ± sqrt)/(2a)
        for sgn in (1, -1):
            num, den = -b + sgn * r, 2 * a
            g = gcd(num, den)
            num, den = num // g, den // g
            # x = (num/den)*y, i.e. den*x - num*y = 0, through the linear
            # base solution (upstream's sign conventions).
            out.add(base_solution_linear(0, den, -num, t))
        return out
    if disc < 0:
        # bounded ellipse: enumerate x where the quadratic in y has roots
        out = set()
        # c*y**2 + (b*x + e)*y + (a*x**2 + d*x + f) = 0 has real y iff
        # (b*x + e)**2 - 4*c*(a*x**2 + d*x + f) >= 0
        A2 = b * b - 4 * a * c
        B2 = 2 * b * e - 4 * c * d
        C2 = e * e - 4 * c * f
        # A2 < 0: real x range from the roots of A2*x**2 + B2*x + C2
        D2 = B2 * B2 - 4 * A2 * C2
        if D2 < 0:
            return set()
        lo = int((-B2 + D2 ** 0.5) / (2 * A2)) - 2
        hi = int((-B2 - D2 ** 0.5) / (2 * A2)) + 2
        for xv in range(min(lo, hi), max(lo, hi) + 1):
            qa, qb, qc = c, b * xv + e, a * xv * xv + d * xv + f
            if qa == 0:
                continue
            dd = qb * qb - 4 * qa * qc
            if dd < 0:
                continue
            s = isqrt(dd)
            if s * s != dd:
                continue
            for root_num in {-qb + s, -qb - s}:
                if root_num % (2 * qa) == 0:
                    out.add((Integer(xv), Integer(root_num // (2 * qa))))
        return out
    raise NotImplementedError("diophantine: this binary quadratic class (Pell-type) is not supported")


def _line(b: int, c: int, t: Any) -> list:
    from ..core import Integer

    # b*x + c*y = 0
    g = gcd(b, c)
    return [(Integer(c // g) * t if c else t, Integer(-b // g) * t)]


def diophantine(eq: Any, param: Any = None, syms: Any = None, permute: bool = False) -> set:
    """Set of integer solution tuples of ``eq = 0`` in the order of
    ``syms`` (default: the free symbols sorted by name)."""
    from ..core import Add, Eq, Integer, Mul, Pow, Symbol, expand, sympify

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    eq = expand(sympify(eq))
    var = list(syms) if syms else sorted(eq.free_symbols, key=lambda s: s.name)
    from ..polys.polytools import Poly

    try:
        p = Poly(eq, *var)
    except Exception:
        raise NotImplementedError("diophantine: not a polynomial equation")
    terms = p.terms()
    if any(not isinstance(c, Integer) for _, c in terms):
        raise NotImplementedError("diophantine: integer coefficients required")
    deg = max(sum(m) for m, _ in terms)
    if len(var) == 1 and deg >= 2:
        from ..polys.polytools import roots as _roots

        return {(r,) for r in _roots(p) if isinstance(r, Integer)}
    if deg <= 1:
        coeff = {v: 0 for v in var}
        const = 0
        for m, c in terms:
            if sum(m) == 0:
                const = int(c)
            else:
                coeff[var[m.index(1)]] = int(c)
        return _diop_linear(coeff, var, const)
    if deg == 2 and len(var) == 2:
        x, y = var
        coeffs = {}
        names = {(2, 0): "xx", (1, 1): "xy", (0, 2): "yy", (1, 0): "x", (0, 1): "y", (0, 0): "1"}
        for m, c in terms:
            coeffs[names[tuple(m)]] = int(c)
        return _diop_quadratic(coeffs, x, y)
    if deg == 2 and len(var) == 3:
        sq = {tuple(m): int(c) for m, c in terms}
        diag = {i: sq.get(tuple(2 if j == i else 0 for j in range(3)), 0) for i in range(3)}
        if len(sq) == 3 and sorted(diag.values()) == [-1, 1, 1]:
            neg = [i for i in range(3) if diag[i] == -1][0]
            others = [i for i in range(3) if i != neg]
            pq = Symbol("p", integer=True), Symbol("q", integer=True)
            p_, q_ = pq
            sol = [None, None, None]
            sol[others[0]] = 2 * p_ * q_
            sol[others[1]] = p_**2 - q_**2
            sol[neg] = p_**2 + q_**2
            return {tuple(sol)}
    raise NotImplementedError("diophantine: equation class not supported")


__all__ = ["base_solution_linear", "diophantine"]
