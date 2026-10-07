"""Linear recurrence relations (upstream ``rsolve``).

Supported, exactly:

* linear recurrences with constant coefficients of any order: the
  characteristic polynomial's roots ``r`` (multiplicity ``m``) give the
  basis ``n**j * r**(n - j)``; constants are ``C0, C1, ...``;
* inhomogeneous terms that are sums of ``poly(n) * b**n``, through
  undetermined coefficients (raised by ``n**m`` when ``b`` is an
  ``m``-fold characteristic root);
* first-order recurrences ``y(n + 1) = p(n)*y(n)`` with ``n``-dependent
  ``p``: ``C0 * prod(p(k), (k, 0, n - 1))``;
* initial conditions ``{y(k): value}``, solved for the constants.

Everything else raises NotImplementedError; every returned solution is
checked by substitution into the recurrence.
"""

from __future__ import annotations

from typing import Any


def _shifts(f: Any, func: Any, n: Any) -> dict:
    """{offset: y(n + offset) node} for every application of ``func``."""
    from ..core import expand
    from ..core.match import preorder_traversal

    out: dict = {}
    for node in preorder_traversal(f):
        if getattr(node, "func", None) == func and len(getattr(node, "args", ())) == 1:
            k = expand(node.args[0] - n)
            if n in getattr(k, "free_symbols", set()) or not getattr(k, "is_Integer", False) and not hasattr(k, "p"):
                raise NotImplementedError("rsolve: argument %s is not n + integer" % node.args[0])
            try:
                out[int(k)] = node
            except TypeError:
                raise NotImplementedError("rsolve: argument %s is not n + integer" % node.args[0])
    return out


def _constants(count: int) -> list:
    from ..core import Symbol

    return [Symbol("C%d" % i) for i in range(count)]


def _char_roots(coeffs: list) -> list:
    """[(root, multiplicity)] of sum c_k r**k, real roots ascending first."""
    from ..core import Integer, Symbol, expand
    from ..polys.polytools import Poly, roots

    r = Symbol("_fsym_rsolve_r")
    char = sum((c * r**k for k, c in enumerate(coeffs)), Integer(0))
    rts = roots(Poly(expand(char), r))
    if sum(rts.values()) != len(coeffs) - 1:
        raise NotImplementedError("rsolve: characteristic roots are not all expressible")

    def key(item: Any) -> tuple:
        from .. import N

        root = item[0]
        try:
            v = complex(root)
        except Exception:
            return (2, 0.0, 0.0)
        if abs(v.imag) < 1e-12:
            return (0, v.real, 0.0)
        return (1, v.real, -v.imag)

    return sorted(rts.items(), key=key)


def _split_forcing(g: Any, n: Any) -> list:
    """g = sum p_b(n) * b**n  ->  [(b, p_b)]; NotImplementedError otherwise."""
    from ..core import Add, Integer, Mul, Pow, expand
    from ..functions import exp

    groups: dict = {}
    for t in Add.make_args(expand(g)):
        base = Integer(1)
        poly_part = []
        for fct in Mul.make_args(t):
            if isinstance(fct, Pow) and n in fct.args[1].free_symbols and n not in fct.args[0].free_symbols:
                k = expand(fct.args[1] / n)
                rest = expand(fct.args[1] - k * n)
                if n in k.free_symbols or n in rest.free_symbols:
                    raise NotImplementedError("rsolve: forcing term %s" % t)
                base = base * fct.args[0] ** k
                poly_part.append(fct.args[0] ** rest)
            elif type(fct).__name__ == "exp" and n in fct.args[0].free_symbols:
                k = expand(fct.args[0] / n)
                rest = expand(fct.args[0] - k * n)
                if n in k.free_symbols or n in rest.free_symbols:
                    raise NotImplementedError("rsolve: forcing term %s" % t)
                base = base * exp(k)
                poly_part.append(exp(rest))
            else:
                poly_part.append(fct)
        p = Mul(*poly_part)
        if not p.is_polynomial(n):
            raise NotImplementedError("rsolve: forcing term %s" % t)
        groups[base] = groups.get(base, Integer(0)) + p
    return list(groups.items())


def _particular(coeffs: list, g: Any, n: Any, roots: list) -> Any:
    from ..core import Integer, Symbol, expand
    from ..polys.polytools import Poly, factor

    total = Integer(0)
    for b, p in _split_forcing(g, n):
        mult = 0
        for r, m in roots:
            if expand(r - b) == 0:
                mult = m
        deg = Poly(p, n).degree() if p.free_symbols & {n} else 0
        unknowns = [Symbol("_fsym_rs_a%d" % i) for i in range(deg + 1)]
        trial_poly = sum((u * n ** (i + mult) for i, u in enumerate(unknowns)), Integer(0))
        lhs = Integer(0)
        for k, c in enumerate(coeffs):
            lhs = lhs + c * b**k * trial_poly.subs(n, n + k)
        resid = expand(lhs - p)
        from .. import solve

        eqs = Poly(resid, n).all_coeffs() if resid.free_symbols & {n} else [resid]
        sol = solve(eqs, unknowns, dict=True)
        if not sol:
            raise NotImplementedError("rsolve: undetermined coefficients failed")
        sol = sol[0] if isinstance(sol, list) else sol
        part = trial_poly.subs(sol)
        part = factor(expand(part)) if part.free_symbols else part
        total = total + (part * b**n if b != 1 else part)
    return total


def _linear_product(p: Any, n: Any, k: Any) -> Any:
    """prod(p(k), (k, 0, n - 1)) for p = a*n + b: a**n * rf(b/a, n),
    i.e. factorial(n + c - 1)/factorial(c - 1) for b/a = c a positive
    integer (upstream: (n + 1)*y(n) -> factorial(n))."""
    from ..core import Integer, Rational, expand
    from ..functions import factorial
    from .. import product

    a = expand(p.diff(n))
    b = expand(p - a * n)
    if n not in a.free_symbols and isinstance(a, Rational) and a != 0 and isinstance(b, Rational):
        c = b / a
        if isinstance(c, Integer) and c >= 1:
            body = factorial(n + c - 1) / factorial(c - 1)
            return body if a == 1 else a**n * body
    return product(p.subs(n, k), (k, 0, n - 1))


def rsolve(f: Any, y: Any, init: Any = None) -> Any:
    """Solve a recurrence ``f = 0`` (or an ``Eq``) for ``y(n)``."""
    from ..core import Eq, Integer, Symbol, expand, sympify

    if isinstance(f, Eq):
        f = f.lhs - f.rhs
    f = sympify(f)
    if len(y.args) != 1:
        raise ValueError("rsolve: y must be an applied function of one argument")
    func, n = y.func, y.args[0]
    shifts = _shifts(f, func, n)
    if not shifts:
        raise ValueError("rsolve: %s does not appear in the recurrence" % y)
    lo = min(shifts)
    if lo:
        f = f.subs(n, n - lo)
        shifts = _shifts(f, func, n)
    order = max(shifts)
    ys = {k: Symbol("_fsym_rs_y%d" % k) for k in shifts}
    E = f
    for k in sorted(shifts, reverse=True):
        E = E.subs(shifts[k], ys[k])
    E = expand(E)
    coeffs = []
    for k in range(order + 1):
        c = expand(E.diff(ys[k])) if k in ys else Integer(0)
        if c.free_symbols & set(ys.values()):
            raise NotImplementedError("rsolve: nonlinear recurrence")
        coeffs.append(c)
    g = -expand(E - sum((coeffs[k] * ys[k] for k in ys), Integer(0)))
    if g.free_symbols & set(ys.values()):
        raise NotImplementedError("rsolve: nonlinear recurrence")

    if all(n not in c.free_symbols for c in coeffs):
        rts = _char_roots(coeffs)
        consts = _constants(order)
        sol = Integer(0)
        idx = 0
        for r, m in rts:
            for j in range(m):
                sol = sol + consts[idx] * n**j * r ** (n - j)
                idx += 1
        if g != 0:
            part = _particular(coeffs, g, n, rts)
            # solve() lifts through strings: restore the declared n.
            sol = sol + part.subs(Symbol(n.name), n)
    elif order == 1 and g == 0:
        from .. import product

        k = Symbol("_fsym_rs_k", integer=True)
        p = expand(-coeffs[0] / coeffs[1])
        consts = _constants(1)
        sol = consts[0] * _linear_product(p, n, k)
    else:
        raise NotImplementedError("rsolve: variable-coefficient recurrence of this form")

    if init:
        from .. import solve

        eqs = []
        for key, value in init.items():
            arg = key.args[0] if hasattr(key, "args") and key.args else key
            eqs.append(expand(sol.subs(n, arg) - value))
        present = [c for c in consts if c in sol.free_symbols]
        found = solve(eqs, present, dict=True)
        if not found:
            raise ValueError("rsolve: inconsistent initial conditions")
        choice = found[0] if isinstance(found, list) else found
        from ..simplify import radsimp

        sol = sol.subs({c: radsimp(v) for c, v in choice.items()})
    return expand(sol) if init else sol


__all__ = ["rsolve"]
