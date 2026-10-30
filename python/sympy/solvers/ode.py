"""Exact ordinary differential equation (ODE) solvers for FrankenSymPy (WS19)."""

from __future__ import annotations

from typing import Any
from ..core import (
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)


def dsolve_linear_first_order(P: Any, Q: Any, x: Any) -> Any:
    """Solve first-order linear ODE: y'(x) + P(x)*y(x) = Q(x)."""
    x_sym = _require_symbol(x)
    p_str = str(_wrap(_native_expr(P)))
    q_str = str(_wrap(_native_expr(Q)))
    res = _native.dsolve_linear_first_order_expr(p_str, q_str, _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_const_coeff_second_order(a: int, b: int, c: int, x: Any) -> Any:
    """Solve homogeneous 2nd-order ODE with constant coefficients: a*y''(x) + b*y'(x) + c*y(x) = 0."""
    x_sym = _require_symbol(x)
    res = _native.dsolve_const_coeff_second_order_expr(int(a), int(b), int(c), _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_const_coeff_second_order_nonhomogeneous(
    a: int, b: int, c: int, f: Any, x: Any
) -> Any:
    """Solve non-homogeneous 2nd-order ODE with constant coefficients: a*y''(x) + b*y'(x) + c*y(x) = f(x)."""
    x_sym = _require_symbol(x)
    f_str = str(_wrap(_native_expr(f)))
    res = _native.dsolve_const_coeff_second_order_nonhomogeneous_expr(
        int(a), int(b), int(c), f_str, _native_symbol_key(x_sym)
    )
    return _parse_result(res)


def dsolve_cauchy_euler(a: int, b: int, c: int, x: Any) -> Any:
    """Solve a*x**2*y'' + b*x*y' + c*y = 0 on the positive real axis x > 0.

    Powers and logarithms use that explicit domain. This helper does not
    impose initial conditions or extend the family across the singularity
    at x = 0.
    """
    x_sym = _require_symbol(x)
    res = _native.dsolve_cauchy_euler_expr(int(a), int(b), int(c), _native_symbol_key(x_sym))
    return _parse_result(res)


def dsolve_separable_linear(f: Any, x: Any) -> Any:
    """Solve separable ODE: y'(x) = f(x)*y(x)."""
    x_sym = _require_symbol(x)
    f_str = str(_wrap(_native_expr(f)))
    res = _native.dsolve_separable_linear_expr(f_str, _native_symbol_key(x_sym))
    return _parse_result(res)


def checkodesol(
    ode: Any,
    sol: Any,
    func: Any = None,
    order: Any = "auto",
    solve_for_func: bool = True,
) -> Any:
    """Check whether a candidate solution satisfies a given ODE.

    Returns (True, 0) if the candidate solution is verified to satisfy the ODE,
    or (False, residual) if the residual is nonzero.

    If sol is a list or tuple of solutions, returns a list of (bool, residual) tuples.
    """
    if isinstance(sol, (list, tuple)):
        return [
            checkodesol(ode, s, func=func, order=order, solve_for_func=solve_for_func)
            for s in sol
        ]

    from ..core import AppliedUndef, Equality, Integer

    if isinstance(sol, Equality):
        target = func
        if target is None:
            if isinstance(sol.lhs, AppliedUndef):
                target = sol.lhs
            elif isinstance(sol.rhs, AppliedUndef):
                target = sol.rhs
            else:
                target = _find_function(ode)
        if sol.lhs != target and sol.rhs != target:
            # Admit both original sides before subtraction can evaluate a
            # held power or erase an excluded point. Keep a refused input
            # unevaluated as the diagnostic residual.
            if _check_exact_ode_relation(ode, sol, target):
                return (True, 0)
            return (False, ode)

    if isinstance(ode, Equality):
        ode_expr = ode.lhs - ode.rhs
    else:
        ode_expr = _wrap(_native_expr(ode))

    if isinstance(sol, Equality):
        if func is None:
            func = target
            rhs = sol.rhs if sol.lhs == target else sol.lhs
        else:
            func = _wrap(_native_expr(func))
            if sol.lhs == func:
                rhs = sol.rhs
            elif sol.rhs == func:
                rhs = sol.lhs
            else:
                rhs = sol.rhs
    else:
        if func is None:
            known_atom_names = {
                "Add", "Mul", "Pow", "Symbol", "Integer", "Rational", "Float",
                "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh",
                "cot", "sec", "csc", "coth", "sech", "csch",
                "asin", "acos", "atan", "acot", "asec", "acsc",
                "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                "sinc", "erf", "erfc", "sign", "gamma", "zeta",
                "factorial", "binomial", "fibonacci", "lucas", "harmonic",
                "catalan", "bernoulli", "bell", "subfactorial", "floor", "ceiling",
                "log", "ln", "sqrt", "diff", "Derivative", "Abs",
            }
            candidates = []
            for atom in ode_expr.atoms():
                fname = getattr(atom, "func", None)
                fname_str = getattr(fname, "__name__", None) or str(fname)
                if fname_str not in known_atom_names and hasattr(atom, "args") and atom.args:
                    candidates.append(atom)
            if candidates:
                func = candidates[0]
            else:
                raise ValueError("func must be specified if sol is not an Equality")
        else:
            func = _wrap(_native_expr(func))
        rhs = _wrap(_native_expr(sol))

    subbed = ode_expr.subs(func, rhs)
    evaluated = subbed.doit()
    residual = evaluated.simplify()

    if getattr(residual, "is_zero", False) or residual == 0 or residual == Integer(0):
        return (True, 0)
    else:
        return (False, residual)


def verify_linear_first_order_solution(sol: Any, P: Any, Q: Any, x: Any) -> bool:
    """Verify candidate solution of y'(x) + P(x)*y(x) = Q(x) via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    p_str = str(_wrap(_native_expr(P)))
    q_str = str(_wrap(_native_expr(Q)))
    return _native.verify_linear_first_order_solution_expr(
        sol_str, p_str, q_str, _native_symbol_key(x_sym)
    )


def verify_const_coeff_second_order_solution(sol: Any, a: int, b: int, c: int, x: Any) -> bool:
    """Verify candidate solution of a*y''(x) + b*y'(x) + c*y(x) = 0 via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    return _native.verify_const_coeff_second_order_solution_expr(
        sol_str, int(a), int(b), int(c), _native_symbol_key(x_sym)
    )


def verify_cauchy_euler_solution(sol: Any, a: int, b: int, c: int, x: Any) -> bool:
    """Verify candidate solution of a*x^2*y''(x) + b*x*y'(x) + c*y(x) = 0 via independent native verifier."""
    x_sym = _require_symbol(x)
    sol_str = str(_wrap(_native_expr(sol)))
    return _native.verify_cauchy_euler_solution_expr(
        sol_str, int(a), int(b), int(c), _native_symbol_key(x_sym)
    )


__all__ = [
    "checkodesol",
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
    "verify_cauchy_euler_solution",
    "verify_const_coeff_second_order_solution",
    "verify_linear_first_order_solution",
]


# ---------------------------------------------------------------------------
# General dsolve
# ---------------------------------------------------------------------------


def _preorder(e):
    stack = [(e, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > 8192 or depth > 4096:
            raise ValueError("dsolve: expression exceeds the bounded traversal envelope")
        yield node
        children = getattr(node, "args", ())
        if count + len(stack) + len(children) > 8192:
            raise ValueError("dsolve: expression exceeds the bounded traversal envelope")
        stack.extend((child, depth + 1) for child in reversed(children))


def _find_function(eq):
    from ..core import AppliedUndef

    found = [n for n in _preorder(eq) if isinstance(n, AppliedUndef)]
    funcs = []
    for n in found:
        if n not in funcs:
            funcs.append(n)
    if len(funcs) != 1:
        raise ValueError(f"The function cannot be automatically detected for {eq}.")
    return funcs[0]


def _derivative_order(d, f, x):
    """Order of Derivative d of f wrt x, or None when it is not one."""
    if type(d).__name__ != "Derivative":
        return None
    args = d.args
    if args[0] != f:
        return None
    variables = list(args[1:])
    if any(v != x for v in variables):
        return None
    return len(variables)


def _ode_to_symbols(eq, f, x):
    """Replace f(x) and its x-derivatives by symbols y0, y1, ..."""
    from ..core import Dummy as _S

    orders = {}
    for n in _preorder(eq):
        k = _derivative_order(n, f, x)
        if k is not None:
            orders[k] = n
    n_order = max(orders) if orders else 0
    ys = [_S(f"_fsym_y{k}") for k in range(n_order + 1)]
    out = eq
    for k in sorted(orders, reverse=True):
        out = out.subs(orders[k], ys[k])
    out = out.subs(f, ys[0])
    return out, ys, n_order


def _constants(start, count):
    from ..core import Symbol as _S

    return [_S(f"C{i}") for i in range(start, start + count)]


def _homogeneous_basis(coeffs, x):
    """Fundamental solutions of sum a_k y^(k) = 0 with constant a_k."""
    from ..core import Symbol as _S, I, Integer, Rational
    from ..functions import exp, sin, cos
    from ..polys.polytools import Poly, roots
    from ..core import expand

    r = _S("_fsym_r")
    char = sum((c * r**k for k, c in enumerate(coeffs)), Integer(0))
    rts = roots(Poly(expand(char), r))
    if sum(rts.values()) != len(coeffs) - 1:
        raise NotImplementedError("dsolve: characteristic roots are not all expressible")
    real = []
    pairs = []
    seen = set()
    for root, mult in rts.items():
        re_part = expand(root.subs(I, 0))
        im_part = expand((root - re_part) / I)
        if im_part == 0:
            real.append((root, mult))
        else:
            key = (str(re_part), str(abs(im_part)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((re_part, abs(im_part), mult))
    real.sort(key=lambda rm: float(rm[0]))
    groups = []
    for root, mult in real:
        groups.append(("real", root, mult))
    for a, b, mult in pairs:
        groups.append(("pair", (a, b), mult))
    return groups


def _assemble(groups, x, start=1):
    from ..core import Integer
    from ..functions import exp, sin, cos

    terms = []
    basis = []
    idx = start
    for kind, data, mult in groups:
        if kind == "real":
            cs = _constants(idx, mult)
            idx += mult
            poly = sum((c * x**j for j, c in enumerate(cs)), Integer(0))
            terms.append(poly * exp(data * x))
            basis.extend(x**j * exp(data * x) for j in range(mult))
        else:
            a, b = data
            sin_poly = Integer(0)
            cos_poly = Integer(0)
            for j in range(mult):
                c1, c2 = _constants(idx, 2)
                idx += 2
                sin_poly = sin_poly + c1 * x**j
                cos_poly = cos_poly + c2 * x**j
                basis.append(x**j * exp(a * x) * sin(b * x))
                basis.append(x**j * exp(a * x) * cos(b * x))
            terms.append((sin_poly * sin(b * x) + cos_poly * cos(b * x)) * exp(a * x))
    total = Integer(0)
    for t in terms:
        total = total + t
    return total, basis


def _char_solutions(coeffs, x):
    """Fundamental solutions and collect terms of sum a_k y^(k) = 0 with
    constant a_k (upstream ``_get_const_characteristic_eq_sols``)."""
    from ..core import Symbol as _S, I, Integer, expand
    from ..functions import exp, sin, cos
    from ..polys.polytools import Poly, roots

    r = _S("_fsym_r")
    char = sum((c * r**k for k, c in enumerate(coeffs)), Integer(0))
    rts = roots(Poly(expand(char), r))
    if sum(rts.values()) != len(coeffs) - 1:
        raise NotImplementedError("dsolve: characteristic roots are not all expressible")
    gensols, collectterms, seen = [], [], set()
    for root, mult in rts.items():
        re_part = expand(root.subs(I, 0))
        im_part = expand((root - re_part) / I)
        for i in range(mult):
            if im_part == 0:
                gensols.append(x**i * exp(re_part * x))
                collectterms.insert(0, (i, re_part, Integer(0)))
                continue
            collectterms.insert(0, (i, re_part, im_part))
            key = (i, str(re_part), str(abs(im_part)))
            if key in seen:
                continue
            seen.add(key)
            gensols.append(x**i * exp(re_part * x) * sin(abs(im_part) * x))
            gensols.append(x**i * exp(re_part * x) * cos(im_part * x))
    return gensols, collectterms


def _euler_solutions(coeffs, x):
    """Fundamental solutions of sum a_k x^k y^(k) = 0 (Cauchy-Euler):
    x**r for each root of sum a_k r(r-1)...(r-k+1)."""
    from ..core import Symbol as _S, I, Integer, expand
    from ..functions import cos, log, sin
    from ..polys.polytools import Poly, roots

    r = _S("_fsym_r")
    char = Integer(0)
    for k, a in enumerate(coeffs):
        falling = Integer(1)
        for j in range(k):
            falling = falling * (r - j)
        char = char + a * falling
    rts = roots(Poly(expand(char), r))
    if sum(rts.values()) != len(coeffs) - 1:
        raise NotImplementedError("dsolve: indicial roots are not all expressible")
    gensols, seen = [], set()
    for root, mult in rts.items():
        re_part = expand(root.subs(I, 0))
        im_part = expand((root - re_part) / I)
        for i in range(mult):
            if im_part == 0:
                gensols.append(log(x) ** i * x**re_part)
                continue
            key = (i, str(re_part), str(abs(im_part)))
            if key in seen:
                continue
            seen.add(key)
            gensols.append(log(x) ** i * x**re_part * sin(abs(im_part) * log(x)))
            gensols.append(log(x) ** i * x**re_part * cos(im_part * log(x)))
    return gensols


def _apply_operator(coeffs, y, x):
    from ..core import Integer, diff

    return sum((c * diff(y, x, k) if k else c * y for k, c in enumerate(coeffs)), Integer(0))


def _uc_test(e, x):
    """Upstream ``_undetermined_coefficients_match`` shape test."""
    from ..core import Add, Mul, Pow, Rational

    if x not in e.free_symbols:
        return True
    if isinstance(e, Add):
        return all(_uc_test(a, x) for a in e.args)
    if isinstance(e, Mul):
        trig = [a for a in e.args if any(type(n).__name__ in ("sin", "cos") for n in _preorder(a))]
        if len(trig) > 1:
            return False
        return all(_uc_test(a, x) for a in e.args)
    name = type(e).__name__
    if name in ("sin", "cos", "exp", "sinh", "cosh"):
        from ..core import diff, expand

        u = e.args[0]
        a = expand(diff(u, x))
        return x not in a.free_symbols and x not in expand(u - a * x).free_symbols
    if isinstance(e, Pow):
        b, k = e.args
        if b == x and isinstance(k, Rational) and k.q == 1 and k >= 0:
            return True
        return False
    return e == x


def _strip_coeff(t, x):
    from ..core import Integer, Mul

    return Mul(*[f for f in Mul.make_args(t) if x in f.free_symbols]) if x in t.free_symbols else Integer(1)


def _uc_trial_set(g, x, coeffs):
    """Trial functions for undetermined coefficients, or None."""
    from ..core import Add, diff, expand
    from ..simplify import powsimp

    g = powsimp(expand(g), combine="exp")
    if not _uc_test(g, x):
        return None
    trial = []
    for term in Add.make_args(g):
        act = [_strip_coeff(term, x)]
        i = 0
        while i < len(act):
            d = expand(diff(act[i], x))
            for t in Add.make_args(d):
                if t == 0:
                    continue
                s = _strip_coeff(t, x)
                if s not in act:
                    act.append(s)
            i += 1
            if len(act) > 40:
                return None
        while any(expand(_apply_operator(coeffs, ts, x)) == 0 for ts in act):
            act = [expand(x * ts) for ts in act]
        for ts in act:
            if ts not in trial:
                trial.append(ts)
    return trial


def _particular_uc(trial, g, coeffs, x):
    from ..core import Add, Integer, Symbol as _S, expand
    from .. import solve as _solve

    a = [_S(f"_fsym_a{i}") for i in range(len(trial))]
    yp = sum((ai * t for ai, t in zip(a, trial)), Integer(0))
    resid = expand(_apply_operator(coeffs, yp, x) - g)
    groups: dict = {}
    for t in Add.make_args(resid):
        key = _strip_coeff(t, x)
        groups[key] = groups.get(key, Integer(0)) + expand(t / key)
    found = _solve(list(groups.values()), a, dict=True)
    if not found:
        raise NotImplementedError("dsolve: undetermined coefficients failed")
    sol = found[0] if isinstance(found, list) else found
    return expand(yp.subs(sol))


def _simplified_linear(sol, x, collectterms):
    """Upstream ``_get_simplified_sol``: collect on the basis functions."""
    from ..core import expand
    from ..functions import cos, exp, sin
    from ..printing.str import sort_key
    from ..simplify import collect, powsimp

    terms = sorted(collectterms, key=lambda t: (t[0], sort_key(t[1]), sort_key(t[2])), reverse=True)
    sol = expand(sol)
    for i, re_part, im_part in terms:
        if im_part != 0:
            sol = collect(sol, x**i * exp(re_part * x) * sin(abs(im_part) * x))
            sol = collect(sol, x**i * exp(re_part * x) * cos(im_part * x))
    for i, re_part, _ in terms:
        sol = collect(sol, x**i * exp(re_part * x))
    return powsimp(sol)


def _count(e, c):
    return sum(1 for n in _preorder(e) if n == c)


def _is_const(e, cs):
    free = getattr(e, "free_symbols", set())
    return free <= cs


def _absorb(expr, cs):
    """One pass of upstream ``constantsimp`` absorption: a subexpression
    built only from numbers and constants that hold no other occurrence
    collapses to its first constant; ``exp(C + R)`` becomes ``C*exp(R)``."""
    from ..core import Add, Mul

    total = {c: _count(expr, c) for c in cs}

    def first(xe, whole=False):
        # ``whole``: xe is a node, so every copy of it collapses alike
        # (upstream reaches the same through cse of repeated subterms).
        syms = sorted(xe.free_symbols & cs, key=str)
        copies = _count(expr, xe) if whole else 1
        if syms and all(_count(xe, c) * max(copies, 1) == total[c] for c in syms):
            return syms[0]
        return None

    def walk(e):
        args = getattr(e, "args", ())
        if not args or not (e.free_symbols & cs):
            return e
        if _is_const(e, cs):
            c = first(e, whole=True)
            if c is not None:
                return c
        if type(e).__name__ == "exp" and isinstance(e.args[0], Add):
            inner = Add.make_args(e.args[0])
            const = [t for t in inner if _is_const(t, cs) and t.free_symbols]
            if const:
                c = first(Add(*const))
                if c is not None:
                    from ..functions import exp

                    rest = Add(*[t for t in inner if t not in const])
                    return c * exp(walk(rest))
        if isinstance(e, (Add, Mul)):
            const = [t for t in e.args if _is_const(t, cs)]
            if len(const) > 1 and any(t.free_symbols for t in const):
                c = first(e.func(*const))
                if c is not None:
                    others = [walk(t) for t in e.args if t not in const]
                    return e.func(c, *others)
        return e.func(*[walk(a) for a in args])

    return walk(expr)


def _scale_constant_sums(expr, cs):
    """Clean a constant-bearing sum under a power (upstream solve output
    shapes): ``(-C - x)**-1 -> -(C + x)**-1``, ``(C + x**2/2)**-1 ->
    2*(C + x**2)**-1`` and ``sqrt(2)*sqrt(C + x) -> sqrt(C + 2*x)``."""
    from math import lcm
    from ..core import Add, Integer, Mul, Pow, Rational

    def split(add):
        const = [t for t in add.args if _is_const(t, cs) and t.free_symbols]
        if len(const) != 1:
            return None
        c = const[0]
        sym = sorted(c.free_symbols, key=str)[0]
        if not (c == sym or c == -sym) or _count(expr, sym) != 1:
            return None
        return sym, [t for t in add.args if t is not c]

    def coeff(t):
        m = Mul.make_args(t)
        return m[0] if isinstance(m[0], Rational) else Integer(1)

    def walk(e):
        args = getattr(e, "args", ())
        if not args:
            return e
        e = e.func(*[walk(a) for a in args])
        if isinstance(e, Pow) and isinstance(e.args[0], Add):
            base, k = e.args
            sp = split(base)
            if sp is not None and isinstance(k, Rational):
                c, rest = sp
                scale = Integer(lcm(*[int(coeff(t).q) for t in rest]))
                if k.q == 1 and all(coeff(t) < 0 for t in rest):
                    scale = -scale
                if scale != 1 and (k.q == 1 or scale > 0):
                    new = Add(c, *[scale * t for t in rest])
                    return scale ** (-k) * Pow(new, k)
        if isinstance(e, Mul):
            facs = list(e.args)
            for i, f in enumerate(facs):
                if not (isinstance(f, Pow) and isinstance(f.args[0], Add)):
                    continue
                base, k = f.args
                if not isinstance(k, Rational) or k.q == 1 or split(base) is None:
                    continue
                for j, g in enumerate(facs):
                    if j != i and isinstance(g, Pow) and isinstance(g.args[0], Rational) and g.args[1] == k and g.args[0] > 0:
                        c, rest = split(base)
                        new = Pow(Add(c, *[g.args[0] * t for t in rest]), k)
                        return Mul(*[h for n, h in enumerate(facs) if n not in (i, j)], new)
        return e

    return walk(expr)


def _monomial_numerator(expr, cs):
    """``N/D`` with an exponential monomial ``N`` (times constants) over a
    constant-bearing sum ``D`` -> ``1/(D/N)`` when that leaves a bare
    number in the sum: ``exp(x)/(C1 + exp(x)) -> 1/(C1*exp(-x) + 1)``
    (the shape upstream's logistic-type solutions take)."""
    from ..core import Add, Integer, Mul, Rational, expand
    from .. import fraction

    num, den = fraction(expr)
    if not isinstance(den, Add) or isinstance(num, Add) or num == 1:
        return expr
    if not all(type(f).__name__ == "exp" or f in cs or isinstance(f, Rational) for f in Mul.make_args(num)):
        return expr
    if not (den.free_symbols & cs) or _count(den, sorted(den.free_symbols & cs, key=str)[0]) != 1:
        return expr
    new = expand(den / num)
    if not isinstance(new, Add) or not any(isinstance(t, Rational) for t in new.args):
        return expr
    return Integer(1) / new


def _constant_renumber(expr, cs):
    """Upstream ``constant_renumber``: constants numbered C1, C2, ... in
    the canonical (default_sort_key) traversal order."""
    from ..core import Add, Mul, Symbol as _S
    from ..printing.str import sort_key

    ones = {c: 1 for c in cs}
    found: list = []

    def walk(e):
        if not (getattr(e, "free_symbols", set()) & cs):
            return
        if e in cs:
            if e not in found:
                found.append(e)
            return
        args = list(getattr(e, "args", ()))
        if isinstance(e, (Add, Mul)):
            args.sort(key=lambda a: sort_key(a.subs(ones)))
        for a in args:
            walk(a)

    walk(expr)
    mapping = {old: _S(f"C{i + 1}") for i, old in enumerate(found)}
    if all(k == v for k, v in mapping.items()):
        return expr
    tmp = {old: _S(f"_fsym_k{i}") for i, old in enumerate(found)}
    out = expr.subs(tmp)
    return out.subs({tmp[old]: mapping[old] for old in found})


def _odesimp(rhs, cs):
    from ..polys.polytools import terms_gcd

    for _ in range(4):
        new = _scale_constant_sums(_absorb(rhs, cs), cs)
        new = _monomial_numerator(new, cs)
        if new == rhs:
            break
        rhs = new
        cs = cs & rhs.free_symbols | {c for c in cs if c in rhs.free_symbols}
    rhs = terms_gcd(rhs, clear=False, deep=True)
    return _constant_renumber(rhs, cs & rhs.free_symbols)


def _particular_vop(basis, g, lead, x):
    """Variation of parameters for order 1 or 2."""
    from ..core import Integer, diff, expand
    from ..simplify import simplify
    from .. import integrate as _integrate
    from ..integrals import Integral

    g = g / lead
    if len(basis) == 1:
        y1 = basis[0]
        integral = _integrate(simplify(g / y1), x)
        if isinstance(integral, Integral):
            raise NotImplementedError("dsolve: particular integral not found")
        return simplify(expand(y1 * integral))
    if len(basis) == 2:
        y1, y2 = basis
        w = simplify(y1 * diff(y2, x) - y2 * diff(y1, x))
        i1 = _integrate(simplify(y2 * g / w), x)
        i2 = _integrate(simplify(y1 * g / w), x)
        if isinstance(i1, Integral) or isinstance(i2, Integral):
            raise NotImplementedError("dsolve: particular integral not found")
        return simplify(expand(-y1 * i1 + y2 * i2))
    raise NotImplementedError("dsolve: inhomogeneous equations of order > 2")


def _euler_coefficients(coeffs, x):
    """``a_k`` with ``coeffs[k] == a_k * x**k`` (Cauchy-Euler), else None."""
    from ..core import expand

    out = []
    for k, c in enumerate(coeffs):
        a = expand(c / x**k)
        if x in a.free_symbols:
            return None
        out.append(a)
    return out


def _solve_separable(left, right, y):
    """Solve ``left(y) = right`` for ``y``; a log-sum ``left`` combines to
    ``log(u)`` first and ``u = exp(right)`` is solved instead."""
    from .. import solve as _solve
    from ..functions import exp
    from ..simplify import logcombine

    combined = logcombine(left, force=True)
    if type(combined).__name__ == "log":
        try:
            return _solve(combined.args[0] - exp(right), y)
        except (NotImplementedError, ValueError):
            pass
    return _solve(left - right, y)


def _exact_ode_source(equation, function, variable, *, values=None):
    """Admit exact polynomial syntax, then bound work before normalization.

    The checked lane permits degree at most 17, at most 128 possible expanded
    monomials, and conservative 4096-bit coefficient growth. Degree 17 admits
    the potential of a degree-16 one-form. Native checks still enforce the
    original one-form bounds. Unsupported syntax returns False; exhausted
    bounds are terminal errors. Optional exact coordinate values preflight
    initial-condition substitution without first evaluating the polynomial.
    """
    from math import comb
    from ..core import (
        Add, AppliedUndef, Derivative, Dummy, Eq, Integer, Mul, Pow, Rational,
        S, Symbol, UndefinedFunction,
    )

    cls = type(function)
    if (type(cls) is not UndefinedFunction or cls.__bases__ != (AppliedUndef,)
            or set(cls.__dict__) - {"__module__", "__dict__", "__weakref__", "__doc__"}):
        return False
    if type(variable) not in (Symbol, Dummy):
        return False
    exact_atoms = (Integer, Rational, type(S.Zero), type(S.One),
                   type(S.NegativeOne), type(S.Half))
    integer_atoms = (Integer, type(S.Zero), type(S.One), type(S.NegativeOne))
    roots = equation.args if type(equation) is Eq else (equation,)
    todo = [("finish", ("add", len(roots)), 0)]
    todo.extend(("visit", node, 0) for node in reversed(roots))
    postfix = []
    seen = 0
    # First establish exact syntax, so unrelated legacy function classes do
    # not inherit polynomial-only degree or coefficient restrictions.
    while todo:
        action, node, depth = todo.pop()
        if action == "finish":
            postfix.append(node)
            continue
        seen += 1
        if seen > 8192 or depth > 4096:
            raise ValueError("bounded exact polynomial ODE: source exceeds the expression envelope")
        if type(node) in exact_atoms:
            postfix.append(("number", node))
            continue
        if type(node) in (Symbol, Dummy):
            if node != variable:
                return False
            postfix.append(("x", None))
            continue
        if type(node) is cls:
            if node != function:
                return False
            postfix.append(("y", None))
            continue
        if type(node) is Derivative:
            if node.args != (function, variable):
                return False
            postfix.append(("dy", None))
            continue
        if type(node) in (Add, Mul):
            children = node.args
            if not children:
                return False
            if seen + len(todo) + len(children) > 8192:
                raise ValueError("bounded exact polynomial ODE: source exceeds the expression envelope")
            todo.append(("finish", ("add" if type(node) is Add else "mul", len(children)), depth))
            todo.extend(("visit", child, depth + 1) for child in reversed(children))
            continue
        if type(node) is Pow:
            base, exponent = node.args
            if type(exponent) not in integer_atoms or exponent.p < 0:
                return False
            todo.append(("finish", ("pow", exponent.p), depth))
            # Even a zero power must inspect its original base.
            todo.append(("visit", base, depth + 1))
            continue
        return False

    def number_bounds(number):
        numerator = number if type(number) is int else number.p
        denominator = 1 if type(number) is int else getattr(number, "q", 1)
        return (0, 0, abs(numerator).bit_length(),
                0 if denominator == 1 else denominator.bit_length(), 0, 1)

    def bounded(bounds):
        low, high, numerator, denominator, variables, terms = bounds
        if high > 17:
            raise ValueError("bounded exact polynomial ODE: source degree exceeds 17")
        if numerator > 4096 or denominator > 4096:
            raise ValueError("bounded exact polynomial ODE: coefficient growth exceeds 4096 bits")
        # A degree band in the used variables also bounds distinct monomials.
        # This admits (x+y)**16 without allocating its expansion.
        count = variables.bit_count()
        dense = (comb(high + count, count)
                 - (comb(low - 1 + count, count) if low else 0)) if count else 1
        terms = min(terms, dense)
        if terms > 128:
            raise ValueError("bounded exact polynomial ODE: source monomial bound exceeds 128")
        return low, high, numerator, denominator, variables, terms

    stack = []
    for kind, payload in postfix:
        if kind == "number":
            result = number_bounds(payload)
        elif kind in ("x", "y", "dy"):
            if values is not None and kind != "dy":
                result = number_bounds(values[0 if kind == "x" else 1])
            else:
                result = (1, 1, 1, 0, {"x": 1, "y": 2, "dy": 4}[kind], 1)
        elif kind == "pow":
            if payload > 64:
                raise ValueError("bounded exact polynomial ODE: literal power exceeds 64")
            low, high, numerator, denominator, variables, terms = stack.pop()
            result = ((low*payload, high*payload, numerator*payload,
                       denominator*payload, variables, terms**payload)
                      if payload else (0, 0, 1, 0, 0, 1))
        else:
            children = stack[-payload:]
            del stack[-payload:]
            result = children[0]
            for other in children[1:]:
                lo, hi, nb, db, mask, nt = result
                olo, ohi, onb, odb, omask, ont = other
                if kind == "add":
                    result = (min(lo, olo), max(hi, ohi),
                              max(nb + odb, onb + db) + 1, db + odb,
                              mask | omask, nt + ont)
                else:
                    result = (lo + olo, hi + ohi, nb + onb, db + odb,
                              mask | omask, nt * ont)
                result = bounded(result)
        stack.append(bounded(result))
    return True


def _exact_ode_coefficients(expression, ys):
    """Extract M + N*y' without dividing by the derivative coefficient."""
    from ..core import diff

    if len(ys) != 2:
        return None
    y, dy = ys
    n = diff(expression, dy)
    if dy in n.free_symbols or n == 0:
        return None
    m = expression.subs(dy, 0)
    return m, n, y


def _exact_ode_relation(expression, ys, variable, function):
    from ..core import Eq
    from .exact_ode import dsolve_exact_first_order

    form = _exact_ode_coefficients(expression, ys)
    if form is None:
        return None
    m, n, y = form
    try:
        relation = dsolve_exact_first_order(m, n, variable, y)
    except ValueError as error:
        # This is the native classifier's mathematical non-match. Resource,
        # input, and verification failures must propagate, never fall through.
        if "the polynomial differential form is not exact: M_y differs from N_x" not in str(error):
            raise
        return None
    return Eq(relation.lhs.subs(y, function), relation.rhs, evaluate=False)


def _check_exact_ode_relation(equation, relation, function):
    """Check a first-order polynomial first integral, without division by F_y.

    Acceptance covers the native potential normalization and nonzero rational
    constant multiples. It asserts the differential identity only: no local
    graph, uniqueness at a singular point, or completeness is certified.
    """
    from ..core import Dummy, Eq, Integer, Rational, S, diff
    from ..simplify import simplify

    if len(function.args) != 1:
        return False
    x = function.args[0]
    if not _exact_ode_source(equation, function, x):
        return False
    original = equation.lhs - equation.rhs if type(equation) is Eq else equation
    expression, ys, order = _ode_to_symbols(original, function, x)
    if order != 1:
        return False
    form = _exact_ode_coefficients(expression, ys)
    if form is None:
        return False
    m, n, y = form
    lhs, rhs = relation.lhs, relation.rhs
    if not lhs.has(function) and rhs.has(function):
        lhs, rhs = rhs, lhs
    if rhs.has(function) or x in rhs.free_symbols:
        return False
    if not _exact_ode_source(lhs, function, x):
        return False
    # The level is a symbol or an exact rational, as returned by dsolve.
    exact_atoms = (Integer, Rational, type(S.Zero), type(S.One),
                   type(S.NegativeOne), type(S.Half))
    if type(rhs) not in exact_atoms + (Symbol, Dummy):
        return False
    potential = lhs.subs(function, y)
    if _native.verify_exact_ode_potential_native(
        _native_expr(potential), _native_expr(m), _native_expr(n),
        _native_expr(x), _native_expr(y),
    ):
        return True
    scale = simplify(diff(potential, y) / n)
    if type(scale) not in exact_atoms or scale == 0:
        return False
    return bool(_native.verify_exact_ode_potential_native(
        _native_expr(potential), _native_expr(scale * m), _native_expr(scale * n),
        _native_expr(x), _native_expr(y),
    ))


def _apply_exact_ode_ics(relation, ics):
    """Bind a polynomial level set using exact value initial conditions."""
    from ..core import Eq, Integer, Rational, S, sympify

    function = _find_function(relation)
    x = function.args[0]
    level = relation.rhs
    if type(level) is not Symbol or level in relation.lhs.free_symbols:
        raise NotImplementedError("dsolve: unsupported implicit integration constant")
    exact_atoms = (Integer, Rational, type(S.Zero), type(S.One),
                   type(S.NegativeOne), type(S.Half))
    chosen = None
    for key, raw_value in ics.items():
        if type(key) is not type(function) or len(key.args) != 1:
            raise NotImplementedError("dsolve: implicit solutions support value initial conditions only")
        if type(raw_value) not in exact_atoms + (int,):
            raise NotImplementedError("dsolve: implicit initial data must be exact rational values")
        if type(raw_value) is int and raw_value.bit_length() > 4096:
            raise ValueError("bounded exact polynomial ODE: initial value exceeds 4096 bits")
        point, value = key.args[0], sympify(raw_value)
        if type(point) not in exact_atoms or type(value) not in exact_atoms:
            raise NotImplementedError("dsolve: implicit initial data must be exact rational values")
        if not _exact_ode_source(relation.lhs, function, x, values=(point, value)):
            raise ValueError("dsolve: implicit initial-condition expression is not polynomial")
        candidate = relation.lhs.subs(function, value).subs(x, point)
        if type(candidate) not in exact_atoms:
            raise ValueError("dsolve: initial condition did not determine an exact level")
        if chosen is not None and candidate != chosen:
            raise ValueError("Couldn't solve for initial conditions")
        chosen = candidate
    return Eq(relation.lhs, chosen if chosen is not None else level, evaluate=False)


def _explicit_exact_ode_solution(relation, ics):
    """Lift checked level sets to rational/quadratic graphs when justified.

    A nonconstant leading coefficient or undecidable branch/initial-value
    check retains the implicit relation. Higher algebraic degrees remain
    implicit; a partial list of algebraic roots is never published.
    """
    from ..core import Dummy, Eq, Integer, Rational, S, diff, expand, sqrt, sympify
    from ..simplify import simplify

    function = _find_function(relation)
    x = function.args[0]
    y = Dummy("exact_ode_y")
    polynomial = relation.lhs.subs(function, y) - relation.rhs
    if diff(polynomial, y, 3) != 0:
        return relation
    exact_atoms = (Integer, Rational, type(S.Zero), type(S.One),
                   type(S.NegativeOne), type(S.Half))
    a = simplify(diff(polynomial, y, 2) / 2)
    b = diff(polynomial, y).subs(y, 0)
    c = polynomial.subs(y, 0)
    if a == 0:
        if type(b) not in exact_atoms or b == 0:
            return relation
        candidates = [-c / b]
    else:
        if type(a) not in exact_atoms:
            return relation
        radical = sqrt(expand(b*b - 4*a*c))
        candidates = [(-b - radical) / (2*a), (-b + radical) / (2*a)]
    accepted = []
    for candidate in candidates:
        if simplify(expand(polynomial.subs(y, candidate))) != 0:
            return relation
        keep = True
        for key, value in (ics or {}).items():
            residual = simplify(candidate.subs(x, key.args[0]) - sympify(value))
            if residual == 0:
                continue
            if residual.is_zero is False:
                keep = False
                break
            return relation
        if keep and candidate not in accepted:
            accepted.append(candidate)
    if not accepted:
        raise ValueError("Couldn't solve for initial conditions")
    answers = [Eq(function, candidate, evaluate=False) for candidate in accepted]
    return answers[0] if len(answers) == 1 else answers


def dsolve(eq, func=None, hint="default", ics=None, **kwargs):
    """Solve an ordinary differential equation for ``func`` (upstream
    ``dsolve``); ``ics`` fixes the integration constants from initial
    conditions ``{f(x0): v, f(x).diff(x).subs(x, x0): v1, ...}``.

    Bounded exact polynomial first-order forms also support ``1st_exact``.
    Their checked level sets stay implicit with ``simplify=False`` or when
    a complete rational/quadratic graph cannot be lifted safely. Exact rational
    value initial conditions bind those level sets; derivative initial data
    for an implicit solution are not currently supported.
    """
    from ..core import AppliedUndef, Eq

    try:
        sol = _dsolve_general(eq, func, hint, **kwargs)
    except NotImplementedError as error:
        # Preserve the established linear/separable default forms. Only these
        # explicit structural non-matches may try the checked exact fallback.
        if hint != "default" or str(error) not in {
            "dsolve: equation is not separable",
            "dsolve: implicit solution only",
            "dsolve: equation class not supported",
        }:
            raise
        function = func if func is not None else _find_function(eq)
        if len(function.args) != 1:
            raise
        variable = function.args[0]
        if not _exact_ode_source(eq, function, variable):
            raise
        original = eq.lhs - eq.rhs if type(eq) is Eq else eq
        expression, ys, order = _ode_to_symbols(original, function, variable)
        sol = _exact_ode_relation(expression, ys, variable, function) if order == 1 else None
        if sol is None:
            raise
    implicit = type(sol) is Eq and not isinstance(sol.lhs, AppliedUndef)
    if ics:
        sol = _apply_exact_ode_ics(sol, ics) if implicit else _apply_ics(sol, ics)
    if implicit and kwargs.get("simplify", True):
        sol = _explicit_exact_ode_solution(sol, ics)
    return sol


def _apply_ics(sol, ics):
    """Solve for the constants C1, C2, ... from initial conditions."""
    import re as _re
    from ..core import Eq as _Eq, diff, Symbol as _Sym
    from .. import solve as _solve

    if isinstance(sol, list):
        return [_apply_ics(s, ics) for s in sol]
    lhs, rhs = sol.lhs, sol.rhs
    x = lhs.args[0]
    consts = sorted((s for s in rhs.free_symbols if _re.fullmatch(r"C\d+", s.name)),
                    key=lambda s: int(s.name[1:]))
    equations = []
    for key, value in ics.items():
        name = type(key).__name__
        if name == "Subs":
            inner = key.expr
            if type(inner).__name__ != "Derivative" or len(key.variables) != 1:
                raise NotImplementedError("unsupported initial condition %s" % key)
            order = sum(1 for v in inner.args[1:] if v == key.variables[0])
            point = key.point[0]
            equations.append(diff(rhs, x, order).subs(x, point) - value)
        elif type(key) is type(lhs) and len(key.args) == 1:
            equations.append(rhs.subs(x, key.args[0]) - value)
        else:
            raise NotImplementedError("unsupported initial condition %s" % key)
    if not consts:
        return sol
    found = _solve(equations, consts, dict=True)
    if not found:
        raise ValueError("Couldn't solve for initial conditions")
    choice = found[0] if isinstance(found, list) else found
    return _Eq(lhs, rhs.subs(choice))


def _dsolve_general(eq, func=None, hint="default", **kwargs):
    """Solve an ordinary differential equation for ``func`` (upstream ``dsolve``).

    Supported classes: linear equations with constant coefficients of any
    order (homogeneous; inhomogeneous through variation of parameters up to
    order 2), first-order linear equations (integrating factor) and
    separable first-order equations. Returns ``Eq(f(x), solution)`` with
    constants ``C1, C2, ...``; other equations raise NotImplementedError.
    """
    from ..core import Eq as _Eq, Integer, diff, expand, Symbol as _S
    from ..simplify import simplify
    from .. import integrate as _integrate, solve as _solve
    from ..integrals import Integral
    from ..functions import exp

    source = eq
    if hint != "1st_exact" and type(eq) is _Eq:
        # Preserve the established default inference after normalization;
        # the explicit exact hint must admit both original sides first.
        eq = eq.lhs - eq.rhs
    f = func if func is not None else _find_function(eq)
    if len(f.args) != 1:
        raise NotImplementedError("dsolve: only ordinary (single-variable) equations")
    x = f.args[0]
    if hint == "1st_exact" and not _exact_ode_source(source, f, x):
        raise ValueError("dsolve: 1st_exact requires an exact polynomial first-order equation")
    if type(eq) is _Eq:
        eq = eq.lhs - eq.rhs
    E, ys, n = _ode_to_symbols(eq, f, x)
    if hint == "1st_exact":
        relation = _exact_ode_relation(E, ys, x, f)
        if relation is None:
            raise ValueError("dsolve: equation does not match the 1st_exact hint")
        return relation
    E = expand(E)
    if n == 0:
        raise ValueError("dsolve: the equation contains no derivative of the function")

    # Linear in y0..yn?
    coeffs = [simplify(diff(E, y)) for y in ys]
    linear = all(not (c.free_symbols & set(ys)) for c in coeffs)
    if linear:
        rest = simplify(expand(E - sum((c * y for c, y in zip(coeffs, ys)), Integer(0))))
        linear = not (rest.free_symbols & set(ys))
    if linear:
        g = -rest
        if all(x not in c.free_symbols for c in coeffs) and (n > 1 or g == 0 or _uc_test(g, x)):
            gensols, collectterms = _char_solutions(coeffs, x)
            consts = _constants(1, len(gensols))
            hom = sum((c * b for c, b in zip(consts, gensols)), Integer(0))
            sol = hom
            if g != 0:
                trial = _uc_trial_set(g, x, coeffs)
                if trial is not None:
                    sol = hom + _particular_uc(trial, g, coeffs, x)
                else:
                    sol = hom + _particular_vop(gensols, g, coeffs[-1], x)
            sol = _simplified_linear(sol, x, collectterms)
            return _Eq(f, _odesimp(sol, set(consts)))
        euler = _euler_coefficients(coeffs, x)
        if euler is not None:
            gensols = _euler_solutions(euler, x)
            consts = _constants(1, len(gensols))
            sol = sum((c * b for c, b in zip(consts, gensols)), Integer(0))
            if g != 0:
                sol = sol + _particular_vop(gensols, g, coeffs[-1], x)
                sol = simplify(sol)
            return _Eq(f, _odesimp(sol, set(consts)))
        if n == 1:
            p = simplify(coeffs[0] / coeffs[1])
            q = simplify(g / coeffs[1])
            ip = _integrate(p, x)
            if isinstance(ip, Integral):
                raise NotImplementedError("dsolve: integrating factor not found")
            mu = exp(ip)
            iq = _integrate(simplify(q * mu), x)
            if isinstance(iq, Integral):
                raise NotImplementedError("dsolve: integral of the forcing term not found")
            (c1,) = _constants(1, 1)
            return _Eq(f, _odesimp(simplify((c1 + iq) / mu), {c1}))
        raise NotImplementedError("dsolve: linear equations with variable coefficients of order > 1")

    if n == 1:
        sols = _solve(E, ys[1])
        if len(sols) == 1:
            F = sols[0]
            if type(F).__name__ == "Add":
                from ..polys import factor as _factor

                F = _factor(F)
            factors = F.args if type(F).__name__ == "Mul" else (F,)
            gx = Integer(1)
            hy = Integer(1)
            for fac in factors:
                syms = fac.free_symbols
                if ys[0] in syms and x in syms:
                    raise NotImplementedError("dsolve: equation is not separable")
                if ys[0] in syms:
                    hy = hy * fac
                else:
                    gx = gx * fac
            left = _integrate(1 / hy, ys[0])
            right = _integrate(gx, x)
            if isinstance(left, Integral) or isinstance(right, Integral):
                raise NotImplementedError("dsolve: separable integrals not found")
            (c1,) = _constants(1, 1)
            solved = _solve_separable(left, right + c1, ys[0])
            if not solved:
                raise NotImplementedError("dsolve: implicit solution only")
            out = [_Eq(f, _odesimp(simplify(s), {c1})) for s in solved]
            return out[0] if len(out) == 1 else out
    raise NotImplementedError("dsolve: equation class not supported")
