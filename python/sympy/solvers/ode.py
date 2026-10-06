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
    """Solve Cauchy-Euler 2nd-order ODE: a*x^2*y''(x) + b*x*y'(x) + c*y(x) = 0."""
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

    from ..core import Equality, Integer

    if isinstance(ode, Equality):
        ode_expr = ode.lhs - ode.rhs
    else:
        ode_expr = _wrap(_native_expr(ode))

    if isinstance(sol, Equality):
        if func is None:
            func = sol.lhs
            rhs = sol.rhs
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
    yield e
    for a in getattr(e, "args", ()):
        yield from _preorder(a)


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
    from ..core import Symbol as _S

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


def dsolve(eq, func=None, hint="default", ics=None, **kwargs):
    """Solve an ordinary differential equation for ``func`` (upstream
    ``dsolve``); ``ics`` fixes the integration constants from initial
    conditions ``{f(x0): v, f(x).diff(x).subs(x, x0): v1, ...}``."""
    sol = _dsolve_general(eq, func, hint, **kwargs)
    if ics:
        sol = _apply_ics(sol, ics)
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

    if type(eq) is _Eq:
        eq = eq.lhs - eq.rhs
    f = func if func is not None else _find_function(eq)
    if len(f.args) != 1:
        raise NotImplementedError("dsolve: only ordinary (single-variable) equations")
    x = f.args[0]
    E, ys, n = _ode_to_symbols(eq, f, x)
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
        if all(x not in c.free_symbols for c in coeffs):
            groups = _homogeneous_basis(coeffs, x)
            hom, basis = _assemble(groups, x)
            if g == 0:
                return _Eq(f, hom)
            part = _particular_vop(basis, g, coeffs[-1], x)
            return _Eq(f, hom + part)
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
            return _Eq(f, simplify((c1 + iq) / mu))
        raise NotImplementedError("dsolve: linear equations with variable coefficients of order > 1")

    if n == 1:
        sols = _solve(E, ys[1])
        if len(sols) == 1:
            F = sols[0]
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
            solved = _solve(left - right - c1, ys[0])
            if not solved:
                raise NotImplementedError("dsolve: implicit solution only")
            out = [_Eq(f, simplify(s)) for s in solved]
            return out[0] if len(out) == 1 else out
    raise NotImplementedError("dsolve: equation class not supported")
