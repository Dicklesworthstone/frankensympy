"""Euler-Lagrange equations for FrankenSymPy calculus of variations."""

from __future__ import annotations
from itertools import combinations_with_replacement
from typing import Any, Iterable

from ..core import Basic, Derivative, Eq, Expr, Function, Integer, S, Symbol, diff, sympify


def euler_equations(L: Any, funcs: Any = (), vars: Any = ()) -> list[Any]:
    r"""Find the Euler-Lagrange equations for a given Lagrangian.

    Parameters
    ==========
    L : Expr
        The Lagrangian expression.
    funcs : Function or iterable of Functions
        The functions the Lagrangian depends on.
    vars : Symbol or iterable of Symbols
        The independent variables.

    Returns
    =======
    list of Eq
        The Euler-Lagrange equations, one for each function.
    """
    L = sympify(L)
    if isinstance(funcs, (Basic, Function)) and not isinstance(funcs, (list, tuple)):
        funcs = (funcs,)
    elif isinstance(funcs, Iterable):
        funcs = tuple(funcs)
    else:
        funcs = (funcs,)

    if not funcs or funcs == ((),):
        funcs = tuple(L.atoms(Function))
    else:
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError(f"Function expected, got: {f}")

    if isinstance(vars, (Basic, Symbol)) and not isinstance(vars, (list, tuple)):
        vars = (vars,)
    elif isinstance(vars, Iterable):
        vars = tuple(vars)
    else:
        vars = (vars,)

    if not vars or vars == ((),):
        if funcs:
            vars = funcs[0].args
        else:
            vars = ()
    else:
        vars = tuple(sympify(v) for v in vars)

    if not all(isinstance(v, Symbol) for v in vars):
        raise TypeError(f"Variables are not symbols, got {vars}")

    for f in funcs:
        if vars != f.args:
            raise ValueError(f"Variables {vars} do not match args: {f}")

    from ..core import Dummy

    # Collect all derivatives of funcs in L and determine maximum order
    all_derivs = [a for a in L.atoms(Derivative) if hasattr(a, "expr") and a.expr in funcs]
    all_derivs.sort(key=lambda d: len(getattr(d, "variables", ())), reverse=True)

    order = 1
    for a in all_derivs:
        order = max(order, len(getattr(a, "variables", ())))

    # In calculus of variations, the Lagrangian L is defined on the jet bundle
    # where fields and their derivatives are treated as independent coordinates
    # when computing partial derivatives dL/df and dL/d(df).
    d_dummies = {d: Dummy(f"_d_{i}") for i, d in enumerate(all_derivs)}
    f_dummies = {fn: Dummy(f"_f_{i}") for i, fn in enumerate(funcs)}

    L_coord = L.subs(d_dummies).subs(f_dummies)
    rev_map = {v: k for k, v in d_dummies.items()}
    rev_map.update({v: k for k, v in f_dummies.items()})

    eqns = []
    for fn in funcs:
        dL_df = diff(L_coord, f_dummies[fn]).subs(rev_map)
        eq = dL_df
        for i in range(1, order + 1):
            for p in combinations_with_replacement(vars, i):
                # Calculate term: (-1)**i * d/dp (dL / d(d_p f))
                d_f_p = diff(fn, *p)
                if d_f_p in d_dummies:
                    dL_dd = diff(L_coord, d_dummies[d_f_p]).subs(rev_map)
                    d_term = diff(dL_dd, *p)
                    if i % 2 == 1:
                        eq = eq - d_term
                    else:
                        eq = eq + d_term
        eqns.append(Eq(eq, Integer(0)))

    return eqns


__all__ = ["euler_equations"]
