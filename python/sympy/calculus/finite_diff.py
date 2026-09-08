"""Finite difference approximation of derivatives for FrankenSymPy."""

from __future__ import annotations
from typing import Any, Iterable

from ..core import Basic, Derivative, Expr, Integer, Rational, S, Symbol, sympify


def finite_diff_weights(order: Any, x_list: Iterable[Any], x0: Any = S.One) -> list[list[list[Any]]]:
    """Calculates finite difference weights for an arbitrarily spaced 1D grid.

    Order of accuracy is at least len(x_list) - order.
    Uses Bengt Fornberg's algorithm (Mathematics of Computation 51(184):699-706, 1988).
    """
    order = sympify(order)
    if not order.is_number:
        raise ValueError("Cannot handle symbolic order.")
    if order < 0:
        raise ValueError("Negative derivative order illegal.")
    if int(order) != order:
        raise ValueError("Non-integer order illegal")

    x_list = [sympify(x) for x in x_list]
    x0 = sympify(x0)
    M = int(order)
    N = len(x_list) - 1

    delta = [[[Integer(0) for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(M + 1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N + 1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n - 1][nu] = Integer(0)
            for m in range(min(n, M) + 1):
                delta[m][n][nu] = (x_list[n] - x0) * delta[m][n - 1][nu] - m * delta[m - 1][n - 1][nu]
                delta[m][n][nu] /= c3
        for m in range(min(n, M) + 1):
            delta[m][n][n] = c1 / c2 * (m * delta[m - 1][n - 1][n - 1] - (x_list[n - 1] - x0) * delta[m][n - 1][n - 1])
        c1 = c2
    return delta


def apply_finite_diff(order: int, x_list: Iterable[Any], y_list: Iterable[Any], x0: Any = S.Zero) -> Any:
    """Calculates finite difference approximation of the derivative of requested order."""
    x_list = list(x_list)
    y_list = list(y_list)
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list not equal in length.")

    N = len(x_list) - 1
    delta = finite_diff_weights(order, x_list, x0)

    derivative = Integer(0)
    for nu in range(len(x_list)):
        derivative += delta[order][N][nu] * sympify(y_list[nu])
    return derivative


def _as_finite_diff(derivative: Any, points: Any = 1, x0: Any = None, wrt: Any = None) -> Any:
    """Returns a finite difference formula for a derivative."""
    if not isinstance(derivative, Derivative):
        return derivative

    if wrt is None:
        old = None
        for v in derivative.variables:
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative

    order = list(derivative.variables).count(wrt)
    if x0 is None:
        x0 = wrt

    if not isinstance(points, (list, tuple)):
        if getattr(points, "is_Function", False) and wrt in points.args:
            points = points.subs(wrt, x0)
        if order % 2 == 0:
            points = [x0 + points * i for i in range(-order // 2, order // 2 + 1)]
        else:
            points = [x0 + points * Rational(i, 2) for i in range(-order, order + 1, 2)]

    others = [wrt, 0]
    for v in set(derivative.variables):
        if v == wrt:
            continue
        others += [v, list(derivative.variables).count(v)]

    if len(points) < order + 1:
        raise ValueError(f"Too few points for order {order}")

    return apply_finite_diff(
        order,
        points,
        [Derivative(derivative.expr.subs({wrt: x}), *others) for x in points],
        x0,
    )


def differentiate_finite(expr: Any, *symbols: Any, points: Any = 1, x0: Any = None, wrt: Any = None, evaluate: bool = False) -> Any:
    """Differentiate expr and replace Derivatives with finite differences."""
    from ..core import diff

    expr = sympify(expr)
    Dexpr = diff(expr, *symbols) if symbols else expr
    if isinstance(Dexpr, Derivative):
        return Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
    if hasattr(Dexpr, "replace"):
        return Dexpr.replace(
            lambda arg: isinstance(arg, Derivative),
            lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt),
        )
    return Dexpr


__all__ = [
    "_as_finite_diff",
    "apply_finite_diff",
    "differentiate_finite",
    "finite_diff_weights",
]
