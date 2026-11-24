"""Finite difference approximation of derivatives for FrankenSymPy."""

from __future__ import annotations
from itertools import islice
from math import isqrt
from typing import Any, Iterable

from ..core import Derivative, Integer, Rational, S, sympify


_MAX_WEIGHT_CELLS = 262_144
_MAX_STENCIL_SAMPLES = 16_384


def _grid(values, maximum):
    items = list(islice(values, maximum + 1))
    if len(items) > maximum:
        raise ValueError("finite difference grid exceeds its weight table limit")
    if not items:
        raise ValueError("finite difference grid must contain at least one point")
    return items


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

    M = int(order)
    # The public result includes every prefix and every derivative order.
    maximum = isqrt(_MAX_WEIGHT_CELLS // (M + 1))
    x_list = [sympify(x) for x in _grid(x_list, maximum)]
    x0 = sympify(x0)
    N = len(x_list) - 1

    delta = [[[Integer(0) for _ in range(N + 1)] for _ in range(N + 1)] for _ in range(M + 1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N + 1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            if c3 == 0:
                raise ValueError("finite difference grid points must be distinct")
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
    x_list = _grid(x_list, isqrt(_MAX_WEIGHT_CELLS))
    y_list = list(islice(y_list, len(x_list) + 1))
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

    variables = derivative.variables
    axes = list(dict.fromkeys(variables)) if wrt is None else [wrt]
    if wrt is not None and wrt not in variables:
        return derivative
    remaining = () if wrt is None else tuple(v for v in variables if v != wrt)
    remaining_axes = set(remaining)
    stencils = []
    samples = 1
    for axis in axes:
        order = variables.count(axis)
        center = axis if x0 is None else sympify(x0)
        if isinstance(points, (tuple, list)):
            nodes = points
        else:
            step = sympify(points)
            if axis in step.free_symbols:
                step = step.subs(axis, center)
            if order + 1 > isqrt(_MAX_WEIGHT_CELLS // (order + 1)):
                raise ValueError("finite difference grid exceeds its weight table limit")
            nodes = [center + step*Rational(2*i-order, 2) for i in range(order + 1)]
        if len(nodes) < order + 1:
            raise ValueError(f"Too few points for order {order}")
        samples *= len(nodes)
        if samples > _MAX_STENCIL_SAMPLES:
            raise ValueError("mixed finite difference exceeds its stencil sample limit")
        # Moving a sample into an undifferenced coordinate would turn a
        # held partial into a total derivative of the sampling map.
        if remaining_axes and any(remaining_axes.intersection(sympify(point).free_symbols)
                                  for point in nodes):
            raise NotImplementedError("sample points depend on a remaining derivative coordinate")
        weights = finite_diff_weights(order, nodes, center)[order][-1]
        stencils.append((axis, tuple(zip(nodes, weights))))

    def sample(bindings):
        if len(bindings) == len(stencils):
            source = derivative.expr
            if len(bindings) == 1:
                source = source.subs(*bindings[0])
            else:
                # Freeze every original coordinate before inserting any
                # sample. A center or step can itself name another axis.
                source = source.subs(bindings, simultaneous=True)
            return Derivative(source, *remaining) if remaining else source
        axis, stencil = stencils[len(bindings)]
        return sum((weight * sample((*bindings, (axis, point)))
                    for point, weight in stencil if weight != 0), S.Zero)

    return sample(())


def differentiate_finite(expr: Any, *symbols: Any, points: Any = 1, x0: Any = None, wrt: Any = None, evaluate: bool = False) -> Any:
    """Approximate the requested derivative of a sampled expression."""
    from ..core import diff

    expr = sympify(expr)

    def stencil(derivative):
        return _as_finite_diff(derivative, points=points, x0=x0, wrt=wrt)

    # An already-specified derivative is itself a request for a stencil.
    if isinstance(expr, Derivative) and not symbols:
        return stencil(expr)
    if evaluate:
        expanded = diff(expr, *symbols)
        return expanded.replace(lambda node: isinstance(node, Derivative), stencil)

    # Discretize embedded derivatives before translating the entire field
    # to each sample position. In particular the default difference of a
    # product samples the product, without a symbolic Leibniz expansion.
    if expr.atoms(Derivative):
        expr = expr.replace(lambda node: isinstance(node, Derivative), stencil)
    held = Derivative(expr, *symbols)
    return stencil(held)


__all__ = [
    "_as_finite_diff",
    "apply_finite_diff",
    "differentiate_finite",
    "finite_diff_weights",
]
