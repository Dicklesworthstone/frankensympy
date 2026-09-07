"""Series expansion and limits for FrankenSymPy."""

from ..core import (
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)


def series(expression, variable=None, point=0, n=6):
    """Compute the Taylor series expansion of expression around variable = point."""
    expr = _wrap(_native_expr(expression))
    if variable is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            symbol = Symbol("x")
        else:
            raise ValueError("variable must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(variable)
    result = _native.taylor_expr(
        str(expr), _native_symbol_key(symbol), int(point), int(n)
    )
    return _parse_result(result)


def limit(expression, variable, point):
    symbol = _require_symbol(variable)
    return _parse_result(
        _native.limit_expr(
            str(_wrap(_native_expr(expression))),
            _native_symbol_key(symbol),
            str(_wrap(_native_expr(point))),
        )
    )


from ..core import Function

Order = Function("Order")
O = Order

__all__ = [
    "O",
    "Order",
    "limit",
    "series",
]

