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


from ..core import Expr, Function


def limit(expression, variable=None, point=None):
    if variable is None and point is None:
        if isinstance(expression, Limit):
            return expression.doit()
        raise TypeError("limit requires variable and point")
    symbol = _require_symbol(variable)
    return _parse_result(
        _native.limit_expr(
            str(_wrap(_native_expr(expression))),
            _native_symbol_key(symbol),
            str(_wrap(_native_expr(point))),
        )
    )


class Limit(Expr):
    """An unevaluated limit."""

    __slots__ = ("_expression", "_variable", "_point", "_dir")

    def __new__(
        cls,
        expression: Any,
        variable: Any,
        point: Any,
        dir: str = "+-",
        evaluate: bool = False,
    ):
        if evaluate:
            return limit(expression, variable, point)
        obj = object.__new__(cls)
        obj._expression = _wrap(_native_expr(expression))
        obj._variable = _require_symbol(variable)
        obj._point = _wrap(_native_expr(point))
        obj._dir = dir
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def expr(self) -> Expr:
        return self._expression

    @property
    def var(self) -> Symbol:
        return self._variable

    @property
    def point(self) -> Expr:
        return self._point

    @property
    def dir(self) -> str:
        return self._dir

    @property
    def args(self) -> tuple[Any, ...]:
        return (self._expression, self._variable, self._point)

    def doit(self, **hints: Any) -> Any:
        return limit(self._expression, self._variable, self._point)

    def __repr__(self) -> str:
        return f"Limit({self._expression}, {self._variable}, {self._point})"

    def __str__(self) -> str:
        return self.__repr__()


Order = Function("Order")
O = Order

__all__ = [
    "Limit",
    "O",
    "Order",
    "limit",
    "series",
]

