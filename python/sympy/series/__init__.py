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


def series(expression, x=None, x0=0, n=6, dir="+", **kwargs):
    """Compute the Taylor series expansion of expression around variable = point."""
    if "point" in kwargs:
        x0 = kwargs["point"]
    if "variable" in kwargs and x is None:
        x = kwargs["variable"]

    expr = _wrap(_native_expr(expression))
    if x is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            symbol = Symbol("x")
        else:
            raise ValueError("variable must be specified when multiple free symbols exist")
    else:
        symbol = _require_symbol(x)
    result = _native.taylor_expr(
        str(expr), _native_symbol_key(symbol), int(x0), int(n)
    )
    return _parse_result(result)


from ..core import Expr, Function


def limit(expression, variable=None, point=None, dir="+-", **kwargs):
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
        try:
            obj._expression = _wrap(_native_expr(expression))
        except (NotImplementedError, TypeError):
            from ..core import sympify
            obj._expression = sympify(expression)
        obj._variable = _require_symbol(variable)
        try:
            obj._point = _wrap(_native_expr(point))
        except (NotImplementedError, TypeError):
            from ..core import sympify
            obj._point = sympify(point)
        obj._dir = dir
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def func(self) -> type:
        return Limit

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

    @property
    def free_symbols(self) -> set[Symbol]:
        expr_syms = set(self._expression.free_symbols) if hasattr(self._expression, "free_symbols") else set()
        if self._variable in expr_syms:
            expr_syms.remove(self._variable)
        if hasattr(self._point, "free_symbols"):
            expr_syms.update(self._point.free_symbols)
        return expr_syms

    @property
    def is_number(self) -> bool:
        return len(self.free_symbols) == 0

    def doit(self, **hints: Any) -> Any:
        return limit(self._expression, self._variable, self._point)

    def _eval_subs(self, old: Any, new: Any) -> Any:
        if self == old:
            return new
        new_point = self._point.subs(old, new) if hasattr(self._point, "subs") else self._point
        if old == self._variable:
            new_expr = self._expression
        else:
            new_expr = self._expression.subs(old, new) if hasattr(self._expression, "subs") else self._expression
        return Limit(new_expr, self._variable, new_point, dir=self._dir)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Limit):
            return False
        return (
            self._expression == other._expression
            and self._variable == other._variable
            and self._point == other._point
            and self._dir == other._dir
        )

    def __hash__(self) -> int:
        return hash((self.__class__, self._expression, self._variable, self._point, self._dir))

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

