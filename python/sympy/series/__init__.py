"""Series expansion and limits for FrankenSymPy."""

from __future__ import annotations

from ..core import (
    _symbol_facts,
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)


def series(expression, x=None, x0=0, n=6, dir="+", **kwargs):
    """Series expansion of ``expression`` about ``x = x0`` up to ``O((x - x0)**n)``.

    Laurent and Puiseux terms, ``log`` terms and expansions at ``oo`` are
    computed natively; the remainder is an ``Order`` term (omitted when the
    expansion is exact, as for polynomials).
    """
    from ..core import sympify

    if "point" in kwargs:
        x0 = kwargs["point"]
    if "variable" in kwargs and x is None:
        x = kwargs["variable"]

    expr = sympify(expression)
    if x is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            return expr
        else:
            raise ValueError("x must be given for multivariate series")
    else:
        symbol = _require_symbol(x)
    x0 = sympify(x0)
    if n is None:
        raise NotImplementedError("lazy series (n=None) is not supported")
    terms, order = _native.series_expansion_expr(
        str(_native_expr(expr)),
        _native_symbol_key(symbol),
        str(_native_expr(x0)),
        int(n),
    )
    from ..core import oo, Rational, Integer

    plain = Symbol(symbol.name)
    if x0 == 0:
        local = plain
    elif x0 == oo:
        local = 1 / plain
    elif x0 == -oo:
        local = -1 / plain
    else:
        local = plain - x0
    total = Integer(0)
    for coeff, exponent in terms:
        total = total + _parse_result(coeff) * local ** _parse_result(exponent)
    if order is not None:
        total = total + Order(local ** _parse_result(order), plain, x0)
    if symbol is not plain:
        total = total.subs({plain: symbol})
    return total


from ..core import Expr, Function


def limit(expression, variable=None, point=None, dir="+", **kwargs):
    """Limit of ``expression`` as ``variable -> point``.

    ``dir`` is ``"+"`` (default, from the right), ``"-"`` or ``"+-"``
    (two-sided; raises ``ValueError`` when the one-sided limits differ).
    Indeterminate forms, poles and limits at infinity are computed natively
    by the Gruntz algorithm.
    """
    if variable is None and point is None:
        if isinstance(expression, Limit):
            return expression.doit()
        raise TypeError("limit requires variable and point")
    symbol = _require_symbol(variable)
    if str(dir) not in ("+", "-", "+-"):
        raise ValueError("direction must be one of '+', '-' or '+-'")
    try:
        native = _native.limit_expr(
            str(_native_expr(expression)),
            _native_symbol_key(symbol),
            str(_native_expr(point)),
            str(dir),
            _symbol_facts(expression),
        )
    except ValueError as exc:
        # Upstream answers an undecidable limit (e.g. an unknown parameter
        # sign) with the unevaluated Limit object.
        if "undetermined" in str(exc):
            return Limit(expression, symbol, point, dir=str(dir))
        raise
    return _parse_result(native)


class Limit(Expr):
    """An unevaluated limit."""

    __slots__ = ("_expression", "_variable", "_point", "_dir")

    def __new__(
        cls,
        expression: Any,
        variable: Any,
        point: Any,
        dir: str = "+",
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
        # Upstream: the direction at +oo is always '-', at -oo always '+'.
        point_text = str(obj._point)
        if point_text == "oo":
            dir = "-"
        elif point_text == "-oo":
            dir = "+"
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
        return limit(self._expression, self._variable, self._point, dir=self._dir)

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

    @property
    def args(self) -> tuple:
        from ..core import Symbol

        return (self._expression, self._variable, self._point, Symbol(self._dir))

    def __repr__(self) -> str:
        return f"Limit({self._expression}, {self._variable}, {self._point}, dir='{self._dir}')"

    def __str__(self) -> str:
        return self.__repr__()


class Order(Function):
    """Big-O remainder of a series: ``O(expr)`` or ``O(expr, (x, x0))``.

    Carried as a structural term so ``series`` results print as upstream
    (``1 + x + O(x**2)``) and ``removeO`` / ``getO`` can strip or read it.
    The kernel does not absorb higher-order terms into it.
    """

    __slots__ = ()

    def __new__(cls, expr, *args, **options):
        from ..core import sympify

        expr = sympify(expr)
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = tuple(args[0])
        if len(args) == 2:
            var, point = sympify(args[0]), sympify(args[1])
            if point != 0:
                return Function.__new__(cls, expr, var, point, evaluate=False)
        return Function.__new__(cls, expr, evaluate=False)

    @property
    def expr(self):
        return self.args[0]

    @property
    def point(self):
        from ..core import Integer

        return (self.args[2],) if len(self.args) == 3 else (Integer(0),)


O = Order


def residue(expr: Any, x: Any, x0: Any) -> Any:
    """Compute the residue of expr with respect to x at point x0.

    The residue is the coefficient of 1/(x - x0) in the Laurent series of expr around x0.
    """
    from ..core import sympify, Integer, diff, oo, zoo
    from ..functions.combinatorial.factorials import factorial
    from ..polys.polytools import Poly

    expr = sympify(expr)
    x = Symbol(str(x)) if not isinstance(x, Symbol) else x
    x0 = sympify(x0)

    # Check if expr is rational in x
    num, den = expr.as_numer_denom()
    try:
        p_den = Poly(den, x)
        p_factor = Poly(x - x0, x)
        k = 0
        cur_den = p_den
        while True:
            q, r = cur_den.div(p_factor)
            if r.is_zero and not q.is_zero:
                cur_den = q
                k += 1
            else:
                break
        if k > 0:
            rem_den = cur_den.as_expr()
            G = num / rem_den
            for _ in range(k - 1):
                G = diff(G, x)
            val = G.subs(x, x0)
            return val / factorial(k - 1)
        elif den.subs(x, x0) != 0:
            return Integer(0)
    except Exception:
        pass

    try:
        if den.subs(x, x0) == 0:
            d_den = diff(den, x)
            d_val = d_den.subs(x, x0)
            if d_val != 0:
                n_val = num.subs(x, x0)
                from ..simplify import simplify
                return simplify(n_val / d_val)
    except Exception:
        pass

    # General limit-based method using variable shift t = x - x0
    t = Symbol("_t")
    shifted = expr.subs(x, t + x0)
    for k in range(1, 6):
        cand = (t**k) * shifted
        try:
            lim = limit(cand, t, 0)
            if lim not in (oo, -oo, zoo) and "oo" not in str(lim):
                deriv = cand
                for _ in range(k - 1):
                    deriv = diff(deriv, t)
                lim_deriv = limit(deriv, t, 0)
                return lim_deriv / factorial(k - 1)
        except Exception:
            continue
    return Integer(0)


class FourierSeries(Expr):
    """Represents a Fourier series of a function."""

    def __init__(self, f: Any, limits: tuple[Any, Any, Any], a0: Any, an: list[Any], bn: list[Any]):
        from ..core import Integer
        from ..simplify import simplify
        self.f = f
        self.x, self.start, self.end = limits
        try:
            self.L = simplify((self.end - self.start) / Integer(2))
        except Exception:
            self.L = (self.end - self.start) / Integer(2)
        self.a0 = a0
        self.an = an
        self.bn = bn

    def truncate(self, n: int = 3) -> Any:
        """Return the partial sum of the Fourier series up to order n."""
        from ..core import Add, Integer, pi
        from ..functions.elementary.trigonometric import cos, sin
        terms = []
        if self.a0 != 0:
            terms.append(self.a0 / Integer(2))
        for k in range(1, min(n + 1, max(len(self.an), len(self.bn)) + 1)):
            idx = k - 1
            arg = (Integer(k) * pi * self.x) / self.L
            if idx < len(self.an) and self.an[idx] != 0:
                terms.append(self.an[idx] * cos(arg))
            if idx < len(self.bn) and self.bn[idx] != 0:
                terms.append(self.bn[idx] * sin(arg))
        if not terms:
            return Integer(0)
        return Add(*terms) if len(terms) > 1 else terms[0]

    def __repr__(self) -> str:
        return f"FourierSeries({self.f}, ({self.x}, {self.start}, {self.end}))"

    def __str__(self) -> str:
        return repr(self)


def _eval_trig_pi(expr: Any) -> Any:
    from ..core import Integer, pi
    from ..functions.elementary.trigonometric import cos, sin
    from ..simplify import simplify, expand_power_base
    res = expr
    try:
        res = expand_power_base(res)
    except Exception:
        pass
    for k in range(-12, 13):
        res = res.subs(sin(Integer(k) * pi), Integer(0))
        res = res.subs(cos(Integer(k) * pi), Integer(1 if k % 2 == 0 else -1))
    try:
        res = simplify(res)
    except Exception:
        pass
    try:
        res = expand_power_base(res)
        res = simplify(res)
    except Exception:
        pass
    return res


def fourier_series(f: Any, limits: Any = None) -> FourierSeries:
    """Compute the Fourier series of f."""
    from ..core import sympify, Symbol, Integer, Rational, pi
    from ..integrals import integrate, Integral
    from ..simplify import simplify
    from ..functions.elementary.trigonometric import cos, sin
    f = sympify(f)
    if limits is None:
        syms = f.free_symbols
        x = next(iter(syms)) if syms else Symbol("x")
        limits = (x, -pi, pi)
    elif len(limits) == 1:
        limits = (limits[0], -pi, pi)
    elif len(limits) == 2:
        syms = f.free_symbols
        x = next(iter(syms)) if syms else Symbol("x")
        limits = (x, limits[0], limits[1])
    x, start, end = limits
    x = Symbol(str(x)) if not isinstance(x, Symbol) else x
    start = sympify(start)
    end = sympify(end)
    try:
        L = simplify((end - start) / Integer(2))
    except Exception:
        L = (end - start) / Integer(2)

    # Compute a0
    int_f = integrate(f, x)
    if isinstance(int_f, Integral):
        int_f = int_f.doit()
    if not isinstance(int_f, Integral):
        a0 = _eval_trig_pi((int_f.subs(x, end) - int_f.subs(x, start)) / L)
    else:
        a0 = Integer(0)

    # Compute first 6 coefficients an and bn
    an_coeffs = []
    bn_coeffs = []
    for k in range(1, 7):
        try:
            arg = simplify((Integer(k) * pi * x) / L)
        except Exception:
            arg = (Integer(k) * pi * x) / L
        # an
        int_cos = integrate(f * cos(arg), x)
        if isinstance(int_cos, Integral):
            int_cos = int_cos.doit()
        if not isinstance(int_cos, Integral):
            ak = _eval_trig_pi((int_cos.subs(x, end) - int_cos.subs(x, start)) / L)
            an_coeffs.append(ak)
        else:
            an_coeffs.append(Integer(0))
        # bn
        int_sin = integrate(f * sin(arg), x)
        if isinstance(int_sin, Integral):
            int_sin = int_sin.doit()
        if not isinstance(int_sin, Integral):
            bk = _eval_trig_pi((int_sin.subs(x, end) - int_sin.subs(x, start)) / L)
            bn_coeffs.append(bk)
        else:
            bn_coeffs.append(Integer(0))

    return FourierSeries(f, (x, start, end), a0, an_coeffs, bn_coeffs)


def fps(f: Any, x: Any = None, x0: Any = 0, dir: int = 1, **kwargs: Any) -> Any:
    """Compute the formal power series of f."""
    return series(f, x=x, x0=x0, **kwargs)


formal_power_series = fps

__all__ = [
    "FourierSeries",
    "Limit",
    "O",
    "Order",
    "formal_power_series",
    "fourier_series",
    "fps",
    "limit",
    "residue",
    "series",
]

