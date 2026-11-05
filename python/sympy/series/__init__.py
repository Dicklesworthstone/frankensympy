"""Series expansion and limits for FrankenSymPy."""

from __future__ import annotations

from ..core import (
    _symbol_binding,
    _symbol_facts,
    _typed_symbol_facts,
    _lift_native_result,
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
    expansion is exact, as for polynomials). At finite points, ``dir="+"``
    approaches from the right and ``dir="-"`` from the left, retaining the
    selected logarithmic and fractional-power branches. At infinities the
    approach is inward, independently of ``dir``.
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
    if type(dir) is not str or dir not in ("+", "-"):
        raise ValueError("series direction must be '+' or '-'")
    if n is None:
        raise NotImplementedError("lazy series (n=None) is not supported")
    terms, order = _native.series_expansion_native(
        _native_expr(expr),
        _symbol_binding(symbol),
        _native_expr(x0),
        int(n),
        dir,
    )
    from ..core import oo, Integer

    if x0 == oo:
        local = 1 / symbol
    elif x0 == -oo:
        local = -1 / symbol
    elif dir == "-":
        local = x0 - symbol
    elif x0 == 0:
        local = symbol
    else:
        local = symbol - x0
    remainder_local = symbol - x0 if dir == "-" and x0 not in (oo, -oo) else local
    total = Integer(0)
    for coeff, exponent in terms:
        total = total + _lift_native_result(coeff, expr, symbol, x0) * local ** _lift_native_result(exponent, expr, symbol, x0)
    if order is not None:
        total = total + Order(remainder_local ** _lift_native_result(order, expr, symbol, x0), symbol, x0)
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
        native = _native.limit_native(
            _native_expr(expression),
            _symbol_binding(symbol),
            _native_expr(point),
            str(dir),
            _typed_symbol_facts(expression, symbol, point),
        )
    except ValueError as exc:
        # Upstream answers an undecidable limit (e.g. an unknown parameter
        # sign) with the unevaluated Limit object.
        if "undetermined" in str(exc):
            return Limit(expression, symbol, point, dir=str(dir))
        raise
    return _lift_native_result(native, expression, symbol, point)


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


class _ResidueAnalysis:
    """Bounded original-source and local meromorphic admission.

    The native Laurent algorithm supplies coefficients. This separate source
    walk establishes that they belong to a single-valued meromorphic germ,
    rather than interpreting an arbitrary one-sided real expansion as one.
    """

    _ENTIRE = ("exp", "sin", "cos", "sinh", "cosh")

    def __init__(self, variable, point):
        from fractions import Fraction
        from ..core import I, E, pi, EulerGamma, Catalan, GoldenRatio

        self.Fraction = Fraction
        self.zero = (Fraction(0), Fraction(0))
        self.one = (Fraction(1), Fraction(0))
        self.variable = _native_expr(variable)
        self.binding = _symbol_binding(variable)
        self.point = point
        self.imaginary = _native_expr(I)
        self.constants = tuple(_native_expr(value) for value in (I, E, pi, EulerGamma, Catalan, GoldenRatio))
        self.nodes = 0
        self.work = 0
        self.calls = 0
        self.center = None

    def step(self, amount=1):
        self.work += amount
        if self.work > 100_000:
            raise ValueError("residue: exact local arithmetic exceeds its work bound")

    def rational(self, value):
        if not (value.is_integer or value.is_rational):
            raise NotImplementedError("residue requires exact integer Laurent exponents")
        numerator, denominator = value.exact_numerator(), value.exact_denominator()
        if max(abs(numerator).bit_length(), denominator.bit_length()) > 4096:
            raise ValueError("residue: rational coefficient exceeds 4096 bits")
        return self.Fraction(numerator, denominator)

    def read(self, native, depth=0):
        """Read the complete closed original tree before algebra or expansion."""
        self.nodes += 1
        self.step()
        if self.nodes > 4096 or depth > 32:
            raise ValueError("residue: original source exceeds its traversal bound")
        if type(native) is not _native.Expr:
            raise TypeError("residue requires exact native source expressions")
        kind = native.func_name
        if native.is_integer or native.is_rational:
            return (native, kind, (), self.rational(native))
        if native.is_symbol:
            if len(native.symbol_binding.name) > 16_384:
                raise ValueError("residue: symbol name exceeds its text bound")
            return (native, kind, (), None)
        if kind == "Constant":
            if not any(native == value for value in self.constants):
                raise NotImplementedError("residue requires finite exact constants")
            value = (self.Fraction(0), self.Fraction(1)) if native == self.imaginary else None
            return (native, kind, (), value)
        if kind not in ("Add", "Mul", "Pow", *self._ENTIRE):
            raise NotImplementedError(
                "residue requires meromorphic integer-power arithmetic and admitted entire functions"
            )
        arguments = native.args
        if len(arguments) > 128 or self.nodes + len(arguments) > 4096:
            raise ValueError("residue: original source exceeds its fanout bound")
        children = tuple(self.read(child, depth + 1) for child in arguments)
        exponent = None
        if kind == "Pow":
            if len(children) != 2:
                raise ValueError("residue: malformed original power")
            exponent = self.rational(arguments[1])
            if exponent.denominator != 1:
                raise NotImplementedError("residue does not admit branch-producing fractional powers")
            exponent = exponent.numerator
            if abs(exponent) > 128:
                raise ValueError("residue: original integer power exceeds 128")
        elif kind in self._ENTIRE and len(children) != 1:
            raise NotImplementedError("residue requires unary entire functions")
        return (native, kind, children, exponent)

    def checked(self, value):
        self.step()
        if any(max(abs(part.numerator).bit_length(), part.denominator.bit_length()) > 4096
               for part in value):
            raise ValueError("residue: exact center arithmetic exceeds 4096 bits")
        return value

    def add(self, first, second):
        return self.checked((first[0] + second[0], first[1] + second[1]))

    def multiply(self, first, second):
        return self.checked((
            first[0]*second[0] - first[1]*second[1],
            first[0]*second[1] + first[1]*second[0],
        ))

    def power(self, value, exponent):
        if exponent <= 0 and value == self.zero:
            raise ValueError("residue: original source has a zero base of a nonpositive power")
        if exponent < 0:
            norm = value[0]*value[0] + value[1]*value[1]
            value = self.checked((value[0]/norm, -value[1]/norm))
            exponent = -exponent
        result = self.one
        while exponent:
            if exponent & 1:
                result = self.multiply(result, value)
            exponent >>= 1
            if exponent:
                value = self.multiply(value, value)
        return result

    def combine(self, kind, values, exponent=None):
        if any(value is None for value in values):
            return None
        if kind == "Pow":
            return self.power(values[0], exponent)
        result = self.zero if kind == "Add" else self.one
        operation = self.add if kind == "Add" else self.multiply
        for value in values:
            result = operation(result, value)
        return result

    def gaussian(self, node):
        """Exact Q(i) value, without numerical probes or assumed nonzero atoms."""
        self.step()
        _, kind, children, data = node
        if kind in ("Integer", "Rational"):
            return (data, self.Fraction(0))
        if kind == "Constant":
            return data
        if kind not in ("Add", "Mul", "Pow"):
            return None
        values = [self.gaussian(child) for child in children]
        return self.combine(kind, values, data)

    def known_nonzero(self, node):
        value = self.gaussian(node)
        if value is not None:
            return value != self.zero
        _, kind, children, _ = node
        if kind in ("Constant", "exp"):
            return True
        if kind == "Mul":
            return all(self.known_nonzero(child) for child in children)
        if kind == "Pow":
            return self.known_nonzero(children[0])
        return False

    def expand(self, native, order):
        self.calls += 1
        if self.calls > 64:
            raise ValueError("residue: local expansion call bound exhausted")
        terms, remainder = _native.series_expansion_native(
            native, self.binding, self.point, order,
        )
        if len(terms) > 256:
            raise ValueError("residue: Laurent coefficient count exceeds its bound")
        if remainder is not None:
            remainder = self.rational(remainder)
            if remainder.denominator != 1:
                raise NotImplementedError("residue requires an integer-order meromorphic remainder")
            remainder = remainder.numerator
        result = []
        previous = None
        for coefficient, exponent in terms:
            exponent = self.rational(exponent)
            if exponent.denominator != 1:
                raise NotImplementedError("residue requires integer Laurent powers")
            exponent = exponent.numerator
            if previous is not None and exponent <= previous:
                raise ValueError("residue: malformed Laurent exponent order")
            previous = exponent
            if remainder is not None and exponent >= remainder:
                raise ValueError("residue: Laurent coefficient exceeds the reported precision")
            node = self.read(coefficient)
            if coefficient.has(self.variable):
                raise ValueError("residue: Laurent coefficient retains the expansion variable")
            if self.gaussian(node) != self.zero:
                result.append((exponent, coefficient, node))
        return result, remainder

    def leading(self, node):
        for order in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            terms, remainder = self.expand(node[0], order)
            if terms:
                exponent, _, coefficient = terms[0]
                if not self.known_nonzero(coefficient):
                    raise NotImplementedError(
                        "residue cannot prove the original denominator's leading coefficient nonzero"
                    )
                return exponent, self.gaussian(coefficient)
            if remainder is None:
                raise ValueError("residue: original nonpositive power has an identically zero base")
        raise ValueError("residue: insufficient precision to establish an original denominator")

    def admit(self, node):
        """Return (holomorphic at center, exact center value, nonzero there)."""
        self.step()
        native, kind, children, data = node
        if kind in ("Integer", "Rational"):
            value = (data, self.Fraction(0))
            return True, value, value != self.zero
        if kind == "Symbol":
            value = self.center if native == self.variable else None
            return True, value, value is not None and value != self.zero
        if kind == "Constant":
            return True, data, True
        states = [self.admit(child) for child in children]
        if kind in self._ENTIRE:
            if not states[0][0]:
                raise NotImplementedError(
                    "residue requires holomorphic original arguments of entire functions"
                )
            value = None
            if states[0][1] == self.zero:
                value = self.one if kind in ("exp", "cos", "cosh") else self.zero
            return True, value, kind == "exp" or (value is not None and value != self.zero)
        if kind in ("Add", "Mul"):
            holomorphic = all(state[0] for state in states)
            value = self.combine(kind, [state[1] for state in states]) if holomorphic else None
            nonzero = (value is not None and value != self.zero) or (
                kind == "Mul" and holomorphic and all(state[2] for state in states)
            )
            return holomorphic, value, nonzero
        base = states[0]
        if data > 0:
            value = self.power(base[1], data) if base[1] is not None else None
            return base[0], value, base[2]
        if base[1] == self.zero and not children[0][0].has(self.variable):
            raise ValueError("residue: original source has a zero base of a nonpositive power")
        if base[0] and base[2]:
            value = self.power(base[1], data) if base[1] is not None else None
            return True, self.one if data == 0 else value, True
        exponent, coefficient = self.leading(children[0])
        if data == 0:
            return True, self.one, True
        valuation = exponent * data
        if valuation > 0:
            return True, self.zero, False
        if valuation == 0:
            value = self.power(coefficient, data) if coefficient is not None else None
            return True, value, True
        return False, None, False


def residue(expr: Any, x: Any, x0: Any) -> Any:
    """Return the coefficient of (x - x0)**-1 in an admitted Laurent germ.

    Exact rational and Gaussian-rational centers are supported. Sources use
    bounded exact integer-power arithmetic and exp, sin, cos, sinh or cosh
    with original arguments proved holomorphic at the center. Inverses
    require a provably nonzero local leading coefficient. Essential,
    branch-dependent and non-holomorphic sources explicitly refuse.

    Rational sources at rational centers use the native independently checked
    rational Laurent algorithm. The analytic lane uses the native generalized
    series after the separate meromorphic-source check; it does not issue a
    complex-analytic proof certificate. Insufficient precision and operational
    failures propagate, rather than being interpreted as zero residues.
    """
    from ..core import Integer, sympify

    symbol = _require_symbol(x)
    expression, point = sympify(expr), sympify(x0)
    native, native_point = _native_expr(expression), _native_expr(point)
    analysis = _ResidueAnalysis(symbol, native_point)
    original = analysis.read(native)
    center_source = analysis.read(native_point)
    analysis.center = analysis.gaussian(center_source)
    if analysis.center is None:
        raise NotImplementedError("residue requires an exact rational or Gaussian-rational center")
    analysis.admit(original)
    terms, remainder = analysis.expand(native, 0)
    if remainder is not None and remainder < 0:
        raise ValueError("residue: insufficient Laurent precision to determine the residue")
    for exponent, coefficient, _ in terms:
        if exponent == -1:
            return _lift_native_result(coefficient, expression, symbol, point)
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

