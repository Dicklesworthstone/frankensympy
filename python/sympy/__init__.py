"""Experimental FrankenSymPy compatibility slice.

Only the names exported here are wired to native behavior. Unsupported SymPy
operations fail explicitly; upstream SymPy is never used as a fallback.
"""

from .core import (
    Add,
    Derivative,
    Expr,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    _native,
    _native_expr,
    _parse_result,
    _require_symbol,
    _wrap,
    diff,
    expand,
    simplify,
    symbols,
)

__version__ = _native.version()

# These are native constant nodes, not ordinary symbols with suggestive names.
pi = Expr("pi")
E = Expr("E")
I = Expr("I")
oo = Expr("oo")
zoo = Expr("zoo")
nan = Expr("nan")


def integrate(expression, *variables):
    """Integrate one implemented univariate form.

    Accepted forms are ``integrate(expr, x)``, ``integrate(expr, (x, a, b))``,
    and the legacy spelling ``integrate(expr, x, a, b)``.
    """
    if len(variables) == 1 and isinstance(variables[0], tuple):
        spec = variables[0]
        if len(spec) != 3:
            raise ValueError("integration tuple must be (variable, lower, upper)")
        variable, lower, upper = spec
        symbol = _require_symbol(variable)
        result = _native.integrate_definite_expr(
            str(_wrap(_native_expr(expression))),
            symbol.name,
            str(_wrap(_native_expr(lower))),
            str(_wrap(_native_expr(upper))),
        )
        return _parse_result(result)
    if len(variables) == 1:
        symbol = _require_symbol(variables[0])
        return _parse_result(
            _native.integrate_expr(str(_wrap(_native_expr(expression))), symbol.name)
        )
    if len(variables) == 3:
        variable, lower, upper = variables
        return integrate(expression, (variable, lower, upper))
    raise TypeError("integrate requires one variable or one definite-integration tuple")


def limit(expression, variable, point):
    symbol = _require_symbol(variable)
    return _parse_result(
        _native.limit_expr(
            str(_wrap(_native_expr(expression))),
            symbol.name,
            str(_wrap(_native_expr(point))),
        )
    )


def solve(expression, variable):
    """Solve the implemented linear ``expression == 0`` case."""
    symbol = _require_symbol(variable)
    result = _native.solve_linear_expr(
        str(_wrap(_native_expr(expression))), symbol.name
    )
    return [_parse_result(result)]


def dsolve(equation, func=None):
    """Refuse the unsupported general SymPy ODE-equation interface."""
    del equation, func
    raise NotImplementedError(
        "general dsolve equation parsing is not implemented; "
        "the native coefficient-form solvers are not a drop-in dsolve interface"
    )


def isprime(value):
    return _native.is_prime(int(value))


def factorint(value):
    return dict(_native.factorize(int(value)))


def totient(value):
    return _native.euler_totient(int(value))


__all__ = [
    "Expr",
    "Symbol",
    "Integer",
    "Rational",
    "Add",
    "Mul",
    "Pow",
    "Derivative",
    "symbols",
    "diff",
    "expand",
    "simplify",
    "integrate",
    "limit",
    "solve",
    "dsolve",
    "isprime",
    "factorint",
    "totient",
    "pi",
    "E",
    "I",
    "oo",
    "zoo",
    "nan",
    "__version__",
]
