"""Experimental FrankenSymPy compatibility slice.

Only the names exported here are wired to native behavior. Unsupported SymPy
operations fail explicitly; upstream SymPy is never used as a fallback.
"""

from .core import (
    Add,
    Application,
    AppliedUndef,
    Atom,
    AtomicExpr,
    Basic,
    ComplexInfinity,
    Derivative,
    Dummy,
    Eq,
    Expr,
    Float,
    Function,
    FunctionClass,
    Ge,
    Gt,
    Integer,
    Le,
    Lt,
    Mul,
    Ne,
    N,
    Number,
    Pow,
    Rational,
    S,
    Symbol,
    UndefinedFunction,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
    diff,
    expand,
    pretty,
    simplify,
    symbols,
)
from .printing import srepr
from .core import zoo as _core_zoo

__version__ = _native.version()

# These are native constant nodes, not ordinary symbols with suggestive names.
pi = Expr("pi")
E = Expr("E")
I = Expr("I")
oo = Expr("oo")
zoo = _core_zoo  # the ComplexInfinity singleton, not a plain Expr (finding 9)
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
            _native_symbol_key(symbol),
            str(_wrap(_native_expr(lower))),
            str(_wrap(_native_expr(upper))),
        )
        return _parse_result(result)
    if len(variables) == 1:
        symbol = _require_symbol(variables[0])
        return _parse_result(
            _native.integrate_expr(
                str(_wrap(_native_expr(expression))), _native_symbol_key(symbol)
            )
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
            _native_symbol_key(symbol),
            str(_wrap(_native_expr(point))),
        )
    )


def solve(expression, variable):
    """Solve the implemented linear ``expression == 0`` case."""
    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    symbol = _require_symbol(variable)
    result = _native.solve_linear_expr(
        str(_wrap(_native_expr(expression))), _native_symbol_key(symbol)
    )
    return [_parse_result(result)]


def dsolve(equation, func=None):
    """Refuse the unsupported general SymPy ODE-equation interface."""
    del equation, func
    raise NotImplementedError(
        "general dsolve equation parsing is not implemented; "
        "the native coefficient-form solvers are not a drop-in dsolve interface"
    )


def _exact_integer_value(value):
    """Return an admitted exact integer without invoking lossy converters."""
    if type(value) is int:
        return value
    if type(value) is Integer:
        return value.p
    return None


def _not_integer_message(value):
    """Format exact built-in floats without executing arbitrary object hooks."""
    if type(value) is float:
        return f"{value!r} is not an integer"
    return "value is not an integer"


def isprime(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise ValueError(_not_integer_message(value))
    if integer < 2:
        return False
    return _native.is_prime(integer)


def factorint(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise ValueError(_not_integer_message(value))
    if integer == 0:
        return {0: 1}
    factors = dict(_native.factorize(abs(integer)))
    if integer < 0:
        factors[-1] = 1
    return factors


def sin(expression):
    return _wrap(_native.py_sin(_native_expr(expression)))


def cos(expression):
    return _wrap(_native.py_cos(_native_expr(expression)))


def exp(expression):
    return _wrap(_native.py_exp(_native_expr(expression)))


def log(expression):
    return _wrap(_native.py_log(_native_expr(expression)))


def totient(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise TypeError("n should be an integer")
    if integer <= 0:
        raise ValueError("n should be a positive integer")
    return _native.euler_totient(integer)


from .matrices import (
    Matrix,
    MatrixBase,
    DenseMatrix,
    MutableDenseMatrix,
    ImmutableMatrix,
    ImmutableDenseMatrix,
    eye,
    zeros,
    diag,
)


__all__ = [
    "Add",
    "Application",
    "AppliedUndef",
    "Atom",
    "AtomicExpr",
    "Basic",
    "ComplexInfinity",
    "DenseMatrix",
    "Derivative",
    "Dummy",
    "E",
    "Eq",
    "Expr",
    "Float",
    "Function",
    "FunctionClass",
    "Ge",
    "Gt",
    "I",
    "ImmutableDenseMatrix",
    "ImmutableMatrix",
    "Integer",
    "Le",
    "Lt",
    "Matrix",
    "MatrixBase",
    "Mul",
    "MutableDenseMatrix",
    "N",
    "Ne",
    "Number",
    "Pow",
    "Rational",
    "S",
    "Symbol",
    "UndefinedFunction",
    "__version__",
    "cos",
    "diag",
    "diff",
    "dsolve",
    "exp",
    "eye",
    "factorint",
    "integrate",
    "isprime",
    "limit",
    "log",
    "nan",
    "oo",
    "pi",
    "pretty",
    "simplify",
    "sin",
    "solve",
    "srepr",
    "symbols",
    "totient",
    "zeros",
    "zoo",
]
