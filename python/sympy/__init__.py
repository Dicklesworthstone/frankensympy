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
    sqrt,
    symbols,
    sympify,
    SympifyError,
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


def solve(expression, variable=None):
    """Solve the algebraic equation ``expression == 0`` for ``variable``."""
    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if variable is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        else:
            raise TypeError("at least one solve variable is required")
    else:
        symbol = _require_symbol(variable)
    results = _native.solve_expr(
        str(expr), _native_symbol_key(symbol)
    )
    return [_parse_result(r) for r in results]


def laplace_transform(expression, t, s):
    t_sym = _require_symbol(t)
    s_sym = _require_symbol(s)
    result = _native.laplace_expr(
        str(_wrap(_native_expr(expression))),
        _native_symbol_key(t_sym),
        _native_symbol_key(s_sym),
    )
    return _parse_result(result)


def fourier_transform(expression, t, w):
    t_sym = _require_symbol(t)
    w_sym = _require_symbol(w)
    result = _native.fourier_expr(
        str(_wrap(_native_expr(expression))),
        _native_symbol_key(t_sym),
        _native_symbol_key(w_sym),
    )
    return _parse_result(result)


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


def tan(expression):
    return _wrap(_native.py_tan(_native_expr(expression)))


def asin(expression):
    return _wrap(_native.py_asin(_native_expr(expression)))


def atan(expression):
    return _wrap(_native.py_atan(_native_expr(expression)))


def sinh(expression):
    return _wrap(_native.py_sinh(_native_expr(expression)))


def cosh(expression):
    return _wrap(_native.py_cosh(_native_expr(expression)))


def tanh(expression):
    return _wrap(_native.py_tanh(_native_expr(expression)))


def exp(expression):
    return _wrap(_native.py_exp(_native_expr(expression)))


def log(expression):
    return _wrap(_native.py_log(_native_expr(expression)))


def floor(expression):
    return _wrap(_native.py_floor(_native_expr(expression)))


def ceiling(expression):
    return _wrap(_native.py_ceiling(_native_expr(expression)))


def factorial(expression):
    return _wrap(_native.py_factorial(_native_expr(expression)))


def gamma(expression):
    return _wrap(_native.py_gamma(_native_expr(expression)))


def fibonacci(expression):
    return _wrap(_native.py_fibonacci(_native_expr(expression)))


def totient(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise TypeError("n should be an integer")
    if integer <= 0:
        raise ValueError("n should be a positive integer")
    return _native.euler_totient(integer)


def mobius(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise TypeError("n should be an integer")
    if integer <= 0:
        raise ValueError("n should be a positive integer")
    return _native.mobius_fn(integer)


def divisor_count(value):
    integer = _exact_integer_value(value)
    if integer is None:
        raise TypeError("n should be an integer")
    if integer <= 0:
        raise ValueError("n should be a positive integer")
    return _native.divisor_count_fn(integer)


def divisor_sigma(value, k=1):
    integer = _exact_integer_value(value)
    if integer is None:
        raise TypeError("n should be an integer")
    if integer <= 0:
        raise ValueError("n should be a positive integer")
    k_int = _exact_integer_value(k)
    if k_int is None or k_int < 0:
        raise TypeError("k should be a non-negative integer")
    return _native.divisor_sum_fn(integer, k_int)


def jacobi_symbol(m, n):
    m_int = _exact_integer_value(m)
    n_int = _exact_integer_value(n)
    if m_int is None or n_int is None:
        raise TypeError("m and n should be integers")
    if n_int <= 0 or n_int % 2 == 0:
        raise ValueError("n should be an odd positive integer")
    return _native.jacobi_symbol_fn(m_int, n_int)


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
from . import (
    functions,
    integrals,
    logic,
    ntheory,
    solvers,
)
from .logic import (
    And,
    Boolean,
    BooleanAtom,
    BooleanFalse,
    BooleanFunction,
    BooleanTrue,
    Equivalent,
    Implies,
    Not,
    Or,
    Xor,
    false,
    satisfiable,
    simplify_logic,
    to_cnf,
    to_dnf,
    true,
)
from .series import limit, series


__all__ = [
    "Add",
    "And",
    "Application",
    "AppliedUndef",
    "Atom",
    "AtomicExpr",
    "Basic",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "ComplexInfinity",
    "DenseMatrix",
    "Derivative",
    "Dummy",
    "E",
    "Eq",
    "Equivalent",
    "Expr",
    "Float",
    "Function",
    "FunctionClass",
    "Ge",
    "Gt",
    "I",
    "ImmutableDenseMatrix",
    "ImmutableMatrix",
    "Implies",
    "Integer",
    "Le",
    "Lt",
    "Matrix",
    "MatrixBase",
    "Mul",
    "MutableDenseMatrix",
    "N",
    "Ne",
    "Not",
    "Number",
    "Or",
    "Pow",
    "Rational",
    "S",
    "Symbol",
    "UndefinedFunction",
    "Xor",
    "__version__",
    "asin",
    "atan",
    "ceiling",
    "cos",
    "cosh",
    "diag",
    "diff",
    "divisor_count",
    "divisor_sigma",
    "dsolve",
    "exp",
    "eye",
    "factorial",
    "factorint",
    "false",
    "fibonacci",
    "floor",
    "fourier_transform",
    "gamma",
    "integrate",
    "isprime",
    "jacobi_symbol",
    "laplace_transform",
    "limit",
    "log",
    "mobius",
    "nan",
    "oo",
    "pi",
    "pretty",
    "satisfiable",
    "series",
    "simplify",
    "simplify_logic",
    "sin",
    "sinh",
    "solve",
    "srepr",
    "sqrt",
    "symbols",
    "sympify",
    "SympifyError",
    "tan",
    "tanh",
    "to_cnf",
    "to_dnf",
    "totient",
    "true",
    "zeros",
    "zoo",
]
