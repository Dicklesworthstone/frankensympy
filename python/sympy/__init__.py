"""FrankenSymPy — Drop-in replacement for SymPy backed by high-performance native Rust."""

from .core import (
    Expr, Symbol, Integer, Rational, Add, Mul, Pow, Derivative,
    symbols, diff, expand, simplify,
)

try:
    import fsym_python as _native
    __version__ = _native.version()
except ImportError:
    __version__ = "0.1.0"
    _native = None

pi = Symbol("pi")
E = Symbol("E")
I = Symbol("I")
oo = Symbol("oo")
zoo = Symbol("zoo")
nan = Symbol("nan")

def integrate(expr, var, *limits):
    """Compute indefinite or definite integral."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    expr_str = str(expr)
    var_str = str(var)
    if not limits:
        return _native.integrate_expr(expr_str, var_str)
    elif len(limits) == 2:
        return _native.integrate_definite_expr(expr_str, var_str, str(limits[0]), str(limits[1]))
    else:
        raise ValueError("Invalid limits for integrate")

def limit(expr, var, point):
    """Compute limit of expression as var -> point."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.limit_expr(str(expr), str(var), str(point))

def solve(expr, var):
    """Solve equation expr == 0 for var."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.solve_linear_expr(str(expr), str(var))

def dsolve(eq, func=None):
    """Solve ordinary differential equation."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.dsolve_linear_first_order_expr("0", str(eq), "x")

def isprime(n):
    """Miller-Rabin primality test."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.is_prime(int(n))

def factorint(n):
    """Prime factorization."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.factorize(int(n))

def totient(n):
    """Euler's totient function."""
    if _native is None:
        raise RuntimeError("fsym_python native module is not available")
    return _native.euler_totient(int(n))

__all__ = [
    "Expr", "Symbol", "Integer", "Rational", "Add", "Mul", "Pow", "Derivative",
    "symbols", "diff", "expand", "simplify", "integrate", "limit", "solve", "dsolve",
    "isprime", "factorint", "totient", "pi", "E", "I", "oo", "zoo", "nan", "__version__",
]
