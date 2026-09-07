"""Experimental FrankenSymPy compatibility slice.

Only the names exported here are wired to native behavior. Unsupported SymPy
operations fail explicitly; upstream SymPy is never used as a fallback.
"""

from .core import (
    Abs,
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
    Tuple,
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
    """Solve the algebraic equation or system of equations ``expression == 0``."""
    if isinstance(expression, (list, tuple)):
        from .solvers.polysys import solve_poly_system as _sps
        if variable is not None and isinstance(variable, (list, tuple)) and len(variable) == 2:
            sols = _sps(expression, *variable)
            if sols is None:
                return []
            var_x, var_y = variable[0], variable[1]
            if len(sols) == 1:
                return {var_x: sols[0][0], var_y: sols[0][1]}
            return [{var_x: sol[0], var_y: sol[1]} for sol in sols]
        elif variable is None:
            all_syms = set()
            for eq in expression:
                if type(eq) is Eq:
                    eq = eq.lhs - eq.rhs
                all_syms.update(_wrap(_native_expr(eq)).free_symbols)
            if len(all_syms) == 2:
                ordered_syms = sorted(list(all_syms), key=lambda s: s.name)
                sols = _sps(expression, *ordered_syms)
                if sols is None:
                    return []
                if len(sols) == 1:
                    return {ordered_syms[0]: sols[0][0], ordered_syms[1]: sols[0][1]}
                return [{ordered_syms[0]: sol[0], ordered_syms[1]: sol[1]} for sol in sols]
            raise TypeError("at least one solve variable is required")
        else:
            raise TypeError(f"unsupported solve system signature with variable {variable}")

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



def solveset(expression, variable=None, domain=None):
    """Solve an algebraic equation for ``variable`` and return a Set of solutions."""
    del domain
    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if variable is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            if expr == 0:
                from .sets import UniversalSet
                return UniversalSet()
            from .sets import EmptySet
            return EmptySet()
        else:
            raise ValueError("at least one solve variable is required")
    else:
        symbol = _require_symbol(variable)

    try:
        results = _native.solve_expr(str(expr), _native_symbol_key(symbol))
    except Exception:
        from .sets import EmptySet
        return EmptySet()

    from .sets import EmptySet, FiniteSet
    if not results:
        return EmptySet()
    roots = [_parse_result(r) for r in results]
    return FiniteSet(*roots)


def checksol(expression, symbol, val=None):
    """Check whether ``val`` (or mapping) satisfies ``expression == 0``."""
    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if isinstance(symbol, dict):
        mapping = symbol
    elif val is not None:
        mapping = {symbol: val}
    else:
        raise ValueError("checksol requires either a mapping or (symbol, val)")
    subbed = expr
    for sym, v in mapping.items():
        subbed = subbed.subs(sym, v)
    simplified = simplify(subbed)
    return bool(simplified == 0)


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
    assumptions,
    functions,
    geometry,
    integrals,
    logic,
    ntheory,
    polys,
    series,
    sets,
    solvers,
    tensor,
)
from .sets import (
    Complement,
    EmptySet,
    FiniteSet,
    Intersection,
    Interval,
    Set,
    Union,
    UniversalSet,
)
from .geometry import (
    Circle,
    Line,
    Line2D,
    Line3D,
    Plane,
    Point,
    Point2D,
    Point3D,
    Polygon,
    Ray,
    Ray2D,
    Ray3D,
    Segment,
    Segment2D,
    Segment3D,
    Sphere,
    Triangle,
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
from .polys import (
    LC,
    Poly,
    degree,
    discriminant,
    gcd,
    groebner,
    lcm,
    monic,
    resultant,
    sqf_list,
    sqf_part,
)
from .series import O, Order, limit, series
from .solvers import (
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
    dsolve_linear_first_order,
    dsolve_separable_linear,
    nonlinsolve,
    solve_poly_system,
)
from .tensor import (
    Metric,
    Tensor,
    TensorIndex,
    tensor_indices,
    tensorcontraction,
    tensorproduct,
)
from .assumptions import (
    AppliedPredicate,
    AssumptionsContext,
    Predicate,
    Q,
    ask,
)


__all__ = [
    "Abs",
    "Add",
    "And",
    "Application",
    "AppliedPredicate",
    "AppliedUndef",
    "AssumptionsContext",
    "Atom",
    "AtomicExpr",
    "Basic",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "Circle",
    "Complement",
    "ComplexInfinity",
    "DenseMatrix",
    "Derivative",
    "Dummy",
    "E",
    "EmptySet",
    "Eq",
    "Equivalent",
    "Expr",
    "FiniteSet",
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
    "Intersection",
    "Interval",
    "LC",
    "Le",
    "Line",
    "Line2D",
    "Line3D",
    "Lt",
    "Matrix",
    "MatrixBase",
    "Metric",
    "Mul",
    "MutableDenseMatrix",
    "N",
    "Ne",
    "Not",
    "Number",
    "O",
    "Or",
    "Order",
    "Plane",
    "Point",
    "Point2D",
    "Point3D",
    "Poly",
    "Polygon",
    "Pow",
    "Predicate",
    "Q",
    "Rational",
    "Ray",
    "Ray2D",
    "Ray3D",
    "S",
    "Segment",
    "Segment2D",
    "Segment3D",
    "Set",
    "Sphere",
    "Symbol",
    "Tensor",
    "TensorIndex",
    "Triangle",
    "Tuple",
    "UndefinedFunction",
    "Union",
    "UniversalSet",
    "Xor",
    "__version__",
    "ask",
    "asin",
    "atan",
    "ceiling",
    "checksol",
    "cos",
    "cosh",
    "degree",
    "diag",
    "diff",
    "discriminant",
    "divisor_count",
    "divisor_sigma",
    "dsolve",
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
    "exp",
    "eye",
    "factorial",
    "factorint",
    "false",
    "fibonacci",
    "floor",
    "fourier_transform",
    "gamma",
    "gcd",
    "groebner",
    "integrate",
    "isprime",
    "jacobi_symbol",
    "laplace_transform",
    "lcm",
    "limit",
    "log",
    "mobius",
    "monic",
    "nan",
    "nonlinsolve",
    "oo",
    "pi",
    "pretty",
    "resultant",
    "satisfiable",
    "series",
    "simplify",
    "simplify_logic",
    "sin",
    "sinh",
    "solve",
    "solve_poly_system",
    "solveset",
    "sqf_list",
    "sqf_part",
    "srepr",
    "sqrt",
    "symbols",
    "sympify",
    "SympifyError",
    "tan",
    "tanh",
    "tensor_indices",
    "tensorcontraction",
    "tensorproduct",
    "to_cnf",
    "to_dnf",
    "totient",
    "true",
    "zeros",
    "zoo",
]
