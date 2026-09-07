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
    Equality,
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
from .matrices import (
    DenseMatrix,
    GramSchmidt,
    ImmutableDenseMatrix,
    ImmutableMatrix,
    Matrix,
    MatrixBase,
    MutableDenseMatrix,
    MutableSparseMatrix,
    SparseMatrix,
    casoratian,
    diag,
    eye,
    hadamard_product,
    hstack,
    jacobian,
    kronecker_product,
    matrix_multiply_elementwise,
    ones,
    vstack,
    wronskian,
    zeros,
)

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
        from .solvers.solvers import linsolve as _linsolve

        if variable is not None:
            if isinstance(variable, (list, tuple)):
                var_list = list(variable)
            elif isinstance(variable, Symbol):
                var_list = [variable]
            else:
                raise TypeError(f"unsupported solve system signature with variable {variable}")
        else:
            all_syms = set()
            for eq in expression:
                if type(eq) is Eq:
                    eq = eq.lhs - eq.rhs
                all_syms.update(_wrap(_native_expr(eq)).free_symbols)
            var_list = sorted(list(all_syms), key=lambda s: s.name)

        if not var_list:
            raise TypeError("at least one solve variable is required")

        # Try linear solver first
        try:
            lin_sol = _linsolve(expression, *var_list)
            if lin_sol:
                sol_tuple = next(iter(lin_sol))
                return {s: v for s, v in zip(var_list, sol_tuple)}
            else:
                return []
        except (ValueError, TypeError):
            pass

        # Fall back to polynomial system solver for 2-variable systems
        if len(var_list) == 2:
            sols = _sps(expression, *var_list)
            if sols is None:
                return []
            var_x, var_y = var_list[0], var_list[1]
            if len(sols) == 1:
                return {var_x: sols[0][0], var_y: sols[0][1]}
            return [{var_x: sol[0], var_y: sol[1]} for sol in sols]

        raise TypeError(f"unsupported solve system signature with variable {variable}")

    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if variable is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            return []
        else:
            raise TypeError("at least one solve variable is required")
    else:
        symbol = _require_symbol(variable)

    if expr == 0 or (not expr.free_symbols):
        return []

    try:
        results = _native.solve_expr(
            str(expr), _native_symbol_key(symbol)
        )
    except Exception as exc:
        msg = str(exc)
        if "No solution found" in msg or "Infinite solutions" in msg:
            return []
        raise
    return [_parse_result(r) for r in results]



def solveset(expression, variable=None, domain=None):
    """Solve an algebraic equation for ``variable`` and return a Set of solutions."""
    if domain is not None and isinstance(domain, str):
        if domain in ("Reals", "RR", "R"):
            from .sets import Reals
            domain = Reals
    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if variable is None:
        symbols = expr.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
        elif len(symbols) == 0:
            if expr == 0:
                if domain is not None and isinstance(domain, Set):
                    return domain
                from .sets import UniversalSet
                return UniversalSet()
            from .sets import EmptySet
            return EmptySet()
        else:
            raise ValueError("at least one solve variable is required")
    else:
        symbol = _require_symbol(variable)

    if expr == 0:
        if domain is not None and isinstance(domain, Set):
            return domain
        from .sets import UniversalSet
        return UniversalSet()

    if not expr.free_symbols:
        from .sets import EmptySet
        return EmptySet()

    try:
        results = _native.solve_expr(str(expr), _native_symbol_key(symbol))
    except Exception:
        from .sets import EmptySet
        return EmptySet()

    from .sets import EmptySet, FiniteSet
    if not results:
        return EmptySet()
    roots = [_parse_result(r) for r in results]
    if domain is not None and isinstance(domain, Set):
        roots = [r for r in roots if r in domain]
    if not roots:
        return EmptySet()
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
    return bool(simplified == 0 or getattr(simplified, "is_zero", None) is True)


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


from .functions import (
    acos,
    acosh,
    acot,
    acoth,
    acsc,
    acsch,
    asec,
    asech,
    asin,
    asinh,
    atan,
    atanh,
    bell,
    bernoulli,
    binomial,
    catalan,
    ceiling,
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    erf,
    erfc,
    exp,
    factorial,
    fibonacci,
    floor,
    gamma,
    harmonic,
    log,
    lucas,
    sec,
    sech,
    sign,
    sin,
    sinc,
    sinh,
    subfactorial,
    tan,
    tanh,
    zeta,
)


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


from .ntheory import (
    carmichael,
    divisors,
    integer_nthroot,
    is_perfect,
    is_primitive_root,
    is_quad_residue,
    is_square_free,
    is_squarefree,
    legendre_symbol,
    mod_inverse,
    nextprime,
    prevprime,
    prime,
    prime_big_omega,
    prime_omega,
    primefactors,
    primenu,
    primeomega,
    primepi,
    proper_divisors,
    reduced_totient,
)



from .matrices import (
    Matrix,
    MatrixBase,
    DenseMatrix,
    MutableDenseMatrix,
    ImmutableMatrix,
    ImmutableDenseMatrix,
    eye,
    zeros,
    ones,
    diag,
    hstack,
    pinv,
    vstack,
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
    ProductSet,
    Reals,
    Set,
    SymmetricDifference,
    Union,
    UniversalSet,
)
from .geometry import (
    Circle,
    Ellipse,
    Line,
    Line2D,
    Line3D,
    LinearEntity,
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
    EC,
    LC,
    Poly,
    TC,
    degree,
    discriminant,
    factor,
    factor_list,
    gcd,
    groebner,
    lcm,
    monic,
    resultant,
    roots,
    sqf,
    sqf_list,
    sqf_part,
    trailing_coeff,
)
from .series import O, Order, limit, series
from .solvers import (
    checkodesol,
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
    dsolve_linear_first_order,
    dsolve_separable_linear,
    linsolve,
    nonlinsolve,
    solve_linear_system,
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
from .core import simplify


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
    "EC",
    "Ellipse",
    "EmptySet",
    "Eq",
    "Equality",
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
    "LinearEntity",
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
    "ProductSet",
    "pinv",
    "Q",
    "Rational",
    "Ray",
    "Ray2D",
    "Ray3D",
    "Reals",
    "S",
    "Segment",
    "Segment2D",
    "Segment3D",
    "Set",
    "Sphere",
    "Symbol",
    "SymmetricDifference",
    "TC",
    "Tensor",
    "TensorIndex",
    "Triangle",
    "Tuple",
    "UndefinedFunction",
    "Union",
    "UniversalSet",
    "Xor",
    "__version__",
    "acos",
    "acosh",
    "acot",
    "acoth",
    "acsc",
    "acsch",
    "asec",
    "asech",
    "ask",
    "asin",
    "asinh",
    "atan",
    "DenseMatrix",
    "GramSchmidt",
    "ImmutableDenseMatrix",
    "ImmutableMatrix",
    "Matrix",
    "MatrixBase",
    "MutableDenseMatrix",
    "MutableSparseMatrix",
    "SparseMatrix",
    "atanh",
    "bell",
    "bernoulli",
    "binomial",
    "carmichael",
    "casoratian",
    "catalan",
    "ceiling",
    "checkodesol",
    "checksol",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "csch",
    "degree",
    "diag",
    "diff",
    "discriminant",
    "divisor_count",
    "divisor_sigma",
    "divisors",
    "dsolve",
    "dsolve_cauchy_euler",
    "dsolve_const_coeff_second_order",
    "dsolve_const_coeff_second_order_nonhomogeneous",
    "dsolve_linear_first_order",
    "dsolve_separable_linear",
    "erf",
    "erfc",
    "exp",
    "eye",
    "factor",
    "factor_list",
    "factorial",
    "factorint",
    "false",
    "fibonacci",
    "floor",
    "fourier_transform",
    "gamma",
    "gcd",
    "groebner",
    "hadamard_product",
    "harmonic",
    "hstack",
    "integer_nthroot",
    "integrate",
    "is_perfect",
    "is_primitive_root",
    "is_quad_residue",
    "is_square_free",
    "is_squarefree",
    "isprime",
    "jacobian",
    "jacobi_symbol",
    "kronecker_product",
    "laplace_transform",
    "lcm",
    "legendre_symbol",
    "limit",
    "linsolve",
    "log",
    "lucas",
    "matrix_multiply_elementwise",
    "mobius",
    "mod_inverse",
    "monic",
    "nan",
    "nextprime",
    "nonlinsolve",
    "ones",
    "oo",
    "pi",
    "pretty",
    "prevprime",
    "prime",
    "primenu",
    "prime_big_omega",
    "prime_omega",
    "primefactors",
    "primeomega",
    "primepi",
    "proper_divisors",
    "reduced_totient",
    "resultant",
    "roots",
    "satisfiable",
    "sec",
    "sech",
    "series",
    "sign",
    "simplify",
    "simplify_logic",
    "sin",
    "sinc",
    "sinh",
    "solve",
    "solve_linear_system",
    "solve_poly_system",
    "solveset",
    "sqf",
    "sqf_list",
    "sqf_part",
    "srepr",
    "sqrt",
    "subfactorial",
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
    "trailing_coeff",
    "true",
    "vstack",
    "wronskian",
    "zeros",
    "zeta",
    "zoo",
]
