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
    cholesky,
    det,
    diag,
    eye,
    hadamard_product,
    hstack,
    jacobian,
    jordan_block,
    jordan_cell,
    kronecker_product,
    matrix_multiply_elementwise,
    ones,
    pinv,
    randMatrix,
    rank,
    shape,
    trace,
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
EulerGamma = Expr("EulerGamma")
Catalan = Expr("Catalan")
GoldenRatio = Expr("GoldenRatio")


def integrate(expression, *variables):
    """Integrate univariate, definite, or iterated multivariate expressions.

    Accepted forms are ``integrate(expr, x)``, ``integrate(expr, (x, a, b))``,
    multivariate ``integrate(expr, x, y)``, and the legacy spelling ``integrate(expr, x, a, b)``.
    """
    if not variables:
        if hasattr(expression, "doit") and type(expression).__name__ == "Integral":
            return expression.doit()
        raise TypeError("integrate requires at least one variable or integration tuple")

    if len(variables) == 3 and not any(isinstance(v, (tuple, list)) for v in variables):
        variable, lower, upper = variables
        return integrate(expression, (variable, lower, upper))

    if len(variables) > 1:
        cur = expression
        for v in variables:
            cur = integrate(cur, v)
        return cur

    spec = variables[0]
    from .integrals import Integral
    if isinstance(spec, (tuple, list)):
        if len(spec) == 1:
            return integrate(expression, spec[0])
        if len(spec) != 3:
            raise ValueError("integration tuple must be (variable, lower, upper)")
        variable, lower, upper = spec
        symbol = _require_symbol(variable)
        try:
            result = _native.integrate_definite_expr(
                str(_wrap(_native_expr(expression))),
                _native_symbol_key(symbol),
                str(_wrap(_native_expr(lower))),
                str(_wrap(_native_expr(upper))),
            )
            return _parse_result(result)
        except (ValueError, NotImplementedError, TypeError):
            return Integral(expression, (variable, lower, upper))

    symbol = _require_symbol(spec)
    try:
        return _parse_result(
            _native.integrate_expr(
                str(_wrap(_native_expr(expression))), _native_symbol_key(symbol)
            )
        )
    except (ValueError, NotImplementedError, TypeError):
        return Integral(expression, symbol)


def solve(expression, *symbols, **flags):
    """Solve the algebraic equation or system of equations ``expression == 0``."""
    dict_flag = bool(flags.get("dict", False))
    set_flag = bool(flags.get("set", False))

    if len(symbols) == 1 and isinstance(symbols[0], (list, tuple)):
        var_list = list(symbols[0])
    elif len(symbols) == 1 and symbols[0] is None:
        var_list = None
    elif symbols:
        var_list = list(symbols)
    else:
        var_list = None

    if isinstance(expression, (list, tuple)):
        from .solvers.polysys import solve_poly_system as _sps
        from .solvers.solvers import linsolve as _linsolve

        if var_list is not None:
            parsed_var_list = []
            for s in var_list:
                if isinstance(s, Symbol):
                    parsed_var_list.append(s)
                else:
                    raise TypeError(f"unsupported solve system signature with variable {s}")
            var_list = parsed_var_list
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
                sol_dict = {s: v for s, v in zip(var_list, sol_tuple)}
                if dict_flag:
                    return [sol_dict]
                if set_flag:
                    return (var_list, {sol_tuple})
                return sol_dict
            else:
                return []
        except (ValueError, TypeError):
            pass

        # Fall back to polynomial system solver
        # Only an established no-solution result is empty. Unsupported
        # systems and faults must remain visible to the caller.
        sols = _sps(expression, *var_list)

        if sols is None:
            return []

        if dict_flag:
            return [{sym: val for sym, val in zip(var_list, sol)} for sol in sols]
        if set_flag:
            return (var_list, set(sols))
        return sols

    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))

    if var_list is None:
        symbols_found = expr.free_symbols
        if len(symbols_found) == 1:
            symbol = next(iter(symbols_found))
        elif len(symbols_found) == 0:
            return []
        else:
            raise TypeError("at least one solve variable is required")
    elif len(var_list) == 1:
        symbol = _require_symbol(var_list[0])
    else:
        from .solvers.polysys import solve_poly_system as _sps
        from .solvers.solvers import linsolve as _linsolve

        try:
            lin_sol = _linsolve([expr], *var_list)
            if lin_sol:
                sol_tuple = next(iter(lin_sol))
                sol_dict = {s: v for s, v in zip(var_list, sol_tuple)}
                if dict_flag:
                    return [sol_dict]
                if set_flag:
                    return (var_list, {sol_tuple})
                return [sol_dict]
        except (ValueError, TypeError):
            pass

        sols = _sps([expr], *var_list)
        if sols is not None:
            if dict_flag:
                return [{sym: val for sym, val in zip(var_list, sol)} for sol in sols]
            if set_flag:
                return (var_list, set(sols))
            return sols
        return []

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
    roots = [_parse_result(r) for r in results]
    if dict_flag:
        return [{symbol: r} for r in roots]
    if set_flag:
        return ([symbol], {(r,) for r in roots})
    return roots



def solveset(expression, variable=None, domain=None):
    """Solve an algebraic equation for ``variable`` and return a Set of solutions."""
    if domain is not None:
        if isinstance(domain, str):
            if domain in ("Reals", "RR", "R"):
                from .sets import Reals
                domain = Reals
        if isinstance(domain, type) and issubclass(domain, Set):
            domain = domain()
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
    except ValueError as exc:
        if str(exc) != "No solution found for equation":
            raise
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
    if isinstance(expression, (list, tuple, set)):
        if not expression:
            raise ValueError("no functions to check")
        return all(checksol(fi, symbol, val) for fi in expression)

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
    if simplified == 0 or getattr(simplified, "is_zero", None) is True:
        return True
    expanded = expand(subbed)
    return bool(expanded == 0 or getattr(expanded, "is_zero", None) is True)


def laplace_transform(expression, t, s, noconds=True, **kwargs):
    from .integrals.transforms import laplace_transform as _lt
    return _lt(expression, t, s, noconds=noconds, **kwargs)


def inverse_laplace_transform(expression, s, t, plane=None, noconds=True, **kwargs):
    from .integrals.transforms import inverse_laplace_transform as _ilt
    return _ilt(expression, s, t, plane=plane, noconds=noconds, **kwargs)


def fourier_transform(expression, t, w, noconds=True, **kwargs):
    from .integrals.transforms import fourier_transform as _ft
    return _ft(expression, t, w, noconds=noconds, **kwargs)


def inverse_fourier_transform(expression, w, t, noconds=True, **kwargs):
    from .integrals.transforms import inverse_fourier_transform as _ift
    return _ift(expression, w, t, noconds=noconds, **kwargs)


def sine_transform(f, t, k, noconds=True, **kwargs):
    from .integrals.transforms import sine_transform as _st
    return _st(f, t, k, noconds=noconds, **kwargs)


def inverse_sine_transform(F, k, t, noconds=True, **kwargs):
    from .integrals.transforms import inverse_sine_transform as _ist
    return _ist(F, k, t, noconds=noconds, **kwargs)


def cosine_transform(f, t, k, noconds=True, **kwargs):
    from .integrals.transforms import cosine_transform as _ct
    return _ct(f, t, k, noconds=noconds, **kwargs)


def inverse_cosine_transform(F, k, t, noconds=True, **kwargs):
    from .integrals.transforms import inverse_cosine_transform as _ict
    return _ict(F, k, t, noconds=noconds, **kwargs)


def hankel_transform(f, r, k, nu, noconds=True, **kwargs):
    from .integrals.transforms import hankel_transform as _ht
    return _ht(f, r, k, nu, noconds=noconds, **kwargs)


def inverse_hankel_transform(F, k, r, nu, noconds=True, **kwargs):
    from .integrals.transforms import inverse_hankel_transform as _iht
    return _iht(F, k, r, nu, noconds=noconds, **kwargs)


def mellin_transform(f, x, s, noconds=True, **kwargs):
    from .integrals.transforms import mellin_transform as _mt
    return _mt(f, x, s, noconds=noconds, **kwargs)


def inverse_mellin_transform(F, s, x, strip=None, noconds=True, **kwargs):
    from .integrals.transforms import inverse_mellin_transform as _imt
    return _imt(F, s, x, strip=strip, noconds=noconds, **kwargs)


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
    Chi,
    Ci,
    Ei,
    FresnelC,
    FresnelS,
    Shi,
    Si,
    acos,
    acosh,
    acot,
    acoth,
    acsc,
    acsch,
    airyai,
    airybi,
    arg,
    asec,
    asech,
    asin,
    asinh,
    atan,
    atanh,
    bell,
    bernoulli,
    besseli,
    besselj,
    besselk,
    bessely,
    beta,
    binomial,
    catalan,
    ceiling,
    conjugate,
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    digamma,
    dirichlet_eta,
    erf,
    erfc,
    erfcinv,
    erfi,
    erfinv,
    exp,
    factorial,
    fibonacci,
    floor,
    gamma,
    hankel1,
    hankel2,
    harmonic,
    hyper,
    im,
    jn,
    lerchphi,
    log,
    loggamma,
    lowergamma,
    lucas,
    meijerg,
    polygamma,
    polylog,
    re,
    sec,
    sech,
    sign,
    sin,
    sinc,
    sinh,
    subfactorial,
    tan,
    tanh,
    trigamma,
    uppergamma,
    yn,
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
    discrete_log,
    divisors,
    integer_nthroot,
    is_nthpow_residue,
    is_perfect,
    is_primitive_root,
    is_quad_residue,
    is_square_free,
    is_squarefree,
    legendre_symbol,
    mod_inverse,
    multiplicity,
    nextprime,
    npartitions,
    perfect_power,
    prevprime,
    prime,
    prime_big_omega,
    prime_omega,
    primefactors,
    primenu,
    primeomega,
    primepi,
    primerange,
    proper_divisors,
    quadratic_residues,
    reduced_totient,
    sqrt_mod,
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
    Set,
    SymmetricDifference,
    Union,
    UniversalSet,
)
Reals = S.Reals
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
    are_collinear,
    are_coplanar,
    are_similar,
    centroid,
    convex_hull,
    idiff,
    intersection,
)
from .integrals import (
    CosineTransform,
    FourierTransform,
    HankelTransform,
    Integral,
    InverseCosineTransform,
    InverseFourierTransform,
    InverseHankelTransform,
    InverseLaplaceTransform,
    InverseMellinTransform,
    InverseSineTransform,
    LaplaceTransform,
    MellinTransform,
    SineTransform,
    Transform,
)
from .logic import (
    And,
    Boolean,
    BooleanAtom,
    BooleanFalse,
    BooleanFunction,
    BooleanTrue,
    Equivalent,
    ITE,
    Implies,
    NAND,
    NOR,
    Nand,
    Nor,
    Not,
    Or,
    POSform,
    SOPform,
    XNOR,
    Xnor,
    Xor,
    false,
    is_nnf,
    pl_true,
    satisfiable,
    simplify_logic,
    to_cnf,
    to_dnf,
    to_nnf,
    true,
    truth_table,
    valid,
)
from .polys import (
    EC,
    LC,
    Poly,
    TC,
    apart,
    cancel,
    compose,
    content,
    decompose,
    degree,
    discriminant,
    div,
    factor,
    factor_list,
    gcd,
    gcdex,
    groebner,
    half_gcdex,
    lcm,
    monic,
    poly,
    primitive,
    quo,
    rem,
    resultant,
    roots,
    sqf,
    sqf_list,
    sqf_part,
    sturm,
    together,
    trailing_coeff,
)
from . import calculus
from .calculus import (
    AccumBounds,
    AccumulationBounds,
    apply_finite_diff,
    continuous_domain,
    differentiate_finite,
    euler_equations,
    finite_diff_weights,
    function_range,
    is_decreasing,
    is_increasing,
    is_monotonic,
    is_strictly_decreasing,
    is_strictly_increasing,
    maximum,
    minimum,
    periodicity,
    singularities,
    stationary_points,
)
from .series import (
    FourierSeries,
    Limit,
    O,
    Order,
    formal_power_series,
    fourier_series,
    fps,
    limit,
    residue,
    series,
)
from .combinatorics import (
    AlternatingGroup,
    Cycle,
    CyclicGroup,
    DihedralGroup,
    GrayCode,
    IntegerPartition,
    Partition,
    Permutation,
    PermutationGroup,
    Subset,
    SymmetricGroup,
)
from .solvers import (
    checkodesol,
    dsolve_cauchy_euler,
    dsolve_const_coeff_second_order,
    dsolve_const_coeff_second_order_nonhomogeneous,
    dsolve_linear_first_order,
    dsolve_separable_linear,
    linsolve,
    nonlinsolve,
    solve_linear,
    solve_linear_system,
    solve_linear_system_LU,
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
    assuming,
    global_assumptions,
    refine,
)
from .simplify import (
    collect,
    combsimp,
    expand_log,
    expand_power_base,
    expand_power_exp,
    expand_trig,
    logcombine,
    nsimplify,
    powsimp,
    radsimp,
    ratsimp,
    separatevars,
    simplify,
    trigsimp,
)


__all__ = [
    "Abs",
    "AccumBounds",
    "AccumulationBounds",
    "Add",
    "AlternatingGroup",
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
    "Catalan",
    "Chi",
    "Ci",
    "Circle",
    "Complement",
    "ComplexInfinity",
    "CosineTransform",
    "Cycle",
    "CyclicGroup",
    "DenseMatrix",
    "Derivative",
    "DihedralGroup",
    "Dummy",
    "E",
    "EC",
    "Ei",
    "Ellipse",
    "EmptySet",
    "Eq",
    "Equality",
    "Equivalent",
    "EulerGamma",
    "Expr",
    "FiniteSet",
    "Float",
    "FourierSeries",
    "FourierTransform",
    "FresnelC",
    "FresnelS",
    "Function",
    "FunctionClass",
    "Ge",
    "GoldenRatio",
    "GramSchmidt",
    "GrayCode",
    "Gt",
    "HankelTransform",
    "I",
    "ITE",
    "ImmutableDenseMatrix",
    "ImmutableMatrix",
    "Implies",
    "Integer",
    "IntegerPartition",
    "Integral",
    "Intersection",
    "Interval",
    "InverseCosineTransform",
    "InverseFourierTransform",
    "InverseHankelTransform",
    "InverseLaplaceTransform",
    "InverseMellinTransform",
    "InverseSineTransform",
    "LC",
    "Le",
    "Line",
    "Line2D",
    "Line3D",
    "Limit",
    "LinearEntity",
    "LaplaceTransform",
    "Lt",
    "Matrix",
    "MatrixBase",
    "MellinTransform",
    "Metric",
    "Mul",
    "MutableDenseMatrix",
    "MutableSparseMatrix",
    "N",
    "NAND",
    "NOR",
    "Nand",
    "Ne",
    "Nor",
    "Not",
    "Number",
    "O",
    "Or",
    "Order",
    "POSform",
    "Partition",
    "Permutation",
    "PermutationGroup",
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
    "SOPform",
    "Segment",
    "Segment2D",
    "Segment3D",
    "Set",
    "Shi",
    "Si",
    "SparseMatrix",
    "Sphere",
    "SineTransform",
    "Subset",
    "Symbol",
    "SymmetricDifference",
    "SymmetricGroup",
    "TC",
    "Tensor",
    "TensorIndex",
    "Triangle",
    "Transform",
    "Tuple",
    "UndefinedFunction",
    "Union",
    "UniversalSet",
    "XNOR",
    "Xnor",
    "Xor",
    "__version__",
    "acos",
    "acosh",
    "acot",
    "acoth",
    "acsc",
    "acsch",
    "airyai",
    "airybi",
    "apart",
    "apply_finite_diff",
    "are_collinear",
    "are_coplanar",
    "are_similar",
    "arg",
    "asec",
    "asech",
    "ask",
    "asin",
    "asinh",
    "assuming",
    "atan",
    "atanh",
    "bell",
    "bernoulli",
    "besseli",
    "besselj",
    "besselk",
    "bessely",
    "beta",
    "binomial",
    "cancel",
    "calculus",
    "carmichael",
    "casoratian",
    "cholesky",
    "catalan",
    "centroid",
    "ceiling",
    "checkodesol",
    "checksol",
    "collect",
    "combsimp",
    "conjugate",
    "content",
    "continuous_domain",
    "convex_hull",
    "compose",
    "cos",
    "cosh",
    "cosine_transform",
    "cot",
    "coth",
    "csc",
    "csch",
    "decompose",
    "degree",
    "det",
    "diag",
    "diff",
    "differentiate_finite",
    "digamma",
    "dirichlet_eta",
    "discrete_log",
    "discriminant",
    "div",
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
    "erfcinv",
    "erfi",
    "erfinv",
    "euler_equations",
    "exp",
    "expand_log",
    "expand_power_base",
    "expand_power_exp",
    "expand_trig",
    "eye",
    "factor",
    "factor_list",
    "factorial",
    "factorint",
    "false",
    "fibonacci",
    "finite_diff_weights",
    "floor",
    "formal_power_series",
    "fourier_series",
    "fourier_transform",
    "fps",
    "function_range",
    "gamma",
    "gcd",
    "gcdex",
    "global_assumptions",
    "groebner",
    "hadamard_product",
    "half_gcdex",
    "hankel1",
    "hankel2",
    "hankel_transform",
    "harmonic",
    "hstack",
    "hyper",
    "idiff",
    "im",
    "integer_nthroot",
    "integrate",
    "intersection",
    "inverse_cosine_transform",
    "inverse_fourier_transform",
    "inverse_hankel_transform",
    "inverse_laplace_transform",
    "inverse_mellin_transform",
    "inverse_sine_transform",
    "is_decreasing",
    "is_increasing",
    "is_monotonic",
    "is_nnf",
    "is_nthpow_residue",
    "is_strictly_decreasing",
    "is_strictly_increasing",
    "is_perfect",
    "is_primitive_root",
    "is_quad_residue",
    "is_square_free",
    "is_squarefree",
    "isprime",
    "jacobian",
    "jacobi_symbol",
    "jn",
    "jordan_block",
    "jordan_cell",
    "kronecker_product",
    "laplace_transform",
    "lcm",
    "legendre_symbol",
    "lerchphi",
    "limit",
    "linsolve",
    "log",
    "logcombine",
    "loggamma",
    "lowergamma",
    "lucas",
    "matrix_multiply_elementwise",
    "maximum",
    "meijerg",
    "mellin_transform",
    "minimum",
    "mobius",
    "mod_inverse",
    "monic",
    "multiplicity",
    "nan",
    "nextprime",
    "nonlinsolve",
    "npartitions",
    "nsimplify",
    "ones",
    "oo",
    "perfect_power",
    "periodicity",
    "pi",
    "pl_true",
    "poly",
    "polygamma",
    "polylog",
    "powsimp",
    "pretty",
    "prevprime",
    "prime",
    "primitive",
    "primenu",
    "prime_big_omega",
    "prime_omega",
    "primefactors",
    "primeomega",
    "primepi",
    "primerange",
    "proper_divisors",
    "quadratic_residues",
    "quo",
    "radsimp",
    "randMatrix",
    "rank",
    "ratsimp",
    "re",
    "reduced_totient",
    "refine",
    "rem",
    "residue",
    "resultant",
    "roots",
    "satisfiable",
    "sec",
    "sech",
    "separatevars",
    "series",
    "shape",
    "sign",
    "simplify",
    "simplify_logic",
    "sin",
    "sinc",
    "sine_transform",
    "singularities",
    "sinh",
    "solve",
    "solve_linear",
    "solve_linear_system",
    "solve_linear_system_LU",
    "solve_poly_system",
    "solveset",
    "sqf",
    "sqf_list",
    "sqf_part",
    "sturm",
    "srepr",
    "sqrt",
    "sqrt_mod",
    "stationary_points",
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
    "to_nnf",
    "together",
    "totient",
    "trace",
    "trailing_coeff",
    "trigamma",
    "trigsimp",
    "true",
    "truth_table",
    "uppergamma",
    "valid",
    "vstack",
    "wronskian",
    "yn",
    "zeros",
    "zeta",
    "zoo",
]
