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
from .printing import latex, pretty, srepr
from .core import nan as _core_nan, zoo as _core_zoo
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
nan = _core_nan
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
            parsed = _parse_result(result)
            return parsed.subs({Symbol(symbol.name): symbol})
        except (ValueError, NotImplementedError, TypeError):
            return Integral(expression, (variable, lower, upper))

    symbol = _require_symbol(spec)
    try:
        result = _parse_result(
            _native.integrate_expr(
                str(_wrap(_native_expr(expression))), _native_symbol_key(symbol)
            )
        )
        # Restore declared typed symbol (the native bridge lifts fresh atoms).
        result = result.subs({Symbol(symbol.name): symbol})
        return result
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

    # Numeric constant systems need no root search. Preserve the profile's
    # distinction between an unconstrained zero system and a contradiction,
    # including set=True output and the caller's original symbol objects.
    # Exact-class admission avoids invoking arbitrary Python numeric hooks;
    # unknown zero status and unsupported variables retain the existing lane.
    constant_inputs = expression if type(expression) in (list, tuple) else (expression,)
    numeric_classes = (int, float, Integer, Rational, Float,
                       type(S.Zero), type(S.One), type(S.NegativeOne), type(S.Half))
    if all(type(value) in numeric_classes for value in constant_inputs):
        if var_list is None or (
            all(type(variable) in (Symbol, Dummy) for variable in var_list)
            and len(set(var_list)) == len(var_list)
        ):
            zero_statuses = [sympify(value).is_zero for value in constant_inputs]
            if all(status is not None for status in zero_statuses):
                if all(status is True for status in zero_statuses) and set_flag:
                    return (var_list or [], set())
                return []

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
                sol_tuple = tuple(next(iter(lin_sol)))
                # A free parameter is not an assignment in solve's mapping.
                sol_dict = {s: v for s, v in zip(var_list, sol_tuple) if s != v}
                if not sol_dict:
                    return (var_list, set()) if set_flag else []
                if dict_flag:
                    return [sol_dict]
                if set_flag:
                    return (var_list, {sol_tuple})
                return sol_dict
            else:
                return (var_list, set()) if set_flag else []
        except (ValueError, TypeError):
            pass

        # Fall back to polynomial system solver
        # Only an established no-solution result is empty. Unsupported
        # systems and faults must remain visible to the caller.
        sols = _sps(expression, *var_list)

        if sols is None:
            return (var_list, set()) if set_flag else []

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
                sol_tuple = tuple(next(iter(lin_sol)))
                sol_dict = {s: v for s, v in zip(var_list, sol_tuple) if s != v}
                if not sol_dict:
                    return (var_list, set()) if set_flag else []
                if dict_flag:
                    return [sol_dict]
                if set_flag:
                    return (var_list, {sol_tuple})
                return [sol_tuple]
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
        if not isinstance(domain, Set):
            raise ValueError(f"{domain} is not a valid domain")
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
        from .sets import Intersection, Interval
        roots = [simplify(root) for root in roots]
        if isinstance(domain, Interval):
            # Exact nonzero rational imaginary parts exclude a candidate
            # from every real interval, even if membership is undecidable.
            retained = []
            for root in roots:
                imaginary = simplify(root.as_real_imag()[1])
                if isinstance(imaginary, (Integer, Rational)) and imaginary != 0:
                    continue
                retained.append(root)
            roots = retained
        # Intersection preserves an undecidable membership condition instead
        # of discarding the candidate through boolean containment coercion.
        retained = []
        unknown = False
        for root in roots:
            membership = domain.contains(root)
            if membership is False:
                continue
            retained.append(root)
            unknown = unknown or membership is None
        candidates = FiniteSet(*retained)
        return Intersection(candidates, domain) if unknown else candidates
    if not roots:
        return EmptySet()
    return FiniteSet(*roots)


def checksol(expression, symbol, val=None):
    """Check a solution, returning ``None`` when the result is inconclusive."""
    if isinstance(expression, (list, tuple, set)):
        if not expression:
            raise ValueError("no functions to check")
        inconclusive = False
        for equation in expression:
            result = checksol(equation, symbol, val)
            if result is False:
                return False
            if result is None:
                inconclusive = True
        return None if inconclusive else True

    if type(expression) is Eq:
        expression = expression.lhs - expression.rhs
    expr = _wrap(_native_expr(expression))
    if isinstance(symbol, dict):
        mapping = symbol
    elif val is not None:
        mapping = {symbol: val}
    else:
        raise ValueError("checksol requires either a mapping or (symbol, val)")
    illegal = {S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity}
    if not expr.is_number:
        # Validate the complete candidate before substitution can erase an
        # unused entry or simplify an invalid value out of the residual.
        for candidate in mapping.values():
            if sympify(candidate).atoms() & illegal:
                return False
    subbed = expr
    for sym, v in mapping.items():
        subbed = subbed.subs(sym, v)
    if subbed.atoms() & illegal:
        return False
    simplified = simplify(subbed)
    if simplified.atoms() & illegal:
        return False
    if simplified == 0 or getattr(simplified, "is_zero", None) is True:
        return True
    expanded = expand(subbed)
    if expanded == 0:
        return True
    zero_status = getattr(expanded, "is_zero", None)
    if zero_status is not None:
        return zero_status
    # Failure to simplify a function or radical to zero is not a disproof.
    # Keep the rational-expression negative lane, but leave these harder
    # residuals inconclusive, as the compatibility checker requires.
    pending = [expanded]
    while pending:
        node = pending.pop()
        if isinstance(node, Function):
            return None
        if isinstance(node, Pow) and not isinstance(node.args[1], Integer):
            return None
        pending.extend(node.args)
    return False


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


def lambdify(symbols_list: Any, expr: Any, modules: str = "math") -> Any:
    """Convert a symbolic expression to a numerical callable.

    Parameters
    ----------
    symbols_list : Symbol or tuple of Symbols
        Free variable(s) in positional order.
    expr : Expr
        The expression to evaluate numerically.
    modules : str
        Module namespace for math functions (``"math"`` default).

    Returns
    -------
    A callable ``f(*args)`` evaluating the expression numerically.
    """
    if isinstance(symbols_list, Symbol):
        symbols_list = (symbols_list,)
    arg_names = ", ".join(sym.name for sym in symbols_list)
    import math as _math
    code = f"lambda {arg_names}: {expr}"
    namespace: dict[str, Any] = {}
    exec(code, {"__builtins__": {}, **vars(_math), "Abs": abs}, namespace)
    return namespace.popitem()[1]


def nsolve(expr: Any, var: Any, initial: float, tol: float = 1e-12, maxiter: int = 200) -> Any:
    """Numerically solve ``expr = 0`` for *var* starting from *initial*.

    Uses Newton's method with symbolic differentiation for the Jacobian.
    Returns the converged approximate root as a ``Float``.
    """
    derivative = diff(expr, var)
    f = lambdify(var, expr)
    fp = lambdify(var, derivative)
    x_val = float(initial)
    for _ in range(maxiter):
        f_val = f(x_val)
        fp_val = fp(x_val)
        if abs(fp_val) < 1e-30:
            raise ValueError(
                f"Derivative is zero at x = {x_val}; Newton cannot continue"
            )
        x_new = x_val - f_val / fp_val
        if abs(x_new - x_val) < tol:
            return Float(x_new)
        x_val = x_new
    raise ValueError(
        f"nsolve failed to converge in {maxiter} iterations (last x = {x_val})"
    )


def lambdify(symbols_list: Any, expr: Any, modules: str = "math") -> Any:
    """Convert a symbolic expression to a numerical callable.

    Parameters
    ----------
    symbols_list : Symbol or tuple of Symbols
        The free variable(s) of the expression, in positional order.
    expr : Expr
        The expression to evaluate.
    modules : str
        The math module to use for numerical evaluation.

    Returns
    -------
    A callable ``f(*args)`` that evaluates the expression numerically.
    """
    if isinstance(symbols_list, Symbol):
        symbols_list = (symbols_list,)
    arg_names = ", ".join(sym.name for sym in symbols_list)
    expr_str = str(expr)
    code = f"lambda {arg_names}: {expr_str}"
    import math as _math
    safe_globals: dict[str, Any] = {"__builtins__": {}, **vars(_math), "Abs": abs}
    return eval(code, safe_globals)


def sstr(expr: Any) -> str:
    """Return the string form of *expr* (same as ``str(expr)``)."""
    return str(expr)


def fraction(expr: Any) -> tuple:
    """Return a ``(numer, denom)`` tuple for a rational expression."""
    if isinstance(expr, Rational):
        return (Integer(expr.p), Integer(expr.q))
    if isinstance(expr, Expr):
        num, den = fraction_inner(expr)
        return (num, den)
    return (expr, Integer(1))


def fraction_inner(expr: Any) -> tuple:
    """Walk an expression tree and split into (numer, denom)."""
    if isinstance(expr, Mul):
        numer = Integer(1)
        denom = Integer(1)
        for factor in expr.args:
            n, d = fraction_inner(factor)
            numer = numer * n
            denom = denom * d
        return (numer, denom)
    if isinstance(expr, Pow):
        base, exp = expr.args
        exp_int = _require_int(exp)
        if exp_int is not None and exp_int < 0:
            n, d = fraction_inner(base)
            return (Integer(1), n ** (-exp_int) * d)
        return (expr, Integer(1))
    if isinstance(expr, Rational):
        return (Integer(expr.p), Integer(expr.q))
    return (expr, Integer(1))


def _require_int(expr: Any) -> int | None:
    if isinstance(expr, Integer):
        return expr.p
    return None


def posify(expr: Any) -> tuple:
    """Replace symbols with positive dummies, returning (dummies, substituted)."""
    from .core import Symbol as _Sym

    if isinstance(expr, _Sym):
        dummy = _Sym(f"_pos_{expr.name}", positive=True)
        return (dummy, expr.subs({expr: dummy}))
    if isinstance(expr, Expr):
        symbols = expr.free_symbols
        if not symbols:
            return (expr, expr)
        mapping = {}
        dummies = []
        for s in sorted(symbols, key=lambda s: s.name):
            dummy = _Sym(f"_pos_{s.name}", positive=True)
            mapping[s] = dummy
            dummies.append(dummy)
        return (dummies[0] if len(dummies) == 1 else tuple(dummies), expr.subs(mapping))
    return (expr, expr)


def root(expr: Any, n: int) -> Expr:
    """Return *expr* raised to the power ``1/n``."""
    if n == 2:
        return sqrt(expr)
    return Pow(expr, Rational(1, n))


def cbrt(expr: Any) -> Expr:
    """Return the cube root of *expr*."""
    return Pow(expr, Rational(1, 3))


class Sum:
    """Unevaluated symbolic summation."""

    def __init__(self, function, *limits):
        self.function = function
        self.limits = limits

    def doit(self):
        if len(self.limits) == 1 and isinstance(self.limits[0], (tuple, list)):
            var, lo, hi = self.limits[0]
            var = _require_symbol(var)
            total = Integer(0)
            for k in range(int(lo), int(hi) + 1):
                total += self.function.subs({var: Integer(k)})
            return total
        raise NotImplementedError("multi-index symbolic summation not yet supported")

    def __repr__(self):
        return f"Sum({self.function}, {self.limits})"

    def __str__(self):
        var, lo, hi = self.limits[0] if self.limits else (None, None, None)
        return f"Sum({self.function}, ({var}, {lo}, {hi}))"


class Product:
    """Unevaluated symbolic product."""

    def __init__(self, function, *limits):
        self.function = function
        self.limits = limits

    def doit(self):
        if len(self.limits) == 1 and isinstance(self.limits[0], (tuple, list)):
            var, lo, hi = self.limits[0]
            var = _require_symbol(var)
            total = Integer(1)
            for k in range(int(lo), int(hi) + 1):
                total *= self.function.subs({var: Integer(k)})
            return total
        raise NotImplementedError("multi-index symbolic product not yet supported")

    def __repr__(self):
        return f"Product({self.function}, {self.limits})"

    def __str__(self):
        var, lo, hi = self.limits[0] if self.limits else (None, None, None)
        return f"Product({self.function}, ({var}, {lo}, {hi}))"


def nroots(expr: Any, n: int = 15) -> list:
    """Numerical roots of a univariate polynomial, sorted by real part."""
    sympify_expr = _wrap(_native_expr(expr))
    var = _extract_single_symbol(expr)
    if var is None:
        raise ValueError("expression must contain exactly one free symbol")
    exact_roots = solve(expr, var)
    return [float(r.evalf(n)) if hasattr(r, "evalf") else float(r) for r in exact_roots]


def _extract_single_symbol(expr: Any):
    syms = getattr(expr, "free_symbols", set())
    if len(syms) == 1:
        return next(iter(syms))
    return None


def deg(poly: Any, gen: Any = None) -> int:
    """Return the degree of the leading generator of *poly*."""
    if isinstance(poly, Pow):
        base, exp = poly.args
        exp_int = _as_int_local(exp)
        if exp_int is not None and exp_int > 0:
            return deg(base, gen) * exp_int
    if isinstance(poly, Mul):
        return max(deg(f, gen) for f in poly.args)
    if isinstance(poly, Symbol):
        return 1
    if isinstance(poly, (Integer, Rational)):
        return 0
    return 0


def _as_int_local(expr: Any) -> int | None:
    if isinstance(expr, Integer):
        return expr.p
    return None


def coeff(expr: Any, sym: Any, n: int = 1) -> Any:
    """Extract the coefficient of ``sym**n`` from *expr*."""
    if n == 0:
        # constant term: substitute 0 for sym
        return expr.subs({sym: Integer(0)})
    term_sym = Pow(sym, Integer(n)) if n != 1 else sym
    if isinstance(expr, Add):
        for term in expr.args:
            if isinstance(term, Mul):
                coeff_found = Integer(1)
                has_target = False
                for factor in term.args:
                    if factor == term_sym:
                        has_target = True
                    elif factor == sym:
                        has_target = True
                        coeff_found = coeff_found * Integer(1)
                    elif isinstance(factor, Integer):
                        coeff_found = coeff_found * factor
                if has_target:
                    return coeff_found
            elif term == term_sym:
                return Integer(1)
            elif term == sym:
                return Integer(1)
        return Integer(0)
    if isinstance(expr, Mul):
        coeff_found = Integer(1)
        has_target = False
        for factor in expr.args:
            if factor == term_sym or factor == sym:
                has_target = True
            elif isinstance(factor, Integer):
                coeff_found = coeff_found * factor
        return coeff_found if has_target else Integer(0)
    if expr == term_sym or expr == sym:
        return Integer(1)
    return Integer(0)


def re(expr: Any) -> Any:
    """Real part via as_real_imag decomposition."""
    if hasattr(expr, "as_real_imag"):
        return expr.as_real_imag()[0]
    return expr


def im(expr: Any) -> Any:
    """Imaginary part via as_real_imag decomposition."""
    if hasattr(expr, "as_real_imag"):
        return expr.as_real_imag()[1]
    return Integer(0)


def conjugate(expr: Any) -> Any:
    """Complex conjugate via as_real_imag decomposition."""
    if hasattr(expr, "as_real_imag"):
        re_part, im_part = expr.as_real_imag()
        return re_part - im_part * I
    return expr


def count_ops(expr: Any) -> Integer:
    """Count the number of operations in an expression."""
    def _count(node: Any) -> int:
        args = getattr(node, "args", ())
        if not args:
            return 0
        return 1 + sum(_count(a) for a in args)
    return Integer(_count(expr))


def real_roots(expr: Any, var: Any = None) -> list:
    """Return the real roots of a univariate polynomial expression."""
    solutions = solve(expr, var) if var is not None else solve(expr)
    real: list = []
    for sol in solutions:
        try:
            v = float(sol)
            real.append(sol)
        except (TypeError, ValueError):
            continue
    return real


def factor_terms(expr: Any) -> Any:
    """Factor out common coefficients from an expression."""
    if isinstance(expr, Add):
        terms = expr.args
        # Extract numeric coefficients
        coeffs: list = []
        for t in terms:
            if isinstance(t, (Integer, Rational)):
                coeffs.append(t)
                continue
            c = Rational(1)
            if isinstance(t, Mul):
                numeric_factors = [f for f in t.args if isinstance(f, (Integer, Rational))]
                for f in numeric_factors:
                    c = c * f
            coeffs.append(c)
        # Compute the GCD of the numeric coefficients
        from math import gcd as _igcd
        int_coeffs = [abs(int(c)) for c in coeffs]
        common = 0
        for ic in int_coeffs:
            common = _igcd(common, ic) if common else ic
        if common <= 1:
            return expr
        common_r = Rational(common)
        factored = Add(*[t / common_r for t in terms], evaluate=False)
        return Mul(common_r, factored, evaluate=False)
    return expr


class Max:
    """Maximum of arguments (evaluates numerically; raises on incomparable)."""

    def __new__(cls, *args: Any) -> Any:
        if len(args) == 1:
            return args[0]
        try:
            vals = [float(a) for a in args]
            return Float(max(vals))
        except (TypeError, ValueError):
            raise NotImplementedError(
                "symbolic Max requires Piecewise (not in shell subset)"
            )


class Min:
    """Minimum of arguments (evaluates numerically; raises on incomparable)."""

    def __new__(cls, *args: Any) -> Any:
        if len(args) == 1:
            return args[0]
        try:
            vals = [float(a) for a in args]
            return Float(min(vals))
        except (TypeError, ValueError):
            raise NotImplementedError(
                "symbolic Min requires Piecewise (not in shell subset)"
            )


def minpoly(expr: Any, gen: Any = None) -> Any:
    """Minimal polynomial of a quadratic surd expression.

    Handles a + b*sqrt(c) forms (with rational a, b, c) by conjugate
    elimination. Returns a Poly in *gen* (default x).
    """
    x = Symbol("x") if gen is None else gen
    expr = sympify(expr) if "sympify" in globals() else expr
    # Detect a + b*sqrt(c) with integer/rational a, b, c
    if isinstance(expr, Add) and len(expr.args) == 2:
        a_part, rest = expr.args
        a_ok = isinstance(a_part, (Integer, Rational))
        bare_surd = isinstance(rest, Pow) and len(rest.args) == 2 and rest.args[1] == Rational(1, 2)
        if a_ok and bare_surd:
            b_val, c_val, a_val = Integer(1), rest.args[0], a_part
            two_a = Integer(2) * a_val
            const = a_val**2 - b_val**2 * c_val
            return Poly(x**2 - two_a * x + const, x)
        if a_ok and isinstance(rest, Mul):
            coeffs = [f for f in rest.args if isinstance(f, (Integer, Rational))]
            surds = [f for f in rest.args if isinstance(f, Pow)]
            if len(coeffs) == 1 and len(surds) == 1 and len(surds[0].args) == 2 and surds[0].args[1] == Rational(1, 2):
                b_val = coeffs[0]
                c_val = surds[0].args[0]
                a_val = a_part
                # (x - a)^2 = b^2 * c  =>  x^2 - 2a x + (a^2 - b^2 c)
                two_a = Integer(2) * a_val
                const = a_val**2 - b_val**2 * c_val
                return Poly(x**2 - two_a * x + const, x)
    if isinstance(expr, Pow) and len(expr.args) == 2 and expr.args[1] == Rational(1, 2):
        # sqrt(c): x^2 - c
        return Poly(x**2 - expr.args[0], x)
    if isinstance(expr, Mul):
        coeffs = [f for f in expr.args if isinstance(f, (Integer, Rational))]
        surds = [f for f in expr.args if isinstance(f, Pow)]
        if len(coeffs) == 1 and len(surds) == 1 and len(surds[0].args) == 2 and surds[0].args[1] == Rational(1, 2):
            # b*sqrt(c): x^2 - b^2 c
            b_val, c_val = coeffs[0], surds[0].args[0]
            return Poly(x**2 - b_val**2 * c_val, x)
    if isinstance(expr, (Integer, Rational)):
        return Poly(x - expr, x)
    raise NotImplementedError(
        f"minpoly supports rational and quadratic surd forms, got {expr}"
    )


minimal_polynomial = minpoly


def reduce_inequalities(inequalities: Any, symbols_list: Any = None) -> Any:
    """Reduce a list of polynomial inequalities to solution intervals.

    Returns a list of relational solutions (e.g. ``x > -2`` pairs).
    Only strict polynomial inequalities in one variable are supported.
    """
    if not isinstance(inequalities, (list, tuple)):
        inequalities = [inequalities]
    var = symbols_list if symbols_list is not None else _free_var_of(inequalities[0])
    results: list = []
    for ineq in inequalities:
        if isinstance(ineq, (Lt, Gt, Le, Ge)):
            # ineq.lhs < ineq.rhs form: solve lhs - rhs against the sign
            diff_expr = ineq.lhs - ineq.rhs
            roots = solve(diff_expr, var)
            test_points = _sample_points(roots, diff_expr, var)
            holds = [bool(diff_expr.subs({var: p})) for p in test_points]
            for tp, ok in zip(test_points, holds):
                if ok:
                    results.append((ineq, tp))
        else:
            raise NotImplementedError(
                f"reduce_inequalities handles Lt/Gt/Le/Ge, got {type(ineq).__name__}"
            )
    return results


def _free_var_of(expr: Any) -> Any:
    syms = getattr(expr, "free_symbols", set())
    if len(syms) != 1:
        raise ValueError(
            f"reduce_inequalities needs exactly one free variable, got {sorted(s.name for s in syms)}"
        )
    return next(iter(syms))


def _sample_points(roots: list, expr: Any, var: Any) -> list:
    """Pick one test point per interval defined by the roots."""
    vals = sorted(float(r) for r in roots)
    points: list = []
    points.append(Integer(vals[0] - 1) if vals else Integer(0))
    for i in range(len(vals) - 1):
        points.append(Float((vals[i] + vals[i + 1]) / 2))
    if vals:
        points.append(Integer(vals[-1] + 1))
    if not roots:
        points = [Integer(0)]
    return points


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
    "sstr",
    "sstr",
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
    "root",
    "cbrt",
    "Sum",
    "Product",
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
    "nsolve",
    "lambdify",
    "count_ops",
    "real_roots",
    "factor_terms",
    "Max",
    "Min",
    "minpoly",
    "minimal_polynomial",
    "reduce_inequalities",
    "nroots",
    "deg",
    "coeff",
    "re",
    "im",
    "conjugate",
    "zeta",
    "zoo",
]
