"""Experimental FrankenSymPy compatibility slice.

Only the names exported here are wired to native behavior. Unsupported SymPy
operations fail explicitly; upstream SymPy is never used as a fallback.
"""

from __future__ import annotations

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
    _symbol_facts,
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
from .printing import latex, pprint, pprint_use_unicode, pretty, pretty_print, srepr
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
                str(_native_expr(expression)),
                _native_symbol_key(symbol),
                str(_native_expr(lower)),
                str(_native_expr(upper)),
                _symbol_facts(expression, lower, upper),
            )
            parsed = _parse_result(result)
            return parsed.subs({Symbol(symbol.name): symbol})
        except (ValueError, NotImplementedError, TypeError):
            return Integral(expression, (variable, lower, upper))

    symbol = _require_symbol(spec)
    special = _integrate_symbolic_power(expression, symbol)
    if special is not None:
        return special
    try:
        result = _parse_result(
            _native.integrate_expr(
                str(_native_expr(expression)), _native_symbol_key(symbol),
                _symbol_facts(expression),
            )
        )
        # Restore declared typed symbol (the native bridge lifts fresh atoms).
        result = result.subs({Symbol(symbol.name): symbol})
        return result
    except (ValueError, NotImplementedError, TypeError):
        return Integral(expression, symbol)


def _integrate_symbolic_power(expression, x):
    """Terms ``c*x**e`` with a symbolic exponent that may equal -1 integrate
    to upstream's ``c*Piecewise((x**(e + 1)/(e + 1), Ne(e, -1)),
    (log(x), True))``; the remaining terms go through the native lane.
    None when no term has that shape."""
    from .functions.elementary.piecewise import Piecewise as _Piecewise

    expr = sympify(expression)
    terms = list(expr.args) if isinstance(expr, Add) else [expr]
    special, rest = [], []
    for t in terms:
        factors = list(t.args) if isinstance(t, Mul) else [t]
        powers = [f for f in factors if isinstance(f, Pow) and f.args[0] == x
                  and x not in f.args[1].free_symbols and f.args[1].free_symbols]
        others = [f for f in factors if f not in powers]
        if len(powers) == 1 and all(x not in f.free_symbols for f in others):
            e = powers[0].args[1]
            if (e + 1).is_zero is False:
                rest.append(t)
                continue
            c = Mul(*others)
            cond = Ne(e, -1)
            syms = list(e.free_symbols)
            if len(syms) == 1 and e != syms[0]:
                # Upstream states the condition on the symbol itself:
                # x**(n - 1) -> Ne(n, 0).
                try:
                    sol = solve(e + 1, syms[0])
                    if len(sol) == 1:
                        cond = Ne(syms[0], sol[0])
                except Exception:
                    pass
            pw = _Piecewise((x ** (e + 1) / (e + 1), cond), (log(x), True))
            special.append(c * pw)
        else:
            rest.append(t)
    if not special:
        return None
    total = Add(*special)
    if rest:
        total = integrate(Add(*rest), x) + total
    return total


def _solve_rational_fallback(expr, symbol):
    """Roots of a rational equation ``N/D = 0`` the native lanes reject:
    the numerator of ``together(expr)`` is solved and roots that make the
    denominator vanish are discarded. None when this changes nothing."""
    from .polys import together as _together

    num, den = fraction(_together(expr))
    if symbol not in den.free_symbols:
        return None
    num = expand(num)
    if num == expr or symbol not in num.free_symbols:
        return None
    try:
        cands = solve(num, symbol)
    except (ValueError, NotImplementedError):
        return None
    out = []
    for r in cands:
        d = expand(den.subs(symbol, r))
        if d == 0 or simplify(d) == 0:
            continue
        out.append(r)
    return out


def solve(expression, *symbols, **flags):
    """Solve the algebraic equation or system of equations ``expression == 0``."""
    dict_flag = bool(flags.get("dict", False))
    set_flag = bool(flags.get("set", False))

    ineqs = expression if isinstance(expression, (list, tuple)) else [expression]
    if ineqs and any(_is_inequality(e) for e in ineqs):
        return reduce_inequalities(list(ineqs), symbols[0] if len(symbols) == 1 else (list(symbols) or None))

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

        try:
            sols = _sps([expr], *var_list)
        except NotImplementedError:
            # Upstream: one equation in several symbols is solved for the
            # first symbol it contains; the others stay free.
            target = next((v for v in var_list if v in expr.free_symbols), None)
            if target is None:
                return []
            roots = solve(expr, target)
            if dict_flag:
                return [{target: r} for r in roots]
            tuples = [tuple(r if v == target else v for v in var_list) for r in roots]
            if set_flag:
                return (var_list, set(tuples))
            return tuples
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
            str(_native_expr(expr)), _native_symbol_key(symbol)
        )
    except Exception as exc:
        msg = str(exc)
        if "No solution found" in msg or "Infinite solutions" in msg:
            return []
        roots = _solve_rational_fallback(expr, symbol)
        if roots is None:
            raise
    else:
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
    if _is_inequality(expression):
        if variable is None:
            variable = _inequality_symbol([expression], ())
        from .sets import Reals as _RealsSet
        if domain is None:
            raise ValueError(
                "Inequalities in the complex domain are not supported. "
                "Try the real domain by setting domain=S.Reals"
            )
        sol = _solve_inequality_set(expression, _require_symbol(variable))
        if domain == _RealsSet() or isinstance(domain, _RealsSet):
            return sol
        from .sets import Intersection as _Intersection
        return _Intersection(sol, domain)
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

    # Periodic equations have infinite solution sets (upstream ImageSet
    # unions); the native solver lists principal solutions only, which is
    # solve()'s contract but not solveset's. Refuse rather than truncate.
    periodic = ("sin", "cos", "tan", "cot", "sec", "csc")
    from .sets import Reals as _RealsP

    over_reals = domain is not None and (domain == _RealsP() or isinstance(domain, _RealsP))
    complex_exp = not over_reals and any(
        type(a).__name__ == "exp" and symbol in a.free_symbols for a in _preorder_nodes(expr)
    )
    if complex_exp or any(
        type(a).__name__ in periodic and symbol in a.free_symbols for a in _preorder_nodes(expr)
    ):
        # Complete (infinite) solution sets as ImageSet unions; shapes
        # outside the exact periodic solver raise instead of truncating.
        from .solvers.solveset import solve_periodic

        sol = solve_periodic(expr, symbol, domain)
        if domain is not None and not over_reals and type(domain).__name__ != "Complexes":
            from .sets import Intersection as _IntersectionP

            return _IntersectionP(sol, domain)
        return sol
    try:
        results = _native.solve_expr(str(_native_expr(expr)), _native_symbol_key(symbol))
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
        from .sets import Reals as _Reals
        for root in roots:
            if (isinstance(domain, _Reals) or domain == _Reals()) and _is_real_constant(root):
                retained.append(root)
                continue
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


def _preorder_nodes(e):
    yield e
    for a in getattr(e, "args", ()):
        yield from _preorder_nodes(a)


def _is_real_constant(e) -> bool:
    """Structural real-valuedness of a symbol-free constant (no numerics)."""
    if getattr(e, "free_symbols", None):
        return False
    if isinstance(e, (Integer, Rational, Float)):
        return True
    name = str(e) if type(e) is Expr else None
    if name in ("pi", "E", "EulerGamma", "GoldenRatio", "Catalan"):
        return True
    if name == "I":
        return False
    if type(e) in (Add, Mul):
        return all(_is_real_constant(a) for a in e.args)
    if type(e) is Pow:
        b, p = e.args
        return _is_real_constant(b) and _is_real_constant(p) and _is_positive_constant(b)
    fname = type(e).__name__
    if fname in ("exp", "sin", "cos", "atan", "sinh", "cosh", "tanh", "Abs"):
        return all(_is_real_constant(a) for a in e.args)
    if fname == "log":
        return _is_positive_constant(e.args[0])
    return False


def _is_positive_constant(e) -> bool:
    if isinstance(e, (Integer, Rational)):
        return e.p > 0
    if type(e) is Expr and str(e) in ("pi", "E"):
        return True
    if type(e) is Mul:
        return all(_is_positive_constant(a) for a in e.args)
    if type(e) is Add:
        return all(_is_positive_constant(a) for a in e.args)
    if type(e) is Pow:
        return _is_positive_constant(e.args[0]) and _is_real_constant(e.args[1])
    if type(e).__name__ == "exp":
        return _is_real_constant(e.args[0])
    return False


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


def laplace_transform(expression, t, s, noconds=False, **kwargs):
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


def dsolve(equation, func=None, hint="default", **kwargs):
    """Solve an ordinary differential equation (see sympy.solvers.ode.dsolve)."""
    from .solvers.ode import dsolve as _dsolve

    return _dsolve(equation, func, hint=hint, **kwargs)


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
    DiracDelta,
    Heaviside,
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
    Complexes,
    ConditionSet,
    ImageSet,
    Integers,
    Naturals,
    Naturals0,
    Range,
    Rationals,
    imageset,
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
    GroebnerBasis,
    LM,
    LT,
    interpolate,
    invert,
    EC,
    LC,
    Poly,
    PurePoly,
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
    terms_gcd,
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
    gammasimp,
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


def _limit_triples(limits):
    from .core import sympify

    out = []
    for lim in limits:
        if isinstance(lim, (tuple, list)) or type(lim).__name__ == "Tuple":
            items = tuple(lim.args) if type(lim).__name__ == "Tuple" else tuple(lim)
            if len(items) != 3:
                raise ValueError("limits must be (index, lower, upper) triples")
            var, lo, hi = items
            out.append((_require_symbol(var), sympify(lo), sympify(hi)))
        else:
            raise ValueError("limits must be (index, lower, upper) triples")
    return out


class _ConcreteOperator(Function):
    """Shared shape of Sum / Product: args (function, Tuple(k, a, b), ...)."""

    __slots__ = ()
    _native_fn = ""

    def __new__(cls, function, *limits, **options):
        from .core import sympify

        triples = _limit_triples(limits)
        args = [sympify(function)] + [Tuple(k, a, b) for k, a, b in triples]
        return Function.__new__(cls, *args, evaluate=False)

    @property
    def function(self):
        return self.args[0]

    @property
    def limits(self):
        return tuple(tuple(t.args) for t in self.args[1:])

    @property
    def variables(self):
        return [lim[0] for lim in self.limits]

    @property
    def free_symbols(self):
        syms = set(self.function.free_symbols)
        for k, a, b in self.limits:
            syms.discard(k)
            syms |= set(getattr(a, "free_symbols", set()))
            syms |= set(getattr(b, "free_symbols", set()))
        return syms

    def doit(self, **hints):
        current = self.function
        if hints.get("deep", True) and hasattr(current, "doit"):
            current = current.doit(**hints)
        for index, (k, a, b) in enumerate(self.limits):
            native = getattr(_native, self._native_fn)(
                str(_native_expr(current)),
                _native_symbol_key(k),
                str(_native_expr(a)),
                str(_native_expr(b)),
            )
            if native is None:
                remaining = self.limits[index:]
                return type(self)(current, *remaining)
            current = _parse_result(native).subs({Symbol(k.name): k})
        return current


class Sum(_ConcreteOperator):
    """Unevaluated summation ``Sum(f, (k, a, b))``; ``doit`` finds closed forms."""

    __slots__ = ()
    _native_fn = "summation_expr"


class Product(_ConcreteOperator):
    """Unevaluated product ``Product(f, (k, a, b))``; ``doit`` finds closed forms."""

    __slots__ = ()
    _native_fn = "product_expr"


def summation(f, *symbols, **kwargs):
    """Closed form of a sum (upstream ``summation``); unevaluated ``Sum`` if none."""
    return Sum(f, *symbols).doit(deep=False)


def product(*args, **kwargs):
    """Closed form of a product (upstream ``product``); unevaluated ``Product`` if none."""
    return Product(*args).doit(deep=False)


def _numeric_poly_roots(coeffs: list) -> list:
    """All complex roots of a polynomial (descending coefficients) by
    Aberth-Ehrlich simultaneous iteration, polished by Newton steps."""
    import cmath

    while coeffs and coeffs[0] == 0:
        coeffs = coeffs[1:]
    deg = len(coeffs) - 1
    if deg < 1:
        return []
    lead = coeffs[0]
    a = [c / lead for c in coeffs]
    zeros = 0
    while deg > 0 and a[-1] == 0:
        a.pop()
        deg -= 1
        zeros += 1

    def ev(z: complex) -> tuple:
        pv, dv = 0j, 0j
        for c in a:
            dv = dv * z + pv
            pv = pv * z + c
        return pv, dv

    roots = []
    if deg > 0:
        radius = 1 + max(abs(c) for c in a[1:])
        zs = [radius * 0.5 * cmath.exp(2j * cmath.pi * (k + 0.25) / deg) for k in range(deg)]
        for _ in range(500):
            done = True
            for i in range(deg):
                pv, dv = ev(zs[i])
                if pv == 0:
                    continue
                ratio = pv / dv if dv != 0 else 1e-3
                corr = sum(1 / (zs[i] - zs[j]) for j in range(deg) if j != i and zs[i] != zs[j])
                w = ratio / (1 - ratio * corr)
                zs[i] -= w
                if abs(w) > 1e-17 * max(1.0, abs(zs[i])):
                    done = False
            if done:
                break
        for z in zs:
            for _ in range(3):
                pv, dv = ev(z)
                if dv == 0:
                    break
                step = pv / dv
                if abs(step) > 1e-6 * max(1.0, abs(z)):
                    break
                z -= step
            roots.append(z)
    return roots + [0j] * zeros


def _float_digits(v: float, n: int) -> Any:
    from .core import _ExactDecimalFloat

    if v == 0:
        return Integer(0)
    text = f"{v:#.{n}g}"
    if "e" not in text and text.endswith("."):
        text = text + "0"
    return _ExactDecimalFloat(text, n)


def nroots(f: Any, n: int = 15, maxsteps: int = 50, cleanup: bool = True) -> list:
    """Numerical roots of a univariate polynomial with numeric coefficients
    (upstream ``nroots``): real roots first in increasing order, then the
    complex ones by real and imaginary part. Binary64-backed, so at most
    15 significant digits are honest; more is refused."""
    del maxsteps, cleanup
    if n > 15:
        raise NotImplementedError(
            "precision-honest nroots: this shell computes roots in binary64 and "
            "is honest to at most 15 significant digits; requested %d" % n
        )
    from .polys.polytools import Poly as _Poly

    p = f if isinstance(f, _Poly) else _Poly(f)
    if len(p.gens) != 1:
        raise ValueError("nroots requires a univariate polynomial")
    # Exact square-free parts first: simultaneous iteration only reaches
    # sqrt(eps) on a repeated root.
    parts = [(p, 1)]
    try:
        _, sq = p.sqf_list()
        if sq:
            parts = sq
    except Exception:
        pass
    roots = []
    for part, mult in parts:
        coeffs = []
        for c in part.all_coeffs():
            try:
                coeffs.append(complex(N(c)))
            except Exception:
                raise ValueError("nroots requires numeric coefficients, got %s" % c)
        roots.extend(_numeric_poly_roots(coeffs) * mult)
    scale = max([1.0] + [abs(z) for z in roots])
    real, cplx = [], []
    for z in roots:
        if abs(z.imag) <= 1e-10 * scale:
            real.append(z.real)
        else:
            cplx.append(z)
    real.sort()
    cplx.sort(key=lambda z: (round(z.real, 10), z.imag))
    out = [_float_digits(r, n) for r in real]
    for z in cplx:
        re_part = _float_digits(z.real, n) if abs(z.real) > 1e-14 * scale else Integer(0)
        out.append(re_part + _float_digits(z.imag, n) * I)
    return out


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


from .functions.elementary.complexes import conjugate, im, re


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


from .functions.elementary.miscellaneous import Max, Min
from .functions.elementary.piecewise import ExprCondPair, Piecewise, piecewise_fold
from .functions.elementary.trigonometric import atan2
from .functions.elementary.exponential import LambertW, ln
from .core.mod import Mod


def numer(expr: Any) -> Any:
    """Numerator of ``expr`` (upstream ``numer``)."""
    return sympify(expr).as_numer_denom()[0]


def denom(expr: Any) -> Any:
    """Denominator of ``expr`` (upstream ``denom``)."""
    return sympify(expr).as_numer_denom()[1]


def expand_mul(expr: Any, deep: bool = True) -> Any:
    """Distribute products over sums only (upstream ``expand_mul``)."""
    return expand(expr, power_exp=False, log=False, power_base=False)


def expand_multinomial(expr: Any, deep: bool = True) -> Any:
    """Expand integer powers of sums (upstream ``expand_multinomial``)."""
    return expand(expr, power_exp=False, log=False, power_base=False)


def expand_func(expr: Any, deep: bool = True) -> Any:
    """Expand special functions into elementary ones where possible."""
    return expand(expr, func=True, power_exp=False, log=False)

from .core.lambdify_subs import Lambda, Subs
import sys as _sys
for _obj in (Lambda, Subs):
    _obj.__module__ = "sympy.core.function"
    setattr(_sys.modules["sympy.core.function"], _obj.__name__, _obj)


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


from .core import Relational

_INEQUALITY_OPS = ("<", "<=", ">", ">=")


def _is_inequality(e: Any) -> bool:
    return isinstance(e, Relational) and e.rel_op in _INEQUALITY_OPS


def _solve_inequality_set(rel: Any, symbol: Any) -> Any:
    """Real solution set of a univariate inequality (native, certified)."""
    from .sets.sets import _wrap_set

    f = _wrap(_native_expr(rel.lhs - rel.rhs))
    try:
        native = _native.solve_inequality_expr(str(_native_expr(f)), _native_symbol_key(symbol), rel.rel_op)
    except ValueError as exc:
        raise NotImplementedError(
            f"The inequality, {rel}, cannot be solved by the exact real inequality solver: {exc}"
        ) from None
    return _wrap_set(native)


def _set_as_relational(s: Any, x: Any) -> Any:
    """Upstream ``Set.as_relational`` for real-line sets."""
    from .logic.boolalg import And, Or, false
    from .sets import EmptySet, FiniteSet, Interval, Union

    if s is EmptySet() or getattr(s, "is_empty", None) is True:
        return false
    if isinstance(s, FiniteSet):
        return Or(*[Eq(x, e) for e in s.args])
    if isinstance(s, Interval):
        lo = Lt(s.start, x, evaluate=False) if s.left_open else Le(s.start, x, evaluate=False)
        hi = Lt(x, s.end, evaluate=False) if s.right_open else Le(x, s.end, evaluate=False)
        return And(lo, hi)
    if isinstance(s, Union):
        return Or(*[_set_as_relational(a, x) for a in s.args])
    raise NotImplementedError(f"as_relational for {type(s).__name__}")


def _inequality_symbol(rels: list, symbols: Any) -> Any:
    if symbols:
        syms = list(symbols[0]) if isinstance(symbols[0], (list, tuple, set)) else list(symbols)
        if len(syms) != 1:
            raise NotImplementedError("only univariate inequalities are supported")
        return syms[0]
    free: set = set()
    for r in rels:
        free |= r.free_symbols
    if len(free) != 1:
        raise NotImplementedError(
            "inequality systems need exactly one free variable, got %s"
            % sorted(str(v) for v in free)
        )
    return next(iter(free))


def reduce_inequalities(inequalities: Any, symbols: Any = None) -> Any:
    """Reduce a univariate system of inequalities to a relational form.

    The real solution set of each inequality comes from the native exact
    inequality solver; their intersection is returned as relationals
    (upstream ``reduce_inequalities`` shape). Unsupported systems raise.
    """
    from .logic.boolalg import BooleanAtom
    from .sets import Intersection, Reals

    if not isinstance(inequalities, (list, tuple, set)):
        inequalities = [inequalities]
    rels = []
    for ineq in inequalities:
        ineq = sympify(ineq) if not isinstance(ineq, (Relational, BooleanAtom)) else ineq
        if isinstance(ineq, BooleanAtom):
            if not ineq:
                from .logic.boolalg import false
                return false
            continue
        if type(ineq) is Eq or (isinstance(ineq, Relational) and ineq.rel_op == "=="):
            rels.append(ineq)
            continue
        if not _is_inequality(ineq):
            raise NotImplementedError(f"reduce_inequalities cannot handle {ineq}")
        rels.append(ineq)
    if not rels:
        from .logic.boolalg import true
        return true
    x = _inequality_symbol(rels, (symbols,) if symbols is not None else ())
    sol = Reals()
    for r in rels:
        if _is_inequality(r):
            part = _solve_inequality_set(r, x)
        else:
            part = solveset(r.lhs - r.rhs, x, Reals())
        sol = Intersection(sol, part)
    return _set_as_relational(sol, x)


__all__ = [
    "imageset",
    "Rationals",
    "Range",
    "Naturals0",
    "Naturals",
    "Integers",
    "ImageSet",
    "ConditionSet",
    "Complexes",
    "invert",
    "interpolate",
    "LT",
    "LM",
    "GroebnerBasis",
    "DiracDelta",
    "Heaviside",
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
    "PurePoly",
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
    "gammasimp",
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
    "expand",
    "expand_log",
    "expand_power_base",
    "expand_power_exp",
    "expand_trig",
    "eye",
    "factor",
    "fraction",
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
    "latex",
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
    "pprint",
    "pprint_use_unicode",
    "pretty",
    "pretty_print",
    "prevprime",
    "prime",
    "product",
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
    "terms_gcd",
    "srepr",
    "sstr",
    "sstr",
    "sqrt",
    "summation",
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
    "ExprCondPair",
    "Lambda",
    "LambertW",
    "Mod",
    "Piecewise",
    "Subs",
    "atan2",
    "ln",
    "piecewise_fold",
    "numer",
    "denom",
    "expand_mul",
    "expand_multinomial",
    "expand_func",
]

from .core.match import Wild, WildFunction, bottom_up, postorder_traversal, preorder_traversal  # noqa: E402

__all__ += ["Wild", "WildFunction", "bottom_up", "postorder_traversal", "preorder_traversal"]

from .functions.special.polynomials import (  # noqa: E402
    assoc_laguerre, assoc_legendre, chebyshevt, chebyshevt_poly, chebyshevt_root, chebyshevu,
    chebyshevu_poly, chebyshevu_root, gegenbauer, gegenbauer_poly, hermite, hermite_poly,
    hermite_prob, hermite_prob_poly, jacobi, jacobi_poly, laguerre, laguerre_poly, legendre,
    legendre_poly,
)
from .functions.special import polynomials as _orthopoly  # noqa: E402

__all__ += list(_orthopoly.__all__)

from .solvers.recurr import rsolve  # noqa: E402

__all__ += ["rsolve"]

from .simplify.cse_main import cse, numbered_symbols  # noqa: E402

__all__ += ["cse", "numbered_symbols"]

from .parsing.sympy_parser import parse_expr  # noqa: E402
from . import parsing  # noqa: E402,F401

__all__ += ["parse_expr"]

from .utilities.iterables import (  # noqa: E402
    default_sort_key, flatten, has_dups, has_variety, ordered, postfixes, prefixes, sift,
    subsets, topological_sort, unflatten, variations,
)
from . import utilities  # noqa: E402,F401


def var(names, **args):
    """Create symbols and inject them into the caller's global namespace."""
    import inspect

    frame = inspect.currentframe().f_back
    try:
        syms = symbols(names, **args)
        seq = syms if isinstance(syms, (list, tuple)) else (syms,)

        def inject(s):
            if isinstance(s, (list, tuple)):
                for t in s:
                    inject(t)
            else:
                frame.f_globals[s.name] = s

        inject(seq)
        return syms
    finally:
        del frame


__all__ += [
    "default_sort_key", "flatten", "has_dups", "has_variety", "ordered", "postfixes", "prefixes",
    "sift", "subsets", "topological_sort", "unflatten", "var", "variations",
]

from .printing.codeprinter import (  # noqa: E402
    ccode, fcode, jscode, julia_code, mathematica_code, octave_code, print_ccode, print_fcode,
    pycode, rust_code,
)

__all__ += [
    "ccode", "fcode", "jscode", "julia_code", "mathematica_code", "octave_code", "print_ccode",
    "print_fcode", "pycode", "rust_code",
]

from .solvers.diophantine import diophantine  # noqa: E402

__all__ += ["diophantine"]
