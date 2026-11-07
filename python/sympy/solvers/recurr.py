"""Linear recurrence relations (upstream ``rsolve``).

Supported, exactly:

* linear recurrences with constant coefficients of order at most 32: the
  characteristic polynomial's roots ``r`` (multiplicity ``m``) give the
  basis ``n**j * r**(n - j)``; constants are ``C0, C1, ...``;
* inhomogeneous terms that are sums of ``poly(n) * b**n``, through
  undetermined coefficients (raised by ``n**m`` when ``b`` is an
  ``m``-fold characteristic root);
* first-order recurrences ``y(n + 1) = p(n)*y(n)`` with ``n``-dependent
  ``p``, when the bounded product evaluator supplies a checkable closed form;
* first-order factorial-forced recurrences with ``p(n) = a*(n+c)``, exact
  rational ``a != 0`` and positive integer ``c``: variation of constants
  sums polynomial-geometric multiples of the nonzero factorial product;
* initial conditions ``{y(k): value}``, solved for the constants.

Shifts are exact integer literals of magnitude at most 1024. Polynomial
forcing has degree at most 64. Normalized factorial forcing has degree at
most 32 and at most 64 expanded terms; at most 64 exact ICs are admitted.
Unsupported parameter branches and original source domains are refused.
Every returned expression is checked against the original, unshifted
recurrence and every initial condition. These checks establish an identity
on their common defined integer lattice domain, not a completeness or
singular-point extension certificate.
"""

from __future__ import annotations

from typing import Any


_MAX_SHIFT = 1024
_MAX_ORDER = 32
_MAX_FORCING_DEGREE = 64
_MAX_INITIAL_CONDITIONS = 64
_MAX_EXPANDED_TERMS = 4096
_MAX_EXPRESSION_NODES = 8192
_MAX_EXACT_BITS = 4096
_MAX_PRODUCT_FORCING_TERMS = 64
_MAX_PRODUCT_FORCING_DEGREE = 32


def _expansion_envelope(expression: Any) -> None:
    """Preflight nested distribution without constructing expanded monomials."""
    from ..core import Add, Integer, Mul, Pow
    from .ode import _checkodesol_children

    pending = [(expression, 0, None)]
    terms = {}
    visited = 0
    while pending:
        node, depth, children = pending.pop()
        if children is None:
            visited += 1
            if visited > _MAX_EXPRESSION_NODES or depth > 128:
                raise NotImplementedError("rsolve: expression exceeds the bounded expansion envelope")
            children = _checkodesol_children(node)
            pending.append((node, depth, children))
            pending.extend((child, depth+1, None) for child in reversed(children))
            continue
        counts = [terms[id(child)] for child in children]
        count = 1
        if type(node) is Add:
            count = sum(counts)
        elif type(node) is Mul:
            for value in counts:
                count *= value
                if count > _MAX_EXPANDED_TERMS:
                    break
        elif type(node) is Pow and isinstance(children[1], Integer):
            power = children[1].p
            if power > 0 and counts[0] > 1:
                # A t-term commutative polynomial raised to p has at most
                # binomial(t+p-1, p) monomials. Saturate before large integers
                # or expansion work, retaining useful degree-64 binomials.
                total = counts[0]+power-1
                choose = min(power, counts[0]-1)
                for index in range(1, choose+1):
                    count = count*(total-choose+index)//index
                    if count > _MAX_EXPANDED_TERMS:
                        break
        if count > _MAX_EXPANDED_TERMS:
            raise NotImplementedError("rsolve: predicted polynomial expansion exceeds 4096 terms")
        terms[id(node)] = count


def _expand(expression: Any) -> Any:
    from ..core import expand

    _expansion_envelope(expression)
    result = expand(expression)
    _expansion_envelope(result)
    return result


def _admit_source(expression: Any, func: Any) -> None:
    """Keep original domains and closed source owners before any rewriting."""
    from ..core import Derivative, Float, Function, Integer, Pow, Rational, S
    from ..functions import exp, factorial
    from .ode import _checkodesol_children, _checkodesol_surface

    if not _checkodesol_surface(expression):
        raise NotImplementedError("rsolve: original source domain cannot be preserved")
    pending = [expression]
    while pending:
        node = pending.pop()
        if node in (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity):
            raise NotImplementedError("rsolve: finite exact recurrence data is required")
        if isinstance(node, (Float, Derivative)):
            raise NotImplementedError("rsolve: exact recurrence data without derivatives is required")
        if isinstance(node, Function) and type(node) not in (func, exp, factorial):
            raise NotImplementedError("rsolve: unsupported function in the original source")
        if type(node) is factorial:
            argument = node.args[0]
            if isinstance(argument, Integer) and argument.p > _MAX_SHIFT:
                raise NotImplementedError("rsolve: factorial index exceeds the bounded envelope")
            if argument.is_integer is True and argument.is_negative is True:
                raise NotImplementedError("rsolve: an original factorial pole cannot be discarded")
        if isinstance(node, (Integer, Rational)):
            if max(abs(node.p).bit_length(), node.q.bit_length()) > _MAX_EXACT_BITS:
                raise NotImplementedError("rsolve: exact literal exceeds 4096 bits")
        if isinstance(node, Pow) and isinstance(node.args[1], Integer):
            if abs(node.args[1].p) > _MAX_SHIFT:
                raise NotImplementedError("rsolve: integer power exceeds the bounded envelope")
        pending.extend(_checkodesol_children(node))
    _expansion_envelope(expression)


def _admit_candidate(expression: Any, func: Any) -> None:
    from ..core.match import preorder_traversal

    _admit_source(expression, func)
    if any(type(node) is func for node in preorder_traversal(expression)):
        raise NotImplementedError("rsolve: an explicit candidate cannot contain the solved function")


def _integer_offset(value: Any) -> int:
    from ..core import Integer

    if not isinstance(value, Integer):
        raise NotImplementedError("rsolve: sequence arguments must be n + an exact integer")
    offset = value.p
    if abs(offset) > _MAX_SHIFT:
        raise NotImplementedError("rsolve: exact shift exceeds the bounded shift envelope")
    return offset


def _shifts(f: Any, func: Any, n: Any) -> dict:
    """{offset: y(n + offset) node} for every application of ``func``."""
    from ..core.match import preorder_traversal

    out: dict = {}
    for node in preorder_traversal(f):
        if type(node) is func:
            if len(node.args) != 1:
                raise NotImplementedError("rsolve: sequence applications must have one argument")
            k = _integer_offset(_expand(node.args[0] - n))
            out[k] = node
    return out


def _constants(count: int, sources: tuple) -> list:
    from ..core import Symbol

    used = {symbol.name for source in sources for symbol in source.free_symbols}
    constants = []
    index = 0
    while len(constants) < count:
        name = "C%d" % index
        index += 1
        if name not in used:
            constants.append(Symbol(name))
            used.add(name)
    return constants


def _char_roots(coeffs: list) -> list:
    """[(root, multiplicity)] of sum c_k r**k, real roots ascending first."""
    from ..core import Dummy, Integer
    from ..polys.polytools import Poly, roots

    r = Dummy("rsolve_characteristic")
    char = sum((c * r**k for k, c in enumerate(coeffs)), Integer(0))
    rts = roots(Poly(_expand(char), r))
    if sum(rts.values()) != len(coeffs) - 1:
        raise NotImplementedError("rsolve: characteristic roots are not all expressible")

    def key(item: Any) -> tuple:
        root = item[0]
        try:
            v = complex(root)
        except (TypeError, ValueError, OverflowError):
            return (2, 0.0, 0.0)
        if abs(v.imag) < 1e-12:
            return (0, v.real, 0.0)
        return (1, v.real, -v.imag)

    return sorted(rts.items(), key=key)


def _verify_characteristic(coeffs: list, proposals: Any, func: Any, n: Any) -> tuple:
    """Check the factorization and distinct basis roots independently.

    A list whose multiplicities merely sum to the order can repeat a root,
    omit an independent branch, or collapse on an allowed parameter value.
    Reconstructing coefficients checks the original polynomial without
    trusting the root generator's equations or numerical approximations.
    """
    from ..core import Integer
    from ..simplify import simplify

    order = len(coeffs)-1
    if type(proposals) not in (list, tuple) or len(proposals) > order:
        raise NotImplementedError("rsolve: invalid characteristic-root proposal")
    roots = []
    total = 0
    for item in tuple(proposals):
        if type(item) not in (list, tuple) or len(item) != 2:
            raise NotImplementedError("rsolve: invalid characteristic-root entry")
        root, multiplicity = item
        if type(multiplicity) is not int or not 1 <= multiplicity <= order:
            raise NotImplementedError("rsolve: invalid characteristic-root multiplicity")
        total += multiplicity
        if total > order:
            raise NotImplementedError("rsolve: characteristic multiplicities exceed the order")
        _admit_candidate(root, func)
        if root == 0:
            raise NotImplementedError("rsolve: singular characteristic-root proposal")
        for previous, _ in roots:
            difference = simplify(_expand(root-previous))
            if difference == 0:
                raise NotImplementedError("rsolve: duplicate characteristic-root proposal")
            if (difference.is_zero is not False and difference.is_nonzero is not True
                    and difference.is_real is not False):
                raise NotImplementedError("rsolve: characteristic-root separation is undecidable")
        roots.append((root, multiplicity))
    if total != order:
        raise NotImplementedError("rsolve: characteristic roots do not account for the order")

    # Convolve one linear factor at a time. The degree never exceeds 32;
    # expanding the product as a whole would predict 2**order transient terms.
    reconstructed = [Integer(1)]
    for root, multiplicity in roots:
        for _ in range(multiplicity):
            next_coefficients = [-root*reconstructed[0]]
            next_coefficients.extend(reconstructed[k-1]-root*reconstructed[k]
                                     for k in range(1, len(reconstructed)))
            next_coefficients.append(reconstructed[-1])
            reconstructed = [simplify(_expand(value)) for value in next_coefficients]
    if any(not _exact_zero(original-coeffs[-1]*reconstructed[k], n)
           for k, original in enumerate(coeffs)):
        raise NotImplementedError("rsolve: original characteristic-polynomial verification failed")
    return tuple(roots)


def _split_forcing(g: Any, n: Any) -> list:
    """g = sum p_b(n) * b**n  ->  [(b, p_b)]; NotImplementedError otherwise."""
    from ..core import Add, Integer, Mul, Pow, Rational
    from ..functions import exp

    groups: dict = {}
    for t in Add.make_args(_expand(g)):
        base = Integer(1)
        poly_part = []
        for fct in Mul.make_args(t):
            if isinstance(fct, Pow) and n in fct.args[1].free_symbols and n not in fct.args[0].free_symbols:
                rest = _expand(fct.args[1].subs(n, 0))
                k = _expand(fct.args[1].diff(n))
                if (n in k.free_symbols or n in rest.free_symbols
                        or _expand(fct.args[1] - k*n - rest) != 0):
                    raise NotImplementedError("rsolve: forcing term %s" % t)
                # The original residual checker admits integer-lattice power
                # shifts only. Bound both exact exponents before materializing
                # base**k or base**rest, even when their source was affine.
                _integer_offset(k)
                _integer_offset(rest)
                if isinstance(fct.args[0], Rational):
                    ratio = _rational_power(fct.args[0], k)
                    constant = _rational_power(fct.args[0], rest)
                else:
                    ratio = fct.args[0]**k
                    constant = fct.args[0]**rest
                base = (_rational_product(base, ratio)
                        if isinstance(base, Rational) and isinstance(ratio, Rational)
                        else base*ratio)
                poly_part.append(constant)
            elif type(fct) is exp and n in fct.args[0].free_symbols:
                rest = _expand(fct.args[0].subs(n, 0))
                k = _expand(fct.args[0].diff(n))
                if (n in k.free_symbols or n in rest.free_symbols
                        or _expand(fct.args[0] - k*n - rest) != 0):
                    raise NotImplementedError("rsolve: forcing term %s" % t)
                base = base * exp(k)
                poly_part.append(exp(rest))
            else:
                poly_part.append(fct)
        p = Mul(*poly_part)
        if not p.is_polynomial(n):
            raise NotImplementedError("rsolve: forcing term %s" % t)
        groups[base] = groups.get(base, Integer(0)) + p
    return list(groups.items())


def _particular(coeffs: list, g: Any, n: Any, roots: list) -> Any:
    from ..core import Dummy, Integer
    from ..polys.polytools import Poly, factor

    total = Integer(0)
    for b, p in _split_forcing(g, n):
        mult = 0
        for r, m in roots:
            if _expand(r - b) == 0:
                mult = m
        deg = Poly(p, n).degree() if p.free_symbols & {n} else 0
        if not 0 <= deg <= _MAX_FORCING_DEGREE:
            raise NotImplementedError("rsolve: forcing polynomial exceeds degree 64")
        unknowns = [Dummy("rsolve_coefficient") for _ in range(deg + 1)]
        trial_poly = sum((u * n ** (i + mult) for i, u in enumerate(unknowns)), Integer(0))
        lhs = Integer(0)
        for k, c in enumerate(coeffs):
            lhs = lhs + c * b**k * trial_poly.subs(n, n + k)
        resid = _expand(lhs - p)
        from .. import solve

        eqs = Poly(resid, n).all_coeffs() if resid.free_symbols & {n} else [resid]
        sol = solve(eqs, unknowns, dict=True)
        if not sol:
            raise NotImplementedError("rsolve: undetermined coefficients failed")
        sol = sol[0] if isinstance(sol, list) else sol
        for value in sol.values():
            _admit_candidate(value, None)
        part = trial_poly.subs(sol)
        part = factor(_expand(part)) if part.free_symbols else part
        total = total + (part * b**n if b != 1 else part)
    return total


def _linear_product(p: Any, n: Any, k: Any) -> Any:
    """prod(p(k), (k, 0, n - 1)) for p = a*n + b: a**n * rf(b/a, n),
    i.e. factorial(n + c - 1)/factorial(c - 1) for b/a = c a positive
    integer (upstream: (n + 1)*y(n) -> factorial(n))."""
    from ..core import Integer, Rational
    from ..functions import factorial
    from .. import product

    a = _expand(p.diff(n))
    b = _expand(p - a * n)
    if n not in a.free_symbols and isinstance(a, Rational) and a != 0 and isinstance(b, Rational):
        c = b / a
        if isinstance(c, Integer) and c >= 1:
            offset = _integer_offset(c - 1)
            body = factorial(n + offset) / factorial(offset)
            return body if a == 1 else a**n * body
    return product(p.subs(n, k), (k, 0, n - 1))


def _rational_power(base: Any, exponent: Any) -> Any:
    """Bound exact rational work before materializing integer powers."""
    from ..core import Rational

    if not isinstance(base, Rational) or base == 0:
        raise NotImplementedError("rsolve: geometric normalization requires nonzero rational bases")
    power = _integer_offset(exponent)
    bits = max(abs(base.p).bit_length(), base.q.bit_length())
    if bits*abs(power) > _MAX_EXACT_BITS:
        raise NotImplementedError("rsolve: geometric power exceeds the exact scalar envelope")
    return base**power


def _rational_product(left: Any, right: Any) -> Any:
    """A conservative numerator/denominator bound before multiplication."""
    if (abs(left.p).bit_length()+abs(right.p).bit_length() > _MAX_EXACT_BITS
            or left.q.bit_length()+right.q.bit_length() > _MAX_EXACT_BITS):
        raise NotImplementedError("rsolve: geometric product exceeds the exact scalar envelope")
    return left*right


def _first_order_forcing(p: Any, g: Any, leading: Any, n: Any, func: Any) -> tuple:
    """Generate H(n) and an additive sum for y(n) = H(n)*(C + sum).

    Each expanded forcing term must have one factorial(n+s), with bounded
    integer s, times a polynomial and a nonzero rational geometric factor.
    Factorial shifts are normalized by finite products before asking the
    existing summation engine for a closed form. The caller still verifies
    the resulting solution against the untouched original recurrence.
    """
    from .. import summation
    from ..core import Add, Dummy, Integer, Mul, Rational
    from ..core.match import preorder_traversal
    from ..functions import factorial
    from ..polys.polytools import Poly, cancel

    a = _expand(p.diff(n))
    b = _expand(p-a*n)
    if not isinstance(a, Rational) or a == 0 or not isinstance(b, Rational):
        raise NotImplementedError("rsolve: forced products require an exact rational linear multiplier")
    c = b/a
    if not isinstance(c, Integer) or c.p < 1:
        raise NotImplementedError("rsolve: forced factorial products require a positive integer offset")
    shift = _integer_offset(c)
    scale = factorial(c-1)
    homogeneous = a**n*factorial(n+c-1)/scale
    _admit_candidate(homogeneous, func)

    terms = Add.make_args(_expand(g))
    if len(terms) > _MAX_PRODUCT_FORCING_TERMS:
        raise NotImplementedError("rsolve: factorial forcing exceeds 64 expanded terms")
    groups = {}
    for term in terms:
        factorial_shift = None
        remaining = []
        for factor in Mul.make_args(term):
            if type(factor) is factorial and n in factor.args[0].free_symbols:
                if factorial_shift is not None:
                    raise NotImplementedError("rsolve: one sequence factorial is required per forcing term")
                slope, offset = _affine_argument(factor.args[0], n)
                if slope != 1:
                    raise NotImplementedError("rsolve: forcing factorials require unit index slopes")
                factorial_shift = _integer_offset(offset)
            else:
                if any(type(node) is factorial and n in node.args[0].free_symbols
                       for node in preorder_traversal(factor)):
                    raise NotImplementedError("rsolve: forcing factorials must be direct linear factors")
                remaining.append(factor)
        if factorial_shift is None:
            raise NotImplementedError("rsolve: forcing must contain a shifted sequence factorial")
        distance = abs(factorial_shift-shift)
        if distance > _MAX_FORCING_DEGREE:
            raise NotImplementedError("rsolve: forcing factorial shifts differ by more than 64")
        for base, weight in _split_forcing(Mul(*remaining), n):
            if not isinstance(base, Rational) or base == 0:
                raise NotImplementedError("rsolve: forced geometric ratios must be nonzero rational numbers")
            key = (factorial_shift, base)
            groups[key] = groups.get(key, Integer(0))+weight

    increment = Integer(0)
    for (factorial_shift, base), weight in groups.items():
        # Combine the weight before canceling a lower factorial shift. For
        # example, the expanded terms n*n! and n! jointly contain (n+1)!.
        weight = _expand(weight)
        product = Integer(1)
        for offset in range(min(factorial_shift, shift)+1, max(factorial_shift, shift)+1):
            # Expand incrementally: an order-d linear product has d+1 terms,
            # without an exponentially large transient distribution.
            product = _expand(product*(n+offset))
        if factorial_shift < shift:
            weight = cancel(weight/product, n)
        else:
            weight = _expand(weight*product)
        weight = _expand(weight*scale/leading/a)
        if weight == 0:
            continue
        if not weight.is_polynomial(n):
            raise NotImplementedError("rsolve: normalized factorial forcing must be polynomial-geometric")
        degree = Poly(weight, n).degree() if n in weight.free_symbols else 0
        if not 0 <= degree <= _MAX_PRODUCT_FORCING_DEGREE:
            raise NotImplementedError("rsolve: normalized factorial forcing exceeds degree 32")
        ratio = _rational_product(base, _rational_power(a, Integer(-1)))
        increment += weight if ratio == 1 else weight*ratio**n
    _admit_candidate(increment, func)
    index = Dummy("rsolve_forcing_index", integer=True)
    correction = summation(increment.subs(n, index), (index, 0, n-1))
    _admit_candidate(correction, func)
    return homogeneous, correction


def _initial_conditions(init: Any, func: Any, n: Any) -> tuple:
    """Capture every condition without conflating another function or index."""
    from itertools import islice
    from ..core import Integer, S, sympify

    if init is None:
        return ()
    if type(init) is not dict:
        raise TypeError("rsolve: initial conditions must be an exact dictionary")
    if len(init) > _MAX_INITIAL_CONDITIONS:
        raise NotImplementedError("rsolve: at most 64 initial conditions are supported")
    count = len(init)
    items = tuple(islice(dict.items(init), _MAX_INITIAL_CONDITIONS+1))
    if len(items) != count or len(init) != count:
        raise ValueError("rsolve: initial conditions changed during their bounded snapshot")
    conditions = []
    exact_integers = (Integer, type(S.Zero), type(S.One), type(S.NegativeOne))
    for key, value in items:
        if type(key) is func:
            if len(key.args) != 1:
                raise ValueError("rsolve: an initial application must have one integer index")
            index = key.args[0]
        elif type(key) is int or type(key) in exact_integers:
            index = sympify(key)
        else:
            raise ValueError("rsolve: initial keys must be integers or applications of the solved function")
        if type(index) not in exact_integers:
            raise ValueError("rsolve: initial indices must be exact integer literals")
        if abs(index.p) > _MAX_SHIFT:
            raise NotImplementedError("rsolve: initial index exceeds the bounded shift envelope")
        value = sympify(value)
        _admit_candidate(value, func)
        if n in value.free_symbols:
            raise ValueError("rsolve: initial values must be independent of the sequence index")
        conditions.append((index, value))
    return tuple(conditions)


def _initial_value(solution: Any, n: Any, index: Any, func: Any) -> Any:
    """Keep factorial domains before substitution can erase a zero factor."""
    from ..core import Integer
    from ..core.match import preorder_traversal
    from ..functions import factorial

    for node in preorder_traversal(solution):
        if type(node) is factorial:
            argument = node.args[0].subs(n, index)
            _admit_candidate(argument, func)
            if argument.is_integer is True and argument.is_negative is True:
                raise NotImplementedError("rsolve: an initial index reaches an original factorial pole")
            if isinstance(argument, Integer) and argument.p > _MAX_SHIFT:
                raise NotImplementedError("rsolve: an initial factorial exceeds the bounded envelope")
    value = solution.subs(n, index)
    _admit_candidate(value, func)
    return value


def _affine_argument(expression: Any, n: Any) -> tuple:
    offset = _expand(expression.subs(n, 0))
    slope = _expand(expression.diff(n))
    if n in slope.free_symbols or _expand(expression - slope*n - offset) != 0:
        raise NotImplementedError("rsolve: residual verification requires affine sequence exponents")
    return slope, offset


def _rational_geometric_form(expression: Any, n: Any) -> Any:
    """Combine rational powers only on the admitted integer index lattice.

    Unlike an unrestricted power rewrite, every base here is an exact,
    proved nonzero rational and every exponent has bounded integer slope
    and offset. Negative bases are therefore safe as well. This recognizes
    a**n*(b/a)**n = b**n without making assumptions about symbolic bases.
    """
    from ..core import Add, Integer, Mul, Pow, Rational

    terms = []
    for term in Add.make_args(expression):
        ratio = Integer(1)
        coefficient = Integer(1)
        factors = []
        for factor in Mul.make_args(term):
            if (type(factor) is Pow and isinstance(factor.args[0], Rational)
                    and n in factor.args[1].free_symbols):
                base, exponent = factor.args
                slope, offset = _affine_argument(exponent, n)
                ratio = _rational_product(ratio, _rational_power(base, slope))
                coefficient = _rational_product(coefficient, _rational_power(base, offset))
            else:
                factors.append(factor)
        if ratio != 1:
            factors.append(ratio**n)
        terms.append(Mul(coefficient, *factors))
    result = Add(*terms)
    _expansion_envelope(result)
    return result


def _exact_zero(expression: Any, n: Any) -> bool:
    """Independent integer-shift normalization, followed by exact algebra.

    The input is the substituted original residual, never the characteristic
    coefficients or the undetermined-coefficient equations. Integer shifts
    of one power or factorial are rewritten to a common formal atom. Choosing
    the smallest shift uses only nonnegative powers/products, with no newly
    assumed nonzero denominator. No floating evaluation is an acceptance test.
    """
    from ..core import Dummy, E, Integer, Pow
    from ..core.match import preorder_traversal
    from ..functions import exp, factorial
    from ..simplify import simplify

    expression = _expand(expression)
    if expression == 0:
        return True
    expression = _rational_geometric_form(expression, n)
    if expression == 0:
        return True
    powers = {}
    factorials = []
    exponentials = {}
    for node in preorder_traversal(expression):
        if isinstance(node, Pow) and n in node.args[1].free_symbols:
            base, exponent = node.args
            if n in base.free_symbols:
                raise NotImplementedError("rsolve: residual verification does not admit dynamic bases")
            slope, offset = _affine_argument(exponent, n)
            if not isinstance(slope, Integer):
                raise NotImplementedError("rsolve: residual verification requires integer power slopes")
            shift = _integer_offset(offset)
            if base == E or type(base) is exp:
                logarithm = Integer(1) if base == E else base.args[0]
                exponentials.setdefault(_expand(logarithm*slope), []).append(
                    (node, _expand(logarithm*shift)),
                )
            else:
                powers.setdefault((base, slope), []).append((node, shift))
        elif type(node) is factorial and n in node.args[0].free_symbols:
            slope, offset = _affine_argument(node.args[0], n)
            if slope != 1:
                raise NotImplementedError("rsolve: residual verification requires unit factorial slopes")
            factorials.append((node, _integer_offset(offset)))
        elif type(node) is exp and n in node.args[0].free_symbols:
            slope, offset = _affine_argument(node.args[0], n)
            exponentials.setdefault(slope, []).append((node, offset))
    replacements = {}
    for (base, _), nodes in powers.items():
        minimum = min(shift for _, shift in nodes)
        atom = Dummy("recurrence_power")
        for node, shift in nodes:
            replacements[node] = base**(shift-minimum)*atom
    if factorials:
        minimum = min(shift for _, shift in factorials)
        atom = Dummy("recurrence_factorial")
        for node, shift in factorials:
            value = atom
            for offset in range(minimum+1, shift+1):
                value *= n+offset
            replacements[node] = value
    for _, nodes in exponentials.items():
        atom = Dummy("recurrence_exponential")
        for node, offset in nodes:
            replacements[node] = exp(offset)*atom
    if replacements:
        expression = expression.xreplace(replacements)
    return simplify(_expand(expression)) == 0


def _verify_solution(original: Any, func: Any, n: Any, solution: Any,
                     conditions: tuple, constants: list) -> None:
    from ..core import Eq
    from ..core.match import preorder_traversal

    _admit_candidate(solution, func)
    allowed = set(original.free_symbols) | {n} | set(constants)
    for _, value in conditions:
        allowed.update(value.free_symbols)
    if solution.free_symbols - allowed:
        raise NotImplementedError("rsolve: candidate contains uncaptured internal symbols")
    replacements = {}
    for node in preorder_traversal(original):
        if type(node) is func:
            if len(node.args) != 1:
                raise NotImplementedError("rsolve: invalid original sequence application")
            _integer_offset(_expand(node.args[0]-n))
            replacements[node] = solution.subs(n, node.args[0])
    residual = (original.lhs.xreplace(replacements) - original.rhs.xreplace(replacements)
                if type(original) is Eq else original.xreplace(replacements))
    if not _exact_zero(residual, n):
        raise NotImplementedError("rsolve: original recurrence verification failed")
    for index, value in conditions:
        if not _exact_zero(_initial_value(solution, n, index, func)-value, n):
            raise NotImplementedError("rsolve: original initial-condition verification failed")


def rsolve(f: Any, y: Any, init: Any = None) -> Any:
    """Solve a bounded exact recurrence, checking its original residual and ICs.

    The index denotes an integer lattice even when its Symbol has no declared
    assumptions. Unsupported source domains, parameter-dependent order and
    inconclusive verification raise NotImplementedError, not a no-solution
    assertion. Initial keys must belong to the solved function or be exact
    integers; inconsistent exact conditions raise ValueError.
    """
    from ..core import Dummy, Eq, Integer, sympify
    from .ode import _checkodesol_function, _plain_ode_function_class

    if not _plain_ode_function_class(type(y)):
        raise ValueError("rsolve: y must be an ordinary undefined function of one symbol")
    _checkodesol_function(y)
    func, n = y.func, y.args[0]
    if n.is_integer is False:
        raise ValueError("rsolve: the sequence index cannot be declared noninteger")
    original = sympify(f)
    _admit_source(original, func)
    conditions = _initial_conditions(init, func, n)
    # Keep the original Eq and held source intact for independent replay.
    f = original.lhs-original.rhs if type(original) is Eq else original
    shifts = _shifts(f, func, n)
    if not shifts:
        raise ValueError("rsolve: %s does not appear in the recurrence" % y)
    lo = min(shifts)
    if lo:
        f = f.subs(n, n - lo)
        shifts = _shifts(f, func, n)
    order = max(shifts)
    if order > _MAX_ORDER:
        raise NotImplementedError("rsolve: recurrence order exceeds 32")
    ys = {k: Dummy("rsolve_value") for k in shifts}
    E = f
    for k in sorted(shifts, reverse=True):
        E = E.subs(shifts[k], ys[k])
    E = _expand(E)
    coeffs = []
    for k in range(order + 1):
        c = _expand(E.diff(ys[k])) if k in ys else Integer(0)
        if c.free_symbols & set(ys.values()):
            raise NotImplementedError("rsolve: nonlinear recurrence")
        coeffs.append(c)
    g = -_expand(E - sum((coeffs[k] * ys[k] for k in ys), Integer(0)))
    if g.free_symbols & set(ys.values()):
        raise NotImplementedError("rsolve: nonlinear recurrence")

    sources = (original, n, *(value for _, value in conditions))
    if all(n not in c.free_symbols for c in coeffs):
        for coefficient in (coeffs[0], coeffs[-1]):
            if coefficient == 0 or (coefficient.free_symbols
                                    and coefficient.is_nonzero is not True):
                raise NotImplementedError("rsolve: singular or undecidable parameter-dependent order")
        rts = _verify_characteristic(coeffs, _char_roots(coeffs), func, n)
        consts = _constants(order, sources)
        sol = Integer(0)
        idx = 0
        for r, m in rts:
            for j in range(m):
                sol = sol + consts[idx] * n**j * r ** (n - j)
                idx += 1
        if g != 0:
            part = _particular(coeffs, g, n, rts)
            _admit_candidate(part, func)
            sol = sol + part
    elif order == 1:
        if n in coeffs[1].free_symbols or coeffs[1].is_nonzero is not True:
            raise NotImplementedError("rsolve: singular or undecidable leading coefficient")
        k = Dummy("rsolve_product_index", integer=True)
        p = _expand(-coeffs[0] / coeffs[1])
        consts = _constants(1, sources)
        if g == 0:
            homogeneous = _linear_product(p, n, k)
            correction = Integer(0)
        else:
            homogeneous, correction = _first_order_forcing(p, g, coeffs[1], n, func)
        _admit_candidate(homogeneous, func)
        if homogeneous == 0:
            raise NotImplementedError("rsolve: singular homogeneous product")
        sol = homogeneous * (consts[0] + correction)
    else:
        raise NotImplementedError("rsolve: variable-coefficient recurrence of this form")

    if conditions:
        from .. import solve

        eqs = []
        for index, value in conditions:
            equation = _expand(_initial_value(sol, n, index, func)-value)
            if not _exact_zero(equation, n):
                eqs.append(equation)
        present = [c for c in consts if c in sol.free_symbols]
        if eqs:
            if not present:
                raise ValueError("rsolve: inconsistent initial conditions")
            found = solve(eqs, present, dict=True)
            if not found:
                raise ValueError("rsolve: inconsistent initial conditions")
            choice = found[0] if isinstance(found, list) else found
            from ..simplify import radsimp

            for value in choice.values():
                _admit_candidate(value, func)
            sol = sol.subs({c: radsimp(v) for c, v in choice.items()})
        sol = _expand(sol)
    _verify_solution(original, func, n, sol, conditions, consts)
    return sol


__all__ = ["rsolve"]
