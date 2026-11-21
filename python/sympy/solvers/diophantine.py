"""Integer solutions of Diophantine equations (upstream ``diophantine``).

Supported, exactly, with upstream's parametrizations:

* linear equations in a bounded number of variables (``base_solution_linear``
  and upstream's coefficient-gcd chain; parameters ``t_0, t_1, ...``);
  retained inactive coordinates have independent integer parameters, with
  complete integer-lattice coverage checked by the native Smith solver;
* shifted bilinear equations: every congruence-admitted divisor pair
  within the work bounds, or complete affine integer families;
* homogeneous binary quadratics that factor over QQ into lines;
* definite binary quadratics within explicit integer-work bounds: every
  lattice point, including arbitrarily large translations within the
  coefficient-height limit;
* the Pythagorean equation ``x**2 + y**2 = z**2``: ``(2*p*q, p**2 - q**2,
  p**2 + q**2)``.

Other classes (Pell-type, general ternary) raise NotImplementedError.
"""

from __future__ import annotations

from math import gcd, isqrt
from typing import Any

from .. import core as _core
from ..core._function_capabilities import _implementation
from ..matrices import Matrix as _IntegerMatrix
from ..matrices import normalforms as _integer_forms
from ..matrices._integer_solve import solve_integer as _checked_integer_solve


_INTEGER_SOLVER_IMPLEMENTATION = _implementation(_checked_integer_solve)
_NATIVE_EXPR = _core._native.Expr
_NATIVE_MATRIX = _integer_forms._NATIVE[_IntegerMatrix]
_NATIVE_FLAT = _integer_forms._DICT.__get__(_NATIVE_MATRIX)["flat"]
_NATIVE_NUMERATOR = _integer_forms._DICT.__get__(_NATIVE_EXPR)["exact_numerator"]
_NATIVE_IS_INTEGER = _integer_forms._DICT.__get__(_NATIVE_EXPR)["is_integer"]
_INTEGER_CONVERSIONS = tuple(
    (owner, _integer_forms._reduction_binding(owner, "__int__"),
     _implementation(_integer_forms._reduction_binding(owner, "__int__")))
    for owner in (_core.Integer, _core.Zero, _core.One, _core.NegativeOne)
)
_SOURCE_SLOTS = tuple(
    (owner, name, _integer_forms._DICT.__get__(owner)[name])
    for owner, names in (
        (_core.Basic, ("_value", "_struct_args", "_surface_args")),
        (_core.Add, ("_args",)), (_core.Mul, ("_args",)),
        (_core.Pow, ("_args",)), (_core.Function, ("_args",)),
        (_core.Relational, ("_relational_sides",)),
        (_core.Symbol, ("_assumptions",)),
        (_core.Dummy, ("_dummy_name", "_dummy_number")),
    )
    for name in names
)
_SOURCE_ASSUMPTIONS = _integer_forms._DICT.__get__(_core.Symbol)["_assumptions"]
_SOURCE_VALUE = _integer_forms._DICT.__get__(_core.Basic)["_value"]
_MISSING_SOURCE_SLOT = object()

_MAX_DIOPHANTINE_COEFFICIENT_BITS = 4096
_MAX_DIOPHANTINE_SCAN_POINTS = 100_000
_MAX_DIOPHANTINE_SCAN_BIT_WORK = 20_000_000


def _check_quadratic_coefficients(coefficients: tuple[int, ...]) -> None:
    if any(type(value) is not int for value in coefficients):
        raise NotImplementedError("diophantine: exact integer coefficients required")
    if any(abs(value).bit_length() > _MAX_DIOPHANTINE_COEFFICIENT_BITS
           for value in coefficients):
        raise NotImplementedError("diophantine: quadratic coefficient height exceeds the bounded exact lane")


def _definite_axis_bounds(a: int, b: int, c: int, d: int, e: int, f: int):
    """Exact integer projection of a definite conic onto its first axis.

    A real second coordinate requires
    ``(b*x + e)**2 - 4*c*(a*x**2 + d*x + f) >= 0``. Completing the
    square gives ``(2*q*x - B)**2 <= D`` for positive ``q``. Since the
    squared quantity is integral, ``isqrt(D)`` gives exact integer bounds
    even when the two real endpoints are irrational or have huge offsets.
    """
    q = 4*a*c - b*b
    linear = 2*b*e - 4*c*d
    constant = e*e - 4*c*f
    discriminant = linear*linear + 4*q*constant
    if discriminant < 0:
        return None
    radius = isqrt(discriminant)
    denominator = 2*q
    lower = -((radius - linear) // denominator)
    upper = (linear + radius) // denominator
    return None if lower > upper else (lower, upper)


def _definite_binary_integer_points(
    a: int, b: int, c: int, d: int, e: int, f: int,
) -> set[tuple[int, int]]:
    """All integer points on one bounded definite binary quadratic.

    This shell-only exact lane makes no portable certificate claim. It
    enumerates the shorter exact coordinate projection, proves integral
    second coordinates with integer square roots and divisibility, and
    checks each candidate in the original coefficient equation. All scan
    and arithmetic-height admission happens before the enumeration; a
    refused request cannot return a partial set.
    """
    coefficients = (a, b, c, d, e, f)
    _check_quadratic_coefficients(coefficients)
    if b*b - 4*a*c >= 0:
        raise NotImplementedError("diophantine: the finite quadratic lane requires a definite form")
    x_bounds = _definite_axis_bounds(a, b, c, d, e, f)
    y_bounds = _definite_axis_bounds(c, b, a, e, d, f)
    if x_bounds is None or y_bounds is None:
        return set()
    x_width = x_bounds[1] - x_bounds[0] + 1
    y_width = y_bounds[1] - y_bounds[0] + 1
    swapped = y_width < x_width
    lower, upper = y_bounds if swapped else x_bounds
    width = upper - lower + 1
    coordinate_bits = max(abs(bound).bit_length() for bound in (*x_bounds, *y_bounds))
    coefficient_bits = max(abs(value).bit_length() for value in coefficients)
    # Bound both the number of rows and their integer arithmetic heights.
    # Quadratic residuals have at most coefficient_bits + 2*coordinate_bits
    # magnitude bits, plus a small sum/sign allowance.
    bit_work = width * (coefficient_bits + 2*coordinate_bits + 8)
    if (width > _MAX_DIOPHANTINE_SCAN_POINTS
            or bit_work > _MAX_DIOPHANTINE_SCAN_BIT_WORK):
        raise NotImplementedError("diophantine: exact quadratic enumeration exceeds the bounded work limit")
    if swapped:
        a, c, d, e = c, a, e, d

    out = set()
    # Use exact finite differences of the discriminant instead of
    # reconstructing its large translated quadratic at every scan point.
    quadratic = b*b - 4*a*c
    linear = 2*b*e - 4*c*d
    constant = e*e - 4*c*f
    discriminant = (quadratic*lower + linear)*lower + constant
    delta = quadratic*(2*lower + 1) + linear
    second_linear = b*lower + e
    denominator = 2*c
    for first in range(lower, upper + 1):
        if discriminant >= 0:
            root = isqrt(discriminant)
            if root*root == discriminant:
                numerators = (-second_linear + root,) if root == 0 else (
                    -second_linear - root, -second_linear + root,
                )
                for numerator in numerators:
                    if numerator % denominator == 0:
                        second = numerator // denominator
                        x, y = (second, first) if swapped else (first, second)
                        aa, bb, cc, dd, ee, ff = coefficients
                        if x*(aa*x + bb*y + dd) + y*(cc*y + ee) + ff != 0:
                            raise ArithmeticError("diophantine: exact quadratic residual check failed")
                        out.add((x, y))
        discriminant += delta
        delta += 2*quadratic
        second_linear += b
    return out


def _igcdex(a: int, b: int) -> tuple:
    """(x, y, g) with x*a + y*b = g = gcd(a, b) (upstream igcdex)."""
    if not a and not b:
        return (0, 1, 0)
    if not a:
        return (0, b // abs(b), abs(b))
    if not b:
        return (a // abs(a), 0, abs(a))
    x_sign = -1 if a < 0 else 1
    y_sign = -1 if b < 0 else 1
    a, b = abs(a), abs(b)
    x, y, r, s = 1, 0, 0, 1
    while b:
        c, q = a % b, a // b
        a, b, r, s, x, y = b, c, x - q * r, y - q * s, r, s
    return (x * x_sign, y * y_sign, a)


def base_solution_linear(c: int, a: int, b: int, t: Any = None) -> tuple:
    g = gcd(gcd(a, b), c) if c else gcd(a, b)
    if g > 1:
        a, b, c = a // g, b // g, c // g
    if c == 0:
        if t is None:
            return (0, 0)
        if b < 0:
            t = -t
        return (b * t, -a * t)
    x0, y0, d = _igcdex(abs(a), abs(b))
    x0 *= -1 if a < 0 else 1
    y0 *= -1 if b < 0 else 1
    if c % d == 0:
        if t is not None:
            if b < 0:
                t = -t
            return (c * x0 + b * t, c * y0 - a * t)
        return (c * x0, c * y0)
    return (None, None)


def _params(n: int) -> list:
    from ..core import Symbol

    return [Symbol("t_%d" % i, integer=True) for i in range(n)]


def _diop_linear(coeff: dict, var: list, const: int, parameters=None) -> set:
    from ..core import Add, Integer, Mul

    A = [int(coeff[v]) for v in var]
    g = 0
    for a in A:
        g = gcd(g, a)
    if const % g:
        return set()
    A = [a // g for a in A]
    c = Integer(-const // g)
    params = _params(len(var)) if parameters is None else parameters
    if len(var) == 1:
        if c % A[0]:
            return set()
        return {(Integer(int(c) // A[0]),)}
    B = []
    if len(var) > 2:
        B.append(gcd(A[-2], A[-1]))
        A[-2] //= B[0]
        A[-1] //= B[0]
        for i in range(len(A) - 3, 0, -1):
            gg = gcd(B[0], A[i])
            B[0] //= gg
            A[i] //= gg
            B.insert(0, gg)
    B.append(A[-1])
    solutions = []
    for Ai, Bi in zip(A, B):
        tot_x, tot_y = [], []
        for arg in Add.make_args(c):
            if isinstance(arg, Integer):
                k, p = int(arg), Integer(1)
                pnew = params[0]
            else:
                kk, p = arg.as_coeff_Mul()
                k = int(kk)
                pnew = params[params.index(p) + 1]
            sol_x, sol_y = base_solution_linear(k, Ai, Bi, pnew)
            if p == 1:
                if sol_x is None:
                    return set()
            else:
                sol_x = _scale_param(sol_x, p)
                sol_y = _scale_param(sol_y, p)
            tot_x.append(sol_x)
            tot_y.append(sol_y)
        solutions.append(Add(*tot_x))
        c = Add(*tot_y)
    solutions.append(c)
    return {tuple(solutions)}


def _integer_solver_interfaces():
    current = _implementation(_checked_integer_solve)
    if (len(current) != len(_INTEGER_SOLVER_IMPLEMENTATION)
            or any(a is not b for a, b in zip(current, _INTEGER_SOLVER_IMPLEMENTATION))):
        raise NotImplementedError("diophantine: integer solver implementation changed")
    _integer_forms._interfaces()
    if (_integer_forms._DICT.__get__(_NATIVE_MATRIX).get("flat") is not _NATIVE_FLAT
            or _integer_forms._DICT.__get__(_NATIVE_EXPR).get("exact_numerator")
            is not _NATIVE_NUMERATOR):
        raise NotImplementedError("diophantine: integer result readers changed")
    _integer_conversion_interfaces()


def _integer_conversion_interfaces():
    for owner, original, recorded in _INTEGER_CONVERSIONS:
        current = _integer_forms._reduction_binding(owner, "__int__")
        tokens = _implementation(current)
        if (current is not original or len(tokens) != len(recorded)
                or any(a is not b for a, b in zip(tokens, recorded))):
            raise NotImplementedError("diophantine: integer presentation conversion changed")


def _source_slot(slot, node):
    try:
        return slot.__get__(node, type(node))
    except AttributeError:
        return _MISSING_SOURCE_SLOT


def _source_storage_interfaces():
    if any(_integer_forms._DICT.__get__(owner).get(name) is not slot
           for owner, name, slot in _SOURCE_SLOTS):
        raise NotImplementedError("diophantine: original source storage descriptors changed")
    namespace = _integer_forms._DICT.__get__(_NATIVE_EXPR)
    if (namespace.get("is_integer") is not _NATIVE_IS_INTEGER
            or namespace.get("exact_numerator") is not _NATIVE_NUMERATOR):
        raise NotImplementedError("diophantine: literal integer readers changed")


def _linear_source_snapshot(records, work):
    """Retain original storage before normalization can erase source axes."""
    from ..assumptions._query import typed_symbol

    _source_storage_interfaces()
    _integer_conversion_interfaces()
    snapshot = []
    for node, children in records:
        if type(node) in (_core.Integer, _core.Zero, _core.One, _core.NegativeOne):
            _linear_integer_atom(node, children, work)
        owners = _integer_forms._MRO.__get__(type(node))
        slots = tuple((slot, _source_slot(slot, node))
                      for owner, _name, slot in _SOURCE_SLOTS
                      if any(owner is parent for parent in owners))
        work.charge(len(slots) + 1)
        assumptions = None
        if type(node) in (_core.Symbol, _core.Dummy):
            declared = _SOURCE_ASSUMPTIONS.__get__(node, _core.Symbol)
            if (type(declared) is not dict or len(declared) > 64
                    or any(type(name) is not str or len(name) > 256
                           or type(truth) is not bool for name, truth in dict.items(declared))):
                raise NotImplementedError("diophantine: bounded typed symbol declarations required")
            work.charge(sum(len(name) + 1 for name in dict.keys(declared)) + 1)
            _binding, facts = typed_symbol(node)
            assumptions = (declared, facts)
        snapshot.append((node, type(node), slots, assumptions))
    return tuple(snapshot)


def _linear_source_unchanged(snapshot, work):
    from ._checksol import _require_original

    _require_original()
    _source_storage_interfaces()
    if _core._active_callback_session() is not work.diophantine_session:
        raise RuntimeError("diophantine: integer solve request changed during evaluation")
    for node, kind, slots, assumptions in snapshot:
        work.charge(len(slots) + 1)
        if type(node) is not kind or any(_source_slot(slot, node) is not previous for slot, previous in slots):
            raise RuntimeError("diophantine: original source mutated during integer solving")
        if assumptions is not None:
            declared, facts = assumptions
            work.charge(len(facts) + 1)
            if (len(declared) != len(facts)
                    or any(type(name) is not str or type(truth) is not bool
                           for name, truth in dict.items(declared))
                    or declared != facts):
                raise RuntimeError("diophantine: original symbol assumptions mutated during integer solving")


def _linear_coefficient(value, work):
    if type(value) is not int or abs(value).bit_length() > 4096:
        raise NotImplementedError("diophantine: integer family coefficient height exceeds its bound")
    work.charge(abs(value).bit_length() + 1)
    return value


def _linear_integer_atom(node, children, work):
    """Admit the native literal, not a mutable Python class label alone."""
    native = _source_slot(_SOURCE_VALUE, node)
    if (children or type(native) is not _NATIVE_EXPR
            or _NATIVE_IS_INTEGER.__get__(native, _NATIVE_EXPR) is not True):
        raise ValueError("diophantine: literal integer has invalid native storage")
    value = _linear_coefficient(_NATIVE_NUMERATOR(native), work)
    fixed = {_core.Zero: 0, _core.One: 1, _core.NegativeOne: -1}
    if type(node) in fixed and value != fixed[type(node)]:
        raise ValueError("diophantine: literal integer singleton has inconsistent storage")
    return value


def _integer_family_result(matrix, expected_shape, work):
    """Read a checked lattice without replaceable public matrix accessors."""
    source = _integer_forms._IntegerSource(matrix)
    if source.base is not _IntegerMatrix or source.shape != expected_shape:
        raise ArithmeticError("diophantine: checked integer family has an unexpected shape")
    count = expected_shape[0] * expected_shape[1]
    work.charge(count + 1)
    cells = _NATIVE_FLAT(source.native)
    if type(cells) is not list or len(cells) != count:
        raise ArithmeticError("diophantine: checked integer family has invalid native storage")
    values = tuple(_linear_coefficient(_NATIVE_NUMERATOR(cell), work) for cell in cells)
    source.unchanged()
    _integer_solver_interfaces()
    return values


def _linear_family_coefficients(expression, parameters, work):
    """Read an affine presentation without trusting symbolic simplification.

    This small exact recurrence is independent of the coefficient-gcd
    generator. Only integer atoms, declared parameter atoms, sums and
    products having at most one nonconstant factor are accepted.
    """
    from ..core import Add, Dummy, Integer, Mul, NegativeOne, One, Symbol, Zero
    from ._checksol import _source_nodes

    positions = {parameter: index + 1 for index, parameter in enumerate(parameters)}
    width = len(parameters) + 1
    values = {}
    for node, children in _source_nodes((expression,), work):
        work.charge(width)
        kind = type(node)
        row = [0] * width
        if kind in (Integer, Zero, One, NegativeOne):
            row[0] = _linear_integer_atom(node, children, work)
        elif kind in (Symbol, Dummy) and node in positions:
            row[positions[node]] = 1
        elif kind is Add:
            for child in children:
                for index, coefficient in enumerate(values[id(child)]):
                    row[index] = _linear_coefficient(row[index] + coefficient, work)
        elif kind is Mul:
            row[0] = 1
            for child in children:
                right = values[id(child)]
                if any(row[1:]) and any(right[1:]):
                    raise ArithmeticError("diophantine: linear generator returned nonlinear parameters")
                if any(right[1:]):
                    row, right = right, row
                scale = right[0]
                if any(abs(value).bit_length() + abs(scale).bit_length() > 4097
                       for value in row if value and scale):
                    raise NotImplementedError("diophantine: integer family coefficient product exceeds its bound")
                row = [_linear_coefficient(value * scale, work) for value in row]
        else:
            raise ArithmeticError("diophantine: linear generator returned an undeclared parameter or scalar")
        values[id(node)] = row
    return values[id(expression)]


def _complete_linear_family(coeff, variables, const, original_axes, work, param, source_snapshot):
    """Present the complete checked integer lattice with inactive coordinates.

    The established coefficient-gcd formula is a presentation candidate only.
    First the native solver checks the complete original integer solution
    lattice. A second checked solve proves that the proposed presentation
    contains its particular solution and every primitive kernel direction.
    Exact original-row products prove the reverse containment. Thus a
    residual-zero but index-two sublattice cannot be published as complete.
    """
    from ..core import Dummy, Symbol
    from ._checksol import _require_original, _source_nodes

    _require_original()
    _linear_source_unchanged(source_snapshot, work)
    _integer_solver_interfaces()
    count = len(variables)
    if not 1 <= count <= 64:
        raise NotImplementedError("diophantine: integer families support at most 64 coordinates")
    coefficients = [_linear_coefficient(coeff[variable], work) for variable in variables]
    constant = _linear_coefficient(const, work)
    canonical = sorted(variables, key=lambda variable: variable.name)
    active = [variable for variable in canonical if coeff[variable]]
    if not active:
        raise NotImplementedError("diophantine: zero polynomial requires unconstrained integer families")
    work.charge(count * count + 1)
    checked = _checked_integer_solve(
        _IntegerMatrix([coefficients]), _IntegerMatrix([[-constant]]),
    )
    _integer_solver_interfaces()
    _require_original()
    if checked is None:
        return set(), ()
    particular, kernel = checked
    particular = _integer_family_result(particular, (count, 1), work)
    kernel = _integer_family_result(kernel, (count, count - 1), work)

    if param is None:
        prefix = "t"
    else:
        if type(param) not in (Symbol, Dummy):
            raise NotImplementedError("diophantine: linear family parameter must be a built-in Symbol")
        _source_nodes((param,), work)
        prefix = param.name
    if type(prefix) is not str or len(prefix.encode("utf-8")) > 512:
        raise NotImplementedError("diophantine: linear family parameter name exceeds its bound")
    used = set(original_axes)
    parameter_snapshot = []

    def fresh(name):
        work.charge(len(name) + 1)
        parameter = Symbol(name, integer=True)
        # A typed fresh Dummy preserves independence when an original axis
        # already owns a conventional parameter name. Never sequentially
        # substitute a generated parameter as if it were an original axis.
        if parameter in used:
            parameter = Dummy(name, integer=True)
        parameter_snapshot.extend(_linear_source_snapshot(_source_nodes((parameter,), work), work))
        used.add(parameter)
        return parameter

    active_parameters = [fresh("%s_%d" % (prefix, index)) for index in range(len(active))]
    proposed = _diop_linear(coeff, active, constant, active_parameters)
    _linear_source_unchanged(source_snapshot, work)
    _linear_source_unchanged(parameter_snapshot, work)
    _integer_solver_interfaces()
    if len(proposed) != 1:
        raise ArithmeticError("diophantine: linear presentation contradicts the checked integer lattice")
    active_values = next(iter(proposed))
    if len(active_values) != len(active):
        raise ArithmeticError("diophantine: linear presentation has the wrong coordinate count")
    active_values = dict(zip(active, active_values))
    parameters = active_parameters[:len(active) - 1]
    inactive = [variable for variable in canonical if not coeff[variable]]
    inactive_values = {variable: fresh("n%d" % (index + 1))
                       for index, variable in enumerate(inactive)}
    parameters.extend(inactive_values.values())
    solution = [active_values[variable] if coeff[variable] else inactive_values[variable]
                for variable in variables]
    candidate_snapshot = _linear_source_snapshot(_source_nodes(tuple(solution), work), work)
    rows = [_linear_family_coefficients(value, parameters, work) for value in solution]
    # Pinned DiophantineSolutionSet.add makes an isolated coordinate -t
    # positive unless another coordinate is already t. A whole-column sign
    # flip is unimodular, so this convention preserves the complete lattice.
    positive, negative = set(), set()
    for row in rows:
        work.charge(count)
        support = [index for index, value in enumerate(row) if value]
        if len(support) == 1 and support[0] and abs(row[support[0]]) == 1:
            (positive if row[support[0]] > 0 else negative).add(support[0])
    flip = negative - positive
    for row in rows:
        work.charge(len(flip) + 1)
        for column in flip:
            row[column] = -row[column]
    # Reverse containment uses only exact integer arithmetic and the original
    # coefficient row, not the polynomial generator or its proposed lattice.
    for column in range(count):
        work.charge(count + 1)
        total = sum(coefficient * row[column] for coefficient, row in zip(coefficients, rows))
        if total != (-constant if column == 0 else 0):
            raise ArithmeticError("diophantine: linear family fails the original integer equation")
    work.charge(count * count + 1)
    basis = _IntegerMatrix(count, count - 1, [value for row in rows for value in row[1:]])
    coverage = _IntegerMatrix(count, count, [
        _linear_coefficient(particular[index] - rows[index][0], work)
        if column == 0 else _linear_coefficient(kernel[index * (count - 1) + column - 1], work)
        for index in range(count) for column in range(count)
    ])
    if _checked_integer_solve(basis, coverage) is None:
        raise ArithmeticError("diophantine: linear presentation omits integer solutions")
    _integer_solver_interfaces()
    _require_original()
    _linear_source_unchanged(parameter_snapshot, work)
    _linear_source_unchanged(candidate_snapshot, work)
    # Publish fresh canonical expressions made from the certified integer
    # rows. A generator's retained operands and native payload need not agree;
    # neither is allowed to substitute for this checked presentation.
    work.charge(count * count + 1)
    published = tuple(_core.Add(
        _core.Integer(row[0]),
        *(_core.Mul(_core.Integer(coefficient), parameter)
          for coefficient, parameter in zip(row[1:], parameters) if coefficient),
    ) for row in rows)
    output_snapshot = _linear_source_snapshot(_source_nodes(published, work), work)
    _linear_source_unchanged(parameter_snapshot, work)
    work.charge()
    return {published}, output_snapshot


def _scale_param(s: Any, p: Any) -> Any:
    """Scale the parameter-free part of ``s`` by the parameter ``p``
    (upstream: sol.args[0]*p + sol.args[1] with the constant first)."""
    from ..core import Add, Integer

    if isinstance(s, Add):
        const = Add(*[a for a in s.args if not a.free_symbols])
        return const * p + (s - const)
    return s if getattr(s, "free_symbols", None) else s * p


def _bilinear_integer_solutions(
    b: int, d: int, e: int, f: int,
) -> tuple[set[tuple[int, int]], int | None, int | None]:
    """Exact finite points or affine integer families of a bilinear equation.

    The last two return values are the fixed x and fixed y coordinates
    of complete vertical and horizontal families, respectively. A missing
    family is None, so a zero coordinate remains distinguishable.

    The identity (b*x + e)*(b*y + d) = d*e - b*f proves completeness:
    a nonzero product has only signed divisor pairs, and a zero product
    has only the two possible affine factors. This is a bounded shell
    calculation, not a portable proof-certificate lane.
    """
    coefficients = (b, d, e, f)
    _check_quadratic_coefficients(coefficients)
    if b == 0:
        raise NotImplementedError("diophantine: the bilinear lane requires a nonzero mixed coefficient")
    content = gcd(gcd(b, d), e)
    if f % content:
        return set(), None, None
    b, d, e, f = (value // content for value in coefficients)
    if b < 0:
        b, d, e, f = -b, -d, -e, -f
    product_value = d*e - b*f
    original_b, original_d, original_e, original_f = coefficients
    if product_value == 0:
        x_fixed = -e // b if e % b == 0 else None
        y_fixed = -d // b if d % b == 0 else None
        if x_fixed is not None and (
            original_b*x_fixed + original_e != 0
            or original_d*x_fixed + original_f != 0
        ):
            raise ArithmeticError("diophantine: exact bilinear family check failed")
        if y_fixed is not None and (
            original_b*y_fixed + original_d != 0
            or original_e*y_fixed + original_f != 0
        ):
            raise ArithmeticError("diophantine: exact bilinear family check failed")
        return set(), x_fixed, y_fixed

    magnitude = abs(product_value)
    limit = isqrt(magnitude)
    coefficient_bits = max(abs(value).bit_length() for value in coefficients)
    coordinate_bits = max(magnitude.bit_length(), abs(d).bit_length(), abs(e).bit_length()) + 1
    # Every trial divisor produces at most four ordered signed pairs.
    # The bound includes their coordinate conversion and original residuals.
    bit_work = 4*limit*(coefficient_bits + 2*coordinate_bits + 8)
    if (limit > _MAX_DIOPHANTINE_SCAN_POINTS
            or bit_work > _MAX_DIOPHANTINE_SCAN_BIT_WORK):
        raise NotImplementedError("diophantine: exact bilinear divisor enumeration exceeds the bounded work limit")

    points = set()
    for divisor in range(1, limit + 1):
        if magnitude % divisor:
            continue
        partner = magnitude // divisor
        absolute_factors = (divisor,) if divisor == partner else (divisor, partner)
        for absolute_factor in absolute_factors:
            for left in (-absolute_factor, absolute_factor):
                right = product_value // left
                if (left - e) % b or (right - d) % b:
                    continue
                x, y = (left - e) // b, (right - d) // b
                if x*(original_b*y + original_d) + original_e*y + original_f != 0:
                    raise ArithmeticError("diophantine: exact bilinear residual check failed")
                points.add((x, y))
    return points, None, None


def _diop_quadratic(coeffs: dict, x: Any, y: Any) -> set:
    """a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0."""
    from ..core import Integer, Symbol

    a, b, c, d, e, f = (int(coeffs.get(k, 0)) for k in ("xx", "xy", "yy", "x", "y", "1"))
    _check_quadratic_coefficients((a, b, c, d, e, f))
    if a == 0 and c == 0 and b != 0:
        points, x_fixed, y_fixed = _bilinear_integer_solutions(b, d, e, f)
        out = {(Integer(xv), Integer(yv)) for xv, yv in points}
        if x_fixed is not None or y_fixed is not None:
            # Pinned SymPy uses one free integer named n1 in these
            # factor families, independently of the linear t_0 convention.
            parameter = Symbol("n1", integer=True)
            if x_fixed is not None:
                out.add((Integer(x_fixed), parameter))
            if y_fixed is not None:
                out.add((parameter, Integer(y_fixed)))
        return out
    disc = b * b - 4 * a * c
    # homogeneous: lines through the origin
    if d == 0 and e == 0 and f == 0:
        if disc < 0:
            return {(Integer(0), Integer(0))}
        r = isqrt(disc)
        if r * r != disc:
            return {(Integer(0), Integer(0))}
        t = _params(1)[0]
        out = set()
        if a == 0:
            # y*(b*x + c*y) = 0
            out.add((t, Integer(0)))
            for sol in _line(b, c, t):
                out.add(sol)
            return out
        # a*x**2 + b*x*y + c*y**2 = a*(x - r1*y)*(x - r2*y), r = (-b ± sqrt)/(2a)
        for sgn in (1, -1):
            num, den = -b + sgn * r, 2 * a
            g = gcd(num, den)
            num, den = num // g, den // g
            # x = (num/den)*y, i.e. den*x - num*y = 0, through the linear
            # base solution (upstream's sign conventions).
            out.add(base_solution_linear(0, den, -num, t))
        return out
    if disc < 0:
        return {(Integer(xv), Integer(yv))
                for xv, yv in _definite_binary_integer_points(a, b, c, d, e, f)}
    raise NotImplementedError("diophantine: this binary quadratic class (Pell-type) is not supported")


def _line(b: int, c: int, t: Any) -> list:
    from ..core import Integer

    # b*x + c*y = 0
    g = gcd(b, c)
    return [(Integer(c // g) * t if c else t, Integer(-b // g) * t)]


def _diophantine_source(eq: Any):
    """Retain and admit original domains before polynomial normalization."""
    from ..core import Add, Dummy, Eq, Half, Integer, Mul, NegativeOne, One, Pow, Rational, Symbol, Zero
    from ._checksol import _Unsupported, _Work, _require_original, _scalar, _source_nodes

    work = _Work()
    work.diophantine_session = _core._active_callback_session()
    try:
        _require_original()
        # This bounded admission accepts structured expressions and simple
        # literal/name tokens, without calling external sympification hooks.
        source = _scalar(eq, work)
        records = _source_nodes((source,), work)
        source_snapshot = _linear_source_snapshot(records, work)
    except (_Unsupported, NotImplementedError, ValueError, TypeError, OverflowError, RecursionError) as error:
        if work.failure is not None:
            raise work.failure
        raise NotImplementedError("diophantine: original source exceeds the supported domain-checking lane") from error
    numbers = (Integer, Rational, Zero, One, NegativeOne, Half)
    integer_atoms = (Integer, Zero, One, NegativeOne)
    total = True
    for node, children in records:
        kind = type(node)
        if kind in numbers or kind in (Symbol, Dummy, Add, Mul, Eq):
            continue
        if kind is Pow and type(children[1]) in integer_atoms and children[1].p >= 0:
            continue
        total = False
    # No binding constructs are admitted in this lane, so every retained
    # symbol node is an original coordinate even if normalization drops it.
    axes = {node for node, _ in records if type(node) in (Symbol, Dummy)}
    return source, records[-1][1], axes, work, total, source_snapshot


def _check_diophantine_domains(source: Any, variables: list, solutions: set, work: Any, total: bool) -> set:
    """Publish complete sets only after retaining every original exclusion.

    A polynomial is total at the integer points and polynomial families
    produced here. A source with a reciprocal or another domain guard needs
    a verdict for each complete tuple. Unknown domains of infinite families
    require conditional parameters, which this set-of-tuples API cannot
    represent; refuse instead of publishing an invalid or partial family.
    """
    from ._checksol import _Unsupported, _check_with_work, _require_original, _source_nodes

    # Recheck the enclosing native budget outside all fuzzy checker catches.
    # A nested transformation can charge it directly and leave it exhausted.
    try:
        work.charge()
    except _Unsupported:
        raise NotImplementedError("diophantine: original domain checking exceeds its bounded work limit") from None
    if work.failure is not None:
        raise work.failure
    if total or not solutions:
        return solutions

    accepted = set()
    axes = frozenset(variables)
    for solution in solutions:
        if len(solution) != len(variables):
            raise ArithmeticError("diophantine: solution coordinates do not match the original variables")
        try:
            _require_original()
            _source_nodes(tuple(solution), work)
        except (_Unsupported, NotImplementedError, ValueError, TypeError, OverflowError, RecursionError) as error:
            if work.failure is not None:
                raise work.failure
            raise NotImplementedError("diophantine: candidate domains exceed the supported checking lane") from error
        # checksol deliberately follows its sequential substitution contract.
        # A generated parameter named like an original axis must not become
        # a fixed coordinate under that convention. A guarded family needs
        # a fresh-parameter/conditional representation before it can publish.
        if any(value.free_symbols & axes for value in solution):
            raise NotImplementedError("diophantine: original domains require uncaptured family parameters")
        replacements = dict(zip(variables, solution))
        verdict = _check_with_work(source, replacements, None, work, domain_only=True)
        if work.exhausted:
            raise NotImplementedError("diophantine: original domain checking exceeds its bounded work limit")
        if verdict is None:
            raise NotImplementedError("diophantine: original domains require unresolved parameter conditions")
        if verdict is True:
            accepted.add(solution)
    return accepted


def diophantine(eq: Any, param: Any = None, syms: Any = None, permute: bool = False) -> set:
    """Set of integer solution tuples of ``eq = 0`` in the order of
    ``syms`` (default: the free symbols sorted by name)."""
    from ..core import Dummy, Eq, Integer, NegativeOne, One, Symbol, Zero, expand
    from ._checksol import _MAX_RULES, _Unsupported, _source_nodes

    source, source_children, original_axes, work, total, source_snapshot = _diophantine_source(eq)
    if syms is not None and type(syms) not in (list, tuple):
        raise NotImplementedError("diophantine: syms must be a built-in list or tuple")
    if syms is not None and len(syms) > _MAX_RULES:
        raise NotImplementedError("diophantine: bounded exact symbol coordinates required")
    # Original variables include axes which disappear when held operands
    # cancel. A normalized polynomial cannot redefine the source domain.
    requested = list(syms) if syms else None
    if requested is not None and any(type(v) not in (Symbol, Dummy) for v in requested):
        raise NotImplementedError("diophantine: bounded exact symbol coordinates required")
    if requested is not None and len(set(requested)) != len(requested):
        raise ValueError("diophantine: duplicate solution coordinates")
    # Upstream's syms argument orders the original coordinates; it does not
    # add new unknowns that never occurred in the source. Retained cancelled
    # axes still belong to original_axes and remain independent unknowns.
    var = ([variable for variable in requested if variable in original_axes]
           if requested is not None else sorted(original_axes, key=lambda s: s.name))
    if len(var) > _MAX_RULES or any(type(v) not in (Symbol, Dummy) for v in var):
        raise NotImplementedError("diophantine: bounded exact symbol coordinates required")
    if len(set(var)) != len(var):
        raise ValueError("diophantine: duplicate solution coordinates")
    try:
        _source_nodes(tuple(var), work)
    except _Unsupported:
        raise NotImplementedError("diophantine: solution coordinates exceed the bounded checking lane") from None
    # Equality sides come from the admitted retained owner, not mutable
    # lhs/rhs descriptors which could rewrite the equation during access.
    eq = source_children[0] - source_children[1] if type(source) is Eq else source
    eq = expand(eq)
    required = original_axes if total else eq.free_symbols
    missing = required.difference(var)
    if missing:
        # Pinned upstream rejects omitted original polynomial coordinates;
        # extra syms cannot cause Poly to infer an unrequested unknown.
        raise KeyError(min(missing, key=lambda symbol: symbol.name))
    from ..polys.polytools import Poly

    def admitted(solutions):
        return _check_diophantine_domains(source, var, solutions, work, total)

    try:
        p = Poly(eq, *var)
    except Exception:
        raise NotImplementedError("diophantine: not a polynomial equation")
    terms = p.terms()
    if any(type(c) not in (Integer, Zero, One, NegativeOne) for _, c in terms):
        raise NotImplementedError("diophantine: integer coefficients required")
    deg = max((sum(m) for m, _ in terms), default=0)
    if deg == 0:
        if any(int(c) != 0 for _, c in terms):
            return admitted(set())
        if not total:
            raise NotImplementedError("diophantine: original domains require unresolved parameter conditions")
        raise NotImplementedError("diophantine: zero polynomial requires unconstrained integer families")
    if len(var) == 1 and deg >= 2:
        from ..polys.polytools import roots as _roots

        return admitted({(r,) for r in _roots(p) if isinstance(r, Integer)})
    if deg <= 1:
        coeff = {v: 0 for v in var}
        const = 0
        for m, c in terms:
            if sum(m) == 0:
                const = c.p
            else:
                coeff[var[m.index(1)]] = c.p
        if any(coeff[v] == 0 for v in var):
            try:
                family, candidate_snapshot = _complete_linear_family(
                    coeff, var, const, original_axes, work, param, source_snapshot,
                )
                result = admitted(family)
                _linear_source_unchanged(source_snapshot, work)
                _linear_source_unchanged(candidate_snapshot, work)
            except _Unsupported:
                if work.failure is not None:
                    raise work.failure
                raise NotImplementedError("diophantine: integer family exceeds its bounded work limit") from None
            return result
        return admitted(_diop_linear(coeff, var, const))
    if deg == 2 and len(var) == 2:
        x, y = var
        coeffs = {}
        names = {(2, 0): "xx", (1, 1): "xy", (0, 2): "yy", (1, 0): "x", (0, 1): "y", (0, 0): "1"}
        for m, c in terms:
            coeffs[names[tuple(m)]] = int(c)
        return admitted(_diop_quadratic(coeffs, x, y))
    if deg == 2 and len(var) == 3:
        sq = {tuple(m): int(c) for m, c in terms}
        diag = {i: sq.get(tuple(2 if j == i else 0 for j in range(3)), 0) for i in range(3)}
        if len(sq) == 3 and sorted(diag.values()) == [-1, 1, 1]:
            neg = [i for i in range(3) if diag[i] == -1][0]
            others = [i for i in range(3) if i != neg]
            pq = Symbol("p", integer=True), Symbol("q", integer=True)
            p_, q_ = pq
            sol = [None, None, None]
            sol[others[0]] = 2 * p_ * q_
            sol[others[1]] = p_**2 - q_**2
            sol[neg] = p_**2 + q_**2
            return admitted({tuple(sol)})
    raise NotImplementedError("diophantine: equation class not supported")


__all__ = ["base_solution_linear", "diophantine"]
