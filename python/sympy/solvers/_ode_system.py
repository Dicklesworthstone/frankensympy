"""Bounded exact source lowering for rational constant-coefficient ODE systems.

The shell retains the original undefined-function owners. Only exact rational
linear forms cross the native boundary, where a complete normalized fundamental
matrix is independently checked. The integration constants are the initial
coordinates at the chosen common time, rather than a chosen eigenvector basis.
Systems have one through eight coordinates. Polynomial forcing has degree at
most 16 and a separate coefficient checker. Rational exponential and sine/cosine
forcing uses a checked finite realization with at most eight total physical and
forcing coordinates.
"""

from fractions import Fraction

from ..core import (
    Add, Derivative, Dummy, Eq, Integer, Mul, Pow, Rational, S, Symbol,
    _is_closed_function_class, _lift_native_result, _native, _native_expr, _symbol_binding,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from ..functions.elementary.exponential import exp as _ForcingExp
from ..functions.elementary.trigonometric import cos as _ForcingCos, sin as _ForcingSin

# Capture the defining classes before user code can replace public bindings.
# The registered capability check also validates their current implementations.
_ELEMENTARY = (_ForcingExp, _ForcingSin, _ForcingCos)

_NUMBERS = (Integer, Rational, type(S.Zero), type(S.One),
            type(S.NegativeOne), type(S.Half))
_MAX_BITS = 1024
_MAX_DIMENSION = 8
_MAX_NODES = 1024
_MAX_DEPTH = 32
_MAX_TEXT = 16_384
_MISSING = object()
_VALUE_NAMES = ("p", "q", "name", "__int__", "__str__")


def _value_record(owner):
    namespace = type.__dict__["__dict__"].__get__(owner)
    return tuple(_implementation(namespace.get(name, _MISSING)) for name in _VALUE_NAMES)


# ode imports this module during package initialization. A changed numeric
# getter or symbol identity cannot silently bind a different rational system.
_SURFACE_OWNERS = tuple(
    (owner, _owner_record(owner), _value_record(owner))
    for owner in dict.fromkeys(
        owner for cls in (Add, Derivative, Dummy, Eq, Mul, Pow, Symbol, *_NUMBERS)
        for owner in type.__dict__["__mro__"].__get__(cls)
    )
)


def _require_original_surface():
    for owner, surface, values in _SURFACE_OWNERS:
        if not _owner_matches(owner, surface):
            raise NotImplementedError("dsolve: system source requires original built-in implementations")
        try:
            current = _value_record(owner)
        except ValueError:
            raise NotImplementedError("dsolve: system source implementation exceeds its bounds") from None
        if any(len(tokens) != len(expected)
               or any(left is not right for left, right in zip(tokens, expected))
               for tokens, expected in zip(current, values)):
            raise NotImplementedError("dsolve: system source requires original coefficient and symbol accessors")


def _bounded(value):
    if (abs(value.numerator).bit_length() > _MAX_BITS
            or value.denominator.bit_length() > _MAX_BITS):
        raise ValueError("dsolve: rational system coefficient growth exceeds 1024 bits")
    return value


def _rational(value):
    if type(value) is int:
        if value.bit_length() > _MAX_BITS:
            raise ValueError("dsolve: rational system coefficient exceeds 1024 bits")
        return Fraction(value)
    if type(value) not in _NUMBERS:
        raise NotImplementedError("dsolve: this system requires exact rational data")
    return _bounded(Fraction(value.p, value.q))


def _surface(value):
    return Rational(value.numerator, value.denominator)


def _functions(equations, functions):
    from .ode import (
        _checkodesol_children, _checkodesol_function,
        _checkodesol_function_protocol, _plain_ode_function_class,
    )
    from ..core import _require_default_function_identity

    _checkodesol_function_protocol()
    _require_default_function_identity()
    found = []
    elementary = False
    pending = [(equation, 0) for equation in reversed(equations)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("dsolve: system source exceeds its bounded expression envelope")
        kind = type(node)
        if kind in _NUMBERS or kind is int:
            _rational(node)
            continue
        elementary_kind = any(kind is owner for owner in _ELEMENTARY)
        if elementary_kind:
            if not _is_closed_function_class(kind):
                raise NotImplementedError(
                    "dsolve: elementary forcing requires original built-in implementations"
                )
            elementary = True
        elif _plain_ode_function_class(kind):
            if len(kind.__name__) > _MAX_TEXT:
                raise ValueError("dsolve: system function name exceeds its text bound")
            _checkodesol_function(node)
            if not any(node == other for other in found):
                found.append(node)
            if len(found) > _MAX_DIMENSION:
                raise NotImplementedError("dsolve: this system lane supports at most eight functions")
            continue
        if kind in (Symbol, Dummy):
            continue
        if not elementary_kind and kind not in (Eq, Add, Mul, Pow, Derivative):
            raise NotImplementedError("dsolve: system sources require exact rational linear forms")
        if kind is Eq and depth:
            raise NotImplementedError("dsolve: a system equation cannot be an arithmetic subexpression")
        children = _checkodesol_children(node)
        if type(children) is not tuple or not children:
            raise ValueError("dsolve: malformed system source arguments")
        if nodes + len(pending) + len(children) > _MAX_NODES:
            raise ValueError("dsolve: system source exceeds its bounded expression envelope")
        pending.extend((child, depth + 1) for child in reversed(children))
    if functions is None:
        functions = found
    elif type(functions) is not list:
        # Pinned SymPy 1.14 system dispatch requires a list for explicit funcs.
        raise ValueError("Input to the funcs should be a list of functions.")
    if not 1 <= len(functions) <= _MAX_DIMENSION:
        raise NotImplementedError("dsolve: this system lane supports one through eight functions")
    if len(functions) != len(equations):
        raise ValueError("dsolve: system requires one equation per function")
    functions = [_checkodesol_function(function) for function in functions]
    if any(type(function) is type(previous)
           for index, function in enumerate(functions) for previous in functions[:index]):
        raise ValueError("dsolve: system functions must be distinct")
    time = _checkodesol_children(functions[0])[0]
    if len(time.name) > _MAX_TEXT:
        raise ValueError("dsolve: system time symbol exceeds its text bound")
    if any(_checkodesol_children(function)[0] != time for function in functions[1:]):
        raise ValueError("dsolve: system functions require the same typed time variable")
    if len(found) != len(functions) or any(not any(function == item for item in functions) for function in found):
        raise ValueError("dsolve: supplied functions do not match the original system")
    return functions, time, elementary


class _LinearSource:
    def __init__(self, functions, time):
        self.functions = functions
        self.time = time
        self.nodes = 0
        self.operations = 0

    def step(self, amount=1):
        self.operations += amount
        if self.operations > 200_000:
            raise ValueError("dsolve: polynomial system source exceeds its arithmetic bound")

    @staticmethod
    def trim(poly):
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly

    def poly_scale(self, poly, value):
        self.step(len(poly))
        return self.trim([_bounded(entry*value) for entry in poly])

    def poly_add(self, left, right):
        size = max(len(left), len(right))
        self.step(size)
        return self.trim([_bounded((left[i] if i < len(left) else 0)
                                   + (right[i] if i < len(right) else 0))
                          for i in range(size)])

    def poly_mul(self, left, right):
        size = len(left) + len(right) - 1
        if size > 17:
            raise NotImplementedError("dsolve: polynomial system forcing exceeds degree 16")
        self.step(len(left)*len(right))
        result = [Fraction(0)]*size
        for i, first in enumerate(left):
            for j, second in enumerate(right):
                result[i+j] = _bounded(result[i+j] + _bounded(first*second))
        return self.trim(result)

    @staticmethod
    def scalar(poly):
        if len(poly) != 1:
            raise NotImplementedError("dsolve: system derivative and function coefficients must be rational constants")
        return poly[0]

    def constant(self, value):
        polynomial = value if type(value) is list else [value]
        return [polynomial, *([Fraction(0)] * (2 * len(self.functions)))]

    def basis(self, index):
        result = self.constant(Fraction(0))
        result[index] = Fraction(1)
        return result

    def visit(self, node, depth=0):
        from .ode import _checkodesol_children

        self.nodes += 1
        self.step()
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("dsolve: system source exceeds its bounded expression envelope")
        kind = type(node)
        if kind in _NUMBERS or kind is int:
            return self.constant(_rational(node))
        if kind in (Symbol, Dummy) and node == self.time:
            return self.constant([Fraction(0), Fraction(1)])
        for index, function in enumerate(self.functions):
            if kind is type(function):
                if _checkodesol_children(node) != (self.time,):
                    raise NotImplementedError("dsolve: shifted or foreign system function")
                return self.basis(index + 1)
        if kind is Derivative:
            arguments = _checkodesol_children(node)
            if len(arguments) != 2 or arguments[1] != self.time:
                raise NotImplementedError("dsolve: this system lane supports first derivatives only")
            for index, function in enumerate(self.functions):
                if type(arguments[0]) is type(function) and _checkodesol_children(arguments[0]) == (self.time,):
                    return self.basis(index + len(self.functions) + 1)
            raise NotImplementedError("dsolve: derivative of an unrelated function")
        if kind not in (Add, Mul, Pow, Eq):
            raise NotImplementedError("dsolve: system coefficients must be rational constants")
        children = _checkodesol_children(node)
        # Visit every original child before any zero or cancellation shortcut.
        values = [self.visit(child, depth + 1) for child in children]
        if kind in (Add, Eq):
            if kind is Eq and len(values) != 2:
                raise ValueError("dsolve: malformed equation")
            result = self.constant(Fraction(0))
            for index, value in enumerate(values):
                sign = -1 if kind is Eq and index == 1 else 1
                result = [self.poly_add(result[0], self.poly_scale(value[0], Fraction(sign))),
                          *(_bounded(a + sign*b) for a, b in zip(result[1:], value[1:]))]
            return result
        if kind is Mul:
            result = self.constant(Fraction(1))
            for value in values:
                left_linear, right_linear = any(result[1:]), any(value[1:])
                if left_linear and right_linear:
                    raise NotImplementedError("dsolve: nonlinear coupled systems are outside this lane")
                if right_linear:
                    factor = self.scalar(result[0])
                    result = [self.poly_scale(value[0], factor), *(_bounded(factor*entry) for entry in value[1:])]
                elif left_linear:
                    factor = self.scalar(value[0])
                    result = [self.poly_scale(result[0], factor), *(_bounded(factor*entry) for entry in result[1:])]
                else:
                    result = self.constant(self.poly_mul(result[0], value[0]))
            return result
        if len(values) != 2 or any(values[1][1:]):
            raise NotImplementedError("dsolve: unsupported system power")
        base, exponent = values
        power = self.scalar(exponent[0])
        if power.denominator != 1 or abs(power.numerator) > 32:
            raise NotImplementedError("dsolve: system powers require small literal integers")
        power = power.numerator
        if any(base[1:]):
            if power == 1:
                return base
            raise NotImplementedError("dsolve: nonconstant system powers are outside this lane")
        polynomial = base[0]
        if len(polynomial) > 1:
            if power <= 0:
                raise NotImplementedError("dsolve: original nonpositive polynomial powers require a separate domain proof")
            if (len(polynomial) - 1)*power > 16:
                raise NotImplementedError("dsolve: polynomial system forcing exceeds degree 16")
            result = [Fraction(1)]
            while power:
                if power & 1:
                    result = self.poly_mul(result, polynomial)
                power >>= 1
                if power:
                    polynomial = self.poly_mul(polynomial, polynomial)
            return self.constant(result)
        value = polynomial[0]
        if value == 0 and power <= 0:
            raise ValueError("dsolve: original system contains an undefined nonpositive power of zero")
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) * abs(power) > _MAX_BITS:
            raise ValueError("dsolve: system power coefficient growth exceeds 1024 bits")
        return self.constant(_bounded(value**power))


def _initial_values(functions, time, conditions):
    from .ode import _checkodesol_children

    used_names = {time.name, *(type(function).__name__ for function in functions)}
    constants = []
    index = 1
    while len(constants) < len(functions):
        name = f"C{index}"
        index += 1
        if name not in used_names:
            constants.append(Symbol(name))
            used_names.add(name)
    values = list(constants)
    point = None
    if conditions is not None:
        if type(conditions) is not dict or len(conditions) > len(functions):
            raise ValueError("dsolve: system requires at most one value initial condition per function")
        assigned = set()
        for key, value in conditions.items():
            owners = [index for index, function in enumerate(functions) if type(key) is type(function)]
            if len(owners) != 1 or owners[0] in assigned:
                raise ValueError("dsolve: initial conditions must name distinct system functions")
            arguments = _checkodesol_children(key)
            if len(arguments) != 1:
                raise ValueError("dsolve: unsupported system initial condition")
            current = _rational(arguments[0])
            if point is not None and current != point:
                raise NotImplementedError("dsolve: system initial conditions must share one rational time")
            point = current
            assigned.add(owners[0])
            values[owners[0]] = _surface(_rational(value))
    return _surface(Fraction(0) if point is None else point), values


def _reduce_general_system(rows, parser):
    """Reduce M*y' + B*y + p = 0 and recheck against every original row."""
    dimension = len(rows)
    mass = [list(row[dimension + 1:]) for row in rows]
    coefficients = [[-value for value in row[1:dimension + 1]] for row in rows]
    forcing = [parser.poly_scale(row[0], Fraction(-1)) for row in rows]

    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if mass[row][column]), None)
        if pivot is None:
            raise NotImplementedError("dsolve: singular derivative systems are outside this lane")
        if pivot != column:
            mass[column], mass[pivot] = mass[pivot], mass[column]
            coefficients[column], coefficients[pivot] = coefficients[pivot], coefficients[column]
            forcing[column], forcing[pivot] = forcing[pivot], forcing[column]
        scale = _bounded(1 / mass[column][column])
        parser.step(2 * dimension)
        mass[column] = [_bounded(value * scale) for value in mass[column]]
        coefficients[column] = [_bounded(value * scale) for value in coefficients[column]]
        forcing[column] = parser.poly_scale(forcing[column], scale)
        for row in range(dimension):
            if row == column or not mass[row][column]:
                continue
            factor = mass[row][column]
            parser.step(4 * dimension)
            mass[row] = [_bounded(value - _bounded(factor * pivot_value))
                         for value, pivot_value in zip(mass[row], mass[column])]
            coefficients[row] = [_bounded(value - _bounded(factor * pivot_value))
                                 for value, pivot_value in zip(coefficients[row], coefficients[column])]
            forcing[row] = parser.poly_add(
                forcing[row], parser.poly_scale(forcing[column], -factor),
            )

    # Check the reduced source independently by multiplication through the
    # original derivative matrix. The native receipt then binds this complete
    # constant-coefficient system, including every original forcing term.
    for original in rows:
        for column in range(dimension):
            residual = original[column + 1]
            for index in range(dimension):
                parser.step(2)
                residual = _bounded(residual + _bounded(
                    original[dimension + 1 + index] * coefficients[index][column],
                ))
            if residual:
                raise ValueError("dsolve: derivative matrix reduction failed its source identity")
        residual = original[0].copy()
        for index in range(dimension):
            residual = parser.poly_add(
                residual,
                parser.poly_scale(forcing[index], original[dimension + 1 + index]),
            )
        if residual != parser.constant(Fraction(0))[0]:
            raise ValueError("dsolve: derivative matrix reduction failed its forcing identity")
    return coefficients, forcing


def solve_linear_system(equations, functions=None, *, ics=None, hint="default", **kwargs):
    """Return a flat equation list in the supplied or inferred function order.

    Admit one through eight rational constant-coefficient first-order equations
    with an invertible derivative matrix. Forcing can combine degree-at-most-16
    polynomials with exp(a*t), sin(b*t) and cos(b*t) for rational a and b, including
    polynomial factors and products within an eight-coordinate realization.
    Elementary forcing arguments must be rational multiples of the original
    typed time. The rational-resolvent lane requires rational linear or
    quadratic spectral factors.
    Rational value initial conditions share one time; unspecified coordinates
    remain independent initial-value constants. Unsupported classes refuse.
    """
    _require_original_surface()
    if not 1 <= len(equations) <= _MAX_DIMENSION:
        raise NotImplementedError("dsolve: this system lane supports one through eight equations")
    if hint != "default":
        raise NotImplementedError("dsolve: this system lane supports the default hint")
    functions, time, elementary = _functions(equations, functions)
    dimension = len(functions)
    if elementary:
        from ._ode_forcing import _ElementarySource
        parser = _ElementarySource(functions, time)
    else:
        parser = _LinearSource(functions, time)
    rows = [parser.visit(equation) for equation in equations]
    if elementary:
        coefficients, forcing = _reduce_general_system(rows, parser)
        polynomial, modes = parser.native_forcing(forcing)
        point, values = _initial_values(functions, time, ics)
        native = _native.forced_ode_system_native(
            [[_native_expr(_surface(value)) for value in row] for row in coefficients],
            _symbol_binding(time), _native_expr(point),
            [[_native_expr(_surface(value)) for value in poly] for poly in polynomial],
            [_native_expr(value) for value in values],
            [
                [
                    _native_expr(_surface(Fraction(power))),
                    _native_expr(_surface(rate)),
                    _native_expr(_surface(frequency)),
                    [_native_expr(_surface(value)) for value in cosine],
                    [_native_expr(_surface(value)) for value in sine],
                ]
                for power, rate, frequency, cosine, sine in modes
            ],
        )
    elif dimension == 2:
        # Preserve the established two-coordinate generators and failure paths.
        m00, m01 = rows[0][3:]
        m10, m11 = rows[1][3:]
        determinant = _bounded(m00*m11 - m01*m10)
        if determinant == 0:
            raise NotImplementedError("dsolve: singular derivative systems are outside this lane")
        coefficients = []
        for index in (1, 2):
            coefficients.append(_bounded((-m11*rows[0][index] + m01*rows[1][index])/determinant))
        for index in (1, 2):
            coefficients.append(_bounded((m10*rows[0][index] - m00*rows[1][index])/determinant))
        forcing = [
            parser.poly_add(parser.poly_scale(rows[0][0], _bounded(-m11/determinant)),
                            parser.poly_scale(rows[1][0], _bounded(m01/determinant))),
            parser.poly_add(parser.poly_scale(rows[0][0], _bounded(m10/determinant)),
                            parser.poly_scale(rows[1][0], _bounded(-m00/determinant))),
        ]
        point, values = _initial_values(functions, time, ics)
        arguments = (*(_native_expr(_surface(value)) for value in coefficients),
                     _symbol_binding(time), _native_expr(point),
                     _native_expr(values[0]), _native_expr(values[1]))
        if all(poly == [0] for poly in forcing):
            native = _native.linear_ode_system_native(*arguments)
        else:
            native = _native.polynomial_ode_system_native(
                *arguments,
                *[[_native_expr(_surface(value)) for value in poly] for poly in forcing],
            )
    else:
        coefficients, forcing = _reduce_general_system(rows, parser)
        point, values = _initial_values(functions, time, ics)
        native = _native.general_ode_system_native(
            [[_native_expr(_surface(value)) for value in row] for row in coefficients],
            _symbol_binding(time), _native_expr(point),
            [[_native_expr(_surface(value)) for value in poly] for poly in forcing],
            [_native_expr(value) for value in values],
        )
    if len(native) != dimension:
        raise ValueError("dsolve: checked system returned the wrong coordinate count")
    return [Eq(function, _lift_native_result(value, time, *values))
            for function, value in zip(functions, native)]
