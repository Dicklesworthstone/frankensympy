"""Bounded exact source lowering for homogeneous two-component ODE systems.

The shell retains the original undefined-function owners. Only exact rational
linear forms cross the native boundary, where a complete normalized fundamental
matrix is independently checked. The integration constants are the two initial
coordinates at the chosen common time, rather than a chosen eigenvector basis.
"""

from fractions import Fraction

from ..core import (
    Add, Derivative, Dummy, Eq, Integer, Mul, Pow, Rational, S, Symbol,
    _lift_native_result, _native, _native_expr, _symbol_binding,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record

_NUMBERS = (Integer, Rational, type(S.Zero), type(S.One),
            type(S.NegativeOne), type(S.Half))
_MAX_BITS = 1024
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
        if _plain_ode_function_class(kind):
            if len(kind.__name__) > _MAX_TEXT:
                raise ValueError("dsolve: system function name exceeds its text bound")
            _checkodesol_function(node)
            if not any(node == other for other in found):
                found.append(node)
            if len(found) > 2:
                raise NotImplementedError("dsolve: this system lane supports exactly two functions")
            continue
        if kind in (Symbol, Dummy):
            continue
        if kind not in (Eq, Add, Mul, Pow, Derivative):
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
    if len(functions) != 2:
        raise NotImplementedError("dsolve: this system lane supports exactly two functions")
    functions = [_checkodesol_function(function) for function in functions]
    if type(functions[0]) is type(functions[1]):
        raise ValueError("dsolve: system functions must be distinct")
    time = _checkodesol_children(functions[0])[0]
    if len(time.name) > _MAX_TEXT:
        raise ValueError("dsolve: system time symbol exceeds its text bound")
    if _checkodesol_children(functions[1])[0] != time:
        raise ValueError("dsolve: system functions require the same typed time variable")
    if len(found) != 2 or any(not any(function == item for item in functions) for function in found):
        raise ValueError("dsolve: supplied functions do not match the original system")
    return functions, time


class _LinearSource:
    def __init__(self, functions, time):
        self.functions = functions
        self.time = time
        self.nodes = 0

    @staticmethod
    def constant(value):
        return [value, Fraction(0), Fraction(0), Fraction(0), Fraction(0)]

    @staticmethod
    def basis(index):
        result = _LinearSource.constant(Fraction(0))
        result[index] = Fraction(1)
        return result

    def visit(self, node, depth=0):
        from .ode import _checkodesol_children

        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("dsolve: system source exceeds its bounded expression envelope")
        kind = type(node)
        if kind in _NUMBERS or kind is int:
            return self.constant(_rational(node))
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
                    return self.basis(index + 3)
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
                result = [_bounded(a + sign*b) for a, b in zip(result, value)]
            return result
        if kind is Mul:
            result = self.constant(Fraction(1))
            for value in values:
                left_linear, right_linear = any(result[1:]), any(value[1:])
                if left_linear and right_linear:
                    raise NotImplementedError("dsolve: nonlinear coupled systems are outside this lane")
                result = ([_bounded(result[0]*entry) for entry in value] if right_linear
                          else [_bounded(value[0]*entry) for entry in result])
            return result
        if len(values) != 2 or any(values[1][1:]):
            raise NotImplementedError("dsolve: unsupported system power")
        base, exponent = values
        power = exponent[0]
        if power.denominator != 1 or abs(power.numerator) > 32:
            raise NotImplementedError("dsolve: system powers require small literal integers")
        power = power.numerator
        if any(base[1:]):
            if power == 1:
                return base
            raise NotImplementedError("dsolve: nonconstant system powers are outside this lane")
        if base[0] == 0 and power <= 0:
            raise ValueError("dsolve: original system contains an undefined nonpositive power of zero")
        if max(abs(base[0].numerator).bit_length(), base[0].denominator.bit_length()) * abs(power) > _MAX_BITS:
            raise ValueError("dsolve: system power coefficient growth exceeds 1024 bits")
        return self.constant(_bounded(base[0]**power))


def _initial_values(functions, time, conditions):
    from .ode import _checkodesol_children

    used_names = {time.name, *(type(function).__name__ for function in functions)}
    constants = []
    index = 1
    while len(constants) < 2:
        name = f"C{index}"
        index += 1
        if name not in used_names:
            constants.append(Symbol(name))
            used_names.add(name)
    values = list(constants)
    point = None
    if conditions is not None:
        if type(conditions) is not dict or len(conditions) > 2:
            raise ValueError("dsolve: this system requires at most two value initial conditions")
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


def solve_linear_system_2(equations, functions=None, *, ics=None, hint="default", **kwargs):
    """Return a flat list of two Eq objects, following the supplied function order.

    Exact rational constant homogeneous systems with an invertible derivative
    coefficient matrix are admitted. Value initial conditions may fix either
    or both coordinates at one exact rational time. Other system classes and
    boundary data explicitly refuse instead of publishing a partial solution.
    """
    _require_original_surface()
    if len(equations) != 2:
        raise NotImplementedError("dsolve: this system lane supports exactly two equations")
    if hint != "default":
        raise NotImplementedError("dsolve: this system lane supports the default hint")
    functions, time = _functions(equations, functions)
    parser = _LinearSource(functions, time)
    rows = [parser.visit(equation) for equation in equations]
    if any(row[0] for row in rows):
        raise NotImplementedError("dsolve: nonhomogeneous coupled systems are outside this lane")
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
    point, values = _initial_values(functions, time, ics)
    native = _native.linear_ode_system_native(
        *(_native_expr(_surface(value)) for value in coefficients),
        _symbol_binding(time), _native_expr(point),
        _native_expr(values[0]), _native_expr(values[1]),
    )
    if len(native) != 2:
        raise ValueError("dsolve: checked system returned the wrong coordinate count")
    return [Eq(function, _lift_native_result(value, time, *values))
            for function, value in zip(functions, native)]
