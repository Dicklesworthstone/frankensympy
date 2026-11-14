"""Checked bounded Bernoulli ODEs, with original surface/domain admission.

Only a rational constant linear coefficient and a rational polynomial forcing
of degree at most 16 enter this lane. Integer nonlinear exponents are bounded
by 16 in magnitude. Its implicit relation is valid on nonzero local branches;
explicit reciprocal and paired quadratic roots do not assert a global domain.
"""

from fractions import Fraction

from ..core import (
    Add, Derivative, Dummy, Eq, Integer, Mul, Pow, Rational, Symbol,
    _lift_native_result, _native, _native_expr, _surface_binding_key, _symbol_binding,
)
from ._ode_system import (
    _NUMBERS, _bounded, _functions, _rational, _require_original_surface, _surface,
)

_ASSUMPTIONS = Symbol.__dict__["_assumptions"]
_DUMMY_NAME = Dummy.__dict__["_dummy_name"]
_DUMMY_NUMBER = Dummy.__dict__["_dummy_number"]


def _axis_state(variable):
    assumptions = _ASSUMPTIONS.__get__(variable, Symbol)
    if type(assumptions) is not dict or len(assumptions) > 64:
        raise ValueError("dsolve: Bernoulli symbol assumptions exceed their bound")
    items = tuple(assumptions.items())
    if any(type(key) is not str or len(key) > 1024
           or (type(value) is not bool and value is not None) for key, value in items):
        raise ValueError("dsolve: Bernoulli symbol assumptions require built-in facts")
    dummy = None
    if type(variable) is Dummy:
        name = _DUMMY_NAME.__get__(variable, Dummy)
        number = _DUMMY_NUMBER.__get__(variable, Dummy)
        if type(name) is not str or len(name) > 16_384 or type(number) is not int or number.bit_length() > 4096:
            raise ValueError("dsolve: Bernoulli Dummy metadata exceeds its bounded original form")
        dummy = name, number
    native = _native_expr(variable)
    if type(native) is not _native.Expr or not native.is_symbol:
        raise ValueError("dsolve: Bernoulli symbols require immutable native identities")
    key = _surface_binding_key(_symbol_binding(variable))
    if key != _surface_binding_key(native.symbol_binding):
        raise ValueError("dsolve: Bernoulli symbol assumptions do not match its native identity")
    return assumptions, items, native, key, dummy


def _same_axis(variable, state):
    current = _axis_state(variable)
    if current[0] is not state[0] or current[2] is not state[2] or current[1] != state[1] or current[3:] != state[3:]:
        raise ValueError("dsolve: Bernoulli coordinate or constant changed during evaluation")


class _NotBernoulli(Exception):
    pass


class _Source:
    def __init__(self, function, variable):
        self.function, self.variable = function, variable
        self.nodes = self.operations = 0
        self.snapshots = []
        self.nonzero = False
        self.axis = _axis_state(variable)
        self.constants = []

    def step(self, count=1):
        self.operations += count
        if self.operations > 200_000:
            raise ValueError("dsolve: Bernoulli arithmetic exceeds its bound")

    def combine(self, left, right, multiply=False):
        result = {} if multiply else dict(left)
        for powers, value in right.items():
            sources = left.items() if multiply else [((0, 0, 0), Fraction(1))]
            for other, factor in sources:
                self.step()
                key = tuple(a+b for a, b in zip(powers, other))
                if key[0] > 16 or abs(key[1]) > 16:
                    raise ValueError("dsolve: Bernoulli polynomial degree or exponent exceeds 16")
                if key[2] > 1:
                    raise _NotBernoulli
                product = _bounded(value*factor)
                result[key] = _bounded(result.get(key, Fraction(0)) + product)
                if not result[key]:
                    result.pop(key)
                if len(result) > 128:
                    raise ValueError("dsolve: Bernoulli expansion exceeds 128 monomials")
        return result

    def visit(self, node, depth=0):
        from .ode import _checkodesol_children, _ODE_VALUE_SLOT, _OdeBasic

        self.nodes += 1
        if self.nodes > 1024 or depth > 32:
            raise ValueError("dsolve: Bernoulli source exceeds its bounded expression envelope")
        kind = type(node)
        native = _ODE_VALUE_SLOT.__get__(node, _OdeBasic)
        children = _checkodesol_children(node)
        self.snapshots.append((node, native, children))
        if kind in _NUMBERS:
            number = _rational(node)
            return {(0, 0, 0): number} if number else {}
        if kind in (Symbol, Dummy):
            if node != self.variable:
                raise _NotBernoulli
            return {(1, 0, 0): Fraction(1)}
        if kind is type(self.function):
            if node != self.function:
                raise _NotBernoulli
            self.visit(self.variable, depth+1)
            return {(0, 1, 0): Fraction(1)}
        if kind is Derivative:
            if children != (self.function, self.variable):
                raise _NotBernoulli
            self.visit(self.function, depth+1)
            self.visit(self.variable, depth+1)
            return {(0, 0, 1): Fraction(1)}
        if kind is Eq:
            if depth or len(children) != 2:
                raise _NotBernoulli
            left, right = [self.visit(child, depth+1) for child in children]
            return self.combine(left, {key: -value for key, value in right.items()})
        if kind in (Add, Mul):
            if not children:
                raise _NotBernoulli
            # Visit every child before zero/product simplification.
            values = [self.visit(child, depth+1) for child in children]
            result = {} if kind is Add else {(0, 0, 0): Fraction(1)}
            for value in values:
                result = self.combine(result, value, kind is Mul)
            return result
        if kind is Pow and len(children) == 2:
            base, exponent = children
            value = self.visit(base, depth+1)
            self.visit(exponent, depth+1)
            power = _rational(exponent)
            if power.denominator != 1:
                raise _NotBernoulli
            if abs(power.numerator) > 16:
                raise ValueError("dsolve: Bernoulli exponent exceeds 16")
            power = power.numerator
            if not value and power <= 0:
                raise ValueError("dsolve: Bernoulli source contains an undefined nonpositive power")
            if power < 0:
                if value == {(0, 1, 0): Fraction(1)} and type(base) is type(self.function):
                    self.nonzero = True
                    return {(0, power, 0): Fraction(1)}
                if set(value) == {(0, 0, 0)}:
                    return {(0, 0, 0): _bounded(value[(0, 0, 0)]**power)}
                raise _NotBernoulli
            result = {(0, 0, 0): Fraction(1)}
            for _ in range(power):
                result = self.combine(result, value, True)
            return result
        raise _NotBernoulli

    def recheck(self):
        from .ode import (
            _checkodesol_children, _checkodesol_function, _checkodesol_function_protocol,
            _ODE_VALUE_SLOT, _OdeBasic,
        )
        from ..core import _require_default_function_identity

        _require_original_surface()
        _require_default_function_identity()
        _checkodesol_function_protocol()
        _checkodesol_function(self.function)
        _same_axis(self.variable, self.axis)
        for constant, state in self.constants:
            _same_axis(constant, state)
        for node, native, children in self.snapshots:
            if _ODE_VALUE_SLOT.__get__(node, _OdeBasic) is not native:
                raise ValueError("dsolve: Bernoulli source changed during evaluation")
            current = _checkodesol_children(node)
            if len(current) != len(children) or any(a is not b for a, b in zip(current, children)):
                # Native-only regions recover fresh wrappers; their unchanged
                # immutable native handle is the source identity in that case.
                from ..core._opaque_arithmetic import stored_args
                if stored_args(node) is not None or type(node) is Derivative:
                    raise ValueError("dsolve: Bernoulli retained source changed during evaluation")


def solve_bernoulli(equation, function, hint, ics, simplify):
    """Return None only for a structural non-match on the default route."""
    from .ode import _constants, _checkodesol_children

    if hint not in ("default", "Bernoulli"):
        return None
    _require_original_surface()
    # A cheap bounded syntax classification leaves other existing ODE classes
    # on their own routes. Explicit Bernoulli requests receive full admission.
    if hint == "default":
        todo, found, count = [equation], False, 0
        while todo:
            node = todo.pop()
            count += 1
            if count > 1024:
                if found:
                    raise ValueError("dsolve: Bernoulli source exceeds its bounded expression envelope")
                return None
            if type(node) not in (Eq, Add, Mul, Pow, Derivative):
                continue
            children = _checkodesol_children(node)
            if type(node) is Pow and len(children) == 2:
                from .ode import _plain_ode_function_class
                if _plain_ode_function_class(type(children[0])):
                    found = True
            if count + len(todo) + len(children) > 1024:
                if found:
                    raise ValueError("dsolve: Bernoulli source exceeds its bounded expression envelope")
                return None
            todo.extend(children)
        if not found:
            return None
    source = None
    try:
        functions, variable, elementary = _functions([equation], None if function is None else [function])
        if elementary:
            raise _NotBernoulli
        function = functions[0]
        source = _Source(function, variable)
        terms = source.visit(equation)
        derivative = terms.pop((0, 0, 1), Fraction(0))
        if not derivative or any(key[2] for key in terms):
            raise _NotBernoulli
        p = _bounded(terms.pop((0, 1, 0), Fraction(0))/derivative)
        exponents = {key[1] for key in terms}
        if len(exponents) != 1:
            raise _NotBernoulli
        n = exponents.pop()
        if n in (0, 1) or not -16 <= n <= 16:
            raise _NotBernoulli
        q = [Fraction(0)]*(1+max(key[0] for key in terms))
        for (degree, _, _), coefficient in terms.items():
            q[degree] = _bounded(-coefficient/derivative)
    except (_NotBernoulli, NotImplementedError):
        if source is not None and source.nonzero:
            raise ValueError("dsolve: original Bernoulli reciprocal domains cannot be erased by fallback") from None
        if hint == "default":
            return None
        raise ValueError("dsolve: equation does not match the bounded Bernoulli hint") from None
    # Preserve the previously supported separable default forms and their
    # initial-condition path. Explicit Bernoulli requests use this lane.
    if hint == "default" and (not p or len(q) == 1):
        if source.nonzero:
            raise ValueError("dsolve: original Bernoulli reciprocal domains require the explicit Bernoulli hint")
        return None
    if ics:
        raise NotImplementedError("dsolve: Bernoulli initial conditions are not yet supported")
    (constant,) = _constants(1, 1, {variable})
    source.constants.append((constant, _axis_state(constant)))
    source.recheck()
    native = _native.bernoulli_native(
        _native_expr(_surface(p)), [_native_expr(_surface(v)) for v in q], n,
        _symbol_binding(variable), _symbol_binding(constant),
    )
    source.recheck()
    transformed = _lift_native_result(native, variable, constant)
    if not simplify or n not in (2, 3, -1):
        result = Eq(function**(1-n), transformed, evaluate=False)
    elif n == 2:
        result = Eq(function, 1/transformed, evaluate=False)
    else:
        from ..core import sqrt
        root = sqrt(1/transformed if n == 3 else transformed)
        result = [Eq(function, -root, evaluate=False), Eq(function, root, evaluate=False)]
    source.recheck()
    return result
