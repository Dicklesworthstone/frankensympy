"""Checked bounded Bernoulli ODEs, with original surface/domain admission.

Only a rational constant linear coefficient and a rational polynomial forcing
of degree at most 16 enter this lane. Integer nonlinear exponents are bounded
by 16 in magnitude. Its implicit relation is valid on nonzero local branches;
explicit reciprocal and paired quadratic roots do not assert a global domain.
Exact rational value initial conditions fix the transformed family. Quadratic
graphs retain only the branch through the prescribed nonzero value; zero data
use the separate zero solution only when the original equation permits it.
Nonzero initial data require a reciprocal or explicit quadratic graph; an
implicit power relation cannot identify the prescribed original value branch.
"""

from fractions import Fraction

from ..core import (
    Add, Derivative, Dummy, Eq, Integer, Mul, Pow, Rational, Symbol,
    _lift_native_result, _native, _native_expr, _surface_binding_key, _symbol_binding,
)
from ._ode_system import (
    _NUMBERS, _bounded, _functions, _rational, _require_original_surface, _surface,
)
from ..functions.elementary.exponential import exp as _BernoulliExp

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
        self.initial_state = None

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
        from ..core import _is_closed_function_class, _require_default_function_identity

        _require_original_surface()
        _require_default_function_identity()
        if not _is_closed_function_class(_BernoulliExp):
            raise NotImplementedError("dsolve: Bernoulli requires the original exponential implementation")
        _checkodesol_function_protocol()
        _checkodesol_function(self.function)
        _same_axis(self.variable, self.axis)
        for constant, state in self.constants:
            _same_axis(constant, state)
        if self.initial_state is not None:
            mapping, observations = self.initial_state
            if len(mapping) != len(observations):
                raise ValueError("dsolve: Bernoulli initial conditions changed during evaluation")
            current = tuple(dict.items(mapping))
            if len(current) != len(observations) or any(
                key is not old_key or value is not old_value
                for (key, value), (old_key, old_value) in zip(current, observations)
            ):
                raise ValueError("dsolve: Bernoulli initial conditions changed during evaluation")
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


def _initial_data(source, ics):
    """Freeze exact value observations without invoking user conversion hooks."""
    from .ode import _checkodesol_children, _ODE_VALUE_SLOT, _OdeBasic

    if ics is None:
        return ()
    if type(ics) is not dict:
        raise TypeError("dsolve: Bernoulli initial conditions require a built-in dictionary")
    if len(ics) > 16:
        raise ValueError("dsolve: Bernoulli initial conditions exceed 16 observations")
    observations = tuple(dict.items(ics))
    source.initial_state = ics, observations
    result = []
    for key, raw_value in observations:
        if type(key) is not type(source.function):
            raise NotImplementedError("dsolve: Bernoulli supports value initial conditions only")
        children = _checkodesol_children(key)
        if type(children) is not tuple or len(children) != 1:
            raise ValueError("dsolve: invalid Bernoulli initial-condition function")
        point = children[0]
        if type(point) not in _NUMBERS or type(raw_value) not in (*_NUMBERS, int):
            raise NotImplementedError("dsolve: Bernoulli initial data must be exact rational values")
        time, value = _rational(point), _rational(raw_value)
        source.snapshots.append((key, _ODE_VALUE_SLOT.__get__(key, _OdeBasic), children))
        source.visit(point)
        if type(raw_value) is not int:
            source.visit(raw_value)
        result.append((time, value))
    source.recheck()
    return tuple(result)


def _initial_power(value, exponent):
    # The source exponent is at most 16, so this preflight bounds even the
    # unreduced temporary rational before exponentiation.
    bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    if bits * abs(exponent) > 1024:
        raise ValueError("dsolve: Bernoulli transformed initial value exceeds 1024 bits")
    return _bounded(value**exponent)


def _bind_initial_data(source, transformed, constant, p, n, observations):
    """Specialize the checked linear family without a nonlinear root search.

    Native generation has independently verified u = C*exp((n-1)*p*x)+P(x).
    Centering its nonvanishing homogeneous factor at the first observation
    avoids introducing a division by a symbolic exponential. The original
    value observations are checked again after explicit branch selection.
    """
    point, value = observations[0]
    particular = transformed.subs(constant, 0)
    at_point = _rational(particular.subs(source.variable, _surface(point)))
    target = _initial_power(value, 1-n)
    offset = _bounded(target-at_point)
    rate = _bounded((n-1)*p)
    bound = particular + _surface(offset)*_BernoulliExp(
        _surface(rate)*(source.variable-_surface(point)))
    for time, observed in observations:
        expected = _initial_power(observed, 1-n)
        actual = bound.subs(source.variable, _surface(time))
        if type(actual) not in _NUMBERS:
            # Distinct rational times with nonzero homogeneous coefficient
            # can require a transcendental identity. Unknown is not equality.
            raise NotImplementedError("dsolve: cannot verify Bernoulli initial conditions exactly")
        if _rational(actual) != expected:
            raise ValueError("Couldn't solve for initial conditions")
    source.recheck()
    return bound


def _check_initial_graph(source, candidate, observations):
    """A transformed observation must also select the original value branch."""
    for point, value in observations:
        observed = candidate.subs(source.variable, _surface(point))
        if type(observed) not in _NUMBERS:
            raise NotImplementedError("dsolve: cannot verify the Bernoulli initial branch exactly")
        if _rational(observed) != value:
            raise ValueError("Couldn't solve for initial conditions")
    source.recheck()


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
    observations = _initial_data(source, ics)
    if any(not value for _, value in observations):
        # y -> y^(1-n) is undefined at zero for n>1, but the original
        # polynomial equation has the separate identically-zero solution.
        # A reciprocal anywhere in the retained source keeps its exclusion,
        # even if its coefficient cancelled during bounded classification.
        if n < 0 or source.nonzero:
            raise ValueError("dsolve: zero initial value is outside the original Bernoulli domain")
        if any(value for _, value in observations):
            raise ValueError("Couldn't solve for initial conditions")
        result = Eq(function, Integer(0), evaluate=False)
        source.recheck()
        return result
    if observations and (n not in (2, 3, -1) or (not simplify and n != 2)):
        raise NotImplementedError("dsolve: Bernoulli initial conditions require an unambiguous explicit branch")
    for _, value in observations:
        _initial_power(value, 1-n)
    (constant,) = _constants(1, 1, {variable})
    source.constants.append((constant, _axis_state(constant)))
    source.recheck()
    native = _native.bernoulli_native(
        _native_expr(_surface(p)), [_native_expr(_surface(v)) for v in q], n,
        _symbol_binding(variable), _symbol_binding(constant),
    )
    source.recheck()
    transformed = _lift_native_result(native, variable, constant)
    if observations:
        transformed = _bind_initial_data(source, transformed, constant, p, n, observations)
    selected = None
    if observations and n in (2, 3, -1):
        from ..core import sqrt
        if n == 2:
            selected = 1/transformed
        else:
            root = sqrt(1/transformed if n == 3 else transformed)
            selected = root if observations[0][1] > 0 else -root
        _check_initial_graph(source, selected, observations)
    if not simplify or n not in (2, 3, -1):
        result = Eq(function**(1-n), transformed, evaluate=False)
    elif selected is not None:
        result = Eq(function, selected, evaluate=False)
    elif n == 2:
        result = Eq(function, 1/transformed, evaluate=False)
    else:
        from ..core import sqrt
        root = sqrt(1/transformed if n == 3 else transformed)
        result = [Eq(function, -root, evaluate=False), Eq(function, root, evaluate=False)]
    source.recheck()
    return result
