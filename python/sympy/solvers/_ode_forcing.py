"""Exact entire-function forcing for the bounded constant-coefficient ODE lane.

The source is a finite rational linear combination of
t**k*exp(a*t)*cos(b*t) and t**k*exp(a*t)*sin(b*t), with rational a and b.
Only original registered built-ins are admitted. Every original child is
visited before cancellation; no printer, numeric guess or Python hook defines
the source. The native checker independently verifies the finite realization.
"""

from fractions import Fraction

from ..core import Add, Derivative, Dummy, Eq, Mul, Pow, Symbol
from ..core import _is_closed_function_class
from ._ode_system import (
    _ELEMENTARY, _ForcingCos, _ForcingExp, _ForcingSin, _LinearSource,
    _MAX_BITS, _MAX_DEPTH, _MAX_NODES, _NUMBERS, _bounded, _rational,
)

_ZERO = Fraction(0)
_ONE = Fraction(1)
_CONSTANT = (_ZERO, _ZERO, 0, False)
_TIME = (_ZERO, _ZERO, 1, False)
_MAX_TERMS = 256


class _ElementarySource(_LinearSource):
    """A rational coefficient map in a fixed entire-function basis."""

    def put(self, target, key, value):
        self.step()
        rate, frequency, power, sine = key
        rate, frequency = _bounded(rate), _bounded(frequency)
        if not 0 <= power <= 16:
            raise NotImplementedError("dsolve: elementary forcing exceeds degree 16")
        value = _bounded(value)
        if frequency < 0:
            frequency = -frequency
            if sine:
                value = -value
        if frequency == 0 and sine:
            return
        key = (rate, frequency, power, sine)
        result = _bounded(target.get(key, _ZERO) + value)
        if not result:
            target.pop(key, None)
        else:
            if key not in target and len(target) >= _MAX_TERMS:
                raise ValueError("dsolve: elementary forcing exceeds its source-term bound")
            target[key] = result

    def constant(self, value):
        if type(value) is dict:
            self.step(len(value))
            forcing = value.copy()
        else:
            forcing = {} if not value else {_CONSTANT: _bounded(value)}
        return [forcing, *([_ZERO] * (2 * len(self.functions)))]

    @staticmethod
    def scalar(forcing):
        if any(key != _CONSTANT for key in forcing):
            raise NotImplementedError(
                "dsolve: derivative and function coefficients must be rational constants"
            )
        return forcing.get(_CONSTANT, _ZERO)

    def poly_scale(self, forcing, value):
        result = {}
        for key, coefficient in forcing.items():
            self.put(result, key, _bounded(coefficient * value))
        return result

    def poly_add(self, left, right):
        self.step(len(left))
        result = left.copy()
        for key, value in right.items():
            self.put(result, key, value)
        return result

    def poly_mul(self, left, right):
        result = {}
        for (a, b, p, left_sine), first in left.items():
            for (c, d, q, right_sine), second in right.items():
                self.step(8)
                rate, power = _bounded(a + c), p + q
                plus, minus = _bounded(b + d), _bounded(b - d)
                coefficient = _bounded(_bounded(first * second) / 2)
                if not left_sine and not right_sine:
                    self.put(result, (rate, plus, power, False), coefficient)
                    self.put(result, (rate, minus, power, False), coefficient)
                elif left_sine and right_sine:
                    self.put(result, (rate, minus, power, False), coefficient)
                    self.put(result, (rate, plus, power, False), -coefficient)
                elif left_sine:
                    self.put(result, (rate, plus, power, True), coefficient)
                    self.put(result, (rate, minus, power, True), coefficient)
                else:
                    self.put(result, (rate, plus, power, True), coefficient)
                    self.put(result, (rate, minus, power, True), -coefficient)
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
            return self.constant({_TIME: _ONE})
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
                if (type(arguments[0]) is type(function)
                        and _checkodesol_children(arguments[0]) == (self.time,)):
                    return self.basis(index + len(self.functions) + 1)
            raise NotImplementedError("dsolve: derivative of an unrelated function")
        if any(kind is function for function in _ELEMENTARY):
            if not _is_closed_function_class(kind):
                raise NotImplementedError(
                    "dsolve: elementary forcing requires original built-in implementations"
                )
            children = _checkodesol_children(node)
            if type(children) is not tuple or len(children) != 1:
                raise ValueError("dsolve: malformed elementary forcing function")
            argument = self.visit(children[0], depth + 1)
            if any(argument[1:]) or any(key != _TIME for key in argument[0]):
                raise NotImplementedError(
                    "dsolve: elementary forcing arguments must be rational multiples of time"
                )
            slope = argument[0].get(_TIME, _ZERO)
            result = {}
            if kind is _ForcingExp:
                self.put(result, (slope, _ZERO, 0, False), _ONE)
            elif kind is _ForcingSin:
                self.put(result, (_ZERO, slope, 0, True), _ONE)
            elif kind is _ForcingCos:
                self.put(result, (_ZERO, slope, 0, False), _ONE)
            return self.constant(result)
        if kind not in (Add, Mul, Pow, Eq):
            raise NotImplementedError("dsolve: unsupported original elementary forcing source")
        children = _checkodesol_children(node)
        if type(children) is not tuple or not children:
            raise ValueError("dsolve: malformed original system source")
        # Visit all original children before multiplying by zero, merging
        # equal terms or reducing powers. Unknown/partial nodes stay refused.
        values = [self.visit(child, depth + 1) for child in children]
        if kind in (Add, Eq):
            if kind is Eq and len(values) != 2:
                raise ValueError("dsolve: malformed equation")
            result = self.constant(_ZERO)
            for index, value in enumerate(values):
                sign = -1 if kind is Eq and index == 1 else 1
                result = [
                    self.poly_add(result[0], self.poly_scale(value[0], Fraction(sign))),
                    *(_bounded(a + sign*b) for a, b in zip(result[1:], value[1:])),
                ]
            return result
        if kind is Mul:
            result = self.constant(_ONE)
            for value in values:
                left_linear, right_linear = any(result[1:]), any(value[1:])
                if left_linear and right_linear:
                    raise NotImplementedError("dsolve: nonlinear coupled systems are outside this lane")
                if right_linear:
                    factor = self.scalar(result[0])
                    result = [
                        self.poly_scale(value[0], factor),
                        *(_bounded(factor*entry) for entry in value[1:]),
                    ]
                elif left_linear:
                    factor = self.scalar(value[0])
                    result = [
                        self.poly_scale(result[0], factor),
                        *(_bounded(factor*entry) for entry in result[1:]),
                    ]
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
        forcing = base[0]
        if all(key == _CONSTANT for key in forcing):
            value = self.scalar(forcing)
            if value == 0 and power <= 0:
                raise ValueError("dsolve: original system contains an undefined nonpositive power of zero")
            if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) * abs(power) > _MAX_BITS:
                raise ValueError("dsolve: system power coefficient growth exceeds 1024 bits")
            return self.constant(_bounded(value**power))
        if power <= 0:
            raise NotImplementedError(
                "dsolve: original nonpositive forcing powers require a separate domain proof"
            )
        result = {_CONSTANT: _ONE}
        while power:
            if power & 1:
                result = self.poly_mul(result, forcing)
            power >>= 1
            if power:
                forcing = self.poly_mul(forcing, forcing)
        return self.constant(result)

    def native_forcing(self, rows):
        """Separate polynomial data and bounded canonical elementary modes."""
        dimension = len(rows)
        polynomial = [[_ZERO] for _ in rows]
        modes = {}
        for row, forcing in enumerate(rows):
            for (rate, frequency, power, sine), value in forcing.items():
                self.step()
                if rate == 0 and frequency == 0:
                    polynomial[row].extend([_ZERO] * (power + 1 - len(polynomial[row])))
                    polynomial[row][power] = value
                    continue
                key = (power, rate, frequency)
                if key not in modes:
                    if len(modes) >= 8:
                        raise NotImplementedError("dsolve: at most eight elementary forcing terms")
                    modes[key] = [[_ZERO]*dimension, [_ZERO]*dimension]
                modes[key][int(sine)][row] = value
        states = set()
        result = []
        for (power, rate, frequency), (cosine, sine) in sorted(modes.items()):
            for degree in range(power + 1):
                self.step(2)
                states.add((rate, frequency, degree, False))
                if frequency != 0:
                    states.add((rate, frequency, degree, True))
                if dimension + len(states) > 8:
                    raise NotImplementedError(
                        "dsolve: physical plus forcing-state dimension exceeds eight"
                    )
            result.append((power, rate, frequency, cosine, sine))
        return [self.trim(values) for values in polynomial], result
