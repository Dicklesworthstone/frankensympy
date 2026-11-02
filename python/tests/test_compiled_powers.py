"""Variable real powers through actual native programs and checked Jacobians."""

import math

import pytest

from sympy import Add, Mul, Symbol, compile_numeric, compile_residual_system, log


@pytest.mark.parametrize("kind", ["self", "variable", "constant_base", "compound"])
def test_dynamic_power_values_and_checked_jacobians(kind):
    x, y = Symbol("x"), Symbol("y")
    expressions = {
        "self": (x**x, 4.0, [4 * (math.log(2) + 1), 0.0]),
        "variable": (x**y, 8.0, [12.0, 8 * math.log(2)]),
        "constant_base": (2**x, 4.0, [4 * math.log(2), 0.0]),
        "compound": (x**(x + y), 32.0,
                     [32 * (math.log(2) + 2.5), 32 * math.log(2)]),
    }
    source, expected, gradient = expressions[kind]
    assert compile_numeric((x, y), source)(2, 3) == pytest.approx(expected)
    system = compile_residual_system([source], (x, y))
    values, rows = system.evaluate(2, 3)
    assert values == pytest.approx([expected])
    assert rows[0] == pytest.approx(gradient)
    assert system.jacobian(2, 3)[0] == pytest.approx(gradient)


def test_dynamic_power_requires_positive_base_even_at_integer_exponents():
    x, y = Symbol("x"), Symbol("y")
    function = compile_numeric((x, y), x**y)
    system = compile_residual_system([x**y], (x, y))
    for base in (0, -2):
        with pytest.raises(ValueError):
            function(base, 3)
        with pytest.raises(ValueError):
            system.jacobian(base, 3)
    assert function(2, 3) == 8.0


def test_jacobian_retains_original_held_source_domain():
    x = Symbol("x")
    source = Add(x, Mul(0, log(x), evaluate=False), evaluate=False)
    system = compile_residual_system([source], (x,))
    for point in (0, -1):
        with pytest.raises(ValueError):
            system.jacobian(point)
    assert system.jacobian(2) == [[1.0]]


def test_nonfinite_dynamic_power_does_not_publish_a_value():
    x, y = Symbol("x"), Symbol("y")
    function = compile_numeric((x, y), x**y)
    for point in ((1e308, 2), (2, math.inf), (math.nan, 2)):
        with pytest.raises(ValueError):
            function(*point)


def test_literal_integer_power_keeps_negative_base_support():
    x = Symbol("x")
    assert compile_numeric(x, x**3)(-2) == -8.0
    system = compile_residual_system([x**3], (x,))
    assert system.evaluate(-2) == ([-8.0], [[12.0]])
