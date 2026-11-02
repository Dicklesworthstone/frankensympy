"""Public checked objectives against the actual native second-order engine."""

import math

import pytest

from sympy import (
    Add, Dummy, Expr, Function, Integer, Matrix, Mul, Rational, Symbol,
    compile_objective, exp, log, sin,
)
from sympy.core import _native


def test_rosenbrock_value_gradient_hessian_and_sparse_product():
    x, y = Symbol("x"), Symbol("y")
    program = compile_objective((1 - x)**2 + 100*(y - x**2)**2, (x, y))
    value, gradient, hessian = program.evaluate(-1.2, 1)
    assert value == pytest.approx(24.2)
    assert gradient == pytest.approx([-215.6, -88])
    assert hessian[0] == pytest.approx([1330, 480])
    assert hessian[1] == pytest.approx([480, 200])
    assert program.hessian_vector_product([2, -3], -1.2, 1) == pytest.approx([1220, 360])
    assert program.hessian_pattern == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert program.native_objective.proofs_verified == 6
    assert program.native_objective.hessian_nonzeros == 4
    assert program.native_objective.zero_entries_checked == 0


def test_transcendental_hessian_matches_independent_closed_form():
    x, y = Symbol("x"), Symbol("y")
    program = compile_objective(exp(x*y) + sin(x) + log(y), (x, y))
    value, gradient, hessian = program.evaluate(0.5, 2)
    e = math.e
    assert value == pytest.approx(e + math.sin(0.5) + math.log(2))
    assert gradient == pytest.approx([2*e + math.cos(0.5), 0.5*e + 0.5])
    assert hessian[0] == pytest.approx([4*e - math.sin(0.5), 2*e])
    assert hessian[1] == pytest.approx([2*e, 0.25*e - 0.25])


def test_dynamic_power_rule_composes_through_the_second_derivative():
    x = Symbol("x")
    program = compile_objective(x**x, x)
    value, gradient, hessian = program.evaluate(2)
    second = 4*((math.log(2) + 1)**2 + 0.5)
    assert value == 4
    assert gradient == pytest.approx([4*(math.log(2) + 1)])
    assert hessian[0] == pytest.approx([second])
    assert program.hessian_vector_product([3], 2) == pytest.approx([3*second])
    for point in (0, -2):
        with pytest.raises(ValueError):
            program.hessian_vector_product([0], point)


def test_sparse_pattern_omits_zeros_and_keeps_both_mixed_partials():
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    program = compile_objective(x*y + z**3, (x, y, z))
    assert program.hessian_pattern == ((0, 1), (1, 0), (2, 2))
    assert program.evaluate_sparse(2, 3, 4) == (70.0, [3.0, 2.0, 48.0], [1.0, 1.0, 24.0])
    assert program.hessian_vector_product((7, 11, 13), 2, 3, 4) == [11, 7, 312]
    assert program.native_objective.zero_entries_checked == 6


def test_constant_linear_and_empty_axes_have_correct_shapes():
    x, y = Symbol("x"), Symbol("y")
    empty = compile_objective(7, ())
    assert empty.evaluate() == (7.0, [], [])
    assert empty.hessian_vector_product([]) == []
    constant = compile_objective(Integer(7), (x, y))
    assert constant.evaluate(3, 4) == (7.0, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])
    linear = compile_objective(2*x - y + 7, (x, y))
    assert linear.evaluate_sparse(3, 4) == (9.0, [2.0, -1.0], [])
    assert linear.hessian_vector_product([9, 8], 3, 4) == [0.0, 0.0]


def test_typed_symbols_dummies_and_nested_arguments_keep_identity():
    plain, positive = Symbol("x"), Symbol("x", positive=True)
    left, right = Dummy("x"), Dummy("x")
    program = compile_objective(
        plain**2 + 3*positive**2 + left*right,
        ((plain, positive), (left, right)),
    )
    assert program.variables == (plain, positive, left, right)
    assert program.variables[0] is plain and program.variables[2] is left
    value, gradient, sparse = program.evaluate_sparse((2, 3), (5, 7))
    assert value == 66
    assert gradient == [4, 18, 7, 5]
    assert sparse == [2, 6, 1, 1]
    assert program.hessian_vector_product([1, 2, 3, 4], (2, 3), (5, 7)) == [2, 12, 4, 3]
    with pytest.raises(TypeError):
        program.evaluate(2, 3, 5, 7)


def test_original_held_domain_survives_zero_derivatives_and_zero_direction():
    x = Symbol("x")
    source = Add(x, Mul(0, log(x), evaluate=False), evaluate=False)
    program = compile_objective(source, x)
    assert program.evaluate(2) == (2.0, [1.0], [[0.0]])
    for point in (0, -1):
        with pytest.raises(ValueError):
            program.evaluate(point)
        with pytest.raises(ValueError):
            program.hessian_vector_product([0], point)


def test_first_and_second_derivative_domains_are_not_erased():
    x = Symbol("x")
    for exponent in (Rational(1, 2), Rational(3, 2)):
        program = compile_objective(x**exponent, x)
        with pytest.raises(ValueError):
            program.evaluate(0)
        with pytest.raises(ValueError):
            program.hessian_vector_product([0], 0)
        assert math.isfinite(program.evaluate(1)[0])


def test_dimensions_nonfinite_inputs_overflow_and_reuse():
    x = Symbol("x")
    program = compile_objective(x**2, x)
    for args in ((), (1, 2)):
        with pytest.raises(TypeError):
            program.evaluate(*args)
    for direction in ([], [1, 2], "1"):
        with pytest.raises(TypeError):
            program.hessian_vector_product(direction, 1)
    for point in (math.inf, math.nan, 1e308):
        with pytest.raises(ValueError):
            program.evaluate(point)
    for direction in ([math.nan], [math.inf], [1e308]):
        with pytest.raises(ValueError):
            program.hessian_vector_product(direction, 1)
    with pytest.raises(TypeError):
        program.evaluate("2")
    assert program.evaluate(3) == (9.0, [6.0], [[2.0]])
    assert program.hessian_vector_product([2], 3) == [4.0]


def test_unbound_duplicate_nonscalar_and_custom_sources_refuse_without_hooks():
    x, y = Symbol("x"), Symbol("y")
    for source, arguments in ((x + y, x), (x, (x, x))):
        with pytest.raises(ValueError):
            compile_objective(source, arguments)
    for source in ([x], Matrix([[x]])):
        with pytest.raises((TypeError, NotImplementedError)):
            compile_objective(source, x)
    calls = []

    class Custom(Function):
        def _imp_(self, *args):
            calls.append("implementation")
            return 42

        def __str__(self):
            calls.append("printer")
            return "x"

    with pytest.raises(NotImplementedError):
        compile_objective(Custom(x, evaluate=False), x)
    with pytest.raises(TypeError):
        _native.VerifiedObjective(object(), [])
    assert calls == []


def test_expression_printers_are_never_used_as_numeric_ir(monkeypatch):
    x = Symbol("x")
    source = x**2 + sin(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("printing must not be used as numerical IR")

    monkeypatch.setattr(Expr, "__str__", forbidden)
    monkeypatch.setattr(Symbol, "__str__", forbidden)
    value, gradient, hessian = compile_objective(source, x).evaluate(0)
    assert value == 0 and gradient == [1] and hessian == [[2]]


def test_accepted_objectives_and_metadata_are_immutable():
    x = Symbol("x")
    program = compile_objective(x**2, x)
    for name, value in (("_layout", ()), ("_variables", ()), ("_native_objective", None)):
        with pytest.raises(AttributeError):
            setattr(program, name, value)
        with pytest.raises(AttributeError):
            delattr(program, name)
    with pytest.raises(AttributeError):
        program.native_objective.variable_count = 10
    with pytest.raises(AttributeError):
        program.__init__((), None, ())
    pattern = program.native_objective.hessian_pattern
    pattern.clear()
    assert program.hessian_pattern == ((0, 0),)
    assert program.evaluate(3) == (9.0, [6.0], [[2.0]])


def test_native_variable_limit_is_enforced_before_differentiation():
    variables = tuple(Symbol("x" + str(index)) for index in range(65))
    with pytest.raises(ValueError):
        compile_objective(variables[0]**2, variables)


def test_objective_is_available_from_both_public_exports():
    from sympy.utilities import compile_objective as utility_compile
    assert utility_compile is compile_objective
