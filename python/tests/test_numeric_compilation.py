"""Public typed numerical programs; the actual Rust extension is required."""

import math

import pytest

from sympy import (
    Abs, Add, Dummy, Expr, Float, Function, Integer, Matrix, Mul, Rational, Symbol,
    compile_numeric, compile_residual_system, cos, exp, lambdify, log, pi, sin,
)
from sympy.core import _native, _native_expr, _symbol_binding


def test_math_backend_preserves_python_integer_arithmetic_and_constants():
    x = Symbol("x")
    result = lambdify(x, x + 1)(2)
    assert result == 3 and type(result) is int
    result = lambdify((), Integer(1))()
    assert result == 1 and type(result) is int
    exact = lambdify(x, x**2)(2**60 + 1)
    assert exact == (2**60 + 1)**2 and type(exact) is int
    assert lambdify(x, x + Rational(1, 2))(2) == 2.5
    assert lambdify(x, sin(x), modules=math)(0) == 0.0
    assert lambdify(x, x + 1, modules=["math"])(2) == 3
    assert lambdify(x, x + 1, modules=None)(2) == 3


def test_native_preview_explicitly_returns_binary64_values():
    x = Symbol("x")
    for f in (compile_numeric(x, x + 1), lambdify(x, x + 1, modules="frankensympy")):
        result = f(2)
        assert result == 3.0 and type(result) is float
        assert f.native_program.variable_count == 1
        assert f.native_program.output_count == 1
        assert f.native_program.operation_count >= 3
        assert f(5.5) == 6.5
    assert compile_numeric((), 7)() == 7.0


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_typed_bindings_distinguish_equal_printed_names(backend):
    plain = Symbol("x")
    positive = Symbol("x", positive=True)
    f = lambdify((plain, positive), plain + 2*positive, modules=backend)
    assert f(3, 7) == 17
    with pytest.raises(ValueError):
        lambdify(plain, plain + positive, modules=backend)
    with pytest.raises(ValueError):
        lambdify((plain, plain), plain, modules=backend)


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_names_are_data_and_dummies_do_not_collide(backend):
    unusual = Symbol("not a Python identifier")
    trig_name = Symbol("sin")
    first, second = Dummy("x"), Dummy("x")
    f = lambdify((unusual, trig_name, first, second), unusual + trig_name + first - second,
                 modules=backend)
    assert f(11, 7, 5, 2) == 21


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_nested_arguments_and_output_containers_preserve_structure(backend):
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    f = lambdify([(x, y), z], [x + y, (z**2, [sin(x), 7]), []], modules=backend)
    result = f((0, 3), 4)
    assert result == [3, (16, [0, 7]), []]
    assert type(result) is list and type(result[1]) is tuple
    with pytest.raises(TypeError):
        f(0, 3, 4)
    with pytest.raises(TypeError):
        f((0,), 4)


def test_native_matrix_output_is_a_frozen_source_snapshot():
    x, y = Symbol("x"), Symbol("y")
    matrix = Matrix([[x, 1], [2, y]])
    f = compile_numeric((x, y), matrix)
    matrix[0, 0] = 99
    assert f(3, 4) == [[3.0, 1.0], [2.0, 4.0]]
    with pytest.raises(NotImplementedError):
        lambdify((x, y), matrix, modules="math")


def test_native_evaluates_exact_coefficients_and_float_literals():
    x, y = Symbol("x"), Symbol("y")
    f = compile_numeric((x, y), [Rational(1, 3)*x + sin(y), Float(1.5)*x, pi])
    assert f(6, 0) == pytest.approx([2.0, 9.0, math.pi])
    assert compile_numeric(x, exp(x))(0) == 1.0
    assert compile_numeric(x, Abs(x))(-3) == 3.0


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_compilation_and_evaluation_do_not_call_expression_printers(monkeypatch, backend):
    x = Symbol("x")
    expression = x**2 + sin(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("expression printing is not a numerical IR")

    monkeypatch.setattr(Expr, "__str__", forbidden)
    monkeypatch.setattr(Symbol, "__str__", forbidden)
    assert lambdify(x, expression, modules=backend)(2) == pytest.approx(4 + math.sin(2))


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_custom_functions_and_symbol_subclasses_refuse_without_hooks(backend):
    calls = []

    class sin(Function):
        def __str__(self):
            calls.append("printer")
            return "sin(x)"

        def _imp_(self, *args):
            calls.append("implementation")
            return 42

    class CustomSymbol(Symbol):
        def __str__(self):
            calls.append("symbol printer")
            return "x"

    x = Symbol("x")
    with pytest.raises(NotImplementedError):
        lambdify(x, sin(x, evaluate=False), modules=backend)
    with pytest.raises(NotImplementedError):
        lambdify(CustomSymbol("x"), x, modules=backend)
    assert calls == []


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_custom_function_names_cannot_capture_native_intrinsic_nodes(backend):
    from sympy import sin as builtin_sin

    x = Symbol("x")
    before = builtin_sin(x) + 1
    calls = []

    class sin(Function):
        def _imp_(self, *args):
            calls.append("implementation")
            return 99

    opaque = sin(x, evaluate=False)
    after = builtin_sin(x) + 1
    assert lambdify(x, before, modules=backend)(0) == 1
    assert lambdify(x, after, modules=backend)(0) == 1
    with pytest.raises(NotImplementedError):
        lambdify(x, opaque, modules=backend)
    with pytest.raises(NotImplementedError):
        Add(opaque, 1)
    assert type(opaque) is sin
    assert opaque.args == (x,)
    assert calls == []


@pytest.mark.parametrize("backend", ["math", "frankensympy"])
def test_held_arithmetic_remains_held_after_compilation(backend):
    x = Symbol("x")
    held = Add(x, Mul(1, x, evaluate=False), evaluate=False)
    original = held.args
    f = lambdify(x, held, modules=backend)
    assert f(3) == 6
    assert held.args == original and len(held.args) == 2


@pytest.mark.parametrize("modules", ["numpy", "cmath", {}, ["math", "numpy"]])
def test_unknown_backend_is_not_silently_ignored(modules):
    x = Symbol("x")
    with pytest.raises(NotImplementedError):
        lambdify(x, x + 1, modules=modules)


def test_math_domain_exceptions_and_native_intermediate_domain_guards():
    x = Symbol("x")
    math_f = lambdify(x, 1/log(x))
    native_f = compile_numeric(x, 1/log(x))
    for f in (math_f, native_f):
        with pytest.raises(ValueError):
            f(0)
        with pytest.raises(ValueError):
            f(-1)
    with pytest.raises(ZeroDivisionError):
        lambdify(x, 1/x)(0)
    with pytest.raises(ValueError):
        compile_numeric(x, 1/x)(0)
    with pytest.raises(ValueError):
        compile_numeric(x, exp(x))(1000)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            compile_numeric(x, x)(value)


def test_verified_residual_system_connects_symbolic_derivatives_to_python():
    x, y = Symbol("x"), Symbol("y")
    system = compile_residual_system([x**2 + y, sin(x) - y], (x, y))
    residuals, jacobian = system.evaluate(3, 4)
    assert residuals == pytest.approx([13, math.sin(3) - 4])
    assert jacobian[0] == [6.0, 1.0]
    assert jacobian[1] == pytest.approx([math.cos(3), -1.0])
    assert system.residuals(3, 4) == residuals
    assert system.jacobian(3, 4) == jacobian
    assert system.native_system.proofs_verified == 4
    assert system.native_system.nonzero_derivatives == 4
    assert system.native_system.variable_count == 2
    assert system.native_system.residual_count == 2
    with pytest.raises(TypeError):
        system.evaluate(3)


def test_verified_system_preserves_symbol_identity_and_zero_columns():
    first, second = Symbol("x"), Symbol("x", positive=True)
    system = compile_residual_system([first**2, second], (second, first))
    assert system.evaluate(5, 3) == ([9.0, 5.0], [[0.0, 6.0], [1.0, 0.0]])
    assert system.native_system.nonzero_derivatives == 2
    assert compile_residual_system([], ()).evaluate() == ([], [])


def test_native_boundary_checks_sequences_types_and_dimensions_before_work():
    x = Symbol("x")
    with pytest.raises(TypeError):
        _native.CompiledFunction(["x"], [_symbol_binding(x)])
    with pytest.raises(TypeError):
        _native.CompiledFunction([_native_expr(x)], ["x"])
    with pytest.raises(TypeError):
        _native.CompiledFunction(iter([_native_expr(x)]), [_symbol_binding(x)])
    with pytest.raises(ValueError):
        _native.CompiledFunction([_native_expr(x)] * 4097, [_symbol_binding(x)])
    with pytest.raises(ValueError):
        _native.CompiledFunction([], [_symbol_binding(x)] * 257)
    program = _native.CompiledFunction([_native_expr(x)], [_symbol_binding(x)])
    with pytest.raises(TypeError):
        program.evaluate([])

    class BadFloat:
        def __float__(self):
            raise AssertionError("unregistered conversion hook")

    with pytest.raises(TypeError):
        program.evaluate([BadFloat()])
    for value in ("2", BadFloat(), complex(2, 0)):
        with pytest.raises(TypeError):
            compile_numeric(x, x)(value)


def test_cyclic_containers_and_integer_growth_are_bounded():
    x = Symbol("x")
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        compile_numeric(x, cycle)
    with pytest.raises(ValueError):
        lambdify(x, x**x)(10**9)
