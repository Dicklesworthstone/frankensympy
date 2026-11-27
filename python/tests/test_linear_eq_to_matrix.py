"""Public structural affine extraction and equation-form solver integration."""

import pytest
import sympy as s
from sympy.solvers.solveset import NonlinearError, linear_eq_to_matrix


def test_symbolic_coefficients_constraints_and_reordered_coordinates():
    x, y, z, a = s.symbols("matrix_x matrix_y matrix_z matrix_a")
    equations = [s.Eq(a * (x + y), z + a, evaluate=False), x - y, 3]
    coefficients, rhs = linear_eq_to_matrix(equations, [z, y, x])
    assert coefficients == s.Matrix([[-1, a, a], [0, -1, 1], [0, 0, 0]])
    assert rhs == s.Matrix([a, 0, -3])
    assert type(coefficients) is type(rhs) is s.Matrix
    assert s.linear_eq_to_matrix is linear_eq_to_matrix
    from sympy.solvers import linear_eq_to_matrix as public
    assert public is linear_eq_to_matrix


@pytest.mark.parametrize("matrix", [
    s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix,
])
def test_matrix_equation_input_flattens_row_major_and_preserves_zero_rows(matrix):
    x, y = s.symbols("matrix_x matrix_y")
    equations = matrix([[x - 2, 0], [2 * y, -1]])
    coefficients, rhs = linear_eq_to_matrix(equations, x, y)
    assert coefficients == s.Matrix([[1, 0], [0, 0], [0, 2], [0, 0]])
    assert rhs == s.Matrix([2, 0, 0, 1])


def test_formal_parameter_coefficients_and_constants_retain_axis_identity():
    a, x = s.symbols("matrix_a matrix_x")
    p = s.Dummy("matrix_x")
    t = s.Symbol("matrix_x", positive=True)
    equations = [s.Eq(x / a + s.sin(a) * p, t, evaluate=False)]
    coefficients, rhs = linear_eq_to_matrix(equations, s.Tuple(p, x))
    assert coefficients == s.Matrix([[s.sin(a), 1 / a]])
    assert rhs == s.Matrix([t])


@pytest.mark.parametrize("case", ["product", "divisor", "power", "exp", "held", "both_sides"])
def test_nonlinear_original_structure_refuses_before_cancellation(case):
    x, y = s.symbols("matrix_x matrix_y")
    expr = {
        "product": lambda: (x + 1) * (y - 1),
        "divisor": lambda: x / y,
        "power": lambda: x**2,
        "exp": lambda: s.exp(y),
        "held": lambda: s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False),
        "both_sides": lambda: s.Eq(x**2 + y, x**2 + 1, evaluate=False),
    }[case]()
    with pytest.raises(NonlinearError):
        linear_eq_to_matrix(expr, x, y)


def test_exact_large_coefficients_empty_axes_and_empty_rows():
    x = s.Symbol("matrix_x")
    q = s.Rational(2**165 + 1, 2**119)
    assert linear_eq_to_matrix(q*x - q, x) == (s.Matrix([[q]]), s.Matrix([q]))
    a, b = linear_eq_to_matrix([], (x,))
    assert a.shape == (0, 1)
    assert b.shape == (0, 1)
    a, b = linear_eq_to_matrix([x], [])
    assert a.shape == (1, 0)
    assert b == s.Matrix([-x])


def test_coordinate_admission_and_output_limit():
    x = s.Symbol("matrix_x")
    with pytest.raises(ValueError):
        linear_eq_to_matrix([x])
    with pytest.raises(ValueError):
        linear_eq_to_matrix([x], x, x)
    with pytest.raises(TypeError):
        linear_eq_to_matrix(x, {x})
    with pytest.raises(ValueError, match="limit"):
        linear_eq_to_matrix([0] * 1025, [x] * 256)


def test_linsolve_preserves_parameters_without_numeric_probes(monkeypatch):
    from sympy.solvers import solvers
    a, x, y = s.symbols("matrix_a matrix_x matrix_y")

    def no_probes(*args):
        pytest.fail("linear coefficient extraction sampled the equations")

    monkeypatch.setattr(solvers, "_extract_linear_coeffs", no_probes)
    solution = s.linsolve([s.Eq(a * (x + y), a + 1, evaluate=False), y - 1], x, y)
    assert tuple(next(iter(solution))) == (1/a, 1)
