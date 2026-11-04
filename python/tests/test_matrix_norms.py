"""Exact complex magnitudes and the distinction between spectral/Frobenius norms."""

import pytest

from sympy import Abs, Function, I, ImmutableMatrix, Matrix, Rational, Symbol, eye, oo, simplify, sqrt


@pytest.mark.parametrize("matrix_type", [Matrix, ImmutableMatrix])
def test_complex_frobenius_norm_uses_modulus_squared(matrix_type):
    assert matrix_type([I]).frobenius_norm_squared() == 1
    assert matrix_type([I]).norm() == 1
    matrix = matrix_type([[3 + 4*I, -2*I], [1 + I, -3]])
    assert matrix.frobenius_norm_squared() == 40
    assert matrix.norm("fro") == 2*sqrt(10)
    assert matrix.norm(None) == 2*sqrt(10)


def test_complex_frobenius_norm_does_not_cancel_nonzero_entries():
    matrix = Matrix([[1, I], [I, 1]])
    assert matrix.norm() == 2
    assert matrix.norm(2) == sqrt(2)


def test_complex_phase_preserves_frobenius_and_euclidean_norms():
    matrix = Matrix([[1, -2], [3, -4]])
    assert (I*matrix).norm() == matrix.norm() == sqrt(30)
    for vector in (Matrix([3*I, 4]), Matrix([[3*I, 4]])):
        assert vector.norm(2) == 5
        assert vector.norm() == 5


@pytest.mark.parametrize("assumptions", [{}, {"real": True}])
def test_symbols_retain_absolute_values_and_original_identity(assumptions):
    z = Symbol("complex_norm_z", **assumptions)
    matrix = Matrix([z])
    assert matrix.frobenius_norm_squared() == Abs(z)**2
    result = matrix.norm()
    assert result == Abs(z)
    assert result.args[0] is z
    assert result.subs(z, -3) == 3
    assert matrix[0, 0] is z
    assert matrix.mutation_generation == 0


@pytest.mark.parametrize("matrix_type", [Matrix, ImmutableMatrix])
def test_matrix_order_two_is_spectral(matrix_type):
    matrix = matrix_type([[3, 0], [0, 4]])
    assert matrix.norm() == 5
    assert matrix.norm(2) == 4
    assert matrix_type(eye(4)).norm() == 2
    assert matrix_type(eye(4)).norm(2) == 1


def test_nondiagonal_spectral_norm_uses_the_exact_gram_spectrum():
    matrix = Matrix([[1, 2], [3, 4]])
    assert simplify(matrix.norm(2) - sqrt(15 + sqrt(221))) == 0
    complex_matrix = Matrix([[1, I], [0, 1]])
    assert simplify(complex_matrix.norm(2) - sqrt((3 + sqrt(5))/2)) == 0


@pytest.mark.parametrize("scale", [2, -2, Rational(1, 3)])
def test_scaled_gram_spectra_fold_exact_discriminant_square_factors(scale):
    matrix = scale * Matrix([[1, 2], [3, 4]])
    lower = scale**2 * (15 - sqrt(221))
    upper = scale**2 * (15 + sqrt(221))
    spectrum = (matrix.H * matrix).eigenvalues()
    assert len(spectrum) == 2
    assert spectrum == [lower, upper]
    assert matrix.norm(2) == sqrt(upper)


def test_rectangular_spectral_norm_and_transpose_agree():
    matrix = Matrix([[1, 2, 2], [0, 0, 0]])
    assert matrix.norm(2) == 3
    assert matrix.T.norm(2) == 3


def test_spectral_selection_never_uses_binary64_tie_order():
    larger = 1 + Rational(1, 2**60)
    matrix = Matrix([[1, 0], [0, larger]])
    assert matrix.norm(2) == larger


@pytest.mark.parametrize("shape,expected", [((0, 0), -oo), ((0, 3), 0), ((3, 0), -oo), ((1, 0), 0), ((0, 1), 0)])
def test_empty_matrix_norms_match_the_pinned_public_shapes(shape, expected):
    matrix = Matrix(*shape, [])
    assert matrix.norm() == 0
    assert matrix.norm(2) == expected


def test_unsupported_symbolic_spectrum_does_not_return_a_frobenius_surrogate():
    z = Symbol("unresolved_spectral_parameter")
    matrix = Matrix([[z, 1], [1, 0]])
    with pytest.raises((ValueError, NotImplementedError)):
        matrix.norm(2)


@pytest.mark.parametrize("matrix_type", [Matrix, ImmutableMatrix])
@pytest.mark.parametrize("order", [None, "fro", 2])
def test_opaque_custom_entries_refuse_without_losing_class_identity(matrix_type, order):
    first = type("NormLaw", (Function,), {})
    second = type("NormLaw", (Function,), {})
    x = Symbol("opaque_norm_x")
    a, b = first(x), second(x)
    matrix = matrix_type([[a, 0], [0, b]])
    with pytest.raises(NotImplementedError, match="opaque Python"):
        matrix.norm(order)
    assert matrix[0, 0] is a and matrix[1, 1] is b
    assert type(matrix[0, 0]) is first
    assert type(matrix[1, 1]) is second
    assert type(matrix) is matrix_type
    assert matrix.mutation_generation == 0
