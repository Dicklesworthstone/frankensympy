"""Complete exact all-real matrix spectra through the actual native extension."""

import pytest

import sympy
from sympy import (
    Add, CRootOf, Float, ImmutableMatrix, Integer, Matrix, Mul, Pow,
    Rational, Symbol, diag, sqrt,
)
from sympy.core import _wrap
from sympy.polys import rootoftools


def _jacobi():
    # An irreducible symmetric Jacobi matrix: all five eigenvalues are real
    # and simple. The characteristic polynomial below is also irreducible
    # over Q (for example its reduction modulo 31 is irreducible).
    return Matrix([
        [0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 4, 1],
        [0, 0, 0, 1, 8],
    ])


def _polynomial(x):
    # Independently obtained from p_k=(x-d_k)*p_(k-1)-p_(k-2).
    return x**5 - 15*x**4 + 66*x**3 - 82*x**2 - 31*x + 54


def _roots(polynomial=None):
    x = Symbol("x")
    return [CRootOf(_polynomial(x) if polynomial is None else polynomial, i) for i in range(5)]


def _forbid_approximation(*args, **kwargs):
    raise AssertionError("a complete real spectrum must not use numerical root identities")


def test_irreducible_quintic_spectrum_uses_exact_roots_without_approximation(monkeypatch):
    monkeypatch.setattr(sympy, "_numeric_poly_roots", _forbid_approximation)
    monkeypatch.setattr(rootoftools, "_approximate_factor_roots", _forbid_approximation)
    matrix = _jacobi()
    expected = _roots()
    assert matrix.eigenvalues() == expected
    assert matrix.eigenvals() == {root: 1 for root in expected}
    assert all(root.is_real is True and root.free_symbols == set() for root in expected)


def test_checked_characteristic_coefficients_are_bound_to_original_matrix():
    # This method independently checks the entire characteristic polynomial,
    # rather than merely testing Cayley-Hamilton or trusting a root list.
    assert [_wrap(c) for c in _jacobi()._native.checked_char_poly()] == [
        Integer(c) for c in [1, -15, 66, -82, -31, 54]
    ]


def test_nonsymmetric_companion_matrix_has_the_same_complete_spectrum():
    matrix = Matrix([
        [0, 0, 0, 0, -54],
        [1, 0, 0, 0, 31],
        [0, 1, 0, 0, 82],
        [0, 0, 1, 0, -66],
        [0, 0, 0, 1, 15],
    ])
    assert matrix.eigenvalues() == _roots()


def test_repeated_irreducible_factor_preserves_algebraic_multiplicity():
    matrix = diag(_jacobi(), _jacobi())
    expected = _roots()
    assert matrix.eigenvals() == {root: 2 for root in expected}
    assert matrix.eigenvalues() == [root for root in expected for _ in range(2)]


def test_indexed_rational_and_radical_roots_share_one_complete_spectrum():
    matrix = diag(_jacobi(), Matrix([[0, 2], [1, 0]]), Integer(3))
    expected = {root: 1 for root in _roots()}
    expected.update({-sqrt(2): 1, sqrt(2): 1, Integer(3): 1})
    assert matrix.eigenvals() == expected
    assert len(matrix.eigenvalues()) == 8


def test_rational_scaling_keeps_exact_coefficients_and_root_identity():
    x = Symbol("x")
    scaled = x**5 - Rational(15, 2)*x**4 + Rational(33, 2)*x**3
    scaled += -Rational(41, 4)*x**2 - Rational(31, 16)*x + Rational(27, 16)
    assert (_jacobi() / 2).eigenvals() == {root: 1 for root in _roots(scaled)}


def test_mutation_recomputes_spectrum_without_reusing_old_roots():
    matrix = _jacobi()
    before = matrix.eigenvalues()
    matrix[0, 0] = 1
    x = Symbol("x")
    changed = x**5 - 16*x**4 + 81*x**3 - 149*x**2 + 65*x + 31
    after = matrix.eigenvalues()
    assert after == _roots(changed)
    assert after != before


def test_immutable_matrix_uses_the_same_complete_exact_lane():
    assert ImmutableMatrix(_jacobi()).eigenvals() == {root: 1 for root in _roots()}


@pytest.mark.parametrize("block", [
    Matrix([[0, -1], [1, 0]]),
    # An arbitrarily small imaginary part must not become a real root merely
    # because it lies below a binary64 root-classification threshold.
    Matrix([[1, -Rational(1, 10**80)], [1, 1]]),
])
def test_unresolved_mixed_complex_spectrum_never_returns_only_real_roots(block, monkeypatch):
    monkeypatch.setattr(sympy, "_numeric_poly_roots", _forbid_approximation)
    monkeypatch.setattr(rootoftools, "_approximate_factor_roots", _forbid_approximation)
    matrix = diag(_jacobi(), block)
    for operation in (matrix.eigenvalues, matrix.eigenvals):
        with pytest.raises(ValueError):
            operation()


def test_omitted_real_root_cannot_be_published_as_a_complete_spectrum(monkeypatch):
    original = rootoftools.real_roots

    def omit_one(*args, **kwargs):
        return original(*args, **kwargs)[:-1]

    monkeypatch.setattr(rootoftools, "real_roots", omit_one)
    with pytest.raises(ValueError, match="Exact eigenvalues unsupported"):
        _jacobi().eigenvalues()


@pytest.mark.parametrize("mutation", ["multiplicity", "duplicate"])
def test_malformed_real_root_multiplicity_is_rejected(monkeypatch, mutation):
    original = rootoftools.real_roots

    def corrupt(*args, **kwargs):
        roots = original(*args, **kwargs)
        if mutation == "multiplicity":
            roots[0] = (roots[0][0], 0)
        else:
            roots[-1] = roots[0]
        return roots

    monkeypatch.setattr(rootoftools, "real_roots", corrupt)
    with pytest.raises(ValueError, match="invalid root multiplicities"):
        _jacobi().eigenvalues()


def test_root_isolation_failure_propagates_without_a_weaker_retry(monkeypatch):
    failure = ValueError("exact real-root isolation exhausted its budget")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(rootoftools, "real_roots", fail)
    with pytest.raises(ValueError) as caught:
        _jacobi().eigenvalues()
    assert caught.value is failure


def test_native_resource_failure_does_not_enter_the_algebraic_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a resource failure must remain terminal")

    monkeypatch.setattr(Matrix, "_complete_real_algebraic_spectrum", forbidden)
    with pytest.raises(ValueError, match="dimensions up to 32"):
        Matrix.eye(33).eigenvalues()


@pytest.mark.parametrize("entry", [
    Float(0),
    Symbol("parameter"),
    Add(Integer(0), Integer(0), evaluate=False),
    Mul(0, Pow(Integer(0), Integer(0), evaluate=False), evaluate=False),
])
def test_original_nonliteral_entries_cannot_enter_the_checked_rational_lane(entry):
    matrix = _jacobi()
    matrix[0, 0] = entry
    with pytest.raises((ValueError, NotImplementedError)):
        matrix.eigenvalues()
    with pytest.raises(ValueError, match="exact rational entries only"):
        matrix._native.checked_char_poly()


def test_custom_matrix_subclasses_do_not_enter_the_new_native_fast_path():
    class CustomMatrix(Matrix):
        pass

    with pytest.raises(ValueError, match="Exact eigenvalues unsupported"):
        CustomMatrix(_jacobi()).eigenvalues()


def test_checked_charpoly_preflights_shape_and_coefficient_height():
    with pytest.raises(ValueError, match="square"):
        Matrix([[1, 2]])._native.checked_char_poly()
    with pytest.raises(ValueError, match="dimensions up to 32"):
        Matrix.eye(33)._native.checked_char_poly()
    # Individual payloads remain small enough to construct; the aggregate
    # height must be rejected before multiplying these exact coefficients.
    large = Integer(1 << 4096)
    with pytest.raises(ValueError, match="65536 coefficient bits"):
        diag(*([large] * 16))._native.checked_char_poly()


def test_algebraic_eigenvectors_and_jordan_forms_refuse_unchecked_nullspaces():
    matrix = _jacobi()
    for operation in (matrix.eigenvects, matrix.diagonalize, matrix.jordan_form):
        with pytest.raises(NotImplementedError, match="exact algebraic nullspace"):
            operation()
    with pytest.raises(NotImplementedError, match="exact algebraic nullspace"):
        matrix.jordan_form(calc_transform=False)


def test_distinct_algebraic_eigenvalues_establish_diagonalizability():
    companion = Matrix([
        [0, 0, 0, 0, -54],
        [1, 0, 0, 0, 31],
        [0, 1, 0, 0, 82],
        [0, 0, 1, 0, -66],
        [0, 0, 0, 1, 15],
    ])
    assert companion.is_diagonalizable() is True
    assert companion.is_diagonalizable(reals_only=True) is True
    # Repeated eigenvalues require eigenspace dimensions. An unavailable
    # algebraic nullspace must not be converted to a false mathematical claim.
    with pytest.raises(NotImplementedError, match="exact algebraic nullspace"):
        diag(companion, companion).is_diagonalizable()


def test_algebraic_singular_values_use_exact_order_before_square_roots(monkeypatch):
    matrix = _jacobi()
    x = Symbol("x")
    # det(t*I-A**2) follows independently from -p(x)*p(-x), t=x**2.
    squared = x**5 - 93*x**4 + 1834*x**3 - 9196*x**2 + 9817*x - 2916
    eigenvalues = _roots(squared)
    expected = [sqrt(root) for root in reversed(eigenvalues)]
    monkeypatch.setattr(CRootOf, "evalf", _forbid_approximation)
    assert matrix.singular_values() == expected
    assert matrix.condition_number() == expected[0] / expected[-1]


def test_existing_small_radical_complex_and_empty_spectra_keep_their_forms():
    assert Matrix([]).eigenvalues() == []
    assert Matrix.eye(3).eigenvals() == {Integer(1): 3}
    assert Matrix([[0, 2], [1, 0]]).eigenvals() == {-sqrt(2): 1, sqrt(2): 1}
    assert Matrix([[0, -1], [1, 0]]).eigenvals() == {-sympy.I: 1, sympy.I: 1}


def test_mixed_algebraic_and_radical_singular_values_keep_exact_order(monkeypatch):
    x = Symbol("x")
    squared = x**5 - 93*x**4 + 1834*x**3 - 9196*x**2 + 9817*x - 2916
    roots = _roots(squared)
    smaller = Rational(3, 2) - sqrt(5) / 2
    larger = Rational(3, 2) + sqrt(5) / 2
    # Exact Sturm isolation puts roots in (1/2,2/3), (3/4,1),
    # (5,6), (16,20), (64,70). The quadratic roots interleave as below.
    descending = [roots[4], roots[3], roots[2], larger, roots[1], roots[0], smaller]
    expected = [sqrt(value) for value in descending]
    matrix = diag(_jacobi(), Matrix([[1, 1], [0, 1]]))
    monkeypatch.setattr(CRootOf, "evalf", _forbid_approximation)
    assert matrix.singular_values() == expected
    assert matrix.condition_number() == expected[0] / expected[-1]
