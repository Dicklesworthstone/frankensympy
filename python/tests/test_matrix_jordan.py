"""Complete rational Jordan families and the original public matrix boundary."""

import pytest
import sympy as s
from sympy.core import _native


def _assert_similarity(matrix, expected):
    transform, jordan = matrix.jordan_form()
    assert jordan == expected
    assert matrix * transform == transform * jordan
    assert transform.inv() * transform == s.eye(matrix.rows)
    assert matrix.jordan_form(calc_transform=False) == expected
    return transform, jordan


@pytest.mark.parametrize("sizes", [(2, 2), (3, 2), (3, 3), (3, 2, 1), (2, 2, 2)])
def test_repeated_and_mixed_blocks_do_not_mistake_overlapping_kernels_for_dependence(sizes):
    matrix = s.diag(*(s.jordan_cell(0, size) for size in sizes))
    _assert_similarity(matrix, matrix)


def test_nondiagonal_similarity_preserves_every_generalized_eigenvector():
    expected = s.diag(s.jordan_cell(2, 3), s.jordan_cell(2, 2), s.jordan_cell(2, 1))
    change = s.eye(6)
    for index in range(5):
        change[index, index + 1] = index + 2
    matrix = change * expected * change.inv()
    assert matrix != expected
    _assert_similarity(matrix, expected)


def test_distinct_rational_eigenvalues_and_repeated_blocks_have_canonical_order():
    small, large = s.Rational(-2, 3), s.Rational(1, 2)
    matrix = s.diag(s.jordan_cell(large, 1), s.jordan_cell(small, 2), s.jordan_cell(large, 2))
    expected = s.diag(s.jordan_cell(small, 2), s.jordan_cell(large, 2), s.jordan_cell(large, 1))
    _assert_similarity(matrix, expected)


def test_empty_scalar_zero_and_maximum_native_dimension_are_complete():
    for matrix in (s.zeros(0), s.Matrix([[7]]), s.zeros(5), s.eye(16)):
        _assert_similarity(matrix, matrix)


def test_immutable_and_native_only_sources_preserve_the_public_result_type():
    matrix = s.ImmutableMatrix([[3, 1, 0, 0], [0, 3, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]])
    transform, jordan = _assert_similarity(matrix, matrix)
    assert type(transform) is type(jordan) is s.ImmutableMatrix
    native = _native.Matrix(2, 2, [_native.py_integer(value) for value in (2, 1, 0, 2)])
    native_only = s.Matrix(native)
    assert native_only._surface is None
    _assert_similarity(native_only, s.Matrix([[2, 1], [0, 2]]))


def test_spectrum_proposals_are_not_accepted_as_proof(monkeypatch):
    matrix = s.eye(3)
    monkeypatch.setattr(s.Matrix, "eigenvals", lambda self: {s.Integer(2): 3})
    with pytest.raises(ValueError):
        matrix.jordan_form()


@pytest.mark.parametrize("hint", [{}, {s.Integer(1): 2}, {s.Integer(1): True}])
def test_incomplete_or_malformed_multiplicities_never_publish_padded_jordan_matrices(monkeypatch, hint):
    matrix = s.eye(3)
    monkeypatch.setattr(s.Matrix, "eigenvals", lambda self: hint)
    with pytest.raises(ValueError):
        matrix.jordan_form(calc_transform=False)


def test_source_mutation_during_spectral_search_prevents_publication(monkeypatch):
    matrix = s.eye(3)

    def mutate(self):
        self[0, 0] = 7
        return {s.Integer(1): 3}

    monkeypatch.setattr(s.Matrix, "eigenvals", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.jordan_form()


def test_control_truthiness_cannot_mutate_the_source_after_the_last_check():
    matrix = s.eye(2)

    class MutatingControl:
        def __bool__(self):
            matrix[0, 0] = 7
            return True

    with pytest.raises(RuntimeError, match="mutated"):
        matrix.jordan_form(calc_transform=MutatingControl())


@pytest.mark.parametrize("matrix_type", [s.Matrix, s.ImmutableMatrix])
def test_control_callback_cannot_replace_the_native_view_of_verified_results(monkeypatch, matrix_type):
    matrix = matrix_type(s.eye(2))
    wrong_native = _native.Matrix.zeros(2, 2)

    class MutatingControl:
        def __bool__(self):
            monkeypatch.setattr(matrix_type, "_native", property(lambda self: wrong_native))
            return True

    with pytest.raises(NotImplementedError, match="native storage view"):
        matrix.jordan_form(calc_transform=MutatingControl())


def test_control_callback_cannot_rewrite_the_original_native_getter(monkeypatch):
    matrix = s.eye(2)
    getter = s.Matrix.__dict__["_native"].fget

    def substituted(self):
        return None

    class MutatingControl:
        def __bool__(self):
            monkeypatch.setattr(getter, "__code__", substituted.__code__)
            return True

    with pytest.raises(NotImplementedError, match="native storage view"):
        matrix.jordan_form(calc_transform=MutatingControl())


def test_stale_cells_and_retained_shapes_are_rejected_before_spectral_work(monkeypatch):
    def forbidden(self):
        raise AssertionError("spectral search ran on a stale matrix")

    monkeypatch.setattr(s.Matrix, "eigenvals", forbidden)
    stale = s.Matrix([[2]])
    stale._surface = (s.Integer(3),)
    with pytest.raises(RuntimeError, match="retained cells differ"):
        stale.jordan_form()
    malformed = s.Matrix([[2]])
    malformed._matrix_shape = (2, 2)
    with pytest.raises(ValueError, match="shapes differ"):
        malformed.jordan_form()


def test_replaced_storage_descriptors_are_refused_without_calling_them(monkeypatch):
    matrix = s.Matrix([[2]])
    effects = []

    def forbidden(instance):
        effects.append("getter")
        raise AssertionError("custom getter was called")

    monkeypatch.setattr(s.Matrix, "_native_storage", property(forbidden))
    with pytest.raises(NotImplementedError, match="original"):
        matrix.jordan_form()
    assert effects == []


def test_symbolic_repeated_blocks_keep_the_generic_route():
    variable = s.Symbol("jordan_parameter")
    matrix = s.diag(s.jordan_cell(variable, 2), s.jordan_cell(variable, 2))
    transform, jordan = matrix.jordan_form()
    assert jordan == matrix
    assert matrix * transform == transform * jordan


def test_irrational_repeated_blocks_keep_the_generic_route():
    value = s.sqrt(2)
    matrix = s.diag(s.jordan_cell(value, 2), s.jordan_cell(value, 2))
    transform, jordan = matrix.jordan_form()
    assert jordan == matrix
    assert matrix * transform == transform * jordan


def test_rational_source_with_irrational_spectrum_leaves_the_native_rational_lane():
    matrix = s.Matrix([[0, 2], [1, 0]])
    transform, jordan = matrix.jordan_form()
    assert matrix * transform == transform * jordan
    assert transform.inv() * transform == s.eye(2)
    assert jordan[0, 1] == jordan[1, 0] == 0
    assert {jordan[0, 0], jordan[1, 1]} == {-s.sqrt(2), s.sqrt(2)}


def test_native_bridge_rejects_unbounded_or_nonliteral_spectral_payloads():
    matrix = _native.Matrix.eye(2)
    one = _native.py_integer(1)
    for hints in (((one, 0),), ((one, True),), ((one, 2), (one, 2), (one, 2)),
                  ((_native.py_symbol("eigenvalue"), 2),)):
        with pytest.raises((TypeError, ValueError)):
            matrix.checked_rational_jordan(hints)
    with pytest.raises(ValueError, match="16"):
        _native.Matrix.eye(17).checked_rational_jordan(((one, 17),))
    with pytest.raises(ValueError, match="4096"):
        _native.Matrix(1, 1, [_native.py_integer(1 << 4097)]).checked_rational_jordan(())


def test_rectangular_jordan_requests_remain_errors():
    with pytest.raises(ValueError, match="square"):
        s.Matrix([[1, 2]]).jordan_form()
