"""Complete exact matrix minimal polynomials through the public native lane."""

import pytest

import sympy as sp
import sympy.core as core
from sympy.core import _native


def test_repeated_eigenvalues_do_not_repeat_characteristic_multiplicity():
    t = sp.Symbol("t")
    matrix = sp.eye(8)
    result = matrix.minimal_polynomial(t)
    assert type(result) is sp.PurePoly
    assert result.gens == (t,)
    assert result.all_coeffs() == [1, -1]
    assert matrix.charpoly(t).degree() == 8
    assert matrix.minpoly(t).all_coeffs() == result.all_coeffs()


def test_nilpotent_blocks_retain_the_largest_jordan_multiplicity():
    t = sp.Symbol("t")
    block3 = sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    block2 = sp.Matrix([[0, 1], [0, 0]])
    assert block3.minimal_polynomial(t).all_coeffs() == [1, 0, 0, 0]
    assert sp.diag(block3, block2).minimal_polynomial(t).all_coeffs() == [1, 0, 0, 0]


def test_repeated_and_distinct_factors_are_combined_with_correct_multiplicity():
    t = sp.Symbol("t")
    block = sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    # (t-2)^3*(t+1), with degree four instead of a square-free spectrum.
    assert sp.diag(block, -1).minimal_polynomial(t).all_coeffs() == [1, -5, 6, 4, -8]


def test_irreducible_companion_needs_no_real_or_complex_root_representation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("minimal polynomials must not compute eigenvalues or Jordan vectors")

    for name in ("eigenvalues", "eigenvals", "eigenvects", "jordan_form", "charpoly"):
        monkeypatch.setattr(sp.Matrix, name, forbidden)
    matrix = sp.Matrix([[0, 0, 2], [1, 0, 0], [0, 1, 0]])
    assert matrix.minimal_polynomial().all_coeffs() == [1, 0, 0, -2]


def test_rational_and_nondiagonal_similarity_examples_are_exact():
    half = sp.Rational(1, 2)
    matrix = sp.Matrix([[half, sp.Rational(2, 3)], [0, half]])
    assert matrix.minimal_polynomial().all_coeffs() == [1, -1, sp.Rational(1, 4)]
    assert sp.Matrix([[3, -1], [1, 1]]).minimal_polynomial().all_coeffs() == [1, -4, 4]


def test_empty_zero_and_scalar_matrices_have_explicit_minimal_polynomials():
    assert sp.Matrix(0, 0, []).minimal_polynomial().all_coeffs() == [1]
    assert sp.zeros(4).minimal_polynomial().all_coeffs() == [1, 0]
    assert sp.Matrix([[sp.Rational(-2, 3)]]).minimal_polynomial().all_coeffs() == [1, sp.Rational(2, 3)]


def test_immutable_and_native_only_storage_use_the_same_checked_lane():
    immutable = sp.ImmutableMatrix([[0, 2], [1, 0]])
    assert immutable.minimal_polynomial().all_coeffs() == [1, 0, -2]
    native = _native.Matrix(2, 2, [_native.py_integer(value) for value in (0, 2, 1, 0)])
    matrix = sp.Matrix(native)
    assert matrix._surface is None
    assert matrix.minimal_polynomial().all_coeffs() == [1, 0, -2]


def test_output_generator_preserves_typed_identity_and_default_is_fresh():
    matrix = sp.Matrix([[2]])
    first, second = matrix.minimal_polynomial(), matrix.minimal_polynomial()
    assert type(first.gen) is sp.Dummy and type(second.gen) is sp.Dummy
    assert first.gen != second.gen
    for variable in (sp.Symbol("t", positive=True), sp.Dummy("t")):
        result = matrix.minimal_polynomial(variable)
        assert result.gen is variable
        assert result.as_expr().free_symbols == {variable}
    other = sp.Symbol("t", negative=True)
    result = matrix.minimal_polynomial(other)
    assert result.gen is other
    assert result.gen != sp.Symbol("t", positive=True)


@pytest.mark.parametrize("entry", [
    sp.Float(0), sp.Symbol("a"),
    sp.Add(0, 0, evaluate=False),
    sp.Mul(0, sp.Pow(0, 0, evaluate=False), evaluate=False),
])
def test_nonliteral_original_cells_are_not_erased_or_retried(entry):
    matrix = sp.Matrix([[entry]])
    with pytest.raises((NotImplementedError, ValueError)):
        matrix.minimal_polynomial()


def test_shape_dimension_and_coefficient_limits_are_explicit():
    assert sp.eye(16).minimal_polynomial().all_coeffs() == [1, -1]
    for matrix in (sp.Matrix(0, 1, []), sp.Matrix([[1, 2]]), sp.eye(17)):
        with pytest.raises(ValueError):
            matrix.minimal_polynomial()
    huge = sp.Matrix([[sp.Integer(1 << 4097)]])
    with pytest.raises(ValueError, match="4096"):
        huge.minimal_polynomial()


def test_invalid_output_generator_and_subclasses_do_not_invoke_protocols():
    effects = []

    class ForeignSymbol(sp.Symbol):
        def __str__(self):
            effects.append("string conversion")
            raise AssertionError("custom symbol conversion")

    class ForeignMatrix(sp.Matrix):
        def __getattribute__(self, name):
            if name in ("_native", "_native_storage", "_matrix_shape", "_surface"):
                effects.append(name)
                raise AssertionError("custom matrix getter")
            return super().__getattribute__(name)

    matrix = sp.Matrix([[1]])
    for variable in ("t", 1, sp.Symbol("t") + 1, object.__new__(ForeignSymbol)):
        with pytest.raises(TypeError):
            matrix.minimal_polynomial(variable)
    foreign = object.__new__(ForeignMatrix)
    with pytest.raises(NotImplementedError):
        sp.Matrix.minimal_polynomial(foreign)
    assert effects == []


def test_replaced_matrix_or_scalar_descriptors_are_not_called(monkeypatch):
    matrix = sp.Matrix([[2]])
    effects = []

    def forbidden(instance):
        effects.append("descriptor")
        raise AssertionError("uncontrolled getter")

    for owner, name in ((sp.Matrix, "_native_storage"), (core.Basic, "_value")):
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, property(forbidden))
            with pytest.raises(NotImplementedError):
                matrix.minimal_polynomial()
    assert effects == []


def test_preexisting_stale_surface_or_native_storage_cannot_publish_a_result():
    stale_cell = sp.Matrix([[2]])
    stale_cell._surface = (sp.Integer(3),)
    with pytest.raises(ValueError, match="retained and native entries differ"):
        stale_cell.minimal_polynomial()
    stale_native = sp.Matrix([[2]])
    stale_native._native_storage = _native.Matrix(1, 1, [_native.py_integer(3)])
    with pytest.raises(ValueError, match="retained and native entries differ"):
        stale_native.minimal_polynomial()
    stale_shape = sp.Matrix([[2]])
    stale_shape._matrix_shape = (2, 2)
    with pytest.raises(ValueError, match="shapes differ"):
        stale_shape.minimal_polynomial()


def test_preexisting_output_assumptions_must_match_the_native_typed_atom():
    variable = sp.Symbol("matrix_minpoly_stale_parameter", positive=True)
    variable._assumptions["positive"] = False
    try:
        with pytest.raises(ValueError, match="inconsistent typed assumptions"):
            sp.Matrix([[2]]).minimal_polynomial(variable)
    finally:
        variable._assumptions["positive"] = True


@pytest.mark.parametrize("mutation", ["matrix", "cell", "symbol"])
def test_mutation_during_lifting_is_terminal(monkeypatch, mutation):
    atom = sp.Integer(37)
    matrix = sp.Matrix([[atom]])
    variable = sp.Symbol("matrix_minpoly_mutation", positive=True)
    original = core._lift_native_result

    def mutate(value, *sources):
        result = original(value, *sources)
        if mutation == "matrix":
            matrix[0, 0] = 41
        elif mutation == "cell":
            core.Basic.__dict__["_value"].__set__(atom, _native.py_integer(41))
        else:
            variable._assumptions["positive"] = False
        return result

    monkeypatch.setattr(core, "_lift_native_result", mutate)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            matrix.minimal_polynomial(variable)
    finally:
        variable._assumptions["positive"] = True


def test_native_failure_does_not_retry_a_characteristic_or_eigenvalue_method(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native admission failures must remain terminal")

    monkeypatch.setattr(sp.Matrix, "charpoly", forbidden)
    monkeypatch.setattr(sp.Matrix, "eigenvals", forbidden)
    # Native-only source bypasses shell retained-cell admission deliberately;
    # the complete native checker must refuse the original symbolic entry.
    value = core._native_expr(sp.Symbol("a"))
    matrix = sp.Matrix(_native.Matrix(1, 1, [value]))
    with pytest.raises(ValueError):
        matrix.minimal_polynomial()


def test_recomputing_after_mutation_uses_the_current_complete_matrix():
    matrix = sp.Matrix([[0, 1], [0, 0]])
    first = matrix.minimal_polynomial()
    matrix[1, 1] = 2
    second = matrix.minimal_polynomial()
    assert first.all_coeffs() == [1, 0, 0]
    assert second.all_coeffs() == [1, -2, 0]
