"""Sparse structural operations preserve owners without scanning implicit zeros.

The profile fixtures use SymPy 1.14.0's matrixbase/repmatrix selector and join
contracts. The resource, retained-source and native-only tests exercise this
implementation's bounded public lane; they are not a compatibility certificate.
"""

import importlib
import sys

import pytest

from sympy import (
    Add, Dummy, Function, ImmutableMatrix, ImmutableSparseMatrix, Integer,
    Matrix, Mul, Rational, SparseMatrix, Symbol, Tuple,
)


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_structure_profile_rows_columns_and_repeated_extract(cls):
    source = cls([[1, 0, 2], [0, 3, 4], [5, 6, 0]])
    for result, shape, entries in (
        (source.row(-1), (1, 3), {(0, 0): 5, (0, 1): 6}),
        (source.col(-1), (3, 1), {(0, 0): 2, (1, 0): 4}),
        (source.extract([2, 0, 2], [-1, 0, 0]), (3, 3),
         {(0, 1): 5, (0, 2): 5, (1, 0): 2, (1, 1): 1, (1, 2): 1,
          (2, 1): 5, (2, 2): 5}),
        (source.extract(Tuple(2, 0), Tuple(1, 2)), (2, 2),
         {(0, 0): 6, (1, 1): 2}),
    ):
        assert type(result) is cls
        assert result.shape == shape
        assert result.todok() == entries


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_structure_profile_boolean_masks_and_mixed_indices(cls):
    source = cls([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    masked = source.extract([True, False, True], [False, True, True])
    assert type(masked) is cls and masked.shape == (2, 2)
    assert masked.todok() == {(0, 0): 2, (0, 1): 3, (1, 0): 8, (1, 1): 9}
    # Mixed bool/integer lists are integer selectors, not a boolean mask.
    assert source.extract([True, 0], [0]).todok() == {(0, 0): 4, (1, 0): 1}
    assert source.extract([False]*5, [0]).shape == (0, 1)
    assert source.extract([True, False, False, False], [0]).todok() == {(0, 0): 1}
    with pytest.raises(IndexError):
        source.extract([False, False, False, True], [0])


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_structure_profile_bracket_sequences_slices_and_flattening(cls):
    source = cls([[0, 1, 2, 0], [3, 0, 4, 5], [0, 6, 7, 8]])
    assert source[::-1, ::-2].todok() == {
        (0, 0): 8, (0, 1): 6, (1, 0): 5, (2, 1): 1,
    }
    assert source[[-1, 0, -1], [2, 0]].todok() == {
        (0, 0): 7, (1, 0): 2, (2, 0): 7,
    }
    assert type(source[:, Integer(1)]) is cls
    assert source[Integer(-1), Integer(-1)] == 8
    assert source[Integer(-1)] == 8
    assert source[True, False] == 3
    assert source[::-3] == [8, 0, 0, 2]
    assert isinstance(source[1:8:2], list)
    assert source[1:8:2] == [1, 0, 0, 5]
    assert source[1:1, :].shape == (0, 4)
    assert source[:, []].shape == (3, 0)
    with pytest.raises(ValueError):
        source[::0]
    with pytest.raises(ValueError):
        source[:, ::0]


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_structure_profile_ranges_can_cross_negative_indices(cls):
    source = cls([[1, 2], [3, 4], [5, 6]])
    result = source.extract(range(-2, 3), range(-1, 2))
    assert result.shape == (5, 3)
    assert result.todok() == {
        (0, 0): 4, (0, 1): 3, (0, 2): 4,
        (1, 0): 6, (1, 1): 5, (1, 2): 6,
        (2, 0): 2, (2, 1): 1, (2, 2): 2,
        (3, 0): 4, (3, 1): 3, (3, 2): 4,
        (4, 0): 6, (4, 1): 5, (4, 2): 6,
    }
    assert source.extract(range(2, -3, -2), [1]).todok() == {
        (0, 0): 6, (1, 0): 2, (2, 0): 4,
    }
    with pytest.raises(IndexError):
        source.extract(range(-4, 0), [0])


@pytest.mark.parametrize("left_cls", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("right_cls", [SparseMatrix, ImmutableSparseMatrix, Matrix, ImmutableMatrix])
def test_structure_profile_join_class_and_values(left_cls, right_cls):
    left = left_cls([[1, 0], [0, 2]])
    horizontal = left.row_join(right_cls([[3], [4]]))
    vertical = left.col_join(right_cls([[5, 6]]))
    assert type(horizontal) is left_cls and horizontal.shape == (2, 3)
    assert horizontal.todok() == {(0, 0): 1, (0, 2): 3, (1, 1): 2, (1, 2): 4}
    assert type(vertical) is left_cls and vertical.shape == (3, 2)
    assert vertical.todok() == {(0, 0): 1, (1, 1): 2, (2, 0): 5, (2, 1): 6}


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_structure_profile_empty_axes_and_null_left_joins(cls):
    source = cls(0, 4, {})
    assert source[:].__class__ is list and source[:] == []
    assert source[:, 2].shape == (0, 1)
    assert source[:, ::-1].shape == (0, 4)
    assert source.extract([], [2, 2]).shape == (0, 2)
    assert cls(3, 0, {})[1, :].shape == (1, 0)
    assert source.col_join(cls([[1, 2]])).shape == (1, 2)
    assert cls(5, 0, {}).row_join(cls([[1], [2]])).shape == (2, 1)
    with pytest.raises(IndexError):
        source[0]
    with pytest.raises(IndexError):
        cls(3, 0, {})[:, 0]
    with pytest.raises(ValueError):
        cls(2, 2, {}).row_join(cls(3, 1, {}))
    with pytest.raises(ValueError):
        cls(2, 2, {}).col_join(cls(1, 3, {}))


def test_structure_profile_invalid_indices_do_not_truncate():
    source = SparseMatrix([[1, 2], [3, 4]])

    class IntegerConversionOnly:
        def __int__(self):
            return 0

    for invalid in (0.0, 1.0, "0", Rational(1, 2), IntegerConversionOnly()):
        with pytest.raises(IndexError):
            source.extract([invalid], [0])
        with pytest.raises(IndexError):
            source[invalid, 0]
        with pytest.raises(TypeError):
            source[invalid]
    for invalid in (-3, 2):
        with pytest.raises(IndexError):
            source.row(invalid)
    with pytest.raises(TypeError):
        source.extract((i for i in [0]), [0])


def test_assignment_uses_exact_indices_without_publishing_failed_coercions():
    source = SparseMatrix([[1, 2], [3, 4]])
    for invalid in (0.0, "0", Rational(1, 2)):
        with pytest.raises(IndexError):
            source[invalid, 0] = 9
        with pytest.raises(TypeError):
            source[invalid] = 9
    assert source.mutation_generation == 0
    assert source.todok() == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
    source[Integer(-1), Integer(-1)] = 7
    assert source[1, 1] == 7 and source.mutation_generation == 1


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_huge_sparse_structure_never_densifies_or_scans_scalar_coordinates(cls, monkeypatch):
    x = Symbol("structural_sparse_x")
    value = type("StructuralLaw", (Function,), {})(x)
    source = cls(1_000_000, 1_000_000, {(7, 999_999): value, (999_999, 3): x})
    monkeypatch.setattr(SparseMatrix, "to_dense", lambda self: pytest.fail("dense conversion"))
    monkeypatch.setattr(SparseMatrix, "__getitem__", lambda *args: pytest.fail("scalar scan"))
    column = source.col(-1)
    row = source.row(-1)
    reverse = source.extract(range(999_999, -1, -1), range(1_000_000))
    assert column.shape == (1_000_000, 1)
    assert column.todok()[(7, 0)] is value
    assert row.shape == (1, 1_000_000) and row.todok()[(0, 3)] is x
    assert reverse.shape == source.shape
    assert reverse.todok()[(999_992, 999_999)] is value
    assert reverse.todok()[(0, 3)] is x
    joined = source.row_join(cls(1_000_000, 1, {(5, 0): value}))
    assert joined.shape == (1_000_000, 1_000_001)
    assert joined.todok()[(5, 1_000_000)] is value


def test_compressed_selectors_support_dimensions_above_python_len_limit():
    n = sys.maxsize + 11
    value = type("OversizedStructuralLaw", (Function,), {})(Symbol("x"))
    source = SparseMatrix(n, 2, {(n - 1, 1): value})
    result = source.extract(range(n), [1])
    assert result.shape == (n, 1) and result.todok()[(n - 1, 0)] is value
    reversed_result = source[::-1, :]
    assert reversed_result.shape == (n, 2)
    assert reversed_result.todok()[(0, 1)] is value


def test_selection_and_join_keep_held_custom_and_distinct_symbol_owners():
    x = Symbol("sparse_owner")
    first, second = Dummy("same"), Dummy("same")
    law = type("OwnerLaw", (Function,), {})(x)
    other_law = type("OwnerLaw", (Function,), {})(x)
    held = Add(x, x, evaluate=False)
    held_zero = Mul(0, law, evaluate=False)
    source = ImmutableSparseMatrix(2, 3, {
        (0, 0): held, (0, 1): law, (0, 2): first,
        (1, 0): held_zero, (1, 1): other_law, (1, 2): second,
    })
    selected = source[[1, 0, 1], [2, 1, 0]]
    expected = (second, other_law, held_zero, first, law, held,
                second, other_law, held_zero)
    assert all(selected[i] is value for i, value in enumerate(expected))
    assert selected.nnz == 9
    joined = source.row_join(Matrix([[law], [other_law]]))
    assert type(joined) is ImmutableSparseMatrix
    assert joined[0, 3] is law and joined[1, 3] is other_law
    assert joined[0, 2] is first and joined[1, 2] is second
    with pytest.raises(TypeError):
        selected[0, 0] = 9


def test_native_only_structure_never_lifts_unknown_function_owners(monkeypatch):
    from sympy.core import FunctionClass, _native, _native_expr

    effects = []

    def forbidden_getter(self, name):
        effects.append(name)
        raise AssertionError("native-only structure entered a Python function owner")

    collision = FunctionClass("NativeStructuralCollision", (Function,), {
        "__module__": __name__, "__getattribute__": forbidden_getter,
    })
    raw = _native.py_function("NativeStructuralCollision", _native_expr(Symbol("x")))
    source = SparseMatrix(_native.SparseMatrix(2, 2, {(1, 0): raw}))
    dense = Matrix(_native.Matrix(2, 1, [_native.py_integer(0), raw]))
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_wrap", lambda value: pytest.fail("native-only lift"))
    for result in (
        source.row(1), source.col(0), source[::-1, ::-1],
        source.extract([1, 1], [0]), source.row_join(dense),
        source.col_join(SparseMatrix.zeros(1, 2)),
        source.row_join(SparseMatrix(2, 0, {})),
        SparseMatrix(0, 2, {}).col_join(source),
    ):
        assert result._surface is None
        assert raw in tuple(result._native_storage.todok().values())
    with pytest.raises(NotImplementedError):
        source[1:3]
    with pytest.raises(NotImplementedError):
        source.row_join(SparseMatrix([[1], [0]]))
    assert collision.__name__ == "NativeStructuralCollision"
    assert effects == []


@pytest.mark.parametrize("in_place", [False, True])
def test_index_callbacks_cannot_change_original_structural_source(in_place):
    source = SparseMatrix([[1, 2], [3, 4]])

    class MutatingIndex:
        def __index__(self):
            if in_place:
                source._surface[(0, 0)] = Integer(9)
            else:
                source[0, 0] = 9
            return 0

    with pytest.raises(RuntimeError, match="mutated"):
        source[:, MutatingIndex()]
    assert source[0, 0] == 9


def test_source_mutation_during_native_lifting_refuses_the_whole_join(monkeypatch):
    from sympy.core import _native

    module = importlib.import_module("sympy.matrices.sparse")
    source = SparseMatrix.eye(2)
    rhs = SparseMatrix([[1], [2]])
    original = module._wrap
    triggered = []

    def changing_wrap(value):
        if not triggered:
            triggered.append(True)
            source._native_storage = _native.SparseMatrix.zeros(2, 2)
        return original(value)

    monkeypatch.setattr(module, "_wrap", changing_wrap)
    with pytest.raises(RuntimeError, match="mutated"):
        source.row_join(rhs)
    assert source.nnz == 0


def test_native_source_snapshot_shape_mismatch_is_not_reinterpreted():
    source = SparseMatrix.eye(2)
    source._matrix_shape = (3, 2)
    with pytest.raises(ValueError, match="shapes differ"):
        source[:, :]
    forged = SparseMatrix(2, 0, {})
    forged._surface[(0, 0)] = Integer(1)
    with pytest.raises(ValueError, match="stored coordinate"):
        forged.row_join(SparseMatrix.zeros(2, 1))


def test_selector_output_and_flat_slice_limits_precede_publication(monkeypatch):
    source = SparseMatrix([[7]])
    zeros = SparseMatrix.zeros(1, 1)
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_ENTRIES", 3)
    with pytest.raises(ValueError, match="stored-entry limit"):
        source.extract([0, 0], [0, 0])
    # Output storage, not the area of the selected zero rectangle, is bounded.
    assert zeros.extract([0, 0], [0, 0]).shape == (2, 2)
    with pytest.raises(ValueError, match="selector"):
        zeros.extract([0]*4, [0])
    huge = SparseMatrix.zeros(1, 1_000_000)
    with pytest.raises(ValueError, match="flat slice"):
        huge[:]
    assert huge[-2:] == [0, 0]
    assert huge[:, :].shape == (1, 1_000_000)


def test_join_preflights_stored_fill_and_dimension_overflow(monkeypatch):
    source = SparseMatrix([[1, 2]])
    rhs = SparseMatrix([[3, 4]])
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_ENTRIES", 3)
    with pytest.raises(ValueError, match="stored-entry limit"):
        source.row_join(rhs)
    maximal = SparseMatrix(0, 2*sys.maxsize + 1, {})
    with pytest.raises(OverflowError, match="dimension"):
        maximal.row_join(SparseMatrix(0, 1, {}))


def test_join_slice_solve_pipeline_retains_sparse_storage(monkeypatch):
    n = 300
    coefficients = SparseMatrix.eye(n)
    rhs = ImmutableSparseMatrix(n, 1, {(0, 0): Rational(2, 3), (n - 1, 0): -5})
    monkeypatch.setattr(SparseMatrix, "to_dense", lambda self: pytest.fail("dense conversion"))
    augmented = coefficients.row_join(rhs)
    recovered_coefficients = augmented[:, :n]
    recovered_rhs = augmented[:, -1:]
    solved = recovered_coefficients.LUsolve(recovered_rhs)
    assert type(solved) is SparseMatrix
    assert solved.shape == (n, 1) and solved.nnz == 2
    assert solved.todok() == {(0, 0): Rational(2, 3), (n - 1, 0): -5}


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("position, location", [
    (-99, 0), (-2, 0), (-1, 1), (0, 0), (1, 1), (2, 2), (99, 2),
    (Integer(-1), 1),
])
def test_structure_profile_insertions_clamp_exact_positions(cls, horizontal, position, location):
    source = cls([[1, 2], [3, 4]])
    if horizontal:
        other = Matrix([[5], [6]])
        expected = (
            [[5, 1, 2], [6, 3, 4]],
            [[1, 5, 2], [3, 6, 4]],
            [[1, 2, 5], [3, 4, 6]],
        )[location]
        result = source.col_insert(position, other)
    else:
        other = Matrix([[5, 6]])
        expected = (
            [[5, 6], [1, 2], [3, 4]],
            [[1, 2], [5, 6], [3, 4]],
            [[1, 2], [3, 4], [5, 6]],
        )[location]
        result = source.row_insert(position, other)
    assert type(result) is cls
    assert result == cls(expected)
    assert source == cls([[1, 2], [3, 4]])


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 0)])
def test_structure_profile_null_insertions_ignore_position_and_old_shape(cls, shape):
    class UnusedPosition:
        def __index__(self):
            pytest.fail("zero-area insertion inspected its position")

    source = cls(*shape, {})
    other = ImmutableMatrix([[1, 2], [3, 4]])
    for result in (source.row_insert(UnusedPosition(), other),
                   source.col_insert(UnusedPosition(), other)):
        assert type(result) is cls and result.shape == (2, 2)
        assert result.todok() == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}


def test_structure_profile_insertion_requires_strict_integer_positions():
    source = SparseMatrix([[1, 2], [3, 4]])
    row = SparseMatrix([[5, 6]])
    column = SparseMatrix([[5], [6]])
    for position in (True, False, 0.0, "0", Rational(1, 2)):
        with pytest.raises(ValueError):
            source.row_insert(position, row)
        with pytest.raises(ValueError):
            source.col_insert(position, column)
    with pytest.raises(ValueError):
        source.row_insert(0, SparseMatrix([[1]]))
    with pytest.raises(ValueError):
        source.col_insert(0, SparseMatrix([[1]]))


@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix, Matrix, ImmutableMatrix])
def test_structure_profile_stack_empty_single_and_first_operand_class(cls):
    assert type(cls.hstack()) is cls and cls.hstack().shape == (0, 0)
    assert type(cls.vstack()) is cls and cls.vstack().shape == (0, 0)
    for first_cls in (SparseMatrix, ImmutableSparseMatrix, Matrix, ImmutableMatrix):
        column = first_cls([[1], [2]])
        row = first_cls([[1, 2]])
        assert cls.hstack(column) is column
        assert cls.vstack(row) is row
        horizontal = cls.hstack(column, SparseMatrix([[3, 4], [5, 6]]), Matrix([[7], [8]]))
        vertical = cls.vstack(row, SparseMatrix([[3, 4], [5, 6]]), Matrix([[7, 8]]))
        assert type(horizontal) is first_cls and horizontal.shape == (2, 4)
        assert horizontal == first_cls([[1, 3, 4, 7], [2, 5, 6, 8]])
        assert type(vertical) is first_cls and vertical.shape == (4, 2)
        assert vertical == first_cls([[1, 2], [3, 4], [5, 6], [7, 8]])


def test_structure_profile_stack_null_prefix_matches_sequential_joins():
    for cls in (SparseMatrix, ImmutableSparseMatrix, Matrix, ImmutableMatrix):
        horizontal = cls.hstack(cls(3, 0, []), SparseMatrix(0, 0, {}),
                                SparseMatrix([[1], [2]]))
        vertical = cls.vstack(cls(0, 3, []), SparseMatrix(0, 0, {}),
                              SparseMatrix([[1, 2]]))
        assert type(horizontal) is cls and horizontal.shape == (2, 1)
        assert type(vertical) is cls and vertical.shape == (1, 2)
        assert cls.hstack(cls(2, 0, []), SparseMatrix(5, 0, {})).shape == (5, 0)
        assert cls.vstack(cls(0, 2, []), SparseMatrix(0, 5, {})).shape == (0, 5)
        with pytest.raises(ValueError):
            cls.hstack(cls(0, 3, []), SparseMatrix(0, 0, {}), SparseMatrix([[1]]))
        with pytest.raises(ValueError):
            cls.vstack(cls(3, 0, []), SparseMatrix(0, 0, {}), SparseMatrix([[1]]))


def test_huge_sparse_insertions_and_stacks_keep_retained_owners(monkeypatch):
    value = type("BlockAssemblyLaw", (Function,), {})(Symbol("block_x"))
    held = Add(Symbol("block_x"), Symbol("block_x"), evaluate=False)
    n = 1_000_000
    source = ImmutableSparseMatrix(n, n, {(n - 1, n - 1): value})
    column = SparseMatrix(n, 1, {(7, 0): held})
    row = SparseMatrix(1, n, {(0, 7): held})
    monkeypatch.setattr(SparseMatrix, "to_dense", lambda self: pytest.fail("dense conversion"))
    monkeypatch.setattr(SparseMatrix, "__getitem__", lambda *args: pytest.fail("scalar scan"))
    inserted_column = source.col_insert(5, column)
    inserted_row = source.row_insert(-1, row)
    assert type(inserted_column) is ImmutableSparseMatrix
    assert inserted_column.shape == (n, n + 1)
    assert inserted_column.todok()[(7, 5)] is held
    assert inserted_column.todok()[(n - 1, n)] is value
    assert inserted_row.shape == (n + 1, n)
    assert inserted_row.todok()[(n - 1, 7)] is held
    assert inserted_row.todok()[(n, n - 1)] is value
    assembled = Matrix.hstack(column, source, column)
    assert type(assembled) is SparseMatrix
    assert assembled.shape == (n, n + 2)
    assert assembled.todok()[(7, 0)] is held
    assert assembled.todok()[(7, n + 1)] is held
    assert assembled.todok()[(n - 1, n)] is value


def test_native_only_block_assembly_keeps_unlifted_function_owners(monkeypatch):
    from sympy.core import FunctionClass, _native, _native_expr

    collision = FunctionClass("NativeBlockCollision", (Function,), {})
    raw = _native.py_function("NativeBlockCollision", _native_expr(Symbol("block_native_x")))
    sparse = SparseMatrix(_native.SparseMatrix(2, 1, {(1, 0): raw}))
    dense = Matrix(_native.Matrix(2, 1, [_native.py_integer(0), raw]))
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_wrap", lambda value: pytest.fail("native-only block lift"))
    for result in (
        sparse.col_insert(0, sparse),
        SparseMatrix(0, 9, {}).row_insert(object(), dense),
        SparseMatrix.hstack(sparse, dense, SparseMatrix(2, 0, {})),
        SparseMatrix.hstack(dense, sparse, dense),
        SparseMatrix.vstack(sparse, dense, sparse),
    ):
        assert result._surface is None
    assert collision.__name__ == "NativeBlockCollision"


@pytest.mark.parametrize("hook", ["length", "iteration", "coordinate", "class"])
def test_effectful_key_inspection_cannot_precede_original_source_snapshot(hook):
    source = SparseMatrix([[1, 2], [3, 4]])
    touched = []

    def mutate():
        if not touched:
            touched.append(True)
            if hook == "iteration":
                source.__init__([[9, 8, 7]])
            else:
                source._surface[(0, 0)] = Integer(9)

    if hook == "length":
        class Key(tuple):
            def __len__(self):
                mutate()
                return tuple.__len__(self)
        key = Key((0, 0))
    elif hook == "iteration":
        class Key(tuple):
            def __iter__(self):
                mutate()
                return tuple.__iter__(self)
        key = Key((slice(None), 0))
    elif hook == "coordinate":
        class Key(tuple):
            def __getitem__(self, index):
                mutate()
                return tuple.__getitem__(self, index)
        key = Key((0, 0))
    else:
        class Key:
            @property
            def __class__(self):
                mutate()
                return type(self)

            def __index__(self):
                return 0
        key = Key()
    with pytest.raises(RuntimeError, match="mutated"):
        source[key]
    assert touched


def test_builtin_scalar_access_retains_constant_size_snapshot_work(monkeypatch):
    source = SparseMatrix([[1, 2]])
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_structural_sources",
                        lambda *args: pytest.fail("scalar access copied sparse storage"))
    assert source[0, 1] == 2 and source[-1] == 2


def test_assignment_rejects_retained_mutation_by_index_and_value_converter():
    for during_value in (False, True):
        source = SparseMatrix([[1]])

        class Index:
            def __index__(self):
                source._surface[(0, 0)] = Integer(9)
                return 0

        class Value:
            def _sympy_(self):
                source._surface[(0, 0)] = Integer(9)
                return Integer(7)

        with pytest.raises(RuntimeError, match="mutated"):
            if during_value:
                source[0] = Value()
            else:
                source[Index()] = 7
        assert source[0] == 9 and source.mutation_generation == 0


@pytest.mark.parametrize("rhs_target", [False, True])
def test_insertion_position_cannot_change_either_original_operand(rhs_target):
    source = SparseMatrix([[1, 2], [3, 4]])
    other = SparseMatrix([[5, 6]])

    class Position:
        def __index__(self):
            target = other if rhs_target else source
            target._surface[(0, 0)] = Integer(99)
            return 1

    with pytest.raises(RuntimeError, match="mutated"):
        source.row_insert(Position(), other)


def test_block_assembly_preflights_fill_input_work_and_dense_final_storage(monkeypatch):
    module = importlib.import_module("sympy.matrices.sparse")
    source = SparseMatrix.eye(2)
    other = SparseMatrix([[1], [2]])
    with monkeypatch.context() as patch:
        patch.setattr(module, "_MAX_SPARSE_ENTRIES", 3)
        patch.setattr(module, "_wrap", lambda value: pytest.fail("lifting before fill admission"))
        with pytest.raises(ValueError, match="stored-entry limit"):
            source.col_insert(1, other)
        with pytest.raises(ValueError, match="stored-entry limit"):
            SparseMatrix.hstack(source, other)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_MAX_SPARSE_PRODUCTS", 1)
        with pytest.raises(ValueError, match="work limit"):
            SparseMatrix.hstack(SparseMatrix(1, 0, {}), SparseMatrix(1, 0, {}))
    dense = Matrix([[1]])
    huge = SparseMatrix.zeros(1, 1_000_000)
    with pytest.raises(ValueError, match="limit"):
        SparseMatrix.hstack(dense, huge)
    sparse = Matrix.hstack(SparseMatrix([[1]]), huge)
    assert type(sparse) is SparseMatrix and sparse.shape == (1, 1_000_001)


def test_stacks_preserve_overridden_join_dispatch_and_check_later_operands(monkeypatch):
    first = SparseMatrix([[1]])
    second = SparseMatrix([[2]])
    third = SparseMatrix([[3]])
    original = SparseMatrix.row_join
    calls = []

    def observed(left, right):
        calls.append(right)
        return original(left, right)

    monkeypatch.setattr(SparseMatrix, "row_join", observed)
    result = Matrix.hstack(first, second, third)
    assert calls == [second, third]
    assert result.todok() == {(0, 0): 1, (0, 1): 2, (0, 2): 3}

    def mutating(left, right):
        third._surface[(0, 0)] = Integer(99)
        return original(left, right)

    monkeypatch.setattr(SparseMatrix, "row_join", mutating)
    with pytest.raises(RuntimeError, match="mutated"):
        ImmutableSparseMatrix.hstack(first, second, third)


def test_convenience_stack_entry_points_follow_first_operand_class():
    # These convenience functions are repository APIs, not SymPy1.14 exports.
    from sympy import hstack, vstack

    sparse = ImmutableSparseMatrix([[1]])
    dense = Matrix([[2]])
    for result in (hstack(sparse, dense), vstack(sparse, dense),
                   Matrix.hstack(sparse, dense), Matrix.vstack(sparse, dense)):
        assert type(result) is ImmutableSparseMatrix
    assert type(hstack(dense, sparse)) is Matrix
    assert type(vstack(dense, sparse)) is Matrix


def test_stack_preflight_rejects_changed_native_count_before_any_hook(monkeypatch):
    from sympy.core import _native

    left = SparseMatrix([[1]])
    right = SparseMatrix.eye(1)
    calls = []

    def changing_count(native):
        calls.append(native)
        left._surface[(0, 0)] = Integer(9)
        return 1

    monkeypatch.setattr(_native.SparseMatrix, "nnz", property(changing_count))
    with pytest.raises(NotImplementedError, match="native count interfaces"):
        SparseMatrix.hstack(left, right)
    assert calls == [] and left[0, 0] == 1


def test_stack_preflight_does_not_call_replaced_matrix_accessors(monkeypatch):
    left = SparseMatrix([[1]])
    right = SparseMatrix([[2]])
    original = SparseMatrix.__getattribute__
    calls = []

    def changing_access(self, name):
        if self is right and name == "_surface":
            calls.append(name)
            original(left, "_surface")[(0, 0)] = Integer(9)
        return original(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(SparseMatrix, "__getattribute__", changing_access)
        with pytest.raises(NotImplementedError, match="original storage accessors"):
            SparseMatrix.hstack(left, right)
    assert calls == [] and left[0, 0] == 1
