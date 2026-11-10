"""Checked sparse rank/nullspaces and SymPy 1.14 RREF surface behavior."""

import pytest

from sympy import (
    Add, Function, ImmutableSparseMatrix, Integer, Matrix, Rational, SparseMatrix, Symbol,
)


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
def test_rectangular_rank_nullspace_and_columnspace_do_not_densify(matrix_type, monkeypatch):
    # Its 360,600 implicit cells exceed the dense storage limit, whereas
    # elimination and its one-dimensional nullspace need only O(n) entries.
    dimension = 600
    entries = {(row, row): 2 for row in range(dimension)}
    entries.update({(row, dimension): row + 1 for row in range(dimension)})
    source = matrix_type(dimension, dimension + 1, entries)
    before = source.todok()

    def dense_forbidden(*args, **kwargs):
        pytest.fail("sparse reduction attempted a dense source conversion")

    monkeypatch.setattr(SparseMatrix, "to_dense", dense_forbidden)
    assert source.rank() == dimension
    basis = source.nullspace()
    assert len(basis) == 1 and type(basis[0]) is matrix_type
    assert basis[0].shape == (dimension + 1, 1)
    assert basis[0].todok() == {
        **{(row, 0): -Rational(row + 1, 2) for row in range(dimension)},
        (dimension, 0): 1,
    }
    assert (source*basis[0]).nnz == 0
    columns = source.columnspace()
    assert len(columns) == dimension
    for index, column in enumerate(columns):
        assert type(column) is matrix_type and column.shape == (dimension, 1)
        assert column.todok() == {(index, 0): 2}
    # The profile requires a dense *result* for rational RREF. Refuse that
    # requested representation without disabling sparse rank or nullspace.
    with pytest.raises(ValueError, match="entries.*limit"):
        source.rref()
    assert source.todok() == before and source.mutation_generation == 0


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
def test_exact_bases_pivot_order_and_rref_classes_match_profile(matrix_type, monkeypatch):
    source = matrix_type([[0, 2, 1, 3], [2, 4, -1, 5], [4, 8, -2, 10]])
    expected = Matrix([
        [1, 0, Rational(-3, 2), Rational(-1, 2)],
        [0, 1, Rational(1, 2), Rational(3, 2)],
        [0, 0, 0, 0],
    ])
    monkeypatch.setattr(
        SparseMatrix, "to_dense",
        lambda *args, **kwargs: pytest.fail("dense source conversion"),
    )
    reduced, pivots = source.rref()
    assert type(reduced) is Matrix and reduced == expected and pivots == (0, 1)
    assert source.rank() == 2
    assert source.nullspace() == [
        matrix_type([Rational(3, 2), Rational(-1, 2), 1, 0]),
        matrix_type([Rational(1, 2), Rational(-3, 2), 0, 1]),
    ]
    columns = source.columnspace()
    assert columns == [matrix_type([0, 2, 4]), matrix_type([2, 4, 8])]
    assert all(type(column) is matrix_type for column in columns)
    assert source.colspace() == columns
    assert type(source.rref(pivots=False)) is Matrix
    assert source.rref(pivots=False) == expected
    assert source.rref(simplify=True, normalize_last=False) == (expected, (0, 1))
    assert source.rank(simplify=True) == 2
    assert source.nullspace(simplify=True) == source.nullspace()
    assert source.columnspace(simplify=True) == columns


def test_exact_tiny_pivots_and_nullity_never_use_binary64():
    tiny = Rational(1, 10**400)
    source = SparseMatrix([[tiny, 0, 1], [0, 2*tiny, 3]])
    assert source.rank() == 2
    assert source.nullspace() == [SparseMatrix([-1/tiny, -3/(2*tiny), 1])]
    assert source.rref() == (Matrix([[1, 0, 1/tiny], [0, 1, 3/(2*tiny)]]), (0, 1))


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 3)])
def test_zero_area_and_zero_matrices_have_complete_bases(matrix_type, shape):
    source = matrix_type(*shape, {})
    assert source.rank() == 0
    reduced, pivots = source.rref()
    assert type(reduced) is Matrix and reduced.shape == shape and pivots == ()
    assert source.columnspace() == []
    basis = source.nullspace()
    assert len(basis) == shape[1]
    for column, vector in enumerate(basis):
        assert type(vector) is matrix_type and vector.shape == (shape[1], 1)
        assert vector.todok() == {(column, 0): 1}


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
def test_symbolic_fixed_pivots_retain_the_bounded_existing_lane(matrix_type):
    x = Symbol("sparse_parameter")
    source = matrix_type([[1, x, 0], [0, 0, 1]])
    reduced, pivots = source.rref()
    assert type(reduced) is matrix_type and reduced == source and pivots == (0, 2)
    assert source.rank() == 2
    basis = source.nullspace()
    assert basis == [matrix_type([-x, 1, 0])] and type(basis[0]) is matrix_type
    assert source.columnspace() == [matrix_type([1, 0]), matrix_type([0, 1])]


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_uncertain_symbolic_pivots_remain_explicitly_undetermined(method):
    source = SparseMatrix([[Symbol("unknown_pivot"), 0], [0, 1]])
    with pytest.raises(ValueError, match="symbolic pivot"):
        getattr(source, method)()


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_opaque_cells_keep_their_identity_and_do_not_enter_rational_lane(method):
    x = Symbol("x")
    law = type("SparseReductionLaw", (Function,), {})
    value = law(x)
    source = SparseMatrix([[value]])
    with pytest.raises(NotImplementedError, match="opaque Python"):
        getattr(source, method)()
    assert source[0, 0] is value and type(source[0, 0]) is law
    assert source.mutation_generation == 0


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_custom_simplification_callback_refuses_without_invoking_it(method):
    calls = []

    def callback(value):
        calls.append(value)
        raise AssertionError("unsupervised callback")

    with pytest.raises(NotImplementedError, match="callbacks"):
        getattr(SparseMatrix([[1]]), method)(simplify=callback)
    assert calls == []


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace"])
def test_custom_zero_callback_refuses_without_invoking_it(method):
    calls = []

    def callback(value):
        calls.append(value)
        return True

    with pytest.raises(NotImplementedError, match="callbacks"):
        getattr(SparseMatrix([[1]]), method)(iszerofunc=callback)
    assert calls == []


@pytest.mark.parametrize("option", ["pivots", "normalize_last"])
def test_rref_does_not_coerce_effectful_option_objects(option):
    class Effectful:
        def __bool__(self):
            pytest.fail("row reduction invoked an option hook")

    with pytest.raises(TypeError, match="bool"):
        SparseMatrix([[1]]).rref(**{option: Effectful()})


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_native_resource_failures_do_not_retry_a_dense_lane(method, monkeypatch):
    source = SparseMatrix(8193, 1, {(0, 0): 1})
    monkeypatch.setattr(
        SparseMatrix, "to_dense",
        lambda *args, **kwargs: pytest.fail("resource failure retried with dense storage"),
    )
    with pytest.raises(ValueError, match="bound|limit|8192"):
        getattr(source, method)()


def test_native_sparse_reduction_entrypoints_return_sparse_checked_objects():
    source = SparseMatrix([[0, 2, 4], [3, 6, 9]])
    native, pivots = source._native.rref()
    assert SparseMatrix(native) == SparseMatrix([[1, 0, -1], [0, 1, 2]])
    assert tuple(pivots) == (0, 1) and source._native.rank() == 2
    assert [SparseMatrix(vector) for vector in source._native.nullspace()] == [
        SparseMatrix([1, -2, 1]),
    ]
    assert [SparseMatrix(vector) for vector in source._native.columnspace()] == [
        SparseMatrix([0, 3]), SparseMatrix([2, 6]),
    ]


def test_mutation_replaces_the_source_snapshot_before_the_next_reduction():
    source = SparseMatrix([[1, 2], [2, 4]])
    frozen = source.as_immutable()
    assert source.rank() == 1
    source[1, 1] = 5
    assert source.rank() == 2 and source.nullspace() == []
    assert frozen.rank() == 1 and frozen.nullspace() == [ImmutableSparseMatrix([-2, 1])]


def test_changed_retained_numeric_atoms_cannot_reuse_a_stale_native_snapshot():
    from sympy.core import _native

    atom = Integer(17)
    source = SparseMatrix([[atom]])
    original = atom._value
    try:
        atom._value = _native.py_integer(0)
        with pytest.raises(RuntimeError, match="native snapshot"):
            source.rank()
    finally:
        atom._value = original
    assert source.rank() == 1


def test_changed_retained_cells_cannot_turn_held_sources_into_checked_numbers():
    source = SparseMatrix([[2]])
    source._surface[(0, 0)] = Add(1, 1, evaluate=False)
    with pytest.raises(NotImplementedError, match="literal exact rational"):
        source.rank()


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_exact_atom_cannot_shadow_its_native_value_slot(method, monkeypatch):
    from sympy.core import _native

    source = SparseMatrix([[Integer(17)]])
    calls = []

    def shadow_value(self):
        calls.append("shadowed atom")
        return _native.py_integer(0)

    monkeypatch.setattr(Integer, "_value", property(shadow_value))
    with pytest.raises(NotImplementedError, match="original scalar accessors"):
        getattr(source, method)()
    assert calls == []


@pytest.mark.parametrize("method", ["rref", "rank", "nullspace", "columnspace"])
def test_immutable_class_cannot_shadow_parent_matrix_storage(method, monkeypatch):
    source = ImmutableSparseMatrix([[17]])
    calls = []

    def shadow_surface(self):
        calls.append("shadowed matrix")
        return {(0, 0): Integer(0)}

    monkeypatch.setattr(ImmutableSparseMatrix, "_surface", property(shadow_surface))
    with pytest.raises(NotImplementedError, match="original matrix accessors"):
        getattr(source, method)()
    assert calls == []


@pytest.mark.parametrize("accessor", ["__getattribute__", "__getattr__"])
def test_numeric_attribute_hooks_do_not_run_during_source_admission(accessor, monkeypatch):
    source = SparseMatrix([[Integer(17)]])
    calls = []

    def forbidden(self, name):
        calls.append(name)
        raise AssertionError("effectful numeric accessor")

    monkeypatch.setattr(Integer, accessor, forbidden, raising=False)
    with pytest.raises(NotImplementedError, match="original scalar accessors"):
        source.rank()
    assert calls == []


@pytest.mark.parametrize("matrix_type", [SparseMatrix, ImmutableSparseMatrix])
def test_matrix_attribute_hooks_do_not_run_during_source_admission(matrix_type, monkeypatch):
    source = matrix_type([[17]])
    calls = []

    def forbidden_storage(self, name):
        if name in ("_surface", "_native_storage", "_matrix_shape", "_mutation_generation"):
            calls.append(name)
            raise AssertionError("effectful matrix accessor")
        return object.__getattribute__(self, name)

    monkeypatch.setattr(matrix_type, "__getattribute__", forbidden_storage)
    with pytest.raises(NotImplementedError, match="original matrix accessors"):
        source.rank()
    assert calls == []


@pytest.mark.parametrize("comparison", ["__eq__", "__ne__"])
def test_native_comparison_overrides_cannot_accept_a_stale_scalar(comparison, monkeypatch):
    from sympy.core import _native

    atom = Integer(17)
    source = SparseMatrix([[atom]])
    original = atom._value
    calls = []

    def comparison_override(self, other):
        calls.append(comparison)
        return comparison == "__eq__"

    try:
        atom._value = _native.py_integer(0)
        monkeypatch.setattr(_native.Expr, comparison, comparison_override)
        with pytest.raises(RuntimeError, match="native snapshot"):
            source.rank()
    finally:
        atom._value = original
    assert calls == []


@pytest.mark.parametrize("member", [
    "shape", "is_exact_rational", "todok", "to_dense", "rref", "rank", "nullspace", "columnspace",
])
def test_native_reader_and_algorithm_overrides_refuse_without_invocation(member, monkeypatch):
    from sympy.core import _native

    source = SparseMatrix([[Integer(17)]])
    calls = []

    def forbidden(*args):
        calls.append(member)
        raise AssertionError("native interface override")

    replacement = property(forbidden) if member in ("shape", "is_exact_rational") else forbidden
    monkeypatch.setattr(_native.SparseMatrix, member, replacement)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        source.rank()
    assert calls == []


def test_matrix_subclasses_keep_the_existing_surface_dispatch():
    calls = []

    class SparseChild(SparseMatrix):
        def to_dense(self):
            calls.append("dense")
            return super().to_dense()

    source = SparseChild([[1, 2]])
    reduced, pivots = source.rref()
    assert type(reduced) is SparseChild and pivots == (0,)
    assert source.rank() == 1
    basis = source.nullspace()
    assert type(basis[0]) is SparseChild and basis[0] == SparseChild([-2, 1])
    assert calls == ["dense", "dense", "dense"]


@pytest.mark.parametrize("method", ["rref", "nullspace", "columnspace"])
@pytest.mark.parametrize("hook", ["__init__", "__new__", "__setattr__", "_native"])
def test_checked_results_refuse_changed_construction_hooks(method, hook, monkeypatch):
    source = SparseMatrix([[1, 2]])
    result_type = Matrix if method == "rref" else SparseMatrix
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted result construction hook")

    replacement = property(changed, changed) if hook == "_native" else changed
    monkeypatch.setattr(result_type, hook, replacement)
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        getattr(source, method)()
    assert calls == []


@pytest.mark.parametrize("method", ["nullspace", "columnspace"])
def test_checked_bases_do_not_dispatch_through_instance_new(method, monkeypatch):
    source = SparseMatrix([[1, 2]])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted instance result constructor")

    monkeypatch.setattr(source, "_new", changed)
    result = getattr(source, method)()
    expected = [SparseMatrix([-2, 1])] if method == "nullspace" else [SparseMatrix([1])]
    assert result == expected and all(type(vector) is SparseMatrix for vector in result)
    assert calls == []


def test_rref_refuses_changed_dense_result_storage_descriptors(monkeypatch):
    source = SparseMatrix([[1, 2]])
    calls = []

    def changed(*args):
        calls.append(args)
        raise AssertionError("unadmitted result storage hook")

    monkeypatch.setattr(Matrix, "_surface", property(changed))
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        source.rref()
    assert calls == []
