"""Sparse cells compose with the real callback and numerical program paths.

These are public-path regressions for the first Jacobian campaign. They require
the actual fsym_python extension; an upstream or mocked native implementation is
not a substitute for execution of these tests.
"""

import copy

import pytest
import sympy as s


def _distinct_laws():
    first = type("SparseCampaignLaw", (s.Function,), {"nargs": 1})
    second = type("SparseCampaignLaw", (s.Function,), {"nargs": 1})
    x = s.Symbol("sparse_campaign_x")
    return x, first, second, first(x, evaluate=False), second(x, evaluate=False)


@pytest.mark.parametrize("form", ["mapping", "nested", "flat", "callable", "dense"])
def test_sparse_construction_preserves_distinct_same_named_python_classes(form):
    _, first, second, a, b = _distinct_laws()
    constructors = {
        "mapping": lambda: s.SparseMatrix(2, 2, {(0, 0): a, (1, 1): b}),
        "nested": lambda: s.SparseMatrix([[a, 0], [0, b]]),
        "flat": lambda: s.SparseMatrix(2, 2, [a, 0, 0, b]),
        "callable": lambda: s.SparseMatrix(2, 2, lambda i, j: (a, b)[i] if i == j else 0),
        "dense": lambda: s.Matrix([[a, 0], [0, b]]).to_sparse(),
    }
    matrix = constructors[form]()
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] is a and matrix[1, 1] is b
    assert type(matrix[0, 0]) is first and type(matrix[1, 1]) is second
    assert matrix.todok() == {(0, 0): a, (1, 1): b}
    assert matrix != s.SparseMatrix([[a, 0], [0, a]])
    dense = matrix.to_dense()
    assert dense[0, 0] is a and dense[1, 1] is b


def test_sparse_storage_and_capture_never_use_custom_getters_or_printers():
    events = []
    armed = False

    class Guarded(s.Function):
        def __getattribute__(self, name):
            if armed and name in ("args", "_args", "_value", "func"):
                events.append(name)
                raise AssertionError("sparse storage must retain the original Python object")
            return super().__getattribute__(name)

        def __str__(self):
            events.append("printer")
            raise AssertionError("printed text is not a sparse zero test")

    x = s.Symbol("sparse_campaign_guarded")
    value = Guarded(x, evaluate=False)
    armed = True
    matrix = s.SparseMatrix(2, 2, {(0, 1): value})
    assert matrix.copy()[0, 1] is value
    assert matrix.T[1, 0] is value
    assert matrix.as_immutable()[0, 1] is value
    program = s.compile_numeric_hybrid(x, matrix, callbacks={Guarded: lambda arg: arg + 1})
    assert program(2) == [[0, 3], [0, 0]]
    assert events == []


def test_sparse_snapshots_remain_independent_of_source_dictionary_and_mutation():
    _, _, _, a, b = _distinct_laws()
    cells = {(0, 1): a, (1, 0): b}
    matrix = s.SparseMatrix(2, 2, cells)
    frozen = matrix.as_immutable()
    mutable = matrix.as_mutable()
    copied = copy.copy(matrix)
    cells[(0, 1)] = b
    assert type(frozen) is s.ImmutableSparseMatrix
    assert type(mutable) is s.SparseMatrix and mutable is not matrix
    assert copied is not matrix
    assert frozen.as_immutable() is frozen
    assert frozen.copy() is frozen
    for saved in (frozen, mutable, copied):
        assert saved[0, 1] is a and saved[1, 0] is b
    matrix[0, 1] = b
    assert matrix[0, 1] is b and frozen[0, 1] is a
    mutable[1, 0] = a
    assert frozen[1, 0] is b and matrix[1, 0] is b
    with pytest.raises(TypeError, match="immutable"):
        frozen[0, 1] = b
    with pytest.raises(TypeError):
        hash(matrix)


@pytest.mark.parametrize("opaque", [False, True])
def test_equal_dense_and_sparse_immutable_snapshots_share_hashes(opaque):
    _, _, _, a, b = _distinct_laws()
    cells = [[a, 0], [0, b]] if opaque else [[s.Rational(2, 3), 0], [0, 7]]
    sparse = s.SparseMatrix(cells).as_immutable()
    dense = s.ImmutableMatrix(cells)
    assert sparse == dense and dense == sparse
    assert hash(sparse) == hash(dense)
    assert len({sparse, dense}) == 1
    assert {sparse: "captured"}[dense] == "captured"


def test_sparse_mutation_is_atomic_and_restores_native_exact_solves():
    _, _, _, a, b = _distinct_laws()
    matrix = s.SparseMatrix([[2, 0], [0, 3]])
    matrix[0, 0] = a
    generation = matrix.mutation_generation
    snapshot = matrix.as_immutable()
    with pytest.raises((TypeError, ValueError)):
        matrix[0, 0] = object()
    with pytest.raises(IndexError):
        matrix[2, 0] = b
    assert matrix.mutation_generation == generation
    assert matrix[0, 0] is a and matrix[1, 1] == 3
    matrix[0, 0] = 4
    assert matrix.mutation_generation == generation + 1
    assert matrix.det() == 12
    assert matrix.solve(s.SparseMatrix([8, 9])) == s.SparseMatrix([2, 3])
    assert snapshot[0, 0] is a


@pytest.mark.parametrize("immutable", [False, True])
def test_closed_sparse_numeric_program_captures_cells_before_later_mutation(immutable):
    x, y = s.symbols("sparse_campaign_closed_x sparse_campaign_closed_y")
    source = s.SparseMatrix([[x, 0], [1, y]])
    captured = source.as_immutable() if immutable else source
    program = s.compile_numeric((x, y), captured)
    source[0, 0], source[1, 1] = 99, 88
    assert program(3, 4) == [[3.0, 0.0], [1.0, 4.0]]
    assert program.native_program.output_count == 4


@pytest.mark.parametrize("immutable", [False, True])
def test_opaque_sparse_sources_cannot_enter_closed_native_compilers(immutable):
    x, _, _, a, _ = _distinct_laws()
    source = s.SparseMatrix([a])
    if immutable:
        source = source.as_immutable()
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        s.compile_numeric(x, source)
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        s.compile_residual_system(source, x)
    assert source[0] is a


@pytest.mark.parametrize("immutable", [False, True])
def test_same_named_sparse_callbacks_keep_distinct_classes_and_occurrences(immutable):
    x, first, second, a, b = _distinct_laws()
    seen = []
    source = s.SparseMatrix([[a, 0, b], [b, a, 0]])
    captured = source.as_immutable() if immutable else source
    program = s.compile_numeric_hybrid(
        x, captured,
        callbacks={first: lambda value: seen.append("first") or 2*value,
                   second: lambda value: seen.append("second") or value+1},
    )
    source[0, 0] = 999
    assert seen == []
    assert program(3) == [[6, 0, 4], [4, 6, 0]]
    assert seen == ["first", "second", "second", "first"]
    assert program.evidence == "approximate_with_user_callbacks"


@pytest.mark.parametrize("immutable", [False, True])
def test_sparse_symbolic_jacobian_feeds_one_captured_supervised_residual_solve(immutable):
    x, y = s.symbols("sparse_campaign_solve_x sparse_campaign_solve_y")
    derivative_calls, numerical_calls = [], []

    class Cubic(s.Function):
        def fdiff(self, argindex=1):
            derivative_calls.append(argindex)
            return 3*self.args[0]**2

    source = s.SparseMatrix([Cubic(x, evaluate=False)+y-9, x-y-1])
    residuals = source.as_immutable() if immutable else source
    jacobian = residuals.jacobian([x, y])
    expected_type = s.ImmutableSparseMatrix if immutable else s.SparseMatrix
    assert type(jacobian) is expected_type
    assert jacobian == s.SparseMatrix([[3*x**2, 1], [1, -1]])
    hook_count = len(derivative_calls)
    assert hook_count > 0
    program = s.compile_hybrid_residual_system(
        residuals, (x, y), jacobian=jacobian,
        callbacks={Cubic: lambda value: numerical_calls.append(value) or value**3},
    )
    source[0, 0] = 999
    if not immutable:
        jacobian[0, 0] = 999
    assert len(derivative_calls) == hook_count and numerical_calls == []
    assert program.evaluate(2, 1) == ([0, 0], [[12, 1], [1, -1]])
    ledger = s.EvaluationBudget(max_steps=1 << 30, max_memory_bytes=1 << 30,
                                max_allocations=1 << 20)
    root = program.solve([1.5, 0.5], budget=ledger)
    values = [float(root[i, 0]) for i in range(2)]
    assert values == pytest.approx([2, 1], abs=1e-12)
    assert abs(values[0]**3+values[1]-9) <= 1e-12
    assert abs(values[0]-values[1]-1) <= 1e-12
    assert ledger.remaining_steps < 1 << 30
    assert len(derivative_calls) == hook_count
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.jacobian_evidence == "user_supplied"


def test_immutable_sparse_vectors_work_in_the_closed_residual_solver():
    x, y = s.symbols("sparse_campaign_native_x sparse_campaign_native_y")
    residuals = s.SparseMatrix([x+y-3, 2*x-y]).as_immutable()
    program = s.compile_residual_system(residuals, (x, y))
    assert program.residuals(1, 2) == [0, 0]
    root = s.nsolve(residuals, (x, y), (0, 0))
    assert [float(root[index, 0]) for index in range(2)] == pytest.approx([1, 2])


def test_jacobian_rejects_a_source_mutated_by_a_custom_derivative_hook():
    x = s.Symbol("sparse_campaign_mutating_diff")
    source = None

    class Mutating(s.Function):
        def fdiff(self, argindex=1):
            source[1, 0] = 99
            return 1

    source = s.SparseMatrix([Mutating(x, evaluate=False), x])
    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        source.jacobian([x])
    assert source[1, 0] == 99


@pytest.mark.parametrize("immutable", [False, True])
def test_large_sparse_jacobian_differentiates_stored_rows_without_dense_access(immutable, monkeypatch):
    x, y = s.symbols("sparse_campaign_large_x sparse_campaign_large_y")
    effects = []

    class Law(s.Function):
        def fdiff(self, argindex=1):
            effects.append(argindex)
            return 2*self.args[0]

    dimension = 1_000_000
    source = s.SparseMatrix(dimension, 1, {(9, 0): x**2,
                                         (dimension-1, 0): Law(y, evaluate=False)})
    if immutable:
        source = source.as_immutable()

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse Jacobians must not materialize or scan implicit residuals")

    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbidden)
    monkeypatch.setattr(s.SparseMatrix, "__getitem__", forbidden)
    jacobian = source.jacobian([x, y])
    assert jacobian.shape == (dimension, 2)
    assert jacobian.todok() == {(9, 0): 2*x, (dimension-1, 1): 2*y}
    assert type(jacobian) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)
    assert effects


@pytest.mark.parametrize("container", ["dense", "sparse"])
def test_jacobian_rejects_coordinate_matrix_mutation_during_a_derivative_hook(container):
    x, y, z = s.symbols("sparse_campaign_coord_x sparse_campaign_coord_y sparse_campaign_coord_z")
    coordinates = s.Matrix([x, y]) if container == "dense" else s.SparseMatrix([x, y])

    class MutatingCoordinates(s.Function):
        def fdiff(self, argindex=1):
            coordinates[1, 0] = z
            return 1

    source = s.SparseMatrix([MutatingCoordinates(x, evaluate=False)+y])
    with pytest.raises(RuntimeError, match="mutated"):
        source.jacobian(coordinates)
    assert coordinates[1, 0] is z


def test_jacobian_captures_plain_list_coordinate_order_before_derivative_hooks():
    x, y, z = s.symbols("sparse_campaign_list_x sparse_campaign_list_y sparse_campaign_list_z")
    coordinates = [x, y]

    class MutatingList(s.Function):
        def fdiff(self, argindex=1):
            coordinates[1] = z
            return 1

    source = s.SparseMatrix([MutatingList(x, evaluate=False)+y])
    result = source.jacobian(coordinates)
    assert coordinates[1] is z
    assert result == s.SparseMatrix([[1, 1]])


def test_applyfunc_rejects_mutation_during_callback_and_result_conversion():
    _, _, _, a, b = _distinct_laws()
    source = s.SparseMatrix([a, b])

    def mutate(value):
        source[1, 0] = a
        return value

    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        source.applyfunc(mutate)
    assert source[1, 0] is a

    class Converted:
        def _sympy_(self):
            source[0, 0] = b
            return a

    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        source.applyfunc(lambda value: Converted())
    assert source[0, 0] is b


@pytest.mark.parametrize("immutable", [False, True])
def test_sparse_applyfunc_visits_only_stored_cells_in_original_order(immutable, monkeypatch):
    # Checked in an isolated SymPy 1.14.0 process: even a nonzero f(0) does
    # not populate implicit sparse zeros, and custom calls follow stored order.
    _, _, _, a, b = _distinct_laws()
    source = s.SparseMatrix(1_000_000, 1_000_000, {(15, 23): b, (9, 999_999): a})
    if immutable:
        source = source.as_immutable()
    calls = []

    def visit(value):
        calls.append(value)
        return s.Integer(7)

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse callback traversal cannot enumerate implicit zeros")

    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbidden)
    monkeypatch.setattr(s.SparseMatrix, "__getitem__", forbidden)
    result = source.applyfunc(visit)
    assert len(calls) == 2 and calls[0] is b and calls[1] is a
    assert result.todok() == {(15, 23): 7, (9, 999_999): 7}
    assert type(result) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)


@pytest.mark.parametrize("immutable", [False, True])
def test_sparse_held_zero_cells_preserve_original_numeric_domain_obligations(immutable):
    x = s.Symbol("sparse_campaign_held_domain")
    held = s.Mul(0, s.log(x), evaluate=False)
    source = s.SparseMatrix(1, 1, {(0, 0): held})
    if immutable:
        source = source.as_immutable()
    assert source[0, 0] is held
    assert source.todok()[(0, 0)] is held
    for program in (s.compile_numeric(x, source),
                    s.compile_numeric_hybrid(x, source, callbacks={})):
        assert program(1) == [[0]]
        with pytest.raises(ValueError):
            program(-1)


def test_huge_opaque_sparse_snapshots_and_coordinate_operations_never_densify(monkeypatch):
    _, _, _, a, b = _distinct_laws()
    dimension = 1_000_000
    matrix = s.SparseMatrix(dimension, dimension, {(9, dimension-1): a, (15, 23): b})

    def forbidden(*args, **kwargs):
        raise AssertionError("stored-cell operations must not scan or allocate the dense shape")

    with monkeypatch.context() as patcher:
        patcher.setattr(s.SparseMatrix, "to_dense", forbidden)
        patcher.setattr(s.SparseMatrix, "__getitem__", forbidden)
        copied = matrix.copy()
        frozen = matrix.as_immutable()
        transposed = matrix.T
        assert frozen.todok() == {(9, dimension-1): a, (15, 23): b}
        assert copied.todok() == frozen.todok()
        assert transposed.todok() == {(dimension-1, 9): a, (23, 15): b}
        assert hash(frozen) == hash(frozen.copy())
    matrix[9, dimension-1] = b
    assert frozen[9, dimension-1] is a and matrix[9, dimension-1] is b


def test_huge_sparse_numeric_output_refuses_before_callback_or_dense_expansion(monkeypatch):
    x, first, _, a, _ = _distinct_laws()
    source = s.SparseMatrix(1_000_000, 1_000_000, {(1, 1): a})
    callbacks = []

    def forbidden(*args, **kwargs):
        raise AssertionError("output admission must precede dense matrix access")

    monkeypatch.setattr(s.SparseMatrix, "__getitem__", forbidden)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbidden)
    with pytest.raises(ValueError, match="output limit"):
        s.compile_numeric(x, source)
    with pytest.raises(ValueError, match="output limit"):
        s.compile_numeric_hybrid(x, source,
                                 callbacks={first: lambda value: callbacks.append(value) or value})
    assert callbacks == []


def test_custom_sparse_subclasses_cannot_inherit_a_native_numeric_capability():
    events = []

    class CustomSparse(s.SparseMatrix):
        def __getitem__(self, key):
            events.append(key)
            raise AssertionError("an unadmitted matrix subclass must not be traversed")

    x = s.Symbol("sparse_campaign_custom_container")
    source = CustomSparse([[x]])
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        s.compile_numeric(x, source)
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        s.compile_numeric_hybrid(x, source, callbacks={})
    assert events == []


@pytest.mark.parametrize("direction", ["sparse_to_dense", "dense_to_sparse"])
def test_native_only_matrix_conversion_cannot_invent_custom_classes_from_names(direction):
    from sympy.core import FunctionClass, _native, _native_expr

    effects = []

    def setter(self, name, value):
        effects.append(name)
        object.__setattr__(self, name, value)

    custom = FunctionClass("SparseNativeOnlyLaw", (s.Function,), {
        "__module__": __name__, "__setattr__": setter,
    })
    x = s.Symbol("sparse_campaign_native_only")
    raw = _native.py_function("SparseNativeOnlyLaw", _native_expr(x))
    if direction == "sparse_to_dense":
        source = s.SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): raw}))
        convert = source.to_dense
    else:
        source = s.Matrix(_native.Matrix(1, 1, [raw]))
        convert = source.to_sparse
    effects.clear()
    converted = convert()
    assert effects == []
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        s.compile_numeric(x, converted)
    assert custom.__name__ == "SparseNativeOnlyLaw"
    assert effects == []


def test_sparse_jacobian_callback_order_is_logical_row_order():
    x = s.Symbol("sparse_campaign_derivative_order")
    events = []

    class First(s.Function):
        def fdiff(self, argindex=1):
            events.append("first")
            return 1

    class Second(s.Function):
        def fdiff(self, argindex=1):
            events.append("second")
            return 2

    # The original coordinate insertion order is intentionally reversed.
    source = s.SparseMatrix(10, 1, {(9, 0): Second(x, evaluate=False),
                                   (1, 0): First(x, evaluate=False)})
    result = source.jacobian([x])
    assert result.todok() == {(1, 0): 1, (9, 0): 2}
    assert events == ["first", "second"]


@pytest.mark.parametrize("immutable", [False, True])
def test_sparse_substitution_preserves_implicit_zeros_and_stored_held_cells(immutable, monkeypatch):
    # Pinned SymPy MatrixBase.subs calls SparseRepMatrix.applyfunc, which
    # traverses only stored entries. Implicit zero cells are not substituted.
    x, y = s.symbols("sparse_campaign_subs_x sparse_campaign_subs_y")
    held = s.Add(x, x, evaluate=False)
    source = s.SparseMatrix(1_000_000, 1_000_000, {(9, 1): held})
    if immutable:
        source = source.as_immutable()

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse substitution must not visit implicit zeros")

    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbidden)
    monkeypatch.setattr(s.SparseMatrix, "__getitem__", forbidden)
    result = source.subs(x, y)
    assert result.todok() == {(9, 1): held.subs(x, y)}
    assert type(result) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)
    assert source.todok()[(9, 1)] is held
    empty = s.SparseMatrix.zeros(1_000_000, 1_000_000)
    assert empty.subs(0, 1).todok() == {}


@pytest.mark.parametrize("immutable", [False, True])
def test_sparse_applyfunc_groups_interleaved_coordinates_by_first_seen_row(immutable):
    # Pinned SparseRepMatrix -> DomainMatrix -> SDM storage preserves first-seen
    # rows, then first-seen columns within each row when exposing todok().
    x, _, _, a, b = _distinct_laws()
    source = s.SparseMatrix(3, 2, {(2, 0): a, (1, 0): b, (2, 1): x})
    if immutable:
        source = source.as_immutable()
    seen = []
    result = source.applyfunc(lambda value: seen.append(value) or value)
    assert len(seen) == 3
    assert seen[0] is a and seen[1] is x and seen[2] is b
    assert result.todok() == {(2, 0): a, (2, 1): x, (1, 0): b}
    assert type(result) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)


@pytest.mark.parametrize("target", ["source", "dense_coordinates", "sparse_coordinates"])
def test_sparse_jacobian_rejects_reinitialization_that_resets_generation(target):
    x, y, z = s.symbols("sparse_campaign_reinit_x sparse_campaign_reinit_y sparse_campaign_reinit_z")
    coordinates = s.SparseMatrix([x, y]) if target == "sparse_coordinates" else s.Matrix([x, y])
    source = None

    class Reinitializing(s.Function):
        def fdiff(self, argindex=1):
            if target == "source":
                source.__init__([99, y])
            else:
                coordinates.__init__([x, z])
            return 1

    source = s.SparseMatrix([Reinitializing(x, evaluate=False)+y, x])
    with pytest.raises(RuntimeError, match="mutated"):
        source.jacobian(coordinates)
    if target == "source":
        assert source[0, 0] == 99
    else:
        assert coordinates[1, 0] is z


@pytest.mark.parametrize("immutable", [False, True])
def test_symbolic_left_scalar_defers_to_retained_sparse_multiplication(immutable):
    _, _, _, a, b = _distinct_laws()
    factor = s.Symbol("sparse_campaign_left_factor")
    source = s.SparseMatrix(2, 3, {(0, 1): a, (1, 2): b})
    if immutable:
        source = source.as_immutable()
    result = factor*source
    assert type(result) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)
    assert result.shape == (2, 3)
    assert result.todok() == {(0, 1): factor*a, (1, 2): factor*b}
    assert source[0, 1] is a and source[1, 2] is b


@pytest.mark.parametrize("immutable", [False, True])
def test_custom_function_left_scalar_defers_to_retained_sparse_multiplication(immutable):
    x, _, _, a, b = _distinct_laws()
    source = s.SparseMatrix(2, 2, {(0, 1): b, (1, 0): x})
    if immutable:
        source = source.as_immutable()
    result = a*source
    assert type(result) is (s.ImmutableSparseMatrix if immutable else s.SparseMatrix)
    assert result.shape == (2, 2)
    assert result.todok() == {(0, 1): a*b, (1, 0): a*x}
    assert source[0, 1] is b and source[1, 0] is x
