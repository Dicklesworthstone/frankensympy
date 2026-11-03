"""Custom scalar identities survive dense storage, algebra and numerical capture."""

import copy
import pickle

import pytest
import sympy as s

def same_name_values():
    first = type("MatrixLaw", (s.Function,), {"nargs": 1})
    second = type("MatrixLaw", (s.Function,), {"nargs": 1})
    x = s.Symbol("opaque_matrix_x")
    return x, first, second, first(x, evaluate=False), second(x, evaluate=False)


@pytest.mark.parametrize("form", ["nested", "flat", "shape", "callable"])
def test_matrix_constructors_keep_actual_same_named_function_objects(form):
    _, _, _, a, b = same_name_values()
    constructors = {
        "nested": lambda: s.Matrix([[a], [b]]),
        "flat": lambda: s.Matrix([a, b]),
        "shape": lambda: s.Matrix(2, 1, [a, b]),
        "callable": lambda: s.Matrix(2, 1, lambda i, j: (a, b)[i]),
    }
    matrix = constructors[form]()
    assert matrix.shape == (2, 1) and len(matrix) == 2
    assert matrix[0] is a and matrix[-1] is b
    assert matrix.tolist() == [[a], [b]]
    assert matrix != s.Matrix([b, b])
    assert matrix._native_storage is None
    with pytest.raises(NotImplementedError, match="opaque Python"):
        _ = matrix._native


def test_copy_transpose_extract_and_immutable_snapshot_retain_aliases():
    _, _, _, a, b = same_name_values()
    source = [[a, b]]
    matrix = s.Matrix(source)
    source[0][0] = 99
    frozen = matrix.as_immutable()
    for result in (s.Matrix(matrix), matrix.copy(), copy.copy(matrix), frozen,
                   frozen.as_mutable(), matrix.applyfunc(lambda entry: entry)):
        assert result[0, 0] is a and result[0, 1] is b
    assert frozen.copy() is frozen
    assert frozen == s.ImmutableMatrix([[a, b]])
    assert hash(frozen) == hash(s.ImmutableMatrix([[a, b]]))
    assert {frozen: "retained"}[s.ImmutableMatrix([[a, b]])] == "retained"
    assert frozen != s.ImmutableMatrix([[b, b]])
    assert matrix.T[0, 0] is a and matrix.T[1, 0] is b
    assert matrix[:, 1:][0, 0] is b
    assert matrix.extract([0], [1, 0])[0, 1] is a
    assert matrix.reshape(2, 1)[1, 0] is b
    matrix[0, 0] = b
    assert frozen[0, 0] is a and matrix[0, 0] is b
    with pytest.raises(TypeError, match="immutable"):
        frozen[0, 0] = b


@pytest.mark.parametrize("kind", ["integer", "rational", "held_add", "held_product"])
def test_immutable_closed_matrices_hash_identically_after_native_roundtrip(kind):
    from sympy.core import _native, _native_expr

    x = s.Symbol("opaque_matrix_hash_x")
    value = {
        "integer": s.Integer(1),
        "rational": s.Rational(2, 3),
        "held_add": s.Add(x, x, evaluate=False),
        "held_product": s.Mul(2, x, evaluate=False),
    }[kind]
    surface = s.ImmutableMatrix([[value]])
    native = s.ImmutableMatrix(_native.Matrix(1, 1, [_native_expr(value)]))
    assert surface == native
    assert hash(surface) == hash(native)
    assert len({surface, native}) == 1
    assert {surface: "same"}[native] == "same"


def test_storage_transitions_are_atomic_and_restore_closed_native_algorithms():
    _, _, _, a, b = same_name_values()
    matrix = s.Matrix([[1, 0], [0, 2]])
    assert matrix.det() == 2
    matrix[0, 0] = a
    assert matrix.mutation_generation == 1 and matrix._native_storage is None
    frozen = matrix.as_immutable()
    matrix[0, :] = [b, a]
    assert matrix.mutation_generation == 2
    assert matrix[0, 0] is b and matrix[0, 1] is a
    before = tuple(matrix), matrix.shape, matrix.mutation_generation
    with pytest.raises((TypeError, ValueError)):
        matrix[1, 1] = object()
    assert (tuple(matrix), matrix.shape, matrix.mutation_generation) == before
    with pytest.raises(IndexError):
        matrix[5, :] = [1, 2]
    assert (tuple(matrix), matrix.shape, matrix.mutation_generation) == before
    matrix[:2] = [3, 0]
    assert matrix.mutation_generation == 3 and matrix.det() == 6
    assert matrix._native_storage is not None
    assert frozen[0, 0] is a and frozen._native_storage is None


def test_shape_mutations_keep_source_cells_and_zero_area_dimensions():
    _, _, _, a, b = same_name_values()
    matrix = s.Matrix([[a, b], [b, a]])
    matrix.row_swap(0, 1)
    matrix.col_swap(0, 1)
    assert matrix[0, 0] is a and matrix[1, 0] is b
    matrix.row_op(0, lambda value, column: value)
    matrix.col_op(1, lambda value, row: value)
    assert matrix.mutation_generation == 4
    matrix.col_del(1)
    assert matrix.shape == (2, 1) and matrix[1, 0] is b
    matrix.row_del(0)
    assert matrix.shape == (1, 1) and matrix[0, 0] is b
    matrix.col_del(0)
    assert matrix.shape == (1, 0) and len(matrix) == 0
    assert matrix.T.shape == (0, 1)
    assert pickle.loads(pickle.dumps(matrix)).shape == (1, 0)
    matrix.row_del(0)
    assert matrix.shape == (0, 0) and matrix.mutation_generation == 8


def test_elementary_matrix_algebra_preserves_distinct_function_classes():
    _, _, _, a, b = same_name_values()
    matrix = s.Matrix([[a, b], [0, 1]])
    assert (matrix + s.zeros(2))[0, 0] is a
    assert (matrix * s.eye(2))[0, 1] is b
    assert (s.eye(2) @ matrix)[0, 0] is a
    assert (matrix - s.Matrix([[b, a], [0, 1]]))[0, 0] == a - b
    assert (matrix + matrix)[0, 1] == 2*b
    assert (matrix**2)[0, 1] == a*b + b
    assert matrix**0 == s.eye(2)
    assert (matrix*3)[0, 0] == 3*a
    assert (3*matrix)[0, 1] == 3*b
    assert (matrix/2)[0, 0] == a/2
    assert (s.eye(2)*a)[0, 0] is a
    assert matrix.trace() == a + 1
    assert matrix.hadamard(s.ones(2))[0, 1] is b
    assert matrix.kron(s.eye(2))[0, 2] is b
    assert matrix.minor_submatrix(1, 0)[0, 0] is b
    assert s.diag(a, b)[0, 0] is a and s.diag(a, b)[1, 1] is b
    assert bool(s.diag(a, b).is_diagonal)
    assert bool(s.diag(a, b).is_symmetric)
    assert matrix.is_upper and not matrix.is_lower
    with pytest.raises(ValueError, match="Shape mismatch"):
        matrix + s.Matrix([a])
    with pytest.raises(ValueError, match="Shape mismatch"):
        matrix @ s.Matrix([[a]])


@pytest.mark.parametrize("method", ["det", "inv", "rank", "rref", "nullspace", "charpoly",
                                   "QRdecomposition", "pinv", "to_sparse"])
def test_unimplemented_native_matrix_algorithms_refuse_opaque_cells(method):
    _, _, _, a, _ = same_name_values()
    matrix = s.Matrix([[a]])
    with pytest.raises(NotImplementedError, match="opaque Python"):
        getattr(matrix, method)()
    assert matrix[0, 0] is a and matrix.mutation_generation == 0


def test_matrix_capture_never_uses_custom_argument_or_value_getters():
    events = []
    armed = False

    class Guarded(s.Function):
        def __getattribute__(self, name):
            if armed and name in ("args", "_args", "_value", "func"):
                events.append(name)
                raise AssertionError("matrix storage must retain the original object")
            return super().__getattribute__(name)

    x = s.Symbol("opaque_matrix_guarded")
    value = Guarded(x, evaluate=False)
    armed = True
    matrix = s.Matrix([[value]])
    assert matrix.copy()[0, 0] is value
    assert matrix.T[0, 0] is value
    program = s.compile_numeric_hybrid(x, matrix, callbacks={Guarded: lambda x: x + 1})
    assert program(2) == [[3]] and events == []


def test_supervised_custom_jacobian_survives_normal_matrix_arithmetic():
    x = s.Symbol("opaque_matrix_jacobian")
    effects = []

    class Law(s.Function):
        def fdiff(self, argindex=1):
            effects.append(("derivative", argindex))
            return 2*self.args[0]

    matrix = s.Matrix([Law(x)-4, 3*Law(x)+1])
    jacobian = matrix.jacobian([x])
    derivative_effects = tuple(effects)
    assert jacobian == s.Matrix([2*x, 6*x])
    program = s.compile_numeric_hybrid(
        x, (matrix, jacobian), callbacks={Law: lambda value: effects.append(("numeric", value)) or value**2},
    )
    assert tuple(effects) == derivative_effects
    matrix[0, 0] = 999
    assert program(2) == ([[0], [13]], [[4], [12]])
    assert effects[len(derivative_effects):] == [("numeric", 2.0), ("numeric", 2.0)]


def test_distinct_same_named_matrix_callbacks_feed_a_complete_residual_solve():
    x, first, second, a, b = same_name_values()
    calls = []
    residuals = s.Matrix([a-b-2])
    program = s.compile_hybrid_residual_system(
        residuals, x, jacobian=s.Matrix([[1]]),
        callbacks={first: lambda value: calls.append("first") or 2*value,
                   second: lambda value: calls.append("second") or value},
    )
    residuals[0, 0] = 999
    assert program.residuals(3) == [1]
    # Canonical Add places the negative term first; each recorded occurrence
    # must execute once in that retained argument order.
    assert calls == ["second", "first"]
    root = program.solve(0)
    assert float(root[0, 0]) == pytest.approx(2)
    assert program.evidence == "approximate_with_user_callbacks"


def test_mutation_during_traversal_still_rejects_opaque_results():
    _, _, _, a, b = same_name_values()
    matrix = s.Matrix([a, b])

    def mutate(value):
        matrix[1] = a
        return value

    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        matrix.applyfunc(mutate)
    assert matrix[0] is a and matrix[1] is a


def test_mutation_during_result_coercion_cannot_publish_a_traversal():
    _, _, _, a, b = same_name_values()
    matrix = s.Matrix([a, b])

    class Converted:
        def _sympy_(self):
            matrix[0] = b
            return a

    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        matrix.applyfunc(lambda value: Converted())
    assert matrix[0] is b


def test_matrix_shape_preflight_precedes_entry_callbacks():
    effects = []
    with pytest.raises(ValueError, match="limit"):
        s.Matrix(0, 262_145, lambda i, j: effects.append((i, j)))
    with pytest.raises(ValueError, match="limit"):
        s.Matrix(513, 512, lambda i, j: effects.append((i, j)))
    assert effects == []
    for shape in ((0, 3), (4, 0)):
        matrix = s.Matrix(*shape, [])
        assert pickle.loads(pickle.dumps(matrix)).shape == shape
