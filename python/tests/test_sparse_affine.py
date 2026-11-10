"""Complete sparse affine families, shared dense dispatch, and source ownership."""

import pytest

from _native_subprocess import isolated_heap_new_probe
from sympy import (
    Expr, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Rational,
    SparseMatrix, Symbol,
)


MATRIX_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)


def _forbid_dense(*args, **kwargs):
    pytest.fail("sparse affine solve attempted a dense source conversion")


@pytest.mark.parametrize("coefficient_type", MATRIX_CLASSES)
@pytest.mark.parametrize("rhs_type", MATRIX_CLASSES)
def test_family_keeps_coefficient_class_and_row_major_parameter_identity(coefficient_type, rhs_type):
    coefficients = coefficient_type([[1, 2, 3], [2, 4, 6]])
    rhs = rhs_type([[1, 2], [2, 4]])
    solution, parameters, free = coefficients.gauss_jordan_solve(rhs, freevar=True)
    tau0, tau1, tau2, tau3 = (Symbol(f"tau{i}") for i in range(4))
    assert type(solution) is type(parameters) is coefficient_type
    assert parameters == coefficient_type([[tau0, tau1], [tau2, tau3]])
    assert solution == coefficient_type([
        [1 - 2*tau0 - 3*tau2, 2 - 2*tau1 - 3*tau3],
        [tau0, tau1], [tau2, tau3],
    ])
    assert free == [1, 2]
    for row in range(2):
        for col in range(2):
            assert parameters[row, col] is Symbol(f"tau{row*2 + col}")
            assert solution[free[row], col] is parameters[row, col]
    assert parameters[0, 0] is not parameters[0, 1]
    for assignments in ({tau0: 0, tau1: 1, tau2: -2, tau3: 3},
                        {tau0: 4, tau1: -5, tau2: 6, tau3: 7}):
        assert coefficients*solution.subs(assignments) == rhs
    assert coefficients.mutation_generation == rhs.mutation_generation == 0
    assert len(coefficients.gauss_jordan_solve(rhs)) == 2


@pytest.mark.parametrize("coefficient_type", [Matrix, ImmutableMatrix])
def test_dense_affine_family_extends_beyond_the_old_128_column_limit(coefficient_type):
    source = coefficient_type(1, 129, [0]*129)
    solution, parameters, free = source.gauss_jordan_solve(Matrix([0]), freevar=True)
    assert type(solution) is type(parameters) is coefficient_type
    assert solution.shape == parameters.shape == (129, 1)
    assert solution == parameters and free == list(range(129))


@pytest.mark.parametrize("coefficient_type", [SparseMatrix, ImmutableSparseMatrix])
def test_large_sparse_complete_family_does_not_densify(coefficient_type, monkeypatch):
    size = 600
    cells = {(row, row): 1 for row in range(size)}
    cells.update({(row, size): 1 for row in range(size)})
    source = coefficient_type(size, size + 1, cells)
    rhs = SparseMatrix(size, 2, {
        (row, col): (row + 1 if col == 0 else 2*row + 1)
        for row in range(size) for col in range(2)
    })
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    solution, parameters, free = source.gauss_jordan_solve(rhs, freevar=True)
    assert type(solution) is type(parameters) is coefficient_type
    assert solution.shape == (size + 1, 2) and parameters.shape == (1, 2)
    assert free == [size]
    tau0, tau1 = parameters[0, 0], parameters[0, 1]
    assert solution[size, 0] is tau0 and solution[size, 1] is tau1
    for row in range(size):
        assert solution[row, 0] == row + 1 - tau0
        assert solution[row, 1] == 2*row + 1 - tau1
    assert source*solution.subs({tau0: 0, tau1: 0}) == rhs


@pytest.mark.parametrize("coefficient_type", MATRIX_CLASSES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 3)])
def test_zero_shapes_and_zero_rank_keep_every_free_parameter(coefficient_type, shape):
    rows, cols = shape
    source = coefficient_type(rows, cols, [0]*(rows*cols))
    rhs = Matrix(rows, 2, [0]*(rows*2))
    solution, parameters, free = source.gauss_jordan_solve(rhs, freevar=True)
    assert type(solution) is type(parameters) is coefficient_type
    assert solution.shape == parameters.shape == (cols, 2)
    assert solution == parameters and free == list(range(cols))
    for row in range(cols):
        for col in range(2):
            assert parameters[row, col] is Symbol(f"tau{row*2 + col}")


@pytest.mark.parametrize("coefficient_type", MATRIX_CLASSES)
def test_empty_rhs_keeps_the_explicit_profile_boundary(coefficient_type):
    for source in (coefficient_type([[1]]), coefficient_type([[0, 0]])):
        with pytest.raises(ValueError, match="nonempty right-hand side"):
            source.gauss_jordan_solve(Matrix(source.rows, 0, []))
    solution, parameters, free = coefficient_type(3, 0, []).gauss_jordan_solve(
        Matrix(3, 0, []), freevar=True,
    )
    assert solution.shape == parameters.shape == (0, 0) and free == []


@pytest.mark.parametrize("coefficient_type", MATRIX_CLASSES)
def test_any_inconsistent_rhs_prevents_a_partial_family(coefficient_type):
    with pytest.raises(ValueError, match="no solution"):
        coefficient_type([[1, 2], [2, 4]]).gauss_jordan_solve(Matrix([[1, 1], [2, 3]]))
    with pytest.raises(ValueError, match="no solution"):
        coefficient_type(2, 0, []).gauss_jordan_solve(Matrix([0, 1]))


def test_tiny_exact_pivots_remain_rational_through_parameter_rendering():
    tiny = Rational(1, 10**400)
    source = SparseMatrix([[tiny, 0, 1], [0, 2*tiny, 3]])
    solution, parameters, free = source.gauss_jordan_solve(Matrix([1, 2]), freevar=True)
    tau = parameters[0, 0]
    assert solution == SparseMatrix([1/tiny - tau/tiny, 1/tiny - 3*tau/(2*tiny), tau])
    assert free == [2]


@pytest.mark.parametrize("coefficient_type", [Matrix, SparseMatrix])
@pytest.mark.parametrize("source_side", ["coefficients", "rhs"])
def test_both_original_sources_reject_stale_retained_atoms(coefficient_type, source_side):
    from sympy.core import _native

    retained = Integer(17)
    source = coefficient_type([[retained if source_side == "coefficients" else 17, 34]])
    rhs = Matrix([retained if source_side == "rhs" else 51])
    original = retained._value
    try:
        retained._value = _native.py_integer(0)
        with pytest.raises(RuntimeError, match="native snapshot"):
            source.gauss_jordan_solve(rhs)
    finally:
        retained._value = original


def test_native_generation_and_dense_reader_overrides_never_run(monkeypatch):
    from sympy.core import _native

    source, rhs = Matrix([[1, 2]]), Matrix([3])
    calls = []

    def changed(*args):
        calls.append(args)
        raise AssertionError("unadmitted native callback")

    with monkeypatch.context() as context:
        context.setattr(_native.SparseMatrix, "gauss_jordan_solve", changed)
        with pytest.raises(NotImplementedError, match="original native interfaces"):
            source.gauss_jordan_solve(rhs)
    with monkeypatch.context() as context:
        context.setattr(_native.Matrix, "flat", changed)
        with pytest.raises(NotImplementedError, match="original native interfaces"):
            source.gauss_jordan_solve(rhs)
    assert calls == []


@pytest.mark.parametrize("poison", ["native", "assumptions"])
def test_cached_parameter_corruption_refuses_without_silent_repair(poison):
    from sympy.core import _native

    source, rhs = SparseMatrix([[1, 1]]), Matrix([1])
    parameter = Symbol("tau0")
    original_native, original_assumptions = parameter._value, parameter._assumptions
    poisoned_native = _native.py_integer(0)
    try:
        if poison == "native":
            parameter._value = poisoned_native
        else:
            parameter._assumptions = {"positive": True}
        with pytest.raises(RuntimeError, match="cached affine parameter"):
            source.gauss_jordan_solve(rhs)
        if poison == "native":
            assert parameter._value is poisoned_native
        else:
            assert parameter._assumptions == {"positive": True}
    finally:
        parameter._value = original_native
        parameter._assumptions = original_assumptions


def test_conflicting_parameter_registry_owners_are_not_overwritten(monkeypatch):
    from sympy.core import _surface_symbols

    parameter = Symbol("tau0")
    conflicting = object.__new__(Symbol)
    conflicting._value = parameter._value
    conflicting._assumptions = {}
    monkeypatch.setitem(_surface_symbols, ("tau0", None), conflicting)
    with pytest.raises(RuntimeError, match="registries disagree"):
        SparseMatrix([[1, 1]]).gauss_jordan_solve(Matrix([1]))
    assert _surface_symbols[("tau0", None)] is conflicting


@pytest.mark.parametrize("hook", [
    "__new__", "__init__", "__setattr__", "_value", "_assumptions", "__getattribute__",
])
def test_symbol_hooks_refuse_without_invocation(hook, monkeypatch, request):
    if isolated_heap_new_probe(request, hook):
        return
    source, rhs = SparseMatrix([[1, 1]]), Matrix([1])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted Symbol hook")

    replacement = property(changed, changed) if hook.startswith("_") and not hook.startswith("__") else changed
    monkeypatch.setattr(Symbol, hook, replacement)
    with pytest.raises(NotImplementedError, match="original Symbol interfaces"):
        source.gauss_jordan_solve(rhs)
    assert calls == []


def test_native_family_rendering_never_invokes_python_scalar_arithmetic(monkeypatch):
    source, rhs = SparseMatrix([[1, 2]]), Matrix([3])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Python scalar arithmetic entered native family construction")

    with monkeypatch.context() as context:
        for name in ("__add__", "__mul__", "__sub__"):
            context.setattr(Expr, name, changed)
        solution, parameters = source.gauss_jordan_solve(rhs)
        assert calls == []
    tau = parameters[0, 0]
    assert solution == SparseMatrix([3 - 2*tau, tau])


@pytest.mark.parametrize("coefficient_type", MATRIX_CLASSES)
@pytest.mark.parametrize("hook", ["__init__", "_native"])
def test_affine_result_constructor_hooks_cannot_substitute_another_family(
    coefficient_type, hook, monkeypatch,
):
    source, rhs = coefficient_type([[1, 2]]), Matrix([3])
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unadmitted affine result constructor")

    replacement = property(changed, changed) if hook == "_native" else changed
    monkeypatch.setattr(coefficient_type, hook, replacement)
    with pytest.raises(NotImplementedError, match="original result construction interfaces"):
        source.gauss_jordan_solve(rhs)
    assert calls == []


def test_parameter_amplification_refuses_before_dense_conversion(monkeypatch):
    source = SparseMatrix(0, 600, {})
    rhs = SparseMatrix(0, 600, {})
    monkeypatch.setattr(SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="limit|bound"):
        source.gauss_jordan_solve(rhs)


def test_shape_domain_subclasses_and_effectful_freevar_are_explicit():
    source = SparseMatrix([[1, 2]])
    with pytest.raises(ValueError, match="rows"):
        source.gauss_jordan_solve(Matrix([1, 2]))
    with pytest.raises(ValueError, match="exact rational"):
        SparseMatrix([[Symbol("a")]]).gauss_jordan_solve(Matrix([1]))
    with pytest.raises(TypeError, match="built-in Matrix"):
        source.gauss_jordan_solve([1])

    class CustomSparse(SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomSparse([[1, 2]]).gauss_jordan_solve(Matrix([1]))

    class Flag:
        def __bool__(self):
            raise AssertionError("effectful freevar truth test")

    with pytest.raises(TypeError, match="freevar.*bool"):
        source.gauss_jordan_solve(Matrix([1]), freevar=Flag())
