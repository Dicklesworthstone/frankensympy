"""Matrix-form linsolve and solve_linear_system over checked sparse families."""

from unittest.mock import patch

import pytest

from sympy import (
    Dummy, EmptySet, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix,
    Rational, SparseMatrix, Symbol, Tuple, FiniteSet, Add, Mul,
    linsolve, solve_linear_system, symbols,
)
from sympy.core import Basic, _native, _surface_symbols, _ZERO
from sympy.polys.polyerrors import GeneratorsError
from _native_subprocess import isolated_heap_new_probe


MATRIX_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)


def solution(system, *variables):
    return tuple(next(iter(linsolve(system, *variables))))


@pytest.mark.parametrize("left_class", MATRIX_CLASSES)
@pytest.mark.parametrize("right_class", MATRIX_CLASSES)
def test_pair_forms_preserve_exact_unique_values(left_class, right_class):
    x, y = symbols("linear_pair_x linear_pair_y")
    a = left_class([[2, 0], [0, 3], [2, 3]])
    b = right_class([1, 1, 2])
    before_a, before_b = a.copy(), b.copy()
    assert solution((a, b), x, y) == (Rational(1, 2), Rational(1, 3))
    assert a == before_a and b == before_b


@pytest.mark.parametrize("matrix_class", MATRIX_CLASSES)
def test_augmented_family_and_mapping_omit_only_free_coordinates(matrix_class):
    x, y, z = symbols("linear_aug_x linear_aug_y linear_aug_z")
    source = matrix_class([[1, 2, 3, 1], [2, 4, 6, 2]])
    values = solution(source, x, y, z)
    assert values == (1 - 2*y - 3*z, y, z)
    assert values[1] is y and values[2] is z
    assert solve_linear_system(source, x, y, z) == {x: 1 - 2*y - 3*z}
    assert solve_linear_system(matrix_class.zeros(2, 4), x, y, z) == {}


@pytest.mark.parametrize("right_class", MATRIX_CLASSES)
def test_row_rhs_is_adapted_without_public_transpose(right_class, monkeypatch):
    x, y = symbols("linear_row_x linear_row_y")
    a = SparseMatrix([[1, 0], [0, 1], [1, 1]])
    b = right_class([[2, 3, 5]])
    monkeypatch.setattr(right_class, "T", property(lambda self: pytest.fail("public transpose used")))
    assert solution((a, b), x, y) == (2, 3)


@pytest.mark.parametrize("form", ("pair", "augmented", "mapping"))
def test_large_sparse_system_never_materializes_dense_matrix(form, monkeypatch):
    x, y = symbols("linear_large_x linear_large_y")
    entries = {(0, 0): 1, (1, 1): 1, (999, 0): 1, (999, 1): 1}
    a = SparseMatrix(1000, 2, entries)
    b = SparseMatrix(1000, 1, {(0, 0): 2, (1, 0): 3, (999, 0): 5})
    augmented = SparseMatrix(1000, 3, {
        **entries, (0, 2): 2, (1, 2): 3, (999, 2): 5,
    })
    def dense_forbidden(*args, **kwargs):
        raise AssertionError("dense conversion or construction used")
    monkeypatch.setattr(SparseMatrix, "to_dense", dense_forbidden)
    monkeypatch.setattr(Matrix, "__init__", dense_forbidden)
    monkeypatch.setattr(SparseMatrix, "row_join", dense_forbidden)
    if form == "mapping":
        assert solve_linear_system(augmented, x, y) == {x: 2, y: 3}
    else:
        assert solution((a, b) if form == "pair" else augmented, x, y) == (2, 3)


def test_wide_sparse_family_preserves_the_only_free_caller_atom(monkeypatch):
    variables = symbols("linear_wide_0:601")
    a = SparseMatrix(600, 601, {
        **{(index, index): 1 for index in range(600)}, (0, 600): 1,
    })
    b = SparseMatrix.zeros(600, 1)
    monkeypatch.setattr(SparseMatrix, "to_dense", lambda self: pytest.fail("dense area allocated"))
    values = solution((a, b), variables)
    assert len(values) == 601
    assert values[0] == -variables[600]
    assert all(value == 0 for value in values[1:600])
    assert values[600] is variables[600]


def test_same_name_assumptions_and_dummy_identity_need_no_printer():
    positive = Symbol("linear_same", positive=True)
    negative = Symbol("linear_same", negative=True)
    first, second = Dummy("linear_dummy"), Dummy("linear_dummy")
    source = SparseMatrix([[1, 1, 2]])
    with patch.object(Symbol, "__str__", side_effect=AssertionError("coordinate printer called")):
        assumed = solution(source, positive, negative)
        dummy = solution(source, first, second)
    assert assumed == (2 - negative, negative)
    assert assumed[1] is negative
    assert dummy == (2 - second, second)
    assert dummy[1] is second


def test_reconstructed_dummy_retains_caller_and_existing_registry_owner():
    first = Dummy("linear_reconstructed")
    copied = Dummy("linear_reconstructed", dummy_index=first.dummy_index)
    pivot = Dummy("linear_pivot")
    from sympy.solvers._sparse_linear import _binding_key
    key = _binding_key(first._value)
    assert _surface_symbols[key] is first
    values = solution(SparseMatrix([[1, 1, 2]]), pivot, copied)
    assert values == (2 - copied, copied)
    assert values[1] is copied
    assert _surface_symbols[key] is first
    assert solve_linear_system(SparseMatrix([[1, 1, 2]]), pivot, copied) == {pivot: 2 - copied}


def test_exact_tiny_perturbation_and_large_coefficients():
    x, y = symbols("linear_exact_x linear_exact_y")
    epsilon = Rational(1, 10**80)
    a = SparseMatrix([[1, 1], [1, 1 + epsilon]])
    assert solution((a, SparseMatrix([3, 3 + 2*epsilon])), x, y) == (1, 2)
    large = 10**100 + 137
    assert solution(SparseMatrix([[large, Rational(1, 3), large*5 + Rational(2, 3)],
                                  [0, Rational(2, 7), Rational(4, 7)]]), x, y) == (5, 2)


def test_zero_equation_pairs_and_empty_augmented_inputs_keep_distinct_contracts():
    x, y = symbols("linear_empty_x linear_empty_y")
    assert solution((SparseMatrix.zeros(0, 2), SparseMatrix.zeros(0, 1)), x, y) == (x, y)
    assert solution((SparseMatrix.zeros(1, 0), SparseMatrix.zeros(1, 1))) == ()
    assert linsolve((SparseMatrix.zeros(1, 0), SparseMatrix([[1]]))) == EmptySet()
    assert linsolve(SparseMatrix.zeros(0, 3), x, y) == EmptySet()
    assert linsolve([], x, y) == EmptySet()
    assert solve_linear_system(SparseMatrix.zeros(0, 3), x, y) == {}


def test_automatic_parameters_follow_free_columns():
    values = solution(SparseMatrix([[0, 1, 0, 2]]))
    tau0, tau1 = symbols("tau0 tau1")
    assert values == (tau0, 2, tau1)
    assert values[0] is tau0 and values[2] is tau1


@pytest.mark.parametrize("source", (
    [[1, 0, 2], [0, 1, 3]],
    [[1, 0, 2], [0, 0, 1]],
    [[0, 0, 0]],
))
@pytest.mark.parametrize("invalid", ("constant", "expression", "duplicate"))
def test_all_coordinates_are_admitted_even_unique_or_inconsistent(source, invalid):
    x = Symbol("linear_invalid_x")
    variables = (x, 1) if invalid == "constant" else ((x, x + 1) if invalid == "expression" else (x, x))
    expected = GeneratorsError if invalid == "duplicate" else TypeError
    with pytest.raises(expected):
        linsolve(SparseMatrix(source), *variables)
    with pytest.raises(expected):
        solve_linear_system(SparseMatrix(source), *variables)


@pytest.mark.parametrize("source", ([[1, 2]], [[0, 1]]))
def test_unused_coordinate_name_bound_is_still_enforced(source):
    variable = Symbol("linear_oversized_" + "x"*4097)
    with pytest.raises(ValueError, match="limit"):
        linsolve(SparseMatrix(source), variable)


@pytest.mark.parametrize("cols", (0, 2))
def test_pair_requires_one_rhs_column(cols):
    x, y = symbols("linear_shape_x linear_shape_y")
    with pytest.raises(ValueError, match="dimensions"):
        linsolve((SparseMatrix.eye(2), SparseMatrix.zeros(2, cols)), x, y)


def test_native_limit_is_terminal_without_generic_or_dense_retry(monkeypatch):
    import sympy.solvers.solvers as solvers_module
    x = Symbol("linear_bound_x")
    def forbidden(*args, **kwargs):
        raise AssertionError("fallback attempted after exact native entry")
    monkeypatch.setattr(solvers_module, "_generic_rref", forbidden)
    monkeypatch.setattr(Matrix, "rref", forbidden)
    monkeypatch.setattr(SparseMatrix, "to_dense", forbidden)
    with pytest.raises(ValueError, match="bound|limit"):
        linsolve(SparseMatrix.zeros(8193, 2), x)


@pytest.mark.parametrize("matrix_class", MATRIX_CLASSES)
@pytest.mark.parametrize("side", ("coefficients", "rhs"))
def test_both_original_retained_sources_are_bound(matrix_class, side, monkeypatch):
    atom = Integer(17)
    a = matrix_class([[atom if side == "coefficients" else 1]])
    b = matrix_class([[atom if side == "rhs" else 1]])
    x = Symbol("linear_stale_x")
    monkeypatch.setattr(atom, "_value", _native.py_integer(0))
    with pytest.raises(RuntimeError, match="retained"):
        linsolve((a, b), x)


@pytest.mark.parametrize("target", ("matrix_accessor", "symbol_accessor", "symbol_storage"))
def test_changed_accessors_never_enter_the_native_lane(target, monkeypatch):
    x = Symbol("linear_accessor_x")
    a = SparseMatrix([[1, 2]])
    calls = []
    def changed(self, name):
        calls.append(name)
        return object.__getattribute__(self, name)
    if target == "matrix_accessor":
        monkeypatch.setattr(SparseMatrix, "__getattribute__", changed)
    elif target == "symbol_accessor":
        monkeypatch.setattr(Symbol, "__getattribute__", changed)
    else:
        monkeypatch.setattr(Symbol, "_value", property(lambda self: _native.py_integer(0)), raising=False)
    with pytest.raises(NotImplementedError, match="original"):
        linsolve(a, x)
    assert calls == []


@pytest.mark.parametrize("member", ("todok", "solve_affine_parameters", "solve_augmented_affine_parameters"))
def test_native_source_and_generator_overrides_are_refused(member, monkeypatch):
    a = SparseMatrix([[1, 2]])
    x = Symbol("linear_native_x")
    calls = []
    def changed(*args, **kwargs):
        calls.append(True)
        return None
    monkeypatch.setattr(_native.SparseMatrix, member, changed)
    with pytest.raises(NotImplementedError, match="original"):
        linsolve(a, x)
    assert calls == []


@pytest.mark.parametrize("cls", (FiniteSet, Tuple, Integer, Add, Mul))
@pytest.mark.parametrize("hook", ("__new__", "__init__"))
def test_solution_publication_refuses_changed_constructors(cls, hook, monkeypatch, request):
    if isolated_heap_new_probe(request, hook):
        return
    x, y = symbols("linear_constructor_x linear_constructor_y")
    source = SparseMatrix([[1, 1, 2]])
    calls = []
    def changed(*args, **kwargs):
        calls.append(True)
        return Integer(99)
    monkeypatch.setattr(cls, hook, staticmethod(changed) if hook == "__new__" else changed)
    with pytest.raises(NotImplementedError, match="original"):
        linsolve(source, x, y)
    assert calls == []


@pytest.mark.parametrize("slot", ("_struct_args", "_surface_args"))
@pytest.mark.parametrize("owner", ("explicit", "automatic", "singleton"))
def test_poisoned_atomic_surface_storage_cannot_enter_solutions(slot, owner, monkeypatch):
    x = Symbol("linear_auxiliary_x")
    source = SparseMatrix([[0, 0]])
    atom = x if owner == "explicit" else (Symbol("tau0") if owner == "automatic" else _ZERO)
    monkeypatch.setattr(atom, slot, (Integer(99),), raising=False)
    with pytest.raises(RuntimeError, match="non-atomic"):
        linsolve(source, x) if owner == "explicit" else linsolve(source)


def test_poisoned_automatic_tau_native_value_is_refused(monkeypatch):
    tau = Symbol("tau0")
    source = SparseMatrix([[0, 0]])
    monkeypatch.setattr(tau, "_value", _native.py_integer(0))
    with pytest.raises(RuntimeError, match="parameter"):
        linsolve(source)


@pytest.mark.parametrize("field", ("name", "index"))
def test_mutated_dummy_fields_are_refused(field, monkeypatch):
    variable = Dummy("linear_corrupt_dummy")
    source = SparseMatrix([[1, 2]])
    if field == "name":
        monkeypatch.setattr(variable, "_dummy_name", "other")
    else:
        # The core contract admits built-in int/bool indices only.
        monkeypatch.setattr(variable, "_dummy_number", Integer(variable.dummy_index))
    with pytest.raises((RuntimeError, NotImplementedError), match="Dummy"):
        linsolve(source, variable)


def test_mutated_assumptions_cannot_change_the_typed_coordinate(monkeypatch):
    variable = Symbol("linear_changed_facts", positive=True)
    source = SparseMatrix([[0, 0]])
    monkeypatch.setattr(variable, "_assumptions", {"negative": True})
    with pytest.raises(RuntimeError, match="assumptions"):
        linsolve(source, variable)


@pytest.mark.parametrize("corruption", ("key", "value"))
def test_assumption_recheck_rejects_callbacks_before_comparison(corruption):
    from sympy.solvers._sparse_linear import _coordinates, _check_coordinates
    variable = Symbol("linear_postcheck_" + corruption, positive=True)
    records = _coordinates((variable,), 1)
    assumptions = variable._assumptions
    original = dict(assumptions)
    calls = []
    class Key(str):
        def __lt__(self, other):
            calls.append("comparison")
            return False
    class Value:
        def __eq__(self, other):
            calls.append("equality")
            return True
    try:
        assumptions.clear()
        assumptions.update({Key("positive"): True} if corruption == "key" else {"positive": Value()})
        with pytest.raises(RuntimeError, match="nonliteral"):
            _check_coordinates(records)
        assert calls == []
    finally:
        assumptions.clear()
        assumptions.update(original)


def test_corrupted_existing_dummy_owner_is_not_replaced(monkeypatch):
    first = Dummy("linear_owner")
    copied = Dummy("linear_owner", dummy_index=first.dummy_index)
    source = SparseMatrix([[0, 0]])
    monkeypatch.setattr(first, "_value", _native.py_integer(0))
    with pytest.raises((TypeError, RuntimeError)):
        linsolve(source, copied)


def test_inconsistent_system_returns_empty_set_and_none():
    x, y = symbols("linear_inconsistent_x linear_inconsistent_y")
    augmented = SparseMatrix([[1, 2, 1], [2, 4, 3]])
    assert linsolve(augmented, x, y) == EmptySet()
    assert solve_linear_system(augmented, x, y) is None
