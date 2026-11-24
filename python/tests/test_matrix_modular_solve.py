"""Complete modular families through checked native dense/sparse boundaries."""

from itertools import product
import threading
from math import prod
from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import (
    Half, NegativeOne, One, Zero, SupervisedCallbackSession, _native,
    supervised_callback_session,
)
from sympy.matrices import _modular_solve, normalforms
from sympy.matrices._modular_solve import solve_mod
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.printing.str import StrPrinter


CLASSES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)
AMPLE_STEPS = 10_000_000


def _zero(cls, rows, columns):
    entries = {} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else [0] * (rows * columns)
    return cls(rows, columns, entries)


def _entries(matrix):
    return tuple(int(matrix[row, column])
                 for row in range(matrix.rows) for column in range(matrix.cols))


def _brute_force(source, rhs, modulus):
    coefficients, right = _entries(source), _entries(rhs)
    rows, variables, columns = source.rows, source.cols, rhs.cols
    return {
        values for values in product(range(modulus), repeat=variables * columns)
        if all((sum(coefficients[row * variables + inner] * values[inner * columns + column]
                    for inner in range(variables)) - right[row * columns + column]) % modulus == 0
               for row in range(rows) for column in range(columns))
    }


def _family(result, modulus):
    if result is None:
        return set()
    particular, generators, periods = result
    assert type(periods) is tuple
    assert all(type(period) is s.Integer and period.is_integer is True for period in periods)
    orders = tuple(int(period) for period in periods)
    assert all(2 <= period <= modulus and modulus % period == 0 for period in orders)
    variables, columns = particular.shape
    assert generators.shape == (variables, len(orders))
    base, directions = _entries(particular), _entries(generators)
    assert all(0 <= value < modulus for value in (*base, *directions))
    solutions = set()
    ranges = [range(period) for _ in range(columns) for period in orders]
    for parameters in product(*ranges):
        values = tuple(
            (base[row * columns + column]
             + sum(directions[row * len(orders) + direction]
                   * parameters[column * len(orders) + direction]
                   for direction in range(len(orders)))) % modulus
            for row in range(variables) for column in range(columns)
        )
        assert values not in solutions, "finite parameterization must have no repetitions"
        solutions.add(values)
    assert len(solutions) == prod(orders) ** columns
    return solutions


def _complete(source, rhs, modulus, result):
    expected = _brute_force(source, rhs, modulus)
    assert _family(result, modulus) == expected
    assert (result is None) == (not expected)
    return len(expected)


@pytest.mark.parametrize("coefficient_class", CLASSES)
@pytest.mark.parametrize("rhs_class", CLASSES)
def test_all_mixed_classes_preserve_owner_and_independent_finite_rhs_parameters(
        coefficient_class, rhs_class):
    source = coefficient_class([[2, 0]])
    rhs = rhs_class([[4, 2]])
    result = source.solve_mod(rhs, 6)
    particular, generators, periods = result
    assert type(particular) is type(generators) is coefficient_class
    assert particular.shape == (2, 2)
    assert generators.cols == len(periods)
    assert _complete(source, rhs, 6, result) == 144
    assert source == coefficient_class([[2, 0]])
    assert rhs == rhs_class([[4, 2]])


@pytest.mark.parametrize("values,right,modulus,count", [
    ([[2, 4], [0, 6]], [[4], [6]], 12, 12),
    ([[6]], [[0]], 6, 6),
    ([[-3]], [[9]], 12, 3),
    ([[3]], [[6]], 9, 3),
    ([[5]], [[10]], 25, 5),
    ([[2, 0], [0, 4]], [[-2], [4]], 8, 8),
    ([[2, 3], [3, 2]], [[1], [5]], 6, 1),
    ([[2], [4]], [[4], [2]], 6, 2),
])
def test_prime_power_composite_signed_and_rectangular_families_match_every_residue(
        values, right, modulus, count):
    source, rhs = s.Matrix(values), s.SparseMatrix(right)
    assert _complete(source, rhs, modulus, solve_mod(source, rhs, modulus)) == count


@pytest.mark.parametrize("cls", CLASSES)
def test_checked_inconsistency_is_distinct_from_successful_nonunique_families(cls):
    assert cls([[2]]).solve_mod(cls([[1]]), 6) is None
    assert cls([[2, 4]]).solve_mod(cls([[2, 1]]), 6) is None
    assert cls([[1, 0], [0, 0]]).solve_mod(cls([[0], [1]]), 6) is None
    assert _complete(cls([[2]]), cls([[2]]), 6, cls([[2]]).solve_mod(cls([[2]]), 6)) == 2


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 2, 1), (0, 0, 0),
                                    (2, 0, 1), (0, 0, 3)])
def test_zero_area_shapes_retain_finite_kernel_and_unique_empty_solution(cls, shape):
    rows, variables, columns = shape
    source, rhs = _zero(cls, rows, variables), _zero(cls, rows, columns)
    result = source.solve_mod(rhs, 6)
    particular, generators, periods = result
    assert type(particular) is type(generators) is cls
    assert particular.shape == (variables, columns)
    assert generators.shape == (variables, variables)
    assert tuple(map(int, periods)) == (6,) * variables
    assert _complete(source, rhs, 6, result) == 6 ** (variables * columns)
    if columns == 0:
        assert _family(result, 6) == {()}


@pytest.mark.parametrize("cls", CLASSES)
def test_no_unknowns_cannot_satisfy_nonzero_rhs(cls):
    assert _zero(cls, 2, 0).solve_mod(cls([[0], [1]]), 6) is None


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("literal", (int, s.Integer))
def test_large_exact_modulus_and_periods_are_never_rounded_or_narrowed(cls, literal):
    modulus, step = 1 << 256, 1 << 255
    particular, generators, periods = cls([[6]]).solve_mod(cls([[-12]]), literal(modulus))
    assert type(particular) is type(generators) is cls
    assert int(particular[0, 0]) == step - 2
    assert int(generators[0, 0]) == step
    assert periods == (s.Integer(2),)
    # A zero diagonal must transport the entire 256-bit modulus as a period.
    particular, generators, periods = cls([[0]]).solve_mod(cls([[0]]), literal(modulus))
    assert int(particular[0, 0]) == 0 and int(generators[0, 0]) == 1
    assert len(periods) == 1 and int(periods[0]) == modulus


def test_native_only_mixed_sources_and_native_integer_modulus_are_supported():
    source = s.ImmutableMatrix(_native.Matrix(
        1, 2, [_native.py_integer(2), _native.py_integer(0)],
    ))
    rhs = s.SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): _native.py_integer(4)}))
    assert source._surface is None and rhs._surface is None
    result = source.solve_mod(rhs, s.Integer(6))
    assert type(result[0]) is type(result[1]) is s.ImmutableMatrix
    assert _complete(source, rhs, 6, result) == 12
    native = source._native_storage.solve_mod(rhs._native_storage, _native.py_integer(6))
    assert native is not None and len(native) == 3
    assert [int(value.exact_numerator()) for value in native[2]] == [2, 6]


@pytest.mark.parametrize("modulus", [True, False, 6.0, "6", s.Rational(3, 2),
                                     s.Symbol("modulus")])
def test_noninteger_modulus_types_are_not_coerced(modulus):
    with pytest.raises(TypeError, match="modulus m must be an integer"):
        s.Matrix([[2]]).solve_mod(s.Matrix([[2]]), modulus)


@pytest.mark.parametrize("modulus", [0, 1, -6, s.Integer(0), s.Integer(-6)])
def test_modulus_requires_a_nontrivial_positive_residue_ring(modulus):
    with pytest.raises(ValueError, match="at least two"):
        s.Matrix([[2]]).solve_mod(s.Matrix([[2]]), modulus)


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
@pytest.mark.parametrize("entry", [
    s.Rational(1, 2), s.Float(0), s.Symbol("unknown_integer"),
    s.Add(1, -1, evaluate=False),
    s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
])
def test_both_original_domains_are_checked_before_mathematical_outcomes(operand, entry):
    source = s.Matrix([[entry if operand == "coefficients" else 0]])
    rhs = s.ImmutableSparseMatrix([[entry if operand == "rhs" else 1]])
    with pytest.raises((ValueError, NotImplementedError)):
        source.solve_mod(rhs, 6)


def test_zero_rhs_cannot_erase_invalid_coefficients_and_shape_mismatch_is_not_none():
    hidden = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.Matrix([[hidden]]).solve_mod(_zero(s.SparseMatrix, 1, 0), 6)
    with pytest.raises(ValueError, match="same number of rows"):
        s.eye(2).solve_mod(s.Matrix([[1]]), 6)


def test_source_and_modulus_bounds_precede_coefficient_copies():
    source, rhs = s.Matrix([[2]]), s.Matrix([[2]])
    with patch.object(normalforms, "_rational_source_snapshot",
                      side_effect=AssertionError("copied inadmissible original")):
        with pytest.raises(ValueError, match="4096"):
            source.solve_mod(rhs, 1 << 4096)
        with pytest.raises(ValueError, match="height"):
            s.Matrix([[1 << 4096]]).solve_mod(rhs, 6)
        with pytest.raises(ValueError, match="64"):
            _zero(s.SparseMatrix, 0, 65).solve_mod(_zero(s.Matrix, 0, 1), 6)
    with pytest.raises(ValueError, match="height"):
        source.solve_mod(s.Matrix([[1 << 4096]]), 6)
    with pytest.raises(ValueError, match="64"):
        _zero(s.Matrix, 0, 0).solve_mod(_zero(s.SparseMatrix, 0, 65), 6)


def test_custom_scalar_and_matrix_protocols_are_not_called():
    observed = []

    class Modulus:
        def __int__(self):
            observed.append("int")
            raise AssertionError("custom coercion")

        def __index__(self):
            observed.append("index")
            raise AssertionError("custom indexing")

    class IntegerSubclass(int):
        def bit_length(self):
            observed.append("bits")
            raise AssertionError("custom bit length")

    class MatrixSubclass(s.Matrix):
        def __getitem__(self, index):
            observed.append("entry")
            raise AssertionError("custom matrix indexing")

    for value in (Modulus(), IntegerSubclass(6)):
        with pytest.raises(TypeError):
            s.eye(1).solve_mod(s.ones(1, 1), value)
    for source, rhs in [
        (MatrixSubclass([[2]]), s.Matrix([[2]])),
        (s.Matrix([[2]]), MatrixSubclass([[2]])),
    ]:
        with pytest.raises(NotImplementedError, match="built-in"):
            source.solve_mod(rhs, 6)
    assert observed == []


@pytest.mark.parametrize("cls", CLASSES)
def test_changed_builtin_constructor_is_refused_without_executing_it(cls, monkeypatch):
    source, rhs = cls([[2]]), s.Matrix([[2]])
    observed = []

    def forbidden(*args, **kwargs):
        observed.append("constructor")
        raise AssertionError("changed constructor ran")

    monkeypatch.setattr(cls, "__init__", forbidden)
    with pytest.raises(NotImplementedError, match="original"):
        solve_mod(source, rhs, 6)
    assert observed == []


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
def test_stale_retained_cells_cannot_rebind_either_original(operand):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    if operand == "coefficients":
        source._surface = (s.Integer(3),)
    else:
        rhs._surface = {(0, 0): s.Integer(3)}
    with pytest.raises(RuntimeError, match="retained cells differ"):
        source.solve_mod(rhs, 6)


@pytest.mark.parametrize("kind", (Zero, One, NegativeOne, Half))
def test_fixed_modulus_owners_cannot_hide_another_integer(kind, monkeypatch):
    modulus = kind()
    monkeypatch.setattr(modulus, "_value", _native.py_integer(6))
    with pytest.raises(ValueError, match="fixed modulus constant"):
        s.eye(1).solve_mod(s.ones(1, 1), modulus)


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
def test_literal_source_owner_cannot_hide_a_forged_fixed_constant(operand, monkeypatch):
    value = One()
    monkeypatch.setattr(value, "_value", _native.py_integer(2))
    source = s.Matrix([[value if operand == "coefficients" else 2]])
    rhs = s.SparseMatrix([[value if operand == "rhs" else 2]])
    with pytest.raises(ValueError, match="fixed .* constant"):
        source.solve_mod(rhs, 6)


@pytest.mark.parametrize("slot", ("_surface_args", "_struct_args"))
def test_retained_modulus_domain_is_not_erased_by_its_native_integer(slot, monkeypatch):
    modulus = s.Integer(1009)
    monkeypatch.setattr(modulus, slot, (s.Pow(0, -1, evaluate=False),), raising=False)
    with pytest.raises(ValueError, match="literal integer atom"):
        s.eye(1).solve_mod(s.ones(1, 1), modulus)


@pytest.mark.parametrize("operand", ["coefficients", "rhs", "modulus"])
@pytest.mark.parametrize("consistent", [False, True])
def test_mutation_after_native_acceptance_blocks_families_and_checked_none(
        operand, consistent, monkeypatch):
    source, rhs = s.Matrix([[2]]), s.Matrix([[2 if consistent else 1]])
    modulus = s.Integer(6)
    original = normalforms._check_rational_source
    seen = []

    def changed(snapshot):
        if snapshot[0] is rhs:
            seen.append(1)
            if len(seen) == 3:
                if operand == "modulus":
                    monkeypatch.setattr(modulus, "_value", _native.py_integer(7))
                else:
                    selected = source if operand == "coefficients" else rhs
                    selected[0, 0] = selected[0, 0] + 1
        original(snapshot)

    with patch.object(normalforms, "_check_rational_source", changed):
        with pytest.raises(RuntimeError, match="mutated"):
            source.solve_mod(rhs, modulus)
    assert len(seen) == 3


@pytest.mark.parametrize("operand", ["coefficients", "rhs", "modulus"])
def test_mutation_during_period_lifting_prevents_final_publication(operand, monkeypatch):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    modulus = s.Integer(6)
    original = _modular_solve._lift_periods

    def changed(values, count):
        result = original(values, count)
        if operand == "modulus":
            monkeypatch.setattr(modulus, "_value", _native.py_integer(7))
        else:
            selected = source if operand == "coefficients" else rhs
            selected[0, 0] = selected[0, 0] + 1
        return result

    with patch.object(_modular_solve, "_lift_periods", changed):
        with pytest.raises(RuntimeError, match="mutated"):
            source.solve_mod(rhs, modulus)


def test_coefficient_mutation_during_rhs_capture_cannot_rebind_the_request():
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    original = normalforms._rational_source_snapshot

    def changed(matrix, **kwargs):
        captured = original(matrix, **kwargs)
        if matrix is rhs:
            source[0, 0] = 3
        return captured

    with patch.object(normalforms, "_rational_source_snapshot", changed):
        with pytest.raises(RuntimeError, match="mutated"):
            source.solve_mod(rhs, 6)


@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_any_source_copy(cancelled):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()
    with patch.object(normalforms, "_rational_source_snapshot",
                      side_effect=AssertionError("terminal request copied a source")):
        with supervised_callback_session(session):
            with pytest.raises(RuntimeError if cancelled else ValueError,
                               match="cancelled|budget exhausted"):
                source.solve_mod(rhs, 6)


def _native_cost(source, rhs):
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    assert source._native_storage.solve_mod(rhs._native_storage, 6, session=session) is not None
    assert session.consumed_nested_steps > 0
    return session.consumed_nested_steps


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_modulus_generation_verification_and_lifting_share_the_active_allowance(cls):
    source, rhs = cls([[2, 0]]), cls([[4]])
    cost = _native_cost(source, rhs)
    session = SupervisedCallbackSession(max_nested_steps=2 * cost - 1)
    assert source._native_storage.solve_mod(rhs._native_storage, 6, session=session) is not None
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        source._native_storage.solve_mod(rhs._native_storage, 6, session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps and session.cancelled
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        source._native_storage.solve_mod(rhs._native_storage, 6, session=session)
    assert session.consumed_nested_steps == consumed
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        source.solve_mod(rhs, 6)
    assert measured.consumed_nested_steps > cost


@pytest.mark.parametrize("consistent", [False, True])
def test_late_cancellation_is_never_relabelled_as_a_mathematical_outcome(consistent):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2 if consistent else 1]])
    original = normalforms._check_rational_source
    seen = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancelled(snapshot):
            original(snapshot)
            if snapshot[0] is rhs:
                seen.append(1)
                if len(seen) == 4:
                    session.cancel()

        with patch.object(normalforms, "_check_rational_source", cancelled):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                source.solve_mod(rhs, 6)
    assert len(seen) == 4


def test_shared_modulus_parser_preserves_the_existing_inverse_interface():
    source = s.Matrix([[2, 3], [3, 2]])
    assert source.inv_mod(s.Integer(6)) == source
    with pytest.raises(TypeError, match="inv_mod: modulus m must be an integer"):
        source.inv_mod("6")
    with pytest.raises(TypeError, match="solve_mod: modulus m must be an integer"):
        source.solve_mod(s.eye(2), "6")


def _publication_case(name):
    source = s.Matrix([[5 if name == "inv_mod" else 2]])
    # Use ordinary Integer atoms, avoiding singleton mutation in class-swap tests.
    rhs = s.Matrix([[3 if name.endswith("_none") else 2]])
    modulus = s.Integer(6)

    def call():
        if name.startswith("solve_mod"):
            return source.solve_mod(rhs, modulus)
        if name.startswith("solve_integer"):
            return source.solve_integer(rhs)
        if name.startswith("inv_mod"):
            try:
                return source.inv_mod(modulus)
            except NonInvertibleMatrixError:
                if name != "inv_mod_nonunit":
                    raise
                return None
        functions = {
            "smith": normalforms.smith_normal_form,
            "smith_decomposition": normalforms.smith_normal_decomposition,
            "hermite": normalforms.hermite_normal_form,
            "hermite_decomposition": normalforms.hermite_normal_decomposition,
        }
        return functions[name](source)

    return source, rhs, modulus, call


def _last_ownership_count(call):
    observed = []
    original = threading.get_ident

    def ident():
        observed.append(1)
        return original()

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        with patch.object(threading, "get_ident", ident):
            call()
    assert observed
    return len(observed)


def _mutate_during_last_ownership_check(call, mutate):
    # Measure the real native ownership callbacks, then mutate at the last one
    # on a fresh request. No replacement of the solver or final source guard
    # is involved: charge_nested_steps calls Python threading.get_ident.
    last = _last_ownership_count(call)
    observed, fired = [], []
    original = threading.get_ident

    def ident():
        observed.append(1)
        if len(observed) == last and not fired:
            fired.append(1)
            mutate()
        return original()

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        with patch.object(threading, "get_ident", ident):
            with pytest.raises(RuntimeError, match="mutated"):
                call()
    assert fired == [1]


@pytest.mark.parametrize("name,operand", [
    ("solve_mod", "coefficients"), ("solve_mod", "rhs"), ("solve_mod", "modulus"),
    ("solve_mod_none", "coefficients"), ("solve_mod_none", "rhs"), ("solve_mod_none", "modulus"),
    ("solve_integer", "coefficients"), ("solve_integer", "rhs"),
    ("solve_integer_none", "coefficients"), ("solve_integer_none", "rhs"),
    ("inv_mod", "coefficients"), ("inv_mod", "modulus"),
    ("inv_mod_nonunit", "coefficients"), ("inv_mod_nonunit", "modulus"),
    ("smith", "coefficients"), ("smith_decomposition", "coefficients"),
    ("hermite", "coefficients"), ("hermite_decomposition", "coefficients"),
])
def test_actual_terminal_ownership_hook_cannot_rebind_checked_publication(name, operand):
    source, rhs, modulus, call = _publication_case(name)

    def mutate():
        if operand == "modulus":
            modulus._value = _native.py_integer(7)
        else:
            selected = source if operand == "coefficients" else rhs
            selected[0, 0] = selected[0, 0] + 1

    _mutate_during_last_ownership_check(call, mutate)


@pytest.mark.parametrize("operand", [
    "coefficients", "rhs", "modulus", "coefficient_atom", "rhs_atom",
])
@pytest.mark.parametrize("consistent", [False, True])
def test_actual_terminal_class_swap_cannot_publish_an_unadmitted_owner(operand, consistent):
    source, rhs, modulus, call = _publication_case("solve_mod" if consistent else "solve_mod_none")
    selected = {
        "coefficients": source,
        "rhs": rhs,
        "modulus": modulus,
        "coefficient_atom": source._surface[0],
        "rhs_atom": rhs._surface[0],
    }[operand]
    original_type = type(selected)
    changed_type = type("ChangedModularOwner", (original_type,), {"__slots__": ()})

    def mutate():
        # Layout-compatible __class__ replacement leaves every old slot value
        # intact. Original owner identity is a separate admission obligation.
        object.__setattr__(selected, "__class__", changed_type)

    try:
        _mutate_during_last_ownership_check(call, mutate)
    finally:
        object.__setattr__(selected, "__class__", original_type)


@pytest.mark.parametrize("change", ["native_source", "equal_sparse_cell"])
def test_initial_admission_metadata_cannot_be_rebound_during_preflight_charge(change):
    source, rhs = s.SparseMatrix([[2]]), s.Matrix([[2]])
    equal_replacement = s.Integer(2)
    original_charge = normalforms._charge_nested_if_active
    original_ident = threading.get_ident
    fired = []

    def ident():
        if not fired:
            fired.append(1)
            if change == "native_source":
                source[0, 0] = 4
            else:
                # Same integer and dictionary identity, but a different
                # retained owner than the one admitted before the callback.
                source._surface[(0, 0)] = equal_replacement
        return original_ident()

    def charge(steps=1):
        if steps == 2 and not fired:
            with patch.object(threading, "get_ident", ident):
                return original_charge(steps)
        return original_charge(steps)

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        with patch.object(normalforms, "_charge_nested_if_active", charge):
            with pytest.raises(RuntimeError, match="mutated"):
                source.solve_mod(rhs, 6)
    assert fired == [1]


@pytest.mark.parametrize("operand", ["coefficients", "modulus"])
def test_nonunit_error_printer_cannot_change_sources_after_final_validation(operand):
    source, modulus = s.Matrix([[2]]), s.Integer(6)
    original = StrPrinter.doprint
    fired = []

    def changed(printer, expression):
        if expression is modulus and not fired:
            fired.append(1)
            if operand == "modulus":
                modulus._value = _native.py_integer(5)
            else:
                source[0, 0] = 5
        return original(printer, expression)

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        with patch.object(StrPrinter, "doprint", changed):
            with pytest.raises(RuntimeError, match="mutated"):
                source.inv_mod(modulus)
    assert fired == [1]
