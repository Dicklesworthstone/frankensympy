"""Equation-list linsolve uses the source-bound checked sparse family."""

from fractions import Fraction
from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import Basic, _native, _native_expr, _surface_symbols
from sympy.solvers import _sparse_linear as boundary
from sympy.solvers import solvers as solvers_module


def solution(equations, *variables):
    return tuple(next(iter(s.linsolve(equations, *variables))))


def test_equations_and_expressions_share_exact_sparse_unique_solutions():
    x, y = s.symbols("affine_equation_x affine_equation_y")
    assert solution([s.Eq(2*x, 1), s.Eq(3*y, 1), 2*x + 3*y - 2], x, y) == (
        s.Rational(1, 2), s.Rational(1, 3),
    )
    assert solution([x+y-3, x-y+1], (x, y)) == (1, 2)
    assert s.solve([x+y-3, x-y+1], (x, y)) == {x: 1, y: 2}


def test_sparse_equation_family_has_no_expansion_substitution_or_dense_path(monkeypatch):
    variables = s.symbols("affine_sparse_0:601")
    equations = [variables[index] - index for index in range(1, 600)]
    equations.extend([variables[0] + variables[600] - 2, 0])

    def forbidden(*args, **kwargs):
        raise AssertionError("dense construction, probing, expansion, or generic elimination used")

    monkeypatch.setattr(s.Matrix, "__init__", forbidden)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbidden)
    monkeypatch.setattr(solvers_module, "_extract_linear_coeffs", forbidden)
    monkeypatch.setattr(solvers_module, "_generic_rref", forbidden)
    monkeypatch.setattr(solvers_module, "_native_expr", forbidden)
    monkeypatch.setattr(solvers_module, "expand", forbidden)
    with patch.object(s.Symbol, "__str__", forbidden):
        values = solution(equations, variables)
    assert len(values) == 601
    assert values[0] == 2 - variables[600]
    assert values[1:600] == tuple(range(1, 600))
    assert values[600] is variables[600]


@pytest.mark.parametrize("dummy", [False, True])
def test_same_name_typed_coordinates_and_free_owners_are_preserved(dummy):
    if dummy:
        x, y = s.Dummy("affine_same"), s.Dummy("affine_same")
    else:
        x = s.Symbol("affine_same", positive=True)
        y = s.Symbol("affine_same", negative=True)
    values = solution([x + y - 2], x, y)
    assert values == (2 - y, y)
    assert values[1] is y


def test_reconstructed_dummy_free_parameter_keeps_the_callers_object():
    original = s.Dummy("affine_reconstructed")
    copied = s.Dummy("affine_reconstructed", dummy_index=original.dummy_index)
    x = s.Dummy("affine_pivot")
    key = boundary._binding_key(original._value)
    assert _surface_symbols[key] is original
    values = solution([x + copied - 2], x, copied)
    assert values == (2 - copied, copied)
    assert values[1] is copied
    assert _surface_symbols[key] is original


def test_held_affine_arithmetic_and_equation_sides_are_not_reconstructed():
    x = s.Symbol("affine_held")
    inner = s.Add(x, s.Integer(1), evaluate=False)
    left = s.Mul(s.Integer(2), inner, evaluate=False)
    equation = s.Eq(left, 4, evaluate=False)
    original = (equation.args, left.args, inner.args)
    assert solution([equation], x) == (1,)
    assert equation.args is original[0]
    assert left.args is original[1]
    assert inner.args is original[2]
    repeated = s.Add(x, x, s.Integer(-2), evaluate=False)
    assert solution([repeated], x) == (1,)


def test_exact_tiny_perturbations_and_fraction_input_never_use_float_rank():
    x, y = s.symbols("affine_exact_x affine_exact_y")
    epsilon = s.Rational(1, 10**100)
    assert solution([x+y-3, x+(1+epsilon)*y-3-2*epsilon], x, y) == (1, 2)
    assert s.linsolve([Fraction(1, 3)], x) == s.EmptySet()
    assert solution([Fraction(0)], x) == (x,)


def test_zero_inconsistent_and_empty_equations_keep_distinct_contracts():
    x, y = s.symbols("affine_empty_x affine_empty_y")
    assert s.linsolve([], x, y) == s.EmptySet()
    assert solution([0, s.Integer(0)], x, y) == (x, y)
    assert solution([s.Eq(x, x, evaluate=False)], x, y) == (x, y)
    assert s.linsolve([x-1, x-2], x) == s.EmptySet()
    assert s.linsolve([0, 1], x) == s.EmptySet()


@pytest.mark.parametrize("form", ["zero_product", "cancelled", "power_zero", "equation_sides"])
def test_original_coordinate_denominators_cannot_be_hidden(form, monkeypatch):
    x = s.Symbol("affine_pole")
    reciprocal = s.Pow(x, -1, evaluate=False)
    source = {
        "zero_product": lambda: s.Mul(0, reciprocal, evaluate=False),
        "cancelled": lambda: s.Mul(x, reciprocal, evaluate=False),
        "power_zero": lambda: s.Pow(reciprocal, 0, evaluate=False),
        "equation_sides": lambda: s.Eq(reciprocal, reciprocal, evaluate=False),
    }[form]()
    monkeypatch.setattr(solvers_module, "_extract_linear_coeffs",
                        lambda *args: pytest.fail("domain failure entered symbolic fallback"))
    with pytest.raises(NotImplementedError, match="original denominator"):
        s.linsolve([source], x)


def test_symbolic_coefficient_cannot_hide_a_later_held_domain_failure(monkeypatch):
    a, x = s.symbols("affine_parameter affine_domain_x")
    invalid = s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
    monkeypatch.setattr(solvers_module, "_extract_linear_coeffs",
                        lambda *args: pytest.fail("unchecked fallback occurred"))
    with pytest.raises(NotImplementedError, match="original denominator"):
        s.linsolve([a*x-1, invalid], x)


@pytest.mark.parametrize("value", [s.oo, -s.oo, s.zoo, s.nan])
def test_original_nonfinite_operands_survive_held_zero_admission(value):
    x = s.Symbol("affine_nonfinite")
    source = s.Mul(0, value, evaluate=False)
    with pytest.raises(ValueError, match="finite original"):
        s.linsolve([source], x)


@pytest.mark.parametrize("form", ["power", "product", "function"])
def test_nonlinear_equations_never_enter_the_affine_solver(form):
    x = s.Symbol("affine_nonlinear")
    source = {"power": lambda: x**2,
              "product": lambda: s.Mul(x, x+1, evaluate=False),
              "function": lambda: s.sin(x)}[form]()
    with pytest.raises(ValueError, match="nonlinear"):
        s.linsolve([source], x)


def test_legitimate_symbolic_coefficients_retain_the_existing_generic_lane():
    a, x = s.symbols("affine_generic_a affine_generic_x")
    assert solution([a*x-1], x) == (1/a,)
    assert solution([s.sin(a)*x-1], x) == (1/s.sin(a),)


@pytest.mark.parametrize("value", [0.5, s.Float(0.5)])
def test_finite_float_coefficients_remain_outside_exact_rational_admission(value):
    x = s.Symbol("affine_float")
    assert boundary.checked_linear_equations([value], [x]) is boundary.UNSUPPORTED


@pytest.mark.parametrize("value", [float("inf"), float("nan"), s.Float("inf"), s.Float("nan")])
def test_nonfinite_float_sources_are_rejected_before_symbolic_fallback(value):
    x = s.Symbol("affine_float_nonfinite")
    with pytest.raises(ValueError, match="finite original"):
        s.linsolve([value], x)


def test_malformed_float_payload_cannot_hide_a_coordinate():
    from sympy.core import _FLOAT_INTERN
    x = s.Symbol("affine_float_payload")
    source = s.Expr(_native.py_function(_FLOAT_INTERN, _native_expr(x)))
    with pytest.raises(ValueError, match="malformed native"):
        s.linsolve([source], x)


def test_malformed_entire_function_is_not_a_safe_symbolic_coefficient():
    x = s.Symbol("affine_function_arity")
    source = s.Expr(_native.py_function("sin"))
    with pytest.raises(ValueError, match="one original argument"):
        s.linsolve([source], x)


def test_changed_retained_arguments_are_rejected_without_resolving_them(monkeypatch):
    x = s.Symbol("affine_retained")
    source = s.Add(x, s.Integer(1), evaluate=False)
    monkeypatch.setattr(source, "_args", (x, s.Integer(99)))
    with pytest.raises(ValueError, match="native and retained"):
        s.linsolve([source], x)


def test_changed_equality_operator_cannot_reinterpret_retained_sides(monkeypatch):
    x = s.Symbol("affine_equality_operator")
    source = s.Eq(x, 1, evaluate=False)
    monkeypatch.setattr(s.Eq, "rel_op", "!=")
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.linsolve([source], x)


def test_numeric_class_cannot_impersonate_a_symbol_via_native_storage(monkeypatch):
    x = s.Symbol("affine_impersonation")
    number = s.Integer(17)
    monkeypatch.setattr(number, "_value", _native_expr(x))
    with pytest.raises(ValueError, match="nonrational"):
        s.linsolve([number], x)


def test_native_function_name_does_not_grant_arithmetic_authority():
    x = s.Symbol("affine_native_function")
    source = s.Expr(_native.py_function("Add", _native_expr(x), _native.py_integer(1)))
    with pytest.raises(NotImplementedError, match="domain proof"):
        s.linsolve([source], x)


def test_custom_symbol_is_refused_before_hash_or_getter_hooks():
    calls = []

    class Foreign(s.Symbol):
        def __hash__(self):
            calls.append("hash")
            return super().__hash__()

    x = s.Symbol("affine_custom_x")
    foreign = Foreign("affine_custom_owner")
    calls.clear()
    with pytest.raises(TypeError, match="exact Symbol"):
        s.linsolve([x], foreign)
    assert calls == []


@pytest.mark.parametrize("mutation", ["equation_list", "retained_argument", "symbol_assumption"])
def test_sources_are_rechecked_after_native_solution_before_publication(mutation, monkeypatch):
    x = s.Symbol("affine_changed_" + mutation)
    source = s.Add(x, s.Integer(-1), evaluate=False)
    equations = [source]
    original = boundary.checked_linear_system

    def mutate_after_checked_family(*args, **kwargs):
        result = original(*args, **kwargs)
        if mutation == "equation_list":
            equations[0] = s.Integer(1)
        elif mutation == "retained_argument":
            monkeypatch.setattr(source, "_args", (x, s.Integer(1)))
        else:
            x._assumptions["real"] = True
        return result

    monkeypatch.setattr(boundary, "checked_linear_system", mutate_after_checked_family)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.linsolve(equations, x)
    finally:
        if mutation == "symbol_assumption":
            x._assumptions.pop("real", None)


def test_fraction_storage_is_rechecked_before_publication(monkeypatch):
    x = s.Symbol("affine_changed_fraction")
    source = Fraction(0)
    original = boundary.checked_linear_system

    def mutate_after_checked_family(*args, **kwargs):
        result = original(*args, **kwargs)
        source._numerator = 1
        return result

    monkeypatch.setattr(boundary, "checked_linear_system", mutate_after_checked_family)
    with pytest.raises(RuntimeError, match="mutated"):
        s.linsolve([source], x)


def test_fraction_storage_is_validated_before_arithmetic():
    x = s.Symbol("affine_malformed_fraction")
    source = Fraction(1, 2)
    source._denominator = 0
    with pytest.raises(ValueError, match="normalized original"):
        s.linsolve([source], x)


@pytest.mark.parametrize("member", ["__add__", "__mul__", "__pow__", "numerator", "denominator"])
def test_local_fraction_arithmetic_cannot_change_the_extracted_source(member, monkeypatch):
    x = s.Symbol("affine_fraction_interface_" + member)
    source = x - 1
    calls = []

    def poison(*args, **kwargs):
        calls.append(member)
        raise AssertionError("replacement Fraction arithmetic was invoked")

    replacement = property(poison) if member in ("numerator", "denominator") else poison
    monkeypatch.setattr(Fraction, member, replacement)
    with pytest.raises(NotImplementedError, match="original Fraction"):
        s.linsolve([source], x)
    assert calls == []


def test_fraction_property_implementation_is_checked_beyond_descriptor_identity(monkeypatch):
    x = s.Symbol("affine_fraction_property_code")
    source = 2*x - 2
    getter = Fraction.__dict__["numerator"].fget
    monkeypatch.setattr(getter, "__code__", (lambda self: 0).__code__)
    with pytest.raises(NotImplementedError, match="original Fraction"):
        s.linsolve([source], x)


def test_native_exhaustion_is_terminal_without_a_symbolic_retry(monkeypatch):
    x = s.Symbol("affine_terminal")

    def exhausted(*args, **kwargs):
        raise ValueError("native affine budget exhausted")

    monkeypatch.setattr(boundary, "checked_linear_system", exhausted)
    monkeypatch.setattr(solvers_module, "_extract_linear_coeffs",
                        lambda *args: pytest.fail("native failure triggered a retry"))
    with pytest.raises(ValueError, match="budget exhausted"):
        s.linsolve([x-1], x)


def test_duplicate_and_invalid_unused_coordinates_are_admitted_first():
    x = s.Symbol("affine_invalid")
    with pytest.raises(ValueError, match="duplicate"):
        s.linsolve([1], x, x)
    with pytest.raises(TypeError, match="exact Symbol"):
        s.linsolve([1], x, 3)


def test_source_row_integer_and_power_limits_remain_terminal():
    x = s.Symbol("affine_limits")
    with pytest.raises(ValueError, match="row count"):
        s.linsolve([0]*8193, x)
    with pytest.raises(ValueError, match="4096 bits"):
        s.linsolve([1 << 4096], x)
    power = s.Pow(s.Integer(2), s.Integer(65), evaluate=False)
    with pytest.raises(ValueError, match="power exceeds"):
        s.linsolve([power], x)
