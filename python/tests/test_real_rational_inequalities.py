"""Complete rational sign charts through the actual native/Python boundary."""

import importlib

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def _forbid_approximation(*args, **kwargs):
    raise AssertionError("exact rational inequalities must not use numerical roots")


def _intervals(solution):
    """Read exact endpoints and openness without numerical order comparisons."""
    pieces = solution.args if isinstance(solution, s.Union) else (solution,)
    assert all(isinstance(piece, s.Interval) for piece in pieces)
    return {(piece.start, piece.end, piece.left_open, piece.right_open)
            for piece in pieces}


@pytest.mark.parametrize("operator,left,right", [
    (s.Lt, True, True), (s.Le, True, False),
    (s.Gt, False, True), (s.Ge, False, False),
])
def test_irreducible_quintic_boundaries_are_exact_in_every_relation(monkeypatch, operator, left, right):
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_approximation)
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid_approximation)
    x = s.Symbol("inequality_quintic_x")
    polynomial = x**5 - x + 1
    root = s.CRootOf(polynomial, 0)
    result = s.solveset(operator(polynomial, 0, evaluate=False), x, s.S.Reals)
    expected = ((-s.oo, root, True, right) if left
                else (root, s.oo, right, True))
    assert _intervals(result) == {expected}
    assert root.free_symbols == set()


@pytest.mark.parametrize("operator,inside,weak", [
    (s.Lt, True, False), (s.Le, True, True),
    (s.Gt, False, False), (s.Ge, False, True),
])
def test_general_quartic_has_complete_two_boundary_sign_decomposition(monkeypatch, operator, inside, weak):
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_approximation)
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid_approximation)
    x = s.Symbol("inequality_quartic_x")
    polynomial = x**4 - x - 1
    lo, hi = s.CRootOf(polynomial, 0), s.CRootOf(polynomial, 1)
    result = s.solve_univariate_inequality(
        operator(polynomial, 0, evaluate=False), x, relational=False,
    )
    expected = ({(lo, hi, not weak, not weak)} if inside else {
        (-s.oo, lo, True, not weak), (hi, s.oo, not weak, True),
    })
    assert _intervals(result) == expected


def test_irrational_zeros_and_poles_have_independent_endpoint_ownership():
    x = s.Symbol("inequality_irrational_x")
    expression = s.Mul(x*x-2, s.Pow(x*x-3, -1, evaluate=False), evaluate=False)
    result = s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals)
    assert _intervals(result) == {
        (-s.oo, -s.sqrt(3), True, True),
        (-s.sqrt(2), s.sqrt(2), False, False),
        (s.sqrt(3), s.oo, True, True),
    }
    for point, expected in [(0, True), (2, True), (s.Rational(3, 2), False)]:
        assert result.contains(point) is expected
    for endpoint in (-s.sqrt(3), s.sqrt(3)):
        assert result.contains(endpoint) is False


def test_cancellation_does_not_reintroduce_original_irrational_holes():
    x = s.Symbol("inequality_cancellation_x")
    q = x*x-2
    expression = s.Mul(q*q, s.Pow(q, -1, evaluate=False), evaluate=False)
    above = s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals)
    below = s.solveset(s.Le(expression, 0, evaluate=False), x, s.S.Reals)
    assert _intervals(above) == {
        (-s.oo, -s.sqrt(2), True, True), (s.sqrt(2), s.oo, True, True),
    }
    assert _intervals(below) == {(-s.sqrt(2), s.sqrt(2), True, True)}


def test_repeated_even_roots_can_be_isolated_included_points():
    x = s.Symbol("inequality_isolated_x")
    expression = (x-1)**2 * (x+2)**2
    assert s.solveset(s.Le(expression, 0, evaluate=False), x, s.S.Reals) == s.FiniteSet(-2, 1)
    assert s.solveset(s.Lt(expression, 0, evaluate=False), x, s.S.Reals) == s.S.EmptySet
    assert s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals) == s.S.Reals


def test_original_denominators_survive_both_sides_of_a_held_relation():
    x = s.Symbol("inequality_held_sides_x")
    reciprocal = s.Pow(x-1, -1, evaluate=False)
    weak = s.Ge(reciprocal, reciprocal, evaluate=False)
    result = s.solveset(weak, x, s.S.Reals)
    assert _intervals(result) == {(-s.oo, s.Integer(1), True, True), (s.Integer(1), s.oo, True, True)}
    assert result.contains(1) is False
    assert result.contains(0) is True
    assert result.contains(2) is True
    assert s.solveset(s.Gt(reciprocal, reciprocal, evaluate=False), x, s.S.Reals) == s.S.EmptySet


def test_nested_reciprocals_held_zero_powers_and_zero_products_keep_holes():
    x = s.Symbol("inequality_nested_x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    nested = s.Pow(reciprocal, -1, evaluate=False)
    result = s.solveset(s.Ge(nested, 0, evaluate=False), x, s.S.Reals)
    assert _intervals(result) == {(s.Integer(0), s.oo, True, True)}
    held_one = s.Pow(reciprocal, 0, evaluate=False)
    held_zero = s.Mul(0, held_one, evaluate=False)
    for expression, operation in ((held_one, s.Gt), (held_zero, s.Ge)):
        result = s.solve_univariate_inequality(
            operation(expression, 0, evaluate=False), x, relational=False, continuous=True,
        )
        assert _intervals(result) == {(-s.oo, s.Integer(0), True, True), (s.Integer(0), s.oo, True, True)}
        assert result.contains(0) is False
        assert result.contains(-1) is True
        assert result.contains(1) is True


def test_rational_sums_keep_both_poles_and_the_exact_weak_zero():
    x = s.Symbol("inequality_sum_x")
    expression = s.Add(
        s.Pow(x-1, -1, evaluate=False),
        s.Mul(-2, s.Pow(x+1, -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    result = s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals)
    assert _intervals(result) == {
        (-s.oo, s.Integer(-1), True, True), (s.Integer(1), s.Integer(3), True, False),
    }
    for point, expected in [(-2, True), (-1, False), (0, False), (1, False), (2, True), (3, True), (4, False)]:
        assert result.contains(point) is expected


def test_constants_and_nowhere_defined_held_inputs_never_share_a_result():
    x = s.Symbol("inequality_constants_x")
    assert s.solveset(s.Ge(0, 0, evaluate=False), x, s.S.Reals) == s.S.Reals
    assert s.solveset(s.Gt(0, 0, evaluate=False), x, s.S.Reals) == s.S.EmptySet
    assert s.solveset(s.Gt(7, 0, evaluate=False), x, s.S.Reals) == s.S.Reals
    undefined = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    for operation in (s.Lt, s.Le, s.Gt, s.Ge):
        assert s.solveset(operation(undefined, 0, evaluate=False), x, s.S.Reals) == s.S.EmptySet
    assert s.solveset(s.Lt(x**4+1, 0, evaluate=False), x, s.S.Reals) == s.S.EmptySet


def test_very_close_rational_endpoints_remain_distinct_without_float_rounding():
    x = s.Symbol("inequality_close_x")
    epsilon = s.Rational(1, 10**80)
    expression = x*(x-epsilon)
    result = s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals)
    assert _intervals(result) == {
        (-s.oo, s.Integer(0), True, False), (epsilon, s.oo, False, True),
    }
    assert result.contains(0) is True
    assert result.contains(epsilon) is True
    assert result.contains(epsilon/2) is False


def _relational_atoms(expression):
    if isinstance(expression, s.Relational):
        return {expression.canonical}
    result = set()
    for child in expression.args:
        result.update(_relational_atoms(child))
    return result


def test_algebraic_interval_intersections_preserve_all_constraints_as_conjunctions():
    x = s.Symbol("inequality_intersection_x")
    polynomial = x**4-x-1
    lo, hi = s.CRootOf(polynomial, 0), s.CRootOf(polynomial, 1)
    result = s.reduce_inequalities([
        s.Le(polynomial, 0, evaluate=False), s.Ge(x, 0, evaluate=False),
    ], x)
    assert isinstance(result, s.And)
    atoms = _relational_atoms(result)
    required = {s.Le(0, x, evaluate=False).canonical, s.Le(x, hi, evaluate=False).canonical}
    allowed = required | {
        s.Le(lo, x, evaluate=False).canonical,
        s.Lt(-s.oo, x, evaluate=False).canonical,
        s.Lt(x, s.oo, evaluate=False).canonical,
    }
    assert required <= atoms <= allowed
    # Domain restriction also retains exact algebraic endpoints; comparisons
    # not yet decided by native set algebra remain structural intersections.
    restricted = s.solve_univariate_inequality(
        s.Le(polynomial, 0, evaluate=False), x,
        relational=True, domain=s.Interval(0, s.oo),
    )
    assert required <= _relational_atoms(restricted) <= allowed


def test_rational_domain_restriction_and_public_solver_exports_agree():
    from sympy.solvers import solve_univariate_inequality as exported
    from sympy.solvers.inequalities import solve_univariate_inequality as module_exported

    assert exported is s.solve_univariate_inequality is module_exported
    x = s.Symbol("inequality_domain_x")
    result = exported(s.Le(x*x-4, 0, evaluate=False), x, relational=False, domain=s.Interval(0, 3))
    assert _intervals(result) == {(s.Integer(0), s.Integer(2), False, False)}
    assert result.contains(-1) is False
    assert result.contains(0) is True
    assert result.contains(2) is True
    assert result.contains(3) is False


def test_bridge_uses_real_budgets_and_returns_exact_complete_masks():
    x = s.Symbol("inequality_budget_x")
    expression = s.Mul(x*x-2, s.Pow(x*x-3, -1, evaluate=False), evaluate=False)
    polynomial, gaps, points = _native.solve_real_rational_inequality_native(
        _native_expr(expression), _symbol_binding(x), ">=",
    )
    assert s.Poly(polynomial, x) == s.Poly(x**4-5*x*x+6, x)
    assert gaps == [True, False, True, False, True]
    assert points == [False, True, True, False]
    with pytest.raises(ValueError, match="operator"):
        _native.solve_real_rational_inequality_native(_native_expr(expression), _symbol_binding(x), "!=")


def test_recognized_resource_failures_do_not_fall_back_after_losing_source(monkeypatch):
    x = s.Symbol("inequality_resource_x")

    def forbidden(*args, **kwargs):
        raise AssertionError("a recognized rational resource refusal must remain explicit")

    monkeypatch.setattr(_native, "solve_inequality_native", forbidden)
    for expression in (
        s.Pow(x, 65, evaluate=False),
        s.Mul(0, s.Integer(1 << 4096), evaluate=False),
        s.Mul(x**33+1, s.Pow(x**32+1, -1, evaluate=False), evaluate=False),
    ):
        with pytest.raises(NotImplementedError, match="resource limit"):
            s.solveset(s.Ge(expression, 0, evaluate=False), x, s.S.Reals)
    with pytest.raises(ValueError, match="resource limit"):
        _native.solve_real_rational_inequality_native(
            _native_expr(s.Pow(x, 65, evaluate=False)), _symbol_binding(x), ">",
        )


def test_typed_variable_identity_and_unsupported_grammar_stay_explicit():
    plain = s.Symbol("inequality_identity")
    positive = s.Symbol("inequality_identity", positive=True)
    assert _native.solve_real_rational_inequality_native(
        _native_expr(positive), _symbol_binding(plain), ">",
    ) is None
    for expression in (s.sin(plain), s.sqrt(plain), s.Float("1.5")*plain):
        assert _native.solve_real_rational_inequality_native(
            _native_expr(expression), _symbol_binding(plain), ">",
        ) is None
    dummy = s.Dummy("inequality_dummy")
    result = s.solveset(s.Ge(dummy*dummy-4, 0, evaluate=False), dummy, s.S.Reals)
    assert _intervals(result) == {
        (-s.oo, s.Integer(-2), True, False), (s.Integer(2), s.oo, False, True),
    }


def test_inexact_and_custom_function_surfaces_do_not_enter_closed_exact_lane(monkeypatch):
    module = importlib.import_module("sympy.solvers.inequalities")
    x = s.Symbol("inequality_custom_x")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported surfaces must not enter exact QQ(x) admission")

    class UserFunction(s.Function):
        def _eval_evalf(self, *args):
            raise AssertionError("custom evaluation hook must not run")

    monkeypatch.setattr(_native, "solve_real_rational_inequality_native", forbidden)
    for expression in (s.Float("1.5")*x, UserFunction(x, evaluate=False)):
        assert module._solve_rational_inequality(s.Ge(expression, 0, evaluate=False), x) is None


def test_exact_root_hashing_preserves_bound_generator_independence_and_root_index():
    x, y = s.Symbol("hash_root_x"), s.Symbol("hash_root_y")
    first = s.CRootOf(x**4-x-1, 0)
    renamed = s.CRootOf(y**4-y-1, 0)
    second = s.CRootOf(x**4-x-1, 1)
    assert first == renamed
    assert hash(first) == hash(renamed)
    assert first != second
    assert len({first, renamed, second}) == 2
    mapping = {first: "first", second: "second"}
    assert mapping[renamed] == "first"
    assert mapping[s.CRootOf(y**4-y-1, 1)] == "second"


def test_rational_coefficient_root_hashes_match_equal_scaled_and_renamed_factors():
    x, y = s.Symbol("hash_rational_x"), s.Symbol("hash_rational_y")
    polynomial = x**4-s.Rational(1, 3)*x-s.Rational(1, 2)
    renamed = y**4-s.Rational(1, 3)*y-s.Rational(1, 2)
    first = s.CRootOf(polynomial, 0)
    equal = [s.CRootOf(renamed, 0), s.CRootOf(6*polynomial, 0)]
    for other in equal:
        assert first == other
        assert hash(first) == hash(other)
    assert len(set([first, *equal])) == 1
    # The complete vector retains internal zeros and the local root index.
    assert s.CRootOf(x**4-s.Rational(1, 3)*x*x-s.Rational(1, 2), 0) != first
