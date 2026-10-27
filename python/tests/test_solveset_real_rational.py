"""Complete real rational solveset on the actual native/Python boundary."""

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding


def _forbid_approximation(*args, **kwargs):
    raise AssertionError("exact real solveset must never use numerical roots")


def test_unsolvable_quintic_and_general_quartic_return_complete_exact_sets(monkeypatch):
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_approximation)
    x = s.Symbol("x")
    quintic = x**5 - x + 1
    quartic = x**4 - x - 1
    assert s.solveset(quintic, x, s.S.Reals) == s.FiniteSet(s.CRootOf(quintic, 0))
    assert s.solveset(quartic, x, s.S.Reals) == s.FiniteSet(
        s.CRootOf(quartic, 0), s.CRootOf(quartic, 1)
    )
    root = next(iter(s.solveset(quintic, x, s.S.Reals)))
    assert root.is_real is True
    assert str(root.evalf(30)) == "-1.16730397826141868425604589985"


def test_mixed_repeated_roots_return_distinct_real_solutions_only(monkeypatch):
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_approximation)
    x = s.Symbol("x")
    f = x**5 - x + 1
    expression = (f * (x**2 - 2) * (x - 3))**2 * (x**2 + 1)
    assert s.solveset(expression, x, s.S.Reals) == s.FiniteSet(
        s.CRootOf(f, 0), -s.sqrt(2), s.sqrt(2), 3
    )
    assert s.solveset((x - 1)**2 + s.Rational(1, 10**80), x, s.S.Reals) == s.S.EmptySet


def test_original_irrational_denominator_factors_are_removed_with_all_multiplicities():
    x = s.Symbol("x")
    f = x**5 - x + 1
    numerator = f * (x**2 - 2)**2 * (x - 3)**3
    denominator = (x**2 - 2) * (x - 3)
    expression = s.Mul(numerator, s.Pow(denominator, -1, evaluate=False), evaluate=False)
    assert s.solveset(expression, x, s.S.Reals) == s.FiniteSet(s.CRootOf(f, 0))
    assert s.solveset(s.Pow(x**2 - 2, -1, evaluate=False), x, s.S.Reals) == s.S.EmptySet


def test_nested_reciprocals_and_held_zero_powers_preserve_hidden_holes():
    x = s.Symbol("x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    nested = s.Pow(reciprocal, -1, evaluate=False)
    assert s.solveset(nested, x, s.S.Reals) == s.S.EmptySet
    held_zero = s.Mul(0, s.Pow(reciprocal, 0, evaluate=False), evaluate=False)
    solution = s.solveset(held_zero, x, s.S.Reals)
    assert solution.contains(0) is False
    assert solution.contains(1) is True
    assert solution.contains(-1) is True
    assert solution != s.S.Reals


def test_rational_sums_preserve_both_original_denominators():
    x = s.Symbol("x")
    minus, plus = x - 1, x + 1
    equation = s.Add(
        s.Pow(minus, -1, evaluate=False),
        s.Mul(-2, s.Pow(plus, -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    assert s.solveset(equation, x, s.S.Reals) == s.FiniteSet(3)
    identity = s.Add(
        s.Mul(plus, s.Pow(minus * plus, -1, evaluate=False), evaluate=False),
        s.Mul(-1, s.Pow(minus, -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    solution = s.solveset(identity, x, s.S.Reals)
    assert solution.contains(-1) is False
    assert solution.contains(1) is False
    assert solution.contains(0) is True


def test_identity_equations_preserve_irrational_and_algebraic_exclusions():
    x = s.Symbol("x")
    f = x**5 - x + 1
    reciprocal = s.Pow(f * (x**2 - 2), -1, evaluate=False)
    equation = s.Eq(reciprocal, reciprocal, evaluate=False)
    solution = s.solveset(equation, x, s.S.Reals)
    assert not isinstance(solution, s.ConditionSet)
    assert solution != s.S.Reals
    for excluded in (s.CRootOf(f, 0), -s.sqrt(2), s.sqrt(2)):
        assert solution.contains(excluded) is False
    # Unrelated comparisons may remain unknown, but may not prove false facts.
    assert solution.contains(0) is not False
    assert solution.contains(s.I) is not True
    assert s.solveset(equation, domain=s.S.Reals) == solution


def test_constant_equations_and_nowhere_defined_held_forms():
    x = s.Symbol("x")
    assert s.solveset(0, x, s.S.Reals) == s.S.Reals
    assert s.solveset(0, domain=s.S.Reals) == s.S.Reals
    assert s.solveset(7, x, s.S.Reals) == s.S.EmptySet
    reciprocal_zero = s.Pow(0, -1, evaluate=False)
    undefined_zero = s.Mul(0, reciprocal_zero, evaluate=False)
    assert s.solveset(undefined_zero, x, s.S.Reals) == s.S.EmptySet
    assert s.solveset(s.Eq(reciprocal_zero, reciprocal_zero, evaluate=False), domain=s.S.Reals) == s.S.EmptySet


def test_real_domain_ignores_variable_sign_assumptions_without_losing_identity():
    positive = s.Symbol("x", positive=True)
    assert s.solveset(positive**2 - 4, positive, s.S.Reals) == s.FiniteSet(-2, 2)
    f = positive**5 - positive + 1
    solution = s.solveset(f, positive, s.S.Reals)
    assert solution == s.FiniteSet(s.CRootOf(f, 0))
    plain = s.Symbol("x")
    assert _native.solve_real_rational_native(_native_expr(positive), _symbol_binding(plain)) is None
    dummy = s.Dummy("x")
    assert s.solveset(dummy**2 - 4, dummy, s.S.Reals) == s.FiniteSet(-2, 2)


def test_recognized_exact_resource_failures_retain_the_original_conditions():
    x = s.Symbol("x")
    huge_degree = s.Pow(x, 65, evaluate=False)
    solution = s.solveset(huge_degree, x, s.S.Reals)
    assert isinstance(solution, s.ConditionSet)
    assert solution.sym == x
    assert solution.base_set == s.S.Reals
    assert solution.condition.lhs == huge_degree
    with pytest.raises(ValueError, match="resource limit"):
        _native.solve_real_rational_native(_native_expr(huge_degree), _symbol_binding(x))
    oversized = s.Mul(0, s.Integer(1 << 4096), evaluate=False)
    assert isinstance(s.solveset(oversized, x, s.S.Reals), s.ConditionSet)


def test_float_functions_and_fractional_powers_cannot_enter_the_exact_native_grammar():
    x = s.Symbol("x")
    for expression in (s.Float("1.5") * x - 3, s.sin(x), s.sqrt(x)):
        assert _native.solve_real_rational_native(_native_expr(expression), _symbol_binding(x)) is None


def test_ordinary_linear_quadratic_and_nonreal_domain_routes_remain_available():
    x = s.Symbol("x")
    assert s.solveset(2*x - 1, x, s.S.Reals) == s.FiniteSet(s.Rational(1, 2))
    assert s.solveset(x**2 - 2, x, s.S.Reals) == s.FiniteSet(-s.sqrt(2), s.sqrt(2))
    assert s.solveset(x**2 + 1, x, s.S.Reals) == s.S.EmptySet
    assert s.solveset(x**2 + 1, x) == s.FiniteSet(-s.I, s.I)


def test_custom_integer_subclasses_do_not_enter_the_exact_closed_lane(monkeypatch):
    class CustomInteger(int):
        def _sympy_(self):
            raise AssertionError("custom conversion hook must not run")

    def forbidden(*args, **kwargs):
        raise AssertionError("custom subclasses must not enter exact QQ admission")

    monkeypatch.setattr(_native, "solve_real_rational_native", forbidden)
    x = s.Symbol("x")
    # The existing native literal boundary already requires exact int type;
    # the real-equation lane must preserve that refusal without running hooks.
    with pytest.raises(TypeError, match="exact built-in int"):
        s.solveset(CustomInteger(0), x, s.S.Reals)


def test_exact_quadratic_radicals_keep_rational_arithmetic_and_real_order():
    from sympy.polys.rootoftools import _real_radical

    x = s.Symbol("x")
    third, two_thirds = s.Rational(1, 3), s.Rational(2, 3)
    shifted = s.Rational(3, 2)*x**2 + x - s.Rational(5, 6)
    cases = [
        (shifted, [-third-s.sqrt(6)/3, -third+s.sqrt(6)/3]),
        (-shifted, [-third-s.sqrt(6)/3, -third+s.sqrt(6)/3]),
        (-3*x**2 + 2, [-s.sqrt(6)/3, s.sqrt(6)/3]),
        (-3*x**2 + 2*x, [0, two_thirds]),
    ]
    for expression, expected in cases:
        polynomial = s.Poly(expression, x)
        assert [_real_radical(polynomial, index) for index in (0, 1)] == expected
        assert s.solveset(expression, x, s.S.Reals) == s.FiniteSet(*expected)
    repeated = (2*x-1)**2
    assert s.solveset(repeated, x, s.S.Reals) == s.FiniteSet(s.Rational(1, 2))
    assert s.real_roots(repeated, multiple=False) == [(s.Rational(1, 2), 2)]
