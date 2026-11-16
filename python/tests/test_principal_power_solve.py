"""Complete finite inverse branches of principal rational powers.

The two- and three-root counterexamples were checked independently against
SymPy 1.14.0. Root forms here use exact rational phases; no floating tolerance
establishes membership or completeness.
"""

import pytest
import sympy as s


@pytest.mark.parametrize("power,target,phases", [
    (s.Rational(3, 2), -1, [s.Rational(-2, 3), s.Rational(2, 3)]),
    (s.Rational(5, 2), 1, [0, s.Rational(-4, 5), s.Rational(4, 5)]),
    (s.Rational(7, 3), 1, [0, s.Rational(-6, 7), s.Rational(6, 7)]),
    (s.Rational(-3, 2), -1, [s.Rational(-2, 3), s.Rational(2, 3)]),
    (s.Rational(-5, 2), 1, [0, s.Rational(-4, 5), s.Rational(4, 5)]),
])
def test_solve_returns_all_finite_principal_inverse_branches(power, target, phases):
    x = s.Symbol("principal_power_coordinate")
    source = x**power - target
    roots = s.solve(source, x)
    expected = [s.Integer(-1)**angle for angle in phases]
    assert len(roots) == len(expected)
    assert set(roots) == set(expected)
    for root in roots:
        assert s.simplify(source.subs(x, root)) == 0
    expected_set = s.FiniteSet(*expected)
    assert s.solveset(source, x) == expected_set
    assert s.solveset(source, x, domain=s.S.Complexes) == expected_set


def test_affine_base_and_nonunit_rational_target_preserve_both_branches():
    x = s.Symbol("principal_affine_coordinate")
    source = (2*x + 3)**s.Rational(3, 2) + 8
    roots = s.solve(source, x)
    expected = [
        (4*s.Integer(-1)**s.Rational(-2, 3) - 3)/2,
        (4*s.Integer(-1)**s.Rational(2, 3) - 3)/2,
    ]
    assert len(roots) == 2
    for wanted in expected:
        assert any(s.expand(root - wanted) == 0 for root in roots)
    assert s.solve(source, x, dict=True) == [{x: root} for root in roots]
    assert s.solve(source, x, set=True) == ([x], {(root,) for root in roots})


@pytest.mark.parametrize("power,target,expected", [
    (s.Rational(3, 2), 1, [1]),
    (s.Rational(1, 2), -1, []),
    (s.Rational(2, 3), -1, []),
    (s.Rational(-2, 3), -1, []),
    (s.Rational(3, 2), 0, [0]),
    (s.Rational(-3, 2), 0, []),
])
def test_principal_range_and_zero_poles_do_not_add_other_sheet_roots(power, target, expected):
    x = s.Symbol("principal_range_coordinate")
    assert s.solve(x**power - target, x) == expected


def test_undecided_phase_and_excessive_branch_work_do_not_publish_partial_lists():
    x, y = s.symbols("principal_bound_coordinate principal_unknown_target")
    with pytest.raises(ValueError, match="resolved target phase"):
        s.solve(x**s.Rational(3, 2) - y, x)
    with pytest.raises(ValueError, match="branch count"):
        s.solve(x**s.Rational(13, 2) - 1, x)


@pytest.mark.parametrize("power,target,expected", [
    (s.Rational(3, 2), -1, []),
    (s.Rational(-3, 2), -1, []),
    (s.Rational(5, 2), 1, [1]),
    (s.Rational(-5, 2), 1, [1]),
])
def test_real_coordinate_and_real_domain_exclude_nonreal_inverse_branches(power, target, expected):
    real_x = s.Symbol("real_principal_coordinate", real=True)
    x = s.Symbol("principal_domain_coordinate")
    assert s.solve(real_x**power - target, real_x) == expected
    assert s.solveset(x**power - target, x, domain=s.S.Reals) == s.FiniteSet(*expected)


@pytest.mark.parametrize("power", [s.Rational(2, 3), s.Rational(-2, 3), s.Rational(4, 5)])
def test_negative_base_rational_phases_have_exact_nonreal_assumptions(power):
    negative = s.Symbol("principal_negative_base", negative=True)
    for base in [s.Integer(-1), negative]:
        value = base**power
        assert s.ask(s.Q.real(value)) is False
        assert s.ask(s.Q.positive(value)) is False
        assert s.ask(s.Q.zero(value)) is False
        assert s.ask(s.Q.nonzero(value)) is False
        assert s.ask(s.Q.complex(value)) is True
        assert s.ask(s.Q.finite(value)) is True
        assert s.ask(s.Q.infinite(value)) is False
    positive = s.Symbol("principal_positive_base", positive=True)
    unknown = s.Symbol("principal_unknown_base")
    assert s.ask(s.Q.real(positive**power)) is True
    assert s.ask(s.Q.real(unknown**power)) is None


def test_integer_powers_and_undefined_held_sources_keep_their_domain_rules():
    negative = s.Symbol("principal_integral_power_base", negative=True)
    for exponent in [0, 1, 2, -1, -2]:
        assert s.ask(s.Q.real(negative**exponent)) is True
    held = s.Pow(s.Integer(0), s.Rational(-2, 3), evaluate=False)
    assert s.ask(s.Q.real(held)) is None


def test_nonreal_assumption_alone_does_not_claim_finite_complex_membership():
    unknown = s.Symbol("principal_nonreal_unknown_domain", real=False)
    complex_value = s.Symbol("principal_declared_complex", real=False, complex=True)
    for predicate in [s.Q.complex, s.Q.finite]:
        assert s.ask(predicate(unknown)) is None
        assert s.ask(predicate(complex_value)) is True
    assert s.ask(s.Q.infinite(unknown)) is None
    assert s.ask(s.Q.infinite(complex_value)) is False
