"""Complete finite polynomial systems through the actual typed native boundary."""

import pytest
import sympy as s


def test_rational_overdetermined_system_and_generator_order():
    x, y, z = s.symbols("checked_poly_x checked_poly_y checked_poly_z")
    equations = [2*x/3 + y - s.Rational(1, 3), 4*x - 2*y, z - x - y, 8*z - 3]
    assert s.solve_poly_system(equations, x, y, z) == [(s.Rational(1, 8), s.Rational(1, 4), s.Rational(3, 8))]
    assert s.solve_poly_system(equations, [z, x, y]) == [(s.Rational(3, 8), s.Rational(1, 8), s.Rational(1, 4))]


def test_cartesian_and_degenerate_specialization_branches_are_complete():
    x, y = s.symbols("checked_cartesian_x checked_cartesian_y")
    assert set(s.solve_poly_system([x*x-4, y*y-9], x, y)) == {(-2, -3), (-2, 3), (2, -3), (2, 3)}
    assert set(s.solve_poly_system([x*y, x*x+x, y*y-y], x, y)) == {(-1, 0), (0, 0), (0, 1)}


def test_univariate_constraints_use_common_polynomial_roots():
    x = s.Symbol("checked_common_x")
    expected = {(-s.sqrt(2),), (s.sqrt(2),)}
    assert set(s.solve_poly_system([x*x-2, (x*x-2)*(x+1)], x)) == expected
    assert set(s.solve_poly_system([x*x-2, (x*x-2)*(x+1)])) == expected
    assert s.solve_poly_system([x*x-2, x-1], x) is None


def test_radical_complex_and_repeated_roots_keep_exact_finite_tuples():
    x, y = s.symbols("checked_radical_x checked_radical_y")
    assert set(s.solve_poly_system([x-y, y*y-2], x, y)) == {(-s.sqrt(2), -s.sqrt(2)), (s.sqrt(2), s.sqrt(2))}
    assert set(s.solve_poly_system([x*x+1, y-x], x, y)) == {(-s.I, -s.I), (s.I, s.I)}
    assert s.solve_poly_system([(x-1)**2, (y+2)**3], x, y) == [(1, -2)]
    assert s.nonlinsolve([(x-1)**2, (y+2)**3], x, y) == s.FiniteSet((1, -2))


def test_tiny_exact_inconsistency_and_unsupported_quintic_are_distinct():
    x, y = s.symbols("checked_tiny_x checked_tiny_y")
    tiny = s.Rational(1, 10**80)
    assert s.solve_poly_system([x-1, x-1-tiny], x) is None
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([x**5-x+1, y-x], x, y)


@pytest.mark.parametrize("power", [-1, 0])
def test_held_undefined_children_are_checked_before_zero_products(power):
    x, y = s.symbols("checked_held_x checked_held_y")
    hidden = s.Mul(0, s.Pow(0, power, evaluate=False), evaluate=False)
    equation = s.Add(x, hidden, evaluate=False)
    for sources, generators in [([equation], (x,)), ([equation], ()), ([equation, y], (x, y))]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.solve_poly_system(sources, *generators)


def test_held_equation_preserves_both_original_undefined_sides():
    x = s.Symbol("checked_held_eq_x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    relation = s.Eq(reciprocal, reciprocal, evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([relation, x-1], x)
    assert s.solve_poly_system([s.Eq(x + 1, 3, evaluate=False)], x) == [(2,)]


def test_no_generator_path_cannot_misclassify_held_constant_sources():
    for original in [s.Pow(0, -1, evaluate=False), s.Pow(0, 0, evaluate=False)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.solve_poly_system([original])
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([s.Eq(1, 1, evaluate=False)])
    assert s.solve_poly_system([7]) is None


def test_positive_dimension_remains_explicit_for_the_finite_solver():
    x, y = s.symbols("checked_infinite_x checked_infinite_y")
    for source in [[], [0], [x*y]]:
        with pytest.raises(NotImplementedError):
            s.solve_poly_system(source, x, y)


def test_same_spelling_typed_generators_remain_distinct():
    plain = s.Symbol("checked_same_x")
    positive = s.Symbol("checked_same_x", positive=True)
    assert s.solve_poly_system([plain-1, positive-2], plain, positive) == [(1, 2)]
    with pytest.raises((ValueError, NotImplementedError)):
        s.solve_poly_system([plain-1, positive-2], plain)
    with pytest.raises(ValueError):
        s.solve_poly_system([plain-1], plain, plain)


def test_approximate_and_custom_nodes_cannot_hide_behind_zero():
    x = s.Symbol("checked_exact_source_x")
    opaque = s.Function("checked_opaque")
    for child in [s.Float(0.125), s.sin(x), opaque(x)]:
        held = s.Add(x, s.Mul(0, child, evaluate=False), evaluate=False)
        with pytest.raises((ValueError, NotImplementedError)):
            s.solve_poly_system([held], x)


def test_source_size_and_power_limits_refuse_before_expansion():
    x = s.Symbol("checked_bounded_x")
    for source in [s.Pow(x, 257, evaluate=False), s.Pow(2, 10**9, evaluate=False)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.solve_poly_system([source], x)
    with pytest.raises(ValueError):
        s.solve_poly_system([x]*65, x)
    variables = s.symbols("checked_dim0:9")
    with pytest.raises(ValueError):
        s.solve_poly_system([1], *variables)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_groebner_construction_validates_held_sources_before_zero_or_unit_ideals(exponent):
    x = s.Symbol("checked_groebner_held_x")
    hidden = s.Mul(0, s.Pow(0, exponent, evaluate=False), evaluate=False)
    equation = s.Add(x, hidden, evaluate=False)
    for sources in [[equation], [s.Integer(1), equation]]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.groebner(sources, x)


def test_groebner_construction_preflights_huge_powers_before_domain_inference():
    x = s.Symbol("checked_groebner_power_x")
    for source in [s.Pow(x, 257, evaluate=False), s.Pow(2, 10**9, evaluate=False)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.groebner([source], x)


def test_groebner_inference_preserves_same_spelling_typed_generators():
    from sympy.core import _symbol_binding

    plain = s.Symbol("checked_groebner_same")
    positive = s.Symbol("checked_groebner_same", positive=True)
    integer = s.Symbol("checked_groebner_same", integer=True)
    variables = [plain, positive, integer]
    equations = [plain - 1, positive - 2, integer - 3]

    def declared_order(variable):
        binding = _symbol_binding(variable)
        return variable.name, binding.name, binding.identity_hex

    expected = tuple(sorted(variables, key=declared_order))
    for sources in [equations, list(reversed(equations))]:
        basis = s.groebner(sources, domain="QQ")
        assert basis.gens == expected
        assert basis.is_zero_dimensional
        for original in equations:
            assert basis.contains(original)
        assert not basis.contains(positive - plain)
