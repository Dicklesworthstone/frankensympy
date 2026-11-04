"""Complete bounded prime-field factorization and exact source-content binding."""

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def test_characteristic_two_and_odd_prime_complete_factorizations():
    x = s.Symbol("modular_x")
    assert s.factor_list(x**2 + 1, x, modulus=2) == (1, [(x + 1, 2)])
    assert s.factor(x**2 + 1, x, modulus=2) == (x + 1)**2
    assert s.factor_list(x**4 + 1, x, modulus=3) == (
        1, [(x**2 + x - 1, 1), (x**2 - x - 1, 1)]
    )
    assert s.factor_list(x**4 + 2*x**2 + 1, modulus=5) == (
        1, [(x + 2, 2), (x - 2, 2)]
    )


def test_vanishing_derivatives_and_multiplicity_composition():
    x = s.Symbol("modular_repeated_x")
    assert s.factor_list(x**6 + x**3 + 1, x, modulus=3) == (1, [(x - 1, 6)])
    assert s.factor_list((x**2 + 1)**3 * (x + 1)**2, x, modulus=2) == (
        1, [(x + 1, 8)]
    )
    assert s.factor_list(x**64 + 1, x, modulus=2) == (1, [(x + 1, 64)])


def test_exact_exterior_content_and_modular_leading_units_match_pinned_conventions():
    x = s.Symbol("modular_content_x")
    fixtures = [
        (x/2 + 1, 3, (s.Rational(1, 2), [(x - 1, 1)])),
        (x/3 + 1, 3, (s.Rational(1, 3), [(x, 1)])),
        (2*x**3 + 2*x, 3, (2, [(x, 1), (x**2 + 1, 1)])),
        (2*x + 2, 3, (-1, [(x + 1, 1)])),
        (4*x**2 + 8*x + 4, 5, (-1, [(x + 1, 2)])),
        (3*x**2 + x + 1, 3, (1, [(x + 1, 1)])),
        (3*x + 3, 3, (0, [])),
        (s.Integer(7), 5, (7, [])),
        (s.Integer(0), 5, (0, [])),
    ]
    for source, prime, expected in fixtures:
        assert s.factor_list(source, x, modulus=prime) == expected


def test_factor_and_factor_list_preserve_their_distinct_pinned_content_conventions():
    x = s.Symbol("modular_paired_content_x")
    fixtures = [
        (2*x + 2, 3, s.Mul(2, x + 1, evaluate=False), (-1, [(x + 1, 1)])),
        (-2*x - 2, 3, s.Mul(-2, x + 1, evaluate=False), (1, [(x + 1, 1)])),
        (4*x**2 + 8*x + 4, 5, 4*(x + 1)**2, (-1, [(x + 1, 2)])),
        (x/2 + 1, 3, s.Mul(s.Rational(1, 2), x - 1, evaluate=False),
         (s.Rational(1, 2), [(x - 1, 1)])),
        (-x/2 - 1, 3, s.Mul(s.Rational(-1, 2), x - 1, evaluate=False),
         (s.Rational(-1, 2), [(x - 1, 1)])),
        (x/3 + 1, 3, x/3, (s.Rational(1, 3), [(x, 1)])),
        (3*x**2 + x + 1, 3, x + 1, (1, [(x + 1, 1)])),
        (s.Integer(0), 3, s.Integer(0), (0, [])),
    ]
    for source, prime, expression, listing in fixtures:
        assert s.factor(source, x, modulus=prime) == expression
        assert s.factor_list(source, x, modulus=prime) == listing


def test_original_symbol_identity_survives_modular_lowering_and_lifting():
    for x in (s.Symbol("same", positive=True), s.Dummy("same")):
        _, factors = s.factor_list(x**2 + 1, x, modulus=2)
        assert factors == [(x + 1, 2)]
        assert factors[0][0].free_symbols == {x}
    x = s.Symbol("collision", positive=True)
    y = s.Symbol("collision", negative=True)
    with pytest.raises(ValueError):
        s.factor_list(x + y, x, modulus=3)


@pytest.mark.parametrize("exponent", [-1, 0])
def test_original_undefined_children_cannot_disappear_during_normalization(exponent):
    x = s.Symbol("modular_held_x")
    invalid = s.Pow(0, exponent, evaluate=False)
    source = s.Add(x**2 + 1, s.Mul(0, invalid, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.factor_list(source, x, modulus=2)


def test_direct_bridge_rejects_forged_content_and_missing_source_components():
    x = s.Symbol("modular_bridge_x")
    source = x/2 + 1
    with pytest.raises(ValueError):
        _native.prime_field_factor_source_native(
            _native_expr(source), _symbol_binding(x), 3,
            _native_expr(s.Integer(1)), [(_native_expr(x + 2), 1)],
        )
    with pytest.raises(ValueError):
        _native.prime_field_factor_source_native(
            _native_expr(source), _symbol_binding(x), 3,
            _native_expr(s.Rational(1, 2)), [],
        )


def test_coordinated_content_and_component_rebinding_cannot_change_the_modular_source():
    x = s.Symbol("modular_rebinding_x")
    # Every proposed decomposition reconstructs the source exactly over QQ.
    for source, content, component in [
        (x, s.Rational(1, 3), 3*x),
        (x/2, s.Rational(1, 6), 3*x),
        (3*x + 3, s.Integer(3), x + 1),
        (x**2/2 + x/2 + s.Rational(1, 3), s.Rational(1, 12), 6*x**2 + 6*x + 4),
    ]:
        with pytest.raises(ValueError):
            _native.prime_field_factor_source_native(
                _native_expr(source), _symbol_binding(x), 3,
                _native_expr(content), [(_native_expr(component), 1)],
            )


def test_characteristic_divisible_exterior_coefficients_are_an_explicit_bounded_refusal():
    x = s.Symbol("modular_exterior_refusal_x")
    # Full GF Poly support is not implemented. This lane deliberately refuses
    # pinned normalization forms which retain characteristic factors outside
    # the modular polynomial, while ordinary component zero reduction works.
    for source in [3*x, s.Integer(3), s.Integer(-3), s.Rational(3, 2)]:
        with pytest.raises(ValueError):
            s.factor_list(source, x, modulus=3)
        with pytest.raises(ValueError):
            s.factor(source, x, modulus=3)
    assert s.factor_list(3*x + 3, x, modulus=3) == (0, [])
    # factor retains the extracted exterior 3, whereas factor_list reduces
    # the full integer numerator. Only the latter is admitted by this lane.
    with pytest.raises(ValueError):
        s.factor(3*x + 3, x, modulus=3)


@pytest.mark.parametrize("modulus", [0, 1, 4, 9, 341, -3, 65523, 2.0, True])
def test_only_bounded_exact_prime_moduli_are_admitted(modulus):
    x = s.Symbol("modular_refusal_x")
    with pytest.raises(ValueError):
        s.factor_list(x**2 + 1, x, modulus=modulus)


def test_unsupported_domains_and_poly_modulus_do_not_fall_back_to_rational_arithmetic():
    x, y = s.symbols("modular_limit_x modular_limit_y")
    with pytest.raises(NotImplementedError):
        s.factor_list(x + y, modulus=2)
    with pytest.raises(ValueError):
        s.factor_list(x**65 + 1, x, modulus=2)
    with pytest.raises(ValueError):
        s.factor_list(1/x + 1, x, modulus=2)
    with pytest.raises(NotImplementedError):
        s.Poly(x**2 + 1, x, modulus=2)
    ordinary = s.Poly(x**2 + 1, x)
    with pytest.raises(NotImplementedError):
        s.Poly(ordinary, modulus=2)
    with pytest.raises(NotImplementedError):
        s.factor_list(ordinary, modulus=2)
