"""Exact root arithmetic evaluated before any decimal Float conversion."""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction

import pytest
import sympy as s
from sympy.core import _exact_ratio, _native, _native_expr, _symbol_binding, _wrap
from sympy.polys.rootoftools import _algebraic_expression_value


def sqrt_two(name="algebraic_eval_t", index=1):
    t = s.Symbol(name)
    return s.CRootOf(t*t-2, t, index)


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        return +value


def exact_ratio(native):
    return Fraction(*_exact_ratio(_wrap(native)))


@pytest.mark.parametrize("digits", [16, 50, 100])
def test_root_arithmetic_matches_independent_decimal_square_root(digits):
    root = sqrt_two()
    with localcontext() as context:
        context.prec = digits + 30
        alpha = Decimal(2).sqrt()
        cases = [
            (1 + root, 1 + alpha),
            (3*root-7, 3*alpha-7),
            ((root+1)/(root-1), (alpha+1)/(alpha-1)),
            (root + 1/(root+1), 2*alpha-1),
        ]
    for expression, expected in cases:
        assert Decimal(str(expression.evalf(digits))) == rounded(expected, digits)
        assert Decimal(str(s.N(expression, digits))) == rounded(expected, digits)


def test_N_of_an_exact_root_reaches_its_existing_high_precision_refinement():
    root = sqrt_two()
    assert str(s.N(root, 80)) == str(root.evalf(80))
    assert str(s.N(root, 80)).startswith("1.41421356237309504880168872420969807856967187537694807317667973799")


def test_algebraic_relations_and_rational_values_are_exact():
    root = sqrt_two()
    quotient = s.Mul(
        root,
        s.Pow(s.Mul(2, root, evaluate=False), -1, evaluate=False),
        evaluate=False,
    )
    for expression, value in [
        (root*root-2, 0),
        (1+root**2, 3),
        ((root+1)*(root-1), 1),
        (quotient, s.Rational(1, 2)),
    ]:
        result = _algebraic_expression_value(expression, 80)
        numerator, denominator = _exact_ratio(value)
        assert exact_ratio(result.lower) == Fraction(numerator, denominator)
        assert exact_ratio(result.upper) == Fraction(numerator, denominator)
        assert Decimal(str(s.N(expression, 80))) == Decimal(numerator)/Decimal(denominator)


def test_original_root_index_controls_sign_and_coordinate_value():
    negative, positive = sqrt_two(index=0), sqrt_two(index=1)
    with localcontext() as context:
        context.prec = 90
        alpha = Decimal(2).sqrt()
        expected = [1-alpha, 1+alpha]
    for root, reference in zip([negative, positive], expected):
        assert Decimal(str(s.N(1+root, 60))) == rounded(reference, 60)


def test_bound_generator_renaming_does_not_create_a_second_primitive_root():
    first = sqrt_two("algebraic_alpha_first")
    second = sqrt_two("algebraic_alpha_second")
    expression = s.Add(first, second, evaluate=False)
    with localcontext() as context:
        context.prec = 90
        expected = 2*Decimal(2).sqrt()
    assert Decimal(str(s.N(expression, 60))) == rounded(expected, 60)


def test_large_cancellation_is_refined_before_rounding():
    root = sqrt_two()
    truncated = Fraction("1.414213562373095048801688724209698078569671875376948073176679737990732478462107")
    expression = 10**80*(root-s.Rational(truncated.numerator, truncated.denominator))
    with localcontext() as context:
        context.prec = 150
        expected = Decimal(10)**80*(Decimal(2).sqrt()-Decimal(truncated.numerator)/Decimal(truncated.denominator))
    assert Decimal(str(s.N(expression, 50))) == rounded(expected, 50)


def test_receipt_has_an_independently_checkable_enclosure_and_is_immutable():
    root = sqrt_two()
    receipt = _algebraic_expression_value(1+root, 60)
    lower, upper = exact_ratio(receipt.lower)-1, exact_ratio(receipt.upper)-1
    assert 0 < lower <= upper
    assert lower*lower <= 2 <= upper*upper
    assert upper-lower < Fraction(1, 10**59)
    assert receipt.index == 1
    assert receipt.precision == 60
    with pytest.raises(AttributeError):
        receipt.decimal = "123"
    with pytest.raises(TypeError):
        _native.AlgebraicValue()


@pytest.mark.parametrize("exponent", [-1, 0])
def test_original_zero_base_is_rejected_even_when_hidden_in_a_zero_product(exponent):
    root = sqrt_two()
    base = s.Add(root**2, -2, evaluate=False)
    forbidden = s.Pow(base, exponent, evaluate=False)
    expression = s.Add(root, s.Mul(0, forbidden, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        expression.evalf(50)
    with pytest.raises(ValueError):
        s.N(expression, 50)


def test_cancelled_inverse_keeps_its_original_algebraic_pole():
    root = sqrt_two()
    base = s.Add(root**2, -2, evaluate=False)
    expression = s.Mul(base, s.Pow(base, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        s.N(expression, 50)


def test_distinct_roots_keep_their_selected_embeddings():
    first = sqrt_two()
    t = s.Symbol("algebraic_other_t")
    second = s.CRootOf(t*t-3, t, 1)
    with localcontext() as context:
        context.prec = 90
        expected = Decimal(2).sqrt() + Decimal(3).sqrt()
    assert Decimal(str(s.N(s.Add(first, second, evaluate=False), 50))) == rounded(expected, 50)
    opposite = s.Add(sqrt_two(index=0), first, evaluate=False)
    receipt = _algebraic_expression_value(opposite, 50)
    assert exact_ratio(receipt.lower) == exact_ratio(receipt.upper) == 0


def test_transcendental_and_approximate_root_composites_refuse_this_exact_lane():
    root = sqrt_two()
    for expression in [s.sin(root), s.Add(root, s.Float("0.25"), evaluate=False)]:
        with pytest.raises(NotImplementedError, match="rational arithmetic"):
            s.N(expression, 50)


def test_native_receipt_is_bound_to_the_typed_variable_and_exact_polynomial():
    plain = s.Symbol("algebraic_typed")
    assumed = s.Symbol("algebraic_typed", real=True)
    coefficients = [_native_expr(s.Integer(value)) for value in [-2, 0, 1]]
    assert _native.evaluate_real_algebraic_native(
        _native_expr(1+plain), _symbol_binding(assumed), coefficients, 1, 50
    ) is None
    with pytest.raises(ValueError):
        _native.evaluate_real_algebraic_native(
            _native_expr(1+plain), _symbol_binding(plain), coefficients, 2, 50
        )
    with pytest.raises(ValueError):
        _native.evaluate_real_algebraic_native(
            _native_expr(1+plain), _symbol_binding(plain),
            [_native_expr(s.Integer(0))], 0, 50,
        )


@pytest.mark.parametrize("digits", [0, 513, True])
def test_composite_precision_limits_are_explicit(digits):
    expression = 1+sqrt_two()
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        s.N(expression, digits)


def test_shape_system_coordinates_can_be_evaluated_without_float_substitution():
    x, y = s.symbols("numeric_shape_x numeric_shape_y")
    polynomial = y**5+y**4-4*y**3-3*y**2+3*y+1
    solutions = s.solve_poly_system([x-y*y-1, polynomial], x, y)
    assert len(solutions) == 5
    for coordinate, root in solutions:
        assert not coordinate.free_symbols
        numeric = s.N(coordinate, 50)
        assert isinstance(numeric, s.Float)
        receipt = _algebraic_expression_value(coordinate, 50)
        assert exact_ratio(receipt.lower) > 1
        assert str(numeric) == str(coordinate.evalf(50))
        assert root.is_real is True
