"""Tensor-product stencils and sampling semantics of the public finite differences."""

import pytest
import sympy as s


def test_default_samples_the_field_instead_of_analytically_differentiating_it():
    x = s.Symbol("sample_x")
    # Unit centered differences include a cubic truncation term of 1/4.
    assert s.differentiate_finite(x**3, x, x0=3) == s.Rational(109, 4)
    assert s.differentiate_finite(x**3, x, points=[-1, 1], x0=0) == 1


def test_products_are_translated_as_one_sample():
    x = s.Symbol("sample_x")
    f, g = s.Function("field_f"), s.Function("field_g")
    expected = (f(x+1)*g(x+1) - f(x-1)*g(x-1))/2
    assert (s.differentiate_finite(f(x)*g(x), x, points=2) - expected).expand() == 0


def test_mixed_partial_uses_all_four_corners():
    x, y = s.symbols("sample_x sample_y")
    field = s.Function("sample_field")(x, y)
    half = s.Rational(1, 2)
    f = field.func
    expected = f(x+half, y+half) - f(x+half, y-half) - f(x-half, y+half) + f(x-half, y-half)
    held = s.Derivative(field, x, y)
    for result in (s.differentiate_finite(field, x, y), held.as_finite_difference()):
        assert not result.atoms(s.Derivative)
        assert (result - expected).expand() == 0


def test_mixed_orders_nonuniform_grid_and_offset():
    x, y = s.symbols("sample_x sample_y")
    source = (x**2 + 3*x)*y**2
    # This degree-two grid differentiates quadratics exactly on both axes.
    result = s.Derivative(source, x, y, x).as_finite_difference([0, 1, 4], x0=2)
    assert result == 8


def test_explicit_partial_keeps_other_coordinates_formal():
    x, y = s.symbols("sample_x sample_y")
    f = s.Function("sample_field")
    held = s.Derivative(f(x, y), x, y)
    result = held.as_finite_difference(2, wrt=x)
    expected = (s.Derivative(f(x+1, y), y) - s.Derivative(f(x-1, y), y))/2
    assert (result - expected).expand() == 0


def test_a_derivative_inside_a_product_moves_to_the_sampling_point():
    x = s.Symbol("sample_x")
    f, g = s.Function("field_f"), s.Function("field_g")
    source = f(x)*s.Derivative(g(x), x)
    expected = (f(x+s.S.Half)*(g(x+1)-g(x))
                - f(x-s.S.Half)*(g(x)-g(x-1)))
    assert (s.differentiate_finite(source, x) - expected).expand() == 0


def test_table_and_mixed_grid_limits_precede_sampling(monkeypatch):
    from sympy.calculus import finite_diff
    x, y = s.symbols("sample_x sample_y")
    with pytest.raises(ValueError, match="distinct"):
        s.finite_diff_weights(1, [0, 0], 0)
    with pytest.raises(ValueError, match="at least one"):
        s.finite_diff_weights(1, [], 0)
    monkeypatch.setattr(finite_diff, "_MAX_WEIGHT_CELLS", 32)
    with pytest.raises(ValueError, match="weight table"):
        s.finite_diff_weights(2, range(10**9), 0)
    monkeypatch.setattr(finite_diff, "_MAX_STENCIL_SAMPLES", 3)
    with pytest.raises(ValueError, match="stencil sample"):
        s.Derivative(x*y, x, y).as_finite_difference()


def test_mixed_stencil_samples_coordinates_simultaneously():
    x, y = s.symbols("capture_x capture_y")
    # The x and y samples may be centered on the same symbolic value.
    source = s.Derivative(x**2, x, y)
    assert source.as_finite_difference(2, x0=y).expand() == 0
    with pytest.raises(NotImplementedError, match="remaining derivative coordinate"):
        source.as_finite_difference(2, x0=y, wrt=x)
