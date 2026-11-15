"""Exact and formal Fourier coefficients; actual-shell regression targets."""

import copy
import importlib
import pickle

import pytest
import sympy as s

from sympy.series.fourier import FourierSeries, fourier_series


def test_existing_linear_fourier_series_and_standard_import():
    x = s.Symbol("fourier_x")
    source = fourier_series(x, (x, -s.pi, s.pi))
    assert s.fourier_series is fourier_series
    assert source.truncate(3) == 2*s.sin(x)-s.sin(2*x)+s.Rational(2, 3)*s.sin(3*x)


@pytest.mark.parametrize("harmonic", [8, 29, 1000])
def test_sparse_high_harmonics_are_never_replaced_by_zero(harmonic):
    x = s.Symbol("fourier_high")
    source = fourier_series(s.sin(harmonic*x), (x, -s.pi, s.pi))
    assert source.truncate(1) == s.sin(harmonic*x)
    assert source.term(harmonic) == s.sin(harmonic*x)
    assert source.term(harmonic-1) == 0
    assert source.truncate(10) == s.sin(harmonic*x)


def test_sparse_finite_terms_are_emitted_in_harmonic_order():
    x = s.Symbol("fourier_sparse")
    expression = 2*s.sin(7*x)+3*s.cos(11*x)+s.sin(29*x)
    source = fourier_series(expression, (x, -s.pi, s.pi))
    assert source.truncate(2) == 2*s.sin(7*x)+3*s.cos(11*x)
    assert source.truncate(20) == expression
    assert list(source.truncate(None)) == [2*s.sin(7*x), 3*s.cos(11*x), s.sin(29*x)]


def test_more_than_six_polynomial_harmonics_are_computed():
    x = s.Symbol("fourier_polynomial")
    source = fourier_series(x, (x, -s.pi, s.pi))
    expected = sum((s.Rational(2*(-1)**(k+1), k)*s.sin(k*x)
                    for k in range(1, 13)), s.S.Zero)
    assert source.truncate(12) == expected
    assert source.term(13) == s.Rational(2, 13)*s.sin(13*x)
    assert source.bn[12] == s.Rational(2, 13)
    assert source.an[12] == 0


def test_constant_mean_counts_as_a_truncation_term():
    x = s.Symbol("fourier_square")
    source = fourier_series(x*x, (x, -s.pi, s.pi))
    assert source.a0 == s.pi**2/3
    assert source.truncate(1) == s.pi**2/3
    assert source.truncate(3) == s.pi**2/3-4*s.cos(x)+s.cos(2*x)


def test_one_harmonic_combines_its_sine_and_cosine_parts():
    x = s.Symbol("fourier_mixed")
    source = fourier_series(x*x+x, (x, -s.pi, s.pi))
    assert source.truncate(2) == s.pi**2/3-4*s.cos(x)+2*s.sin(x)


@pytest.mark.parametrize("constant", [0, 3, s.Rational(2, 7)])
def test_constant_series_terminates_without_search(constant):
    x = s.Symbol("fourier_constant")
    assert fourier_series(constant, (x, -s.pi, s.pi)) == constant


def test_scaled_period_and_exact_phase():
    x = s.Symbol("fourier_scaled")
    source = fourier_series(s.sin(7*x/2+s.pi/3), (x, -2*s.pi, 2*s.pi))
    expected = s.sqrt(3)*s.cos(7*x/2)/2+s.sin(7*x/2)/2
    assert s.expand(source.truncate(1)-expected) == 0


def test_rational_interval_and_high_pi_frequency():
    x = s.Symbol("fourier_interval")
    expression = s.cos(9*s.pi*x)
    source = fourier_series(expression, (x, 1, 3))
    assert source.truncate(1) == expression
    assert source.period == 2


def test_product_to_sum_retains_both_high_modes():
    x = s.Symbol("fourier_product")
    source = fourier_series(s.sin(3*x)*s.cos(8*x), (x, -s.pi, s.pi))
    assert source.truncate(1) == -s.sin(5*x)/2
    assert source.truncate(2) == (s.sin(11*x)-s.sin(5*x))/2


@pytest.mark.parametrize("function,sign", [(s.sin, -1), (s.cos, 1)])
def test_trigonometric_powers_have_exact_finite_support(function, sign):
    x = s.Symbol("fourier_power")
    source = fourier_series(function(8*x)**2, (x, -s.pi, s.pi))
    assert source.truncate(1) == s.Rational(1, 2)
    assert source.truncate(10) == s.Rational(1, 2)+sign*s.cos(16*x)/2


def test_polynomial_weighted_trigonometric_coefficients_include_resonance():
    x = s.Symbol("fourier_weight")
    source = fourier_series(x*s.cos(8*x), (x, -s.pi, s.pi))
    assert source.term(8) == -s.sin(8*x)/16
    assert source.term(9) == s.Rational(18, 17)*s.sin(9*x)


def test_noninteger_source_frequency_is_not_falsely_finite():
    x = s.Symbol("fourier_half")
    source = fourier_series(x*s.sin(x/2), (x, -s.pi, s.pi))
    assert source.a0 == 4/s.pi
    assert source.term(1) == -s.Rational(40, 9)*s.cos(x)/s.pi


def test_construction_is_lazy_and_coefficients_are_cached(monkeypatch):
    from sympy.core import _native

    x = s.Symbol("fourier_lazy")
    calls = []
    original = _native.integrate_native

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(_native, "integrate_native", record)
    source = fourier_series(x, (x, -s.pi, s.pi))
    assert calls == []
    first = source.term(3)
    count = len(calls)
    assert count > 0
    assert source.term(3) == first
    assert len(calls) == count
    source.term(4)
    assert len(calls) > count


def test_native_operational_failures_are_terminal(monkeypatch):
    from sympy.core import _native

    x = s.Symbol("fourier_failure")
    source = fourier_series(x, (x, -s.pi, s.pi))

    def fail(*args):
        raise RuntimeError("native integral interrupted")

    monkeypatch.setattr(_native, "integrate_native", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        source.term(1)


def test_a_false_native_primitive_cannot_publish_a_coefficient(monkeypatch):
    from sympy.core import _native, _native_expr

    x = s.Symbol("fourier_bad_primitive")
    source = fourier_series(x, (x, -s.pi, s.pi))
    monkeypatch.setattr(_native, "integrate_native", lambda *args: _native_expr(s.S.Zero))
    with pytest.raises(ValueError, match="derivative check"):
        source.term(1)


def test_unsupported_coefficients_remain_explicit_integrals():
    x = s.Symbol("fourier_formal")
    expression = s.exp(x*x)
    source = fourier_series(expression, (x, -s.pi, s.pi))
    expected = s.Integral(expression, (x, -s.pi, s.pi))/(2*s.pi)
    assert source.truncate(1) == expected
    assert source.term(7).has(s.Integral)
    assert source.truncate(2).has(s.Integral)


@pytest.mark.parametrize("outer", ["zero_product", "zeroth_power"])
def test_original_held_poles_are_preserved_as_formal_coefficients(outer):
    x = s.Symbol("fourier_held")
    reciprocal = s.Pow(x, -1, evaluate=False)
    expression = (s.Mul(0, reciprocal, evaluate=False) if outer == "zero_product"
                  else s.Pow(reciprocal, 0, evaluate=False))
    source = fourier_series(expression, (x, -s.pi, s.pi))
    assert source.function is expression
    assert source.a0.has(s.Integral)
    assert source.a0 != 0


def test_same_printed_parameters_and_dummy_coordinates_keep_identity():
    x = s.Symbol("fourier_same", real=True)
    parameter = s.Symbol("fourier_same", positive=True)
    source = fourier_series(x+parameter, (x, -s.pi, s.pi))
    assert source.x is x
    assert source.a0 == parameter
    assert source.a0.free_symbols == {parameter}
    dummy = s.Dummy("fourier_same")
    dummy_source = fourier_series(s.sin(8*dummy), (dummy, -s.pi, s.pi))
    assert dummy_source.x is dummy
    assert dummy_source.truncate(1).free_symbols == {dummy}


def test_source_reconstruction_copy_and_pickle():
    x = s.Symbol("fourier_copy")
    source = fourier_series(s.sin(8*x)+s.cos(12*x), (x, -s.pi, s.pi))
    for restored in (source.func(*source.args), copy.copy(source), pickle.loads(pickle.dumps(source))):
        assert restored == source
        assert restored.truncate(2) == source.truncate(2)


def test_source_replacement_after_construction_is_rejected():
    x = s.Symbol("fourier_mutation")
    source = fourier_series(s.sin(8*x), (x, -s.pi, s.pi))
    source._struct_args = (s.cos(8*x), source.args[1])
    with pytest.raises(RuntimeError, match="source changed"):
        source.truncate(1)


def test_inconsistent_retained_source_is_rejected_before_generation():
    x = s.Symbol("fourier_retained_source")
    source = s.Pow(x, 2, evaluate=False)
    source._args = (x, s.Integer(3))
    with pytest.raises(ValueError, match="retained arguments"):
        fourier_series(source, (x, -s.pi, s.pi))


@pytest.mark.parametrize("slot", ["_args", "_value"])
def test_nested_limit_tuple_mutation_cannot_reuse_cached_coefficients(slot):
    from sympy.core import _native_expr

    x = s.Symbol("fourier_limit_mutation")
    source = fourier_series(x, (x, -s.pi, s.pi))
    source.term(1)
    signature = source.args[1]
    changed = s.Tuple(x, 0, 2*s.pi)
    setattr(signature, slot, changed.args if slot == "_args" else _native_expr(changed))
    with pytest.raises(RuntimeError, match="source mutated"):
        source.term(1)


def test_mutation_during_native_generation_is_rejected_before_endpoints(monkeypatch):
    from sympy.core import _native

    x = s.Symbol("fourier_midcall_mutation")
    source = fourier_series(x, (x, -s.pi, s.pi))
    original = _native.integrate_native

    def mutate(*args):
        source.args[1]._args = (x, s.S.Zero, 2*s.pi)
        return original(*args)

    monkeypatch.setattr(_native, "integrate_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        source.term(1)


def test_request_and_algebra_bounds_are_explicit():
    x = s.Symbol("fourier_bound")
    source = fourier_series(x, (x, -s.pi, s.pi))
    with pytest.raises(ValueError, match="term count"):
        source.truncate(257)
    with pytest.raises(ValueError, match="harmonic"):
        source.term(4097)
    with pytest.raises(ValueError, match="exponent"):
        fourier_series(x**33, (x, -s.pi, s.pi))
    with pytest.raises(TypeError, match="exact integer"):
        source.truncate(2.5)
    assert source.truncate(0) == 0


def test_cache_exhaustion_does_not_return_a_partial_sum(monkeypatch):
    module = importlib.import_module("sympy.series.fourier")
    x = s.Symbol("fourier_cache_bound")
    source = fourier_series(x, (x, -s.pi, s.pi))
    monkeypatch.setattr(module, "_MAX_COEFFICIENTS", 1)
    source.term(1)
    with pytest.raises(ValueError, match="cache bound"):
        source.term(2)


def test_parameter_only_power_growth_is_bounded_before_expansion():
    x, parameter = s.symbols("fourier_growth_x fourier_growth_parameter")
    coefficient = s.Pow(s.Pow(parameter, 32, evaluate=False), 3, evaluate=False)
    source = s.Mul(x, coefficient, evaluate=False)
    with pytest.raises(ValueError, match="exponent bound"):
        fourier_series(source, (x, -s.pi, s.pi))


def test_held_numeric_power_growth_cannot_exhaust_coefficient_memory():
    x = s.Symbol("fourier_large_scalar")
    coefficient = s.Pow(s.Integer(2**2048), 3, evaluate=False)
    with pytest.raises(ValueError, match="literal"):
        fourier_series(s.Mul(x, coefficient, evaluate=False), (x, -s.pi, s.pi))


@pytest.mark.parametrize("bounds", [(0, 0), (1, -1)])
def test_empty_or_reversed_periods_are_rejected(bounds):
    x = s.Symbol("fourier_bad_interval")
    with pytest.raises(ValueError, match="start < end"):
        fourier_series(x, (x, *bounds))


def test_ambiguous_variable_and_unbounded_period_are_rejected():
    x, y = s.symbols("fourier_v fourier_w")
    with pytest.raises(ValueError, match="multivariate"):
        fourier_series(x+y)
    with pytest.raises(NotImplementedError, match="endpoints"):
        fourier_series(x, (x, 0, s.oo))


def test_custom_source_is_refused_without_argument_or_printer_callbacks():
    calls = []

    class Custom(s.Expr):
        @property
        def args(self):
            calls.append("args")
            raise AssertionError("custom arguments must not run")

        def __str__(self):
            calls.append("str")
            raise AssertionError("custom printing must not run")

    x = s.Symbol("fourier_custom")
    source = Custom(x)
    with pytest.raises(NotImplementedError):
        fourier_series(source, (x, -s.pi, s.pi))
    assert calls == []
