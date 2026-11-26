"""Public complex evalf: justified component digits and original source binding.

Expected values use independently evaluated Decimal identities and exact
arithmetic. These tests require the actual mathematical PyO3 extension; no
mock symbolic engine or upstream production fallback is installed.
"""

from decimal import Decimal, ROUND_HALF_EVEN, localcontext

import pytest
import sympy as s

from sympy.core import (
    _ExactDecimalFloat, _certified_complex_float, _native, _native_expr,
    _wrap, supervised_callback_session,
)


PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651"
)


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        return +value


def components(value):
    real, imaginary = value.as_real_imag()
    return Decimal(str(real)), Decimal(str(imaginary))


def sin_cos(value):
    # Independent small-argument reference: 200 terms at guarded precision.
    sine_term = sine = value
    cosine_term = cosine = Decimal(1)
    squared = value*value
    for k in range(1, 200):
        sine_term *= -squared / ((2*k)*(2*k+1))
        cosine_term *= -squared / ((2*k-1)*(2*k))
        sine += sine_term
        cosine += cosine_term
    return sine, cosine


@pytest.mark.parametrize("digits", [15, 40, 80])
def test_principal_logarithm_has_two_independently_rounded_components(digits):
    source = s.log(1+s.I, evaluate=False)
    with localcontext() as context:
        context.prec = digits + 25
        expected = Decimal(2).ln()/2, PI/4
    for result in (s.N(source, digits), source.evalf(digits), source.n(digits)):
        assert components(result) == tuple(rounded(value, digits) for value in expected)
        real, imaginary = result.as_real_imag()
        assert isinstance(real, s.Float)
        assert isinstance(imaginary, s.Float)
        assert real.dps == imaginary.dps == digits


@pytest.mark.parametrize("sign", [-1, 1])
def test_principal_square_root_keeps_the_correct_half_plane(sign):
    source = s.Pow(3+4*sign*s.I, s.Rational(1, 2), evaluate=False)
    assert components(s.N(source, 60)) == (Decimal(2), Decimal(sign))
    assert components(source.evalf(60)) == (Decimal(2), Decimal(sign))


def test_negative_real_cube_root_uses_the_principal_complex_branch():
    source = s.Pow(-8, s.Rational(1, 3), evaluate=False)
    with localcontext() as context:
        context.prec = 100
        imaginary = Decimal(3).sqrt()
    assert components(s.N(source, 60)) == (Decimal(1), rounded(imaginary, 60))


def test_complex_exponent_uses_the_existing_principal_power_kernel():
    source = s.Pow(1+s.I, 2+s.I, evaluate=False)
    with localcontext() as context:
        context.prec = 90
        angle = Decimal(2).ln()/2
        sine, cosine = sin_cos(angle)
        magnitude = (Decimal(2).ln()-PI/4).exp()
        expected = -magnitude*sine, magnitude*cosine
    assert components(s.N(source, 50)) == tuple(rounded(value, 50) for value in expected)


@pytest.mark.parametrize("function", [s.sin, s.cos])
def test_complex_trigonometric_components_do_not_use_binary64(function):
    source = function(1+s.I, evaluate=False)
    with localcontext() as context:
        context.prec = 90
        sine, cosine = sin_cos(Decimal(1))
        positive, negative = Decimal(1).exp(), Decimal(-1).exp()
        sinh, cosh = (positive-negative)/2, (positive+negative)/2
        expected = (sine*cosh, cosine*sinh) if function is s.sin else (cosine*cosh, -sine*sinh)
    assert components(source.evalf(50)) == tuple(rounded(value, 50) for value in expected)


def test_pure_imaginary_and_exact_zero_components_are_not_noise():
    assert components(s.N(s.I, 80)) == (Decimal(0), Decimal(1))
    assert components(s.N(-s.I, 80)) == (Decimal(0), Decimal(-1))
    negative_axis_log = s.log(-1, evaluate=False)
    assert components(s.N(negative_axis_log, 70)) == (Decimal(0), rounded(PI, 70))
    assert s.N(s.Integer(0), 40) is s.S.Zero
    for signed_zero in (0.0, -0.0):
        source = s.Add(s.Float(signed_zero), s.I, evaluate=False)
        real, imaginary = s.N(source, 40).as_real_imag()
        assert real is s.S.Zero
        assert real.is_zero is True
        assert real.is_nonzero is False
        assert Decimal(str(imaginary)) == 1


@pytest.mark.parametrize("sign", [-1, 1])
def test_a_nonzero_component_below_binary64_is_never_chopped(sign):
    tiny = s.Rational(sign, 10**400)
    source = s.Add(1, s.Mul(tiny, s.I, evaluate=False), evaluate=False)
    result = s.N(source, 50)
    real, imaginary = result.as_real_imag()
    assert Decimal(str(real)) == 1
    assert Decimal(str(imaginary)) == Decimal(sign).scaleb(-400)
    assert type(imaginary) is _ExactDecimalFloat
    assert imaginary.is_zero is False
    assert imaginary.is_nonzero is True
    assert imaginary.is_positive is (sign > 0)
    assert imaginary.is_negative is (sign < 0)
    assert imaginary.is_nonnegative is (sign > 0)
    assert imaginary.is_nonpositive is (sign < 0)
    assert imaginary.is_real is True
    assert imaginary.is_finite is True
    assert imaginary.is_infinite is False
    assert imaginary.is_integer is False
    assert "e-400" in str(result)
    # Re-evaluation must read the retained exact decimal, not its zero f64.
    assert components(result.evalf(25)) == (Decimal(1), Decimal(sign).scaleb(-400))


@pytest.mark.parametrize("sign", [-1, 1])
def test_finite_components_above_binary64_keep_finite_metadata(sign):
    source = s.Add(s.Integer(sign * 10**400), s.I, evaluate=False)
    result = s.N(source, 40)
    real, imaginary = result.as_real_imag()
    assert Decimal(str(real)) == Decimal(sign).scaleb(400)
    assert Decimal(str(imaginary)) == 1
    assert real.is_zero is False
    assert real.is_nonzero is True
    assert real.is_positive is (sign > 0)
    assert real.is_negative is (sign < 0)
    assert real.is_real is True
    assert real.is_finite is True
    assert real.is_infinite is False
    assert real.is_integer is True
    assert components(result.evalf(25)) == (Decimal(sign).scaleb(400), Decimal(1))


def test_decimal_rendering_and_decomposition_keep_all_justified_digits():
    result = s.N(s.log(1+s.I, evaluate=False), 70)
    real, imaginary = result.as_real_imag()
    assert len(str(real).replace(".", "").lstrip("0")) == 70
    assert len(str(imaginary).replace(".", "").lstrip("0")) == 70
    rendered = str(result)
    assert str(real) in rendered
    assert str(imaginary) in rendered
    assert "__fsymFloat" not in rendered
    negative = s.N(s.log(1-s.I, evaluate=False), 70)
    assert " + -" not in str(negative)
    assert str(negative.as_real_imag()[1]).lstrip("-") in str(negative)


@pytest.mark.parametrize("sign, separator", [(1, " + "), (-1, " - ")])
def test_rectangular_rendering_orders_parts_and_trims_only_nested_padding(sign, separator):
    result = s.N(-1 + sign * 2 * s.I, 15)
    real, imaginary = result.as_real_imag()
    assert str(result) == "-1.0" + separator + "2.0*I"
    assert str(real) == "-1.00000000000000"
    assert str(imaginary).lstrip("-") == "2.00000000000000"
    assert str(s.N(sign * 2 * s.I, 15)) == ("-" if sign < 0 else "") + "2.0*I"


@pytest.mark.parametrize("exponent", [-400, 400])
def test_rectangular_rendering_keeps_signed_scientific_coefficients(exponent):
    coefficient = -s.Rational(10) ** exponent
    source = s.Add(1, s.Mul(coefficient, s.I, evaluate=False), evaluate=False)
    result = s.N(source, 40)
    assert str(result) == f"1.0 - 1.0e{exponent:+d}*I"
    assert components(result) == (Decimal(1), Decimal(-1).scaleb(exponent))


@pytest.mark.parametrize("real, imaginary, expected", [
    ("1", "1e-400", "1.0e-400*I + 1.0"),
    ("-1", "1e-400", "-1.0 + 1.0e-400*I"),
    ("2e-400", "1e-400", "2.0e-400 + 1.0e-400*I"),
    ("1", "1e-320", "1.0 + 1.0e-320*I"),
])
def test_rectangular_sort_key_underflow_only_changes_display_order(real, imaginary, expected):
    source = s.Add(s.Rational(real), s.Mul(s.Rational(imaginary), s.I, evaluate=False), evaluate=False)
    result = s.N(source, 40)
    assert str(result) == expected
    assert components(result) == (Decimal(real), Decimal(imaginary))


def test_finite_float_leaves_are_exact_binary64_inputs_not_reparsed_printing():
    source = s.Add(s.Float(0.1), s.Mul(s.Float(-0.2), s.I, evaluate=False), evaluate=False)
    expected = Decimal.from_float(0.1), Decimal.from_float(-0.2)
    assert components(s.N(source, 45)) == tuple(rounded(value, 45) for value in expected)


@pytest.mark.parametrize("exponent", [-1, 0])
@pytest.mark.parametrize("evaluate", [lambda source: s.N(source, 40), lambda source: source.evalf(40)])
def test_original_undefined_children_cannot_be_hidden(exponent, evaluate):
    invalid = s.Pow(0, exponent, evaluate=False)
    source = s.Add(s.I, s.Mul(0, invalid, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        evaluate(source)


def test_float_lowering_preserves_a_held_zero_times_a_pole():
    source = s.Add(
        s.I,
        s.Mul(s.Float(0), s.Pow(0, -1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    before = source.args, _native_expr(source)
    with pytest.raises(ValueError):
        s.N(source, 40)
    assert source.args == before[0]
    assert _native_expr(source) == before[1]


def test_high_precision_never_uses_the_binary64_complex_probe(monkeypatch):
    def forbidden(_source):
        raise AssertionError("binary64 complex probe was used")

    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    assert components(s.N(1+2*s.I, 60)) == (Decimal(1), Decimal(2))


@pytest.mark.parametrize("exception", [MemoryError, OverflowError, RuntimeError, ValueError])
def test_admitted_operational_failures_do_not_fall_back(monkeypatch, exception):
    source = s.log(1+s.I, evaluate=False)

    def fail(_source, _digits):
        raise exception("injected complex evaluation failure")

    def forbidden(*_args):
        raise AssertionError("evaluation restarted after an admitted failure")

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", fail)
    monkeypatch.setattr(_native, "evalf_decimal_native", forbidden)
    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    for evaluate in (lambda: s.N(source, 40), lambda: source.evalf(40)):
        with pytest.raises(exception, match="injected complex evaluation failure"):
            evaluate()


def test_native_unsupported_syntax_is_distinct_from_an_admitted_domain_error():
    x = s.Symbol("complex_decimal_free")
    assert _native.evalf_complex_decimal_native(_native_expr(x), 30) is None
    assert _native.evalf_complex_decimal_native(_native.py_function("unregistered_numeric", _native.py_integer(1)), 30) is None
    with pytest.raises(ValueError):
        _native.evalf_complex_decimal_native(
            _native.Pow(_native.py_integer(0), _native.py_integer(-1), evaluate=False).as_expr(), 30,
        )
    with pytest.raises(NotImplementedError):
        s.N(x+s.I, 30)


def test_real_inverse_function_digits_survive_complex_lane_admission():
    source = s.asin(s.Rational(1, 2), evaluate=False)
    admitted = _certified_complex_float(source, 40)
    assert admitted is not None
    with localcontext() as context:
        context.prec = 80
        expected = PI/6
    assert Decimal(str(s.N(source, 40))) == rounded(expected, 40)


def test_a_genuine_custom_evalf_hook_remains_supervised_and_runs_once():
    calls = []

    class Custom(s.Function):
        def _eval_evalf(self, precision):
            calls.append(precision)
            return 1+2*s.I

    source = Custom(0, evaluate=False)
    assert _certified_complex_float(source, 40) is None
    assert calls == []
    with supervised_callback_session():
        result = s.N(source, 40)
    assert len(calls) == 1
    assert components(result) == (Decimal(1), Decimal(2))


def test_same_named_custom_function_cannot_impersonate_a_native_function():
    calls = []

    class log(s.Function):
        def __str__(self):
            calls.append("printer")
            return "1"

    source = log(1+s.I, evaluate=False)
    assert _certified_complex_float(source, 40) is None
    assert calls == []


def test_stale_retained_children_are_rejected_before_native_evaluation(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    source._args = (s.Integer(2), s.I)

    def forbidden(*_args):
        raise AssertionError("stale original source reached numeric evaluation")

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", forbidden)
    with pytest.raises(ValueError, match="retained arguments"):
        _certified_complex_float(source, 40)


def test_conflicting_retained_decimal_owners_are_rejected(monkeypatch):
    source = s.Add(_ExactDecimalFloat("1e-400", 40), s.I, evaluate=False)
    source._surface_args = (_ExactDecimalFloat("2e-400", 40), s.I)

    def forbidden(*_args):
        raise AssertionError("conflicting decimal source reached numeric evaluation")

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", forbidden)
    with pytest.raises(ValueError, match="retained decimal"):
        _certified_complex_float(source, 40)


def test_invalid_builtin_native_storage_is_terminal(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    source._value = object()

    def forbidden(*_args):
        raise AssertionError("invalid source reached numeric evaluation")

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", forbidden)
    with pytest.raises(ValueError, match="exact native storage"):
        _certified_complex_float(source, 40)


def test_mutation_between_native_evaluation_and_publication_is_terminal(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    original = _native.evalf_complex_decimal_native

    def mutate(native, digits):
        result = original(native, digits)
        source._args = (s.Integer(2), s.I)
        return result

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        _certified_complex_float(source, 40)


def test_changed_arithmetic_implementation_does_not_enter_the_fast_path(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    monkeypatch.setattr(s.Add, "args", property(lambda _self: (s.Integer(7),)))
    with pytest.raises(NotImplementedError, match="original built-in"):
        _certified_complex_float(source, 40)


def test_source_ownership_changes_during_output_charging_are_caught(monkeypatch):
    import sympy.core as core

    source = s.Add(1, s.I, evaluate=False)
    evaluated = [False]
    original = _native.evalf_complex_decimal_native

    def native_call(native, digits):
        result = original(native, digits)
        evaluated[0] = True
        return result

    def charge(_steps=1):
        if evaluated[0]:
            source._args = (s.Integer(3), s.I)

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", native_call)
    monkeypatch.setattr(core, "_charge_nested_if_active", charge)
    with pytest.raises(RuntimeError, match="source mutated"):
        _certified_complex_float(source, 40)


def test_original_held_sources_are_not_rewritten_on_success():
    source = s.Pow(s.Add(3, 4*s.I, evaluate=False), s.Rational(1, 2), evaluate=False)
    arguments, native = source.args, _native_expr(source)
    assert components(s.N(source, 40)) == (Decimal(2), Decimal(1))
    assert source.args is arguments
    assert _native_expr(source) == native


def test_source_depth_limit_precedes_recursive_component_evaluation():
    native = _native.py_integer(1)
    for _ in range(70):
        native = _native.py_function("sin", native)
    source = _wrap(native)
    for evaluate in (
        lambda: _certified_complex_float(source, 30),
        lambda: s.N(source, 30),
        lambda: source.evalf(30),
    ):
        with pytest.raises(ValueError, match="(depth|limit)"):
            evaluate()


@pytest.mark.parametrize("digits", [0, 1001])
def test_native_precision_envelope_is_explicit(digits):
    with pytest.raises(ValueError, match="precision"):
        _native.evalf_complex_decimal_native(_native_expr(s.I), digits)



_INVERSE_FUNCTIONS = (s.asin, s.acos, s.atan, s.asinh, s.acosh, s.atanh)


def _inverse_series_reference(name, real_sign, imaginary_sign, digits):
    """Independent convergent power series in the disk |z| < 0.4.

    The native implementation uses complex enclosures; this reference sums
    scalar Decimal coefficient recurrences. With |z|**2 = 34/225 < 0.16,
    the first omitted term after 240 terms is below 10**-190. The real
    component formulas for acos/acosh fix their principal branches.
    """
    with localcontext() as context:
        context.prec = digits + 45
        real = Decimal(real_sign) / 3
        imaginary = Decimal(imaginary_sign) / 5
        square = real*real - imaginary*imaginary, 2*real*imaginary
        series = "asin" if name in ("acos", "acosh") else name
        term_real = total_real = real
        term_imaginary = total_imaginary = imaginary
        for k in range(1, 240):
            if series in ("asin", "asinh"):
                factor = Decimal((2*k - 1)**2) / ((2*k)*(2*k + 1))
            else:
                factor = Decimal(2*k - 1) / (2*k + 1)
            if series in ("atan", "asinh"):
                factor = -factor
            next_real = (term_real*square[0] - term_imaginary*square[1])*factor
            term_imaginary = (term_real*square[1] + term_imaginary*square[0])*factor
            term_real = next_real
            total_real += term_real
            total_imaginary += term_imaginary
        if name in ("acos", "acosh"):
            total_real, total_imaginary = PI/2 - total_real, -total_imaginary
        if name == "acosh":
            total_real, total_imaginary = (
                -imaginary_sign*total_imaginary,
                imaginary_sign*total_real,
            )
        return rounded(total_real, digits), rounded(total_imaginary, digits)


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
@pytest.mark.parametrize("real_sign, imaginary_sign", [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_complex_inverse_functions_have_fifty_justified_digits_in_every_quadrant(
    function, real_sign, imaginary_sign,
):
    argument = s.Add(s.Rational(real_sign, 3), s.Rational(imaginary_sign, 5)*s.I, evaluate=False)
    source = function(argument, evaluate=False)
    expected = _inverse_series_reference(function.__name__, real_sign, imaginary_sign, 50)
    arguments, native = source.args, _native_expr(source)
    for result in (s.N(source, 50), source.evalf(50), source.n(50)):
        assert components(result) == expected
        assert all(type(part) is _ExactDecimalFloat and part.dps == 50
                   for part in result.as_real_imag())
    assert source.args is arguments
    assert _native_expr(source) == native


def _inverse_cut_reference(name, sign):
    # Exact-axis conventions from the pinned SymPy 1.14.0 elementary
    # function tests and principal logarithmic definitions. No signed-zero
    # convention from the platform's binary64 complex functions is used.
    with localcontext() as context:
        context.prec = 100
        acosh_two = (Decimal(2) + Decimal(3).sqrt()).ln()
        half_log_three = Decimal(3).ln()/2
        if name == "asin":
            value = sign*PI/2, -sign*acosh_two
        elif name == "acos":
            value = Decimal(0) if sign > 0 else PI, sign*acosh_two
        elif name == "atan":
            value = sign*PI/2, sign*half_log_three
        elif name == "asinh":
            value = sign*acosh_two, sign*PI/2
        elif name == "acosh":
            value = acosh_two, Decimal(0) if sign > 0 else PI
        else:
            assert name == "atanh"
            value = sign*half_log_three, -sign*PI/2
        return tuple(rounded(component, 50) for component in value)


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
@pytest.mark.parametrize("sign", [-1, 1])
def test_complex_inverse_functions_use_the_pinned_exact_cut_values(function, sign):
    argument = 2*sign*s.I if function in (s.atan, s.asinh) else s.Integer(2*sign)
    source = function(argument, evaluate=False)
    assert components(s.N(source, 50)) == _inverse_cut_reference(function.__name__, sign)


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
@pytest.mark.parametrize("side", [-1, 1])
def test_complex_inverse_cut_sides_remain_distinct(function, side):
    epsilon = s.Rational(side, 10**24)
    if function in (s.atan, s.asinh):
        argument = s.Add(epsilon, 2*s.I, evaluate=False)
    elif function is s.acosh:
        argument = s.Add(-2, epsilon*s.I, evaluate=False)
    else:
        argument = s.Add(2, epsilon*s.I, evaluate=False)
    with localcontext() as context:
        context.prec = 100
        acosh_two = (Decimal(2) + Decimal(3).sqrt()).ln()
        half_log_three = Decimal(3).ln()/2
        if function is s.asin:
            limit = PI/2, side*acosh_two
        elif function is s.acos:
            limit = Decimal(0), -side*acosh_two
        elif function is s.atan:
            limit = side*PI/2, half_log_three
        elif function is s.asinh:
            limit = side*acosh_two, PI/2
        elif function is s.acosh:
            limit = acosh_two, side*PI
        else:
            limit = half_log_three, side*PI/2
    actual = components(s.N(function(argument, evaluate=False), 50))
    assert all(abs(value - expected) < Decimal("1e-20")
               for value, expected in zip(actual, limit))


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
def test_complex_inverse_functions_preserve_the_callers_cumulative_ledger(function):
    argument = s.Add(s.Rational(1, 3), s.I/5, evaluate=False)
    source = function(argument, evaluate=False)
    expected = _inverse_series_reference(function.__name__, 1, 1, 50)
    budget = s.EvaluationBudget(
        max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30,
    )
    for operation in (
        lambda: s.N(source, 50, budget=budget),
        lambda: source.evalf(50, budget=budget),
        lambda: source.n(50, budget=budget),
    ):
        before = budget.remaining_steps
        assert components(operation()) == expected
        assert budget.remaining_steps < before
    budget.cancel()
    before = budget.remaining_steps
    with pytest.raises(s.EvaluationCancelled):
        source.evalf(50, budget=budget)
    assert budget.remaining_steps == before


@pytest.mark.parametrize("function, argument", [
    (s.atan, s.I), (s.atan, -s.I),
    (s.atanh, s.S.One), (s.atanh, s.S.NegativeOne),
])
@pytest.mark.parametrize("masked", [False, True])
def test_complex_inverse_poles_never_publish_even_under_a_held_zero(function, argument, masked):
    source = function(argument, evaluate=False)
    if masked:
        source = s.Mul(0, source, evaluate=False)
    sufficient = s.EvaluationBudget(
        max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30,
    )
    for budget in (None, sufficient):
        with pytest.raises(ValueError):
            s.N(source, 40, budget=budget)


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
def test_complex_inverse_input_definedness_precedes_cancellation_identities(function):
    invalid = s.Pow(0, -1, evaluate=False)
    argument = s.Add(1, s.I, s.Mul(0, invalid, evaluate=False), evaluate=False)
    source = function(argument, evaluate=False)
    with pytest.raises(ValueError):
        s.N(source, 50)
    with pytest.raises(ValueError):
        s.N(source, 50, budget=s.EvaluationBudget(
            max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30,
        ))


@pytest.mark.parametrize("function", _INVERSE_FUNCTIONS)
def test_complex_inverse_sources_are_rechecked_after_native_return(function, monkeypatch):
    source = function(1+s.I, evaluate=False)
    original = _native.evalf_complex_decimal_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source._args = (s.Integer(2),)
        return result

    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.N(source, 40)


def test_custom_inverse_function_name_does_not_grant_native_evaluation():
    calls = []

    class asin(s.Function):
        def __str__(self):
            calls.append("printer")
            return "0"

    source = asin(1+s.I, evaluate=False)
    assert _certified_complex_float(source, 50) is None
    assert calls == []
    with pytest.raises(NotImplementedError):
        s.N(source, 50, budget=s.EvaluationBudget())
    assert calls == []
