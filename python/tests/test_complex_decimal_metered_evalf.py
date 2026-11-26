"""Principal complex decimal evaluation preserves digits and original domains.

Decimal references use independent rational Taylor sums and Python's decimal
elementary operations. These checks do not use binary64 as a high-precision
oracle. Returned Float components are rounded values, not interval receipts.
"""

from decimal import Decimal, localcontext
import pytest
import sympy as s

from sympy.core import (
    _ExactDecimalFloat, _certified_complex_float, _native, _native_expr,
)


_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651"
)


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        return +value


def sine_cosine_one(digits):
    # Alternating rational Taylor sums at x=1. The first omitted term is
    # far below the extra 30 decimal places retained by this reference.
    with localcontext() as context:
        context.prec = digits + 30
        sine = sine_term = Decimal(1)
        cosine = cosine_term = Decimal(1)
        for k in range(1, digits + 35):
            sine_term /= -(2*k)*(2*k+1)
            cosine_term /= -(2*k-1)*(2*k)
            sine += sine_term
            cosine += cosine_term
        return +sine, +cosine


def decimal_parts(value):
    real, imaginary = value.as_real_imag()
    def decimal_part(part):
        if isinstance(part, s.Float):
            assert type(part) is _ExactDecimalFloat
            return Decimal(str(part))
        assert part == 0
        return Decimal(0)
    return decimal_part(real), decimal_part(imaginary)


def evaluate(source, digits, method):
    return source.evalf(digits) if method else s.N(source, digits)


@pytest.mark.parametrize("digits", [15, 30, 50, 100])
@pytest.mark.parametrize("method", [False, True])
def test_independent_rational_components_keep_requested_significant_digits(digits, method):
    source = s.Rational(1, 3) + s.Rational(2, 7)*s.I
    result = evaluate(source, digits, method)
    with localcontext() as context:
        context.prec = digits + 30
        expected = (rounded(Decimal(1)/3, digits), rounded(Decimal(2)/7, digits))
    assert decimal_parts(result) == expected
    assert all(part.dps == digits for part in result.as_real_imag())


@pytest.mark.parametrize("digits", [30, 50, 100])
@pytest.mark.parametrize("method", [False, True])
def test_complex_exponential_agrees_with_independent_decimal_taylor_sums(digits, method):
    source = s.exp(s.I, evaluate=False)
    sine, cosine = sine_cosine_one(digits)
    assert decimal_parts(evaluate(source, digits, method)) == (
        rounded(cosine, digits), rounded(sine, digits),
    )


@pytest.mark.parametrize("method", [False, True])
def test_composed_complex_exponential_retains_both_components(method):
    digits = 50
    sine, cosine = sine_cosine_one(digits)
    with localcontext() as context:
        context.prec = digits + 30
        exponential = Decimal(1).exp()
        expected = (rounded(exponential*cosine, digits), rounded(exponential*sine, digits))
    assert decimal_parts(evaluate(s.exp(1+s.I, evaluate=False), digits, method)) == expected


@pytest.mark.parametrize("side", [-1, 1])
def test_principal_logarithms_keep_signed_argument_without_binary64(side):
    with localcontext() as context:
        context.prec = 80
        expected = (rounded(Decimal(2).ln()/2, 50), rounded(side*_PI/4, 50))
    source = s.log(1+side*s.I, evaluate=False)
    assert decimal_parts(s.N(source, 50)) == expected
    with localcontext() as context:
        context.prec = 80
        negative_axis = (rounded(Decimal(2).ln(), 50), rounded(_PI, 50))
    assert decimal_parts(s.N(s.log(-2, evaluate=False), 50)) == negative_axis


@pytest.mark.parametrize("method", [False, True])
def test_principal_roots_and_noninteger_powers(method):
    with localcontext() as context:
        context.prec = 80
        sqrt_two = rounded(Decimal(2).sqrt(), 50)
        cube_imaginary = rounded(Decimal(3).sqrt()/2, 50)
    assert decimal_parts(evaluate(s.sqrt(-2, evaluate=False), 50, method)) == (0, sqrt_two)
    assert decimal_parts(evaluate(s.sqrt(3+4*s.I, evaluate=False), 50, method)) == (2, 1)
    assert decimal_parts(evaluate(s.Pow(-1, s.Rational(1, 3), evaluate=False), 50, method)) == (
        Decimal("0.5"), cube_imaginary,
    )


def test_imaginary_trigonometry_and_explicit_complex_projections():
    with localcontext() as context:
        context.prec = 80
        e = Decimal(1).exp()
        sinh_one, cosh_one = (e-1/e)/2, (e+1/e)/2
    assert decimal_parts(s.N(s.sin(s.I, evaluate=False), 50)) == (0, rounded(sinh_one, 50))
    assert decimal_parts(s.N(s.cos(s.I, evaluate=False), 50)) == (rounded(cosh_one, 50), 0)
    source = s.exp(s.I, evaluate=False)
    sine, cosine = sine_cosine_one(50)
    assert Decimal(str(s.N(s.re(source, evaluate=False), 50))) == rounded(cosine, 50)
    assert Decimal(str(s.N(s.im(source, evaluate=False), 50))) == rounded(sine, 50)
    assert decimal_parts(s.N(s.conjugate(source, evaluate=False), 50)) == (
        rounded(cosine, 50), rounded(sine.copy_negate(), 50),
    )


@pytest.mark.parametrize("source, expected, rendered", [
    (s.I, (0, 1), "1.0*I"),
    (-s.I, (0, -1), "-1.0*I"),
    (1+2*s.I, (1, 2), "1.0 + 2.0*I"),
    (1-2*s.I, (1, -2), "1.0 - 2.0*I"),
    (-1-2*s.I, (-1, -2), "-1.0 - 2.0*I"),
])
def test_complex_float_printing_and_exact_zero_components(source, expected, rendered):
    result = s.N(source, 50)
    assert decimal_parts(result) == expected
    assert str(result) == rendered
    for part, integer in zip(result.as_real_imag(), expected):
        if integer == 0:
            assert part is s.S.Zero
            assert part.is_zero is True and part.is_nonzero is False
            assert str(part) == "0"
        else:
            assert type(part) is _ExactDecimalFloat and part.dps == 50
            assert part.is_zero is False and part.is_nonzero is True
            assert str(part) == str(integer) + "." + "0"*49


def test_printing_signs_and_real_imag_projection_do_not_rebuild_decimal_owners():
    result = s.N(s.exp(-s.I, evaluate=False), 50)
    real, imaginary = result.as_real_imag()
    sine, cosine = sine_cosine_one(50)
    assert Decimal(str(real)) == rounded(cosine, 50)
    assert Decimal(str(imaginary)) == rounded(sine.copy_negate(), 50)
    assert str(imaginary).lstrip("-") in str(result)
    assert str(-imaginary) == str(imaginary).lstrip("-")
    assert str(abs(imaginary)) == str(imaginary).lstrip("-")
    assert (-imaginary).dps == imaginary.dps == 50


def test_repeated_evaluation_rounds_retained_decimal_components_without_f64():
    first = s.N(s.exp(s.I, evaluate=False), 80)
    second = s.N(first, 30)
    sine, cosine = sine_cosine_one(80)
    assert decimal_parts(second) == (rounded(cosine, 30), rounded(sine, 30))


@pytest.mark.parametrize("method", [False, True])
def test_complex_path_never_calls_the_legacy_real_or_binary64_fallback(monkeypatch, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("complex decimal evaluation used a legacy fallback")
    monkeypatch.setattr(_native, "evalf_decimal_native", forbidden)
    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    result = evaluate(s.exp(s.I, evaluate=False), 40, method)
    sine, cosine = sine_cosine_one(40)
    assert decimal_parts(result) == (rounded(cosine, 40), rounded(sine, 40))


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("form", ["zero_product", "zero_power", "hidden_exponent"])
def test_original_held_undefined_children_refuse_before_publication(method, form):
    pole = s.Pow(0, -1, evaluate=False)
    source = {
        "zero_product": lambda: s.Mul(
            s.Float(0.0), s.Add(s.I, pole, evaluate=False), evaluate=False,
        ),
        "zero_power": lambda: s.Pow(s.Add(s.I, pole, evaluate=False), 0, evaluate=False),
        "hidden_exponent": lambda: s.Pow(s.I, s.Pow(0, 0, evaluate=False), evaluate=False),
    }[form]()
    with pytest.raises(ValueError):
        evaluate(source, 30, method)


def test_exact_float_leaf_lowering_preserves_original_held_order():
    source = s.Add(
        s.Float(0.125), s.Mul(s.Float(-0.375), s.I, evaluate=False), evaluate=False,
    )
    args, raw = source.args, _native_expr(source)
    assert decimal_parts(s.N(source, 40)) == (Decimal("0.125"), Decimal("-0.375"))
    assert source.args is args
    assert _native_expr(source) == raw


@pytest.mark.parametrize("method", [False, True])
def test_rounding_uncertainty_never_becomes_a_spurious_zero(method):
    cancellation = s.Add(s.pi, s.Mul(-1, s.pi, evaluate=False), evaluate=False)
    source = s.Add(1, s.Mul(s.I, cancellation, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="round|separat|inconclusive"):
        evaluate(source, 30, method)


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("failure", [
    ValueError, RuntimeError, MemoryError, s.EvaluationBudgetExceeded, s.EvaluationCancelled,
])
def test_native_failure_is_terminal_and_preserves_its_error_class(monkeypatch, method, failure):
    def fail(*args, **kwargs):
        raise failure("injected decimal evaluation refusal")
    def forbidden(*args, **kwargs):
        raise AssertionError("native failure selected a binary64 fallback")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", fail)
    monkeypatch.setattr(_native, "complex_value_native", forbidden)
    with pytest.raises(failure, match="injected decimal evaluation refusal"):
        evaluate(s.I, 30, method)


def test_retained_source_mutation_during_native_evaluation_is_rejected(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    original = source.args
    native = _native.evalf_complex_decimal_native
    def mutate(*args, **kwargs):
        result = native(*args, **kwargs)
        s.Add.__dict__["_args"].__set__(source, (s.Integer(2), s.I))
        return result
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mutate)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            s.N(source, 30)
    finally:
        s.Add.__dict__["_args"].__set__(source, original)


def test_inconsistent_retained_source_is_rejected_before_native_work(monkeypatch):
    source = s.Add(1, s.I, evaluate=False)
    s.Add.__dict__["_args"].__set__(source, (s.Integer(2), s.I))
    def forbidden(*args, **kwargs):
        raise AssertionError("inconsistent source reached native evaluation")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", forbidden)
    with pytest.raises(ValueError, match="retained arguments"):
        s.N(source, 30)


def test_custom_classes_and_symbol_names_do_not_enter_the_closed_decimal_lane(monkeypatch):
    effects = []
    class Opaque(s.Function):
        @property
        def args(self):
            effects.append("args")
            raise AssertionError("closed admission inspected an opaque getter")
    custom = Opaque(s.Integer(1), evaluate=False)
    assert _certified_complex_float(custom, 30) is None
    with pytest.raises(NotImplementedError):
        s.N(custom, 30)
    assert effects == []
    def forbidden(*args, **kwargs):
        raise AssertionError("a free symbol was treated as a numeric constant")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", forbidden)
    with pytest.raises(NotImplementedError):
        s.N(s.Symbol("I"), 30)
    with pytest.raises(NotImplementedError):
        s.Symbol("pi").evalf(30)


@pytest.mark.parametrize("source, expected", [
    (s.Add(s.Integer(10**400), s.I, evaluate=False), (Decimal("1e400"), Decimal(1))),
    (s.Add(1, s.Mul(s.Rational(1, 10**400), s.I, evaluate=False), evaluate=False),
     (Decimal(1), Decimal("1e-400"))),
])
def test_components_outside_binary64_preserve_decimal_values_and_metadata(source, expected):
    pair = _native.evalf_complex_decimal_native(_native_expr(source), 30)
    assert len(pair) == 2 and all(type(value) is str for value in pair)
    result = s.N(source, 30)
    assert decimal_parts(result) == expected
    for part in result.as_real_imag():
        assert part.is_finite is True
        assert part.is_zero is False
    assert decimal_parts(result.evalf(20)) == expected


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def test_bridge_budget_is_cumulative_across_decimal_evaluations():
    source = _native_expr(s.Rational(1, 3) + s.Rational(2, 7)*s.I)
    budget = ample_budget()
    expected = _native.evalf_complex_decimal_native(source, 30, budget=budget)
    remaining = budget.remaining_steps
    assert _native.evalf_complex_decimal_native(source, 30, budget=budget) == expected
    assert 0 < budget.remaining_steps < remaining < 1 << 45


@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_bridge_zero_allowances_have_operational_error_types(limit):
    budget = ample_budget(**{limit: 0})
    with pytest.raises(s.EvaluationBudgetExceeded):
        _native.evalf_complex_decimal_native(_native_expr(s.I), 30, budget=budget)


def test_bridge_cancellation_is_terminal_before_any_new_arithmetic():
    budget = ample_budget()
    budget.cancel()
    before = budget.remaining_steps
    for _ in range(2):
        with pytest.raises(s.EvaluationCancelled):
            _native.evalf_complex_decimal_native(_native_expr(s.I), 30, budget=budget)
    assert budget.remaining_steps == before


@pytest.mark.parametrize("digits", [0, 1001])
def test_bridge_precision_envelope_is_explicit(digits):
    with pytest.raises(ValueError):
        _native.evalf_complex_decimal_native(_native_expr(s.I), digits)


@pytest.mark.parametrize("source", [
    s.Integer(2), s.Rational(1, 3), s.sqrt(2), s.E, s.sin(1), s.log(2),
])
def test_existing_real_outputs_remain_single_decimal_floats(source):
    result = s.N(source, 30)
    assert type(result) is _ExactDecimalFloat
    assert result.dps == 30
    assert result.is_real is True
    assert result.as_real_imag()[1] is s.S.Zero


@pytest.mark.parametrize("failure", [
    ValueError, RuntimeError, MemoryError, s.EvaluationBudgetExceeded, s.EvaluationCancelled,
])
def test_retained_decimal_reevaluation_propagates_native_refusal(monkeypatch, failure):
    component = s.N(s.exp(s.I, evaluate=False), 40).as_real_imag()[0]
    def fail(*args, **kwargs):
        raise failure("retained decimal reevaluation refused")
    def forbidden(*args, **kwargs):
        raise AssertionError("retained decimal reevaluation used a legacy fallback")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", fail)
    monkeypatch.setattr(_native, "evalf_decimal_native", forbidden)
    with pytest.raises(failure, match="retained decimal reevaluation refused"):
        component.evalf(50)


def test_retained_decimal_reevaluation_has_an_explicit_precision_envelope():
    component = s.N(s.exp(s.I, evaluate=False), 40).as_real_imag()[0]
    assert component.evalf(40) is component
    with pytest.raises(ValueError):
        component.evalf(1001)


def test_retained_decimal_mutation_by_terminal_budget_charge_is_rejected(monkeypatch):
    import sympy.core as core
    source = s.N(s.exp(s.I, evaluate=False), 40).as_real_imag()[0]
    original = source._decimal
    evaluated = [False]
    native = _native.evalf_complex_decimal_native
    charge = core._charge_nested_if_active
    def mark_evaluation(*args):
        result = native(*args)
        evaluated[0] = True
        return result
    def mutate_on_terminal_charge(*args):
        charge(*args)
        if evaluated[0]:
            _ExactDecimalFloat.__dict__["_decimal"].__set__(source, "0.25")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mark_evaluation)
    monkeypatch.setattr(core, "_charge_nested_if_active", mutate_on_terminal_charge)
    try:
        with pytest.raises(RuntimeError, match="mutated"):
            source.evalf(30)
    finally:
        _ExactDecimalFloat.__dict__["_decimal"].__set__(source, original)


@pytest.mark.parametrize("source", [
    _native_expr(s.Symbol("metered_complex_free")),
    _native.py_function("unregistered_numeric", _native.py_integer(1)),
])
def test_explicit_budget_unsupported_source_is_terminal_and_cancellation_takes_precedence(source):
    assert _native.evalf_complex_decimal_native(source, 30) is None
    budget = ample_budget()
    with pytest.raises(ValueError):
        _native.evalf_complex_decimal_native(source, 30, budget=budget)
    assert budget.remaining_steps < 1 << 45
    budget.cancel()
    before = budget.remaining_steps
    with pytest.raises(s.EvaluationCancelled):
        _native.evalf_complex_decimal_native(source, 30, budget=budget)
    assert budget.remaining_steps == before
