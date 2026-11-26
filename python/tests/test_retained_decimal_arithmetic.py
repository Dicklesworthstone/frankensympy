"""Arithmetic uses retained decimal input values, not their binary64 previews.

Expected results come from exact input decimal strings and guarded independent
Decimal operations. These tests do not reconstruct an original transcendental
constant from a Float or assert a hidden mpmath binary payload contract.
"""

from decimal import Decimal, localcontext
import operator
import pickle

import pytest
import sympy as s

from sympy.core import _ExactDecimalFloat, _native


@pytest.mark.parametrize("text,digits", [
    ("1.000000000000000000000000000000000000001", 50),
    ("1.23456789012345678901234567890123456789", 22),
    ("9.999999999999999999999999999999", 20),
    ("-1.00000000000000000000001e-400", 30),
    ("1.00000000000000000000001e400", 30),
    ("0", 1000),
])
def test_public_float_rounds_the_original_decimal_string(text, digits):
    with localcontext() as context:
        context.prec = digits
        expected = +Decimal(text)
    value = s.Float(value=text, dps=digits)
    assert isinstance(value, s.Float)
    assert value.dps == digits
    assert Decimal(str(value)) == expected
    assert value.is_zero is expected.is_zero()
    assert value.is_finite is True


def test_public_decimal_literal_arithmetic_preserves_sub_binary64_difference():
    value = s.Float("1.000000000000000000000000000001", 40)
    assert value > 1
    assert Decimal(str(value - 1)) == Decimal("1e-30")
    assert type(s.Float("1.25", 15)) is s.Float
    assert type(s.Float(1.25, 40)) is s.Float
    assert type(s.Float("-Infinity", 40)) is s.Float


@pytest.mark.parametrize("text,digits", [("1e4097", 40), ("0" * 4097, 40), ("1.2", 1001)])
def test_public_decimal_literal_refuses_out_of_bounds_requests(text, digits):
    with pytest.raises(ValueError, match="limit|precision"):
        s.Float(text, digits)


def test_public_decimal_literal_propagates_native_rounding_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("literal rounding refused")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", fail)
    with pytest.raises(RuntimeError, match="literal rounding refused"):
        s.Float("1.2", 30)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("text", ["1.2345678901234567890123456789", "1e-400", "-1e400"])
def test_retained_decimal_pickle_preserves_payload_and_precision(text, protocol):
    original = s.Float(text, 40)
    restored = pickle.loads(pickle.dumps(original, protocol))
    assert type(restored) is type(original)
    assert restored.dps == original.dps
    assert str(restored) == str(original)
    assert restored == original and hash(restored) == hash(original)
    assert restored * 2 == original * 2


def test_held_power_pickle_roundtrips_retained_decimal_child():
    original = s.N(s.Rational(1, 3), 50)
    expression = s.Pow(original, s.Symbol("x"), evaluate=False)
    restored = pickle.loads(pickle.dumps(expression))
    assert restored.args == expression.args
    assert str(restored.args[0]) == str(original)
    assert restored.args[0].dps == 50


@pytest.mark.parametrize("text,expected", [
    ("9999999999999999999.9", 9999999999999999999),
    ("-9999999999999999999.9", -9999999999999999999),
    ("-1e400", -(10**400)), ("1e-400", 0), ("-0.99999999999999999999", 0),
])
def test_int_truncates_the_retained_decimal_instead_of_its_preview(text, expected):
    assert int(s.Float(text, 30)) == expected


@pytest.mark.parametrize("source,expected", [
    (10**400, Decimal("1e400")), (s.Integer(-(10**400)), Decimal("-1e400")),
    (s.Rational(1, 10**400), Decimal("1e-400")),
    (s.S.Half, Decimal("0.5")),
    (s.S.Zero, Decimal(0)), (s.S.One, Decimal(1)), (s.S.NegativeOne, Decimal(-1)),
])
def test_high_precision_float_accepts_exact_numeric_sources(source, expected):
    result = s.Float(source, 50)
    assert result.dps == 50
    assert Decimal(str(result)) == expected
    assert result.is_finite is True


def test_high_precision_float_rational_rounds_once_from_the_fraction():
    with localcontext() as context:
        context.prec = 70
        expected = Decimal(1) / Decimal(7)
    result = s.Float(s.Rational(1, 7), 70)
    assert Decimal(str(result)) == expected
    assert result == s.N(s.Rational(1, 7), 70)


def test_retained_float_reconstruction_and_explicit_precision_changes():
    value = s.Float("1.000000000000000000000000000001", 40)
    assert s.Float(value) is value
    assert s.Float(value, 40) is value
    extended = s.Float(value, 60)
    assert extended.dps == 60 and Decimal(str(extended)) == Decimal(str(value))
    rounded_down = s.Float(value, 15)
    assert rounded_down.dps == 15 and Decimal(str(rounded_down)) == 1
    with pytest.raises(TypeError, match="positive int"):
        s.Float(value, 0)
    with pytest.raises(ValueError, match="precision"):
        s.Float(value, 1001)


@pytest.mark.parametrize("source", [1 << 4096, s.Integer(1 << 4096),
                                   s.Rational(1, (1 << 4096) + 1)])
def test_high_precision_float_rejects_oversize_exact_input(source):
    with pytest.raises(ValueError, match="bit limit"):
        s.Float(source, 50)


def test_custom_numeric_subclass_conversion_does_not_call_user_hooks():
    class ForeignInteger(int):
        def __float__(self):
            raise AssertionError("unexpected foreign Float conversion")
    with pytest.raises(TypeError):
        s.Float(ForeignInteger(7), 30)


def retained(text, digits=50):
    return _ExactDecimalFloat(text, digits)


def decimal(value):
    return Decimal(str(value))


@pytest.mark.parametrize("method", ["subs", "xreplace"])
def test_replacement_distinguishes_decimal_atoms_and_composite_patterns(method):
    x = s.Symbol("decimal_replacement")
    first = retained("1.000000000000000000000000000001", 40)
    second = retained("1.000000000000000000000000000002", 40)
    source = s.Mul(x, first, evaluate=False)
    wrong = s.Mul(x, second, evaluate=False)

    def replace(expr, old, new):
        return expr.subs(old, new) if method == "subs" else expr.xreplace({old: new})

    assert replace(first, second, x) is first
    assert replace(source, wrong, x) is source
    assert replace(source, first, second).args == (x, second)
    assert replace(first, retained(str(first), 40), x) is x
    assert replace(first, retained(str(first), 39), x) is first


def test_subs_carries_original_decimal_into_native_arithmetic_and_power_matches():
    x = s.Symbol("decimal_input")
    coefficient = retained("1.000000000000000000000000000001", 40)
    assert (x - 1).subs(x, coefficient) == retained("1e-30", 40)
    tiny = retained("1e-160", 50)
    assert (x**6).subs(x**2, tiny) == retained("1e-480")
    assert x.subs(x, tiny) is tiny
    assert tiny.subs(x, 2) is tiny


def test_simultaneous_decimal_replacements_keep_owners_distinct():
    x, y = s.symbols("decimal_x decimal_y")
    low = retained("1e-400")
    high = retained("2e-400")
    assert (x - y).subs({x: high, y: low}, simultaneous=True) == low


def test_retained_decimal_substitution_callbacks_are_keyed_by_exact_input():
    from sympy.core import supervised_callback_session
    x = s.Symbol("decimal_callback")
    calls = []

    class Law(s.Function):
        def _eval_subs(self, old, new):
            calls.append(new)
            return new

    source = Law(x)
    first, second = retained("1e-400"), retained("2e-400")
    with supervised_callback_session():
        assert source.subs(x, first) is first
        assert source.subs(x, second) is second
        assert source.subs(x, first) is first
    assert calls == [first, second]


def test_ordinary_float_precision_in_compound_rules_and_substitution_callbacks():
    x = s.Symbol("ordinary_precision")
    low, high = s.Float(0.25, 30), s.Float(0.25, 40)
    source = s.Mul(x, low, evaluate=False)
    rule = s.Mul(x, high, evaluate=False)
    assert source != rule
    assert source.xreplace({rule: x}) is source

    class ReturnReplacement(s.Function):
        def _eval_subs(self, old, new):
            return new

    from sympy.core import supervised_callback_session
    source = ReturnReplacement(x)
    with supervised_callback_session():
        assert source.subs(x, low) is low
        assert source.subs(x, high) is high


def rounded(value, digits):
    with localcontext() as context:
        context.prec = digits
        return +value


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("reverse", [False, True])
def test_numeric_operations_preserve_input_digits_and_precision(operation, reverse):
    value = s.N(s.Rational(1, 3), 50)
    lhs, rhs = (s.Integer(7), value) if reverse else (value, s.Integer(7))
    with localcontext() as context:
        context.prec = 100
        exact = operation(decimal(lhs), decimal(rhs))
    result = operation(lhs, rhs)
    assert type(result) is _ExactDecimalFloat
    assert result.dps == 50
    assert decimal(result) == rounded(exact, 50)


def test_rounding_is_from_the_retained_value_and_each_operation_rounds_once():
    value = s.N(s.Rational(1, 3), 50)
    # The input is the displayed 50-digit decimal, not exact rational 1/3.
    with localcontext() as context:
        context.prec = 100
        product = Decimal(str(value)) * 3
        final = rounded(rounded(product, 50) / 7, 50) + Decimal("0.125")
    assert decimal(value * 3) == rounded(product, 50)
    assert decimal((value * 3) / 7 + s.Rational(1, 8)) == rounded(final, 50)


@pytest.mark.parametrize("digits", [20, 50, 100])
def test_mixed_float_precision_promotes_without_inventing_input_digits(digits):
    high = retained("0.2", digits)
    low = s.Float(0.1)
    with localcontext() as context:
        context.prec = digits + 30
        expected = Decimal("0.2") + Decimal.from_float(0.1)
    for result in (high + low, low + high, s.Add(low, high)):
        assert type(result) is _ExactDecimalFloat
        assert result.dps == digits
        assert decimal(result) == rounded(expected, digits)


def test_nary_constructors_and_cancellation_preserve_exact_residuals():
    large = retained("10000000000000000000000000000000000000000000000000", 60)
    residual = retained("0.00000000000000000000000000000000000000000000000001", 60)
    assert decimal(s.Add(large, residual, -large)) == decimal(residual)
    with localcontext() as context:
        context.prec = 100
        expected = rounded(Decimal(1)/7, 60)
    assert decimal(s.Mul(residual, s.Integer(10**50), s.Rational(1, 7))) == expected
    assert large - large is s.S.Zero


@pytest.mark.parametrize("text", ["1e-400", "-1e-400", "1e400", "-1e400"])
def test_extreme_components_remain_finite_nonzero_values_through_arithmetic(text):
    value = retained(text)
    expected = Decimal(text)
    for result in (value * 2, 2 * value, s.Mul(value, 2), value + value):
        assert decimal(result) == expected * 2
        assert result.dps == 50
        assert result.is_zero is False
        assert result.is_finite is True
    assert decimal((value * 2) / 2) == expected


@pytest.mark.parametrize("exponent", [-3, -1, 0, 1, 2, 5])
def test_integer_powers_preserve_precision_and_profile_identities(exponent):
    value = retained("1.2345678901234567890123456789012345678901234567890")
    with localcontext() as context:
        context.prec = 100
        expected = decimal(value) ** exponent
    for result in (value ** exponent, s.Pow(value, exponent)):
        assert decimal(result) == rounded(expected, 50)
        if exponent == 0:
            assert result is s.S.One
        else:
            assert type(result) is _ExactDecimalFloat
            assert result.dps == 50
    assert value ** 1 is value


def test_float_zero_exponent_keeps_float_identity():
    result = retained("2.0") ** s.Float(0)
    assert type(result) is _ExactDecimalFloat
    assert result.dps == 50
    assert decimal(result) == 1


def test_division_zero_contract_distinguishes_float_and_exact_zero():
    value, zero = retained("2"), retained("0")
    assert value / s.S.Zero is s.zoo
    assert zero / s.S.Zero is s.nan
    assert s.S.One / zero is s.zoo
    assert s.S.Zero / value is s.S.Zero
    for left in (value, zero, s.Float(1)):
        with pytest.raises(ZeroDivisionError):
            left / zero


def test_relations_compare_exact_retained_values_from_both_operand_orders():
    above = retained("1.0000000000000000000000000000000000000000000000001")
    below = retained("0.9999999999999999999999999999999999999999999999999")
    assert above > s.S.One
    assert s.S.One < above
    assert below < s.S.One
    assert s.S.One > below
    assert retained("1e-400") > s.S.Zero
    assert retained("-1e-400") < s.S.Zero
    assert retained("1e400") < retained("2e400")
    assert s.Eq(above, below) is s.false
    assert s.Eq(retained("1"), s.S.One) is s.true


def test_float_equality_is_structural_and_hash_consistent():
    close_a = retained("1.0000000000000000000000000000000000000000000000001")
    close_b = retained("1.0000000000000000000000000000000000000000000000002")
    assert close_a != close_b
    assert retained("1e-400") != retained("2e-400")
    assert retained("1e400") != retained("2e400")
    assert retained("1e-400") != s.Float(0)
    assert retained("1") != s.S.One
    assert retained("1", 30) != retained("1", 50)
    assert retained("1", 50) != 1.0
    assert retained("1", 15) == 1.0
    for left, right in ((retained("1.00"), retained("1.0")),
                        (retained("1", 15), s.Float(1)),
                        (retained("1e400"), retained("10e399"))):
        assert left == right and right == left
        assert hash(left) == hash(right)
        assert {left: "same"}[right] == "same"


def test_symbolic_and_noninteger_power_sources_keep_original_decimal_owners():
    value = retained("1.0000000000000000000000000000000000000000000000001")
    x = s.Symbol("retained_decimal_x")
    for result in (value + x, x + value, s.Add(value, x), value * x, x * value,
                   s.Mul(value, x), s.Pow(value, s.Rational(1, 2))):
        assert any(arg is value for arg in result.args)
        assert str(value) in str(result)
    square = retained("4.0000000000000000000000000000000000000000000000000")
    assert decimal(s.N(square ** s.Rational(1, 2), 50)) == 2


def test_retained_held_undefined_children_are_not_erased():
    value = retained("0")
    source = value * s.Pow(0, -1, evaluate=False)
    assert any(arg is value for arg in source.args)
    with pytest.raises(ValueError):
        s.N(source, 40)


def test_closed_arithmetic_does_not_consult_custom_numeric_getters():
    effects = []
    class Custom(s.Float):
        def _as_python_float(self):
            effects.append("converted")
            raise AssertionError("custom numeric conversion entered the closed decimal lane")
    value = retained("0.25")
    custom = Custom(0.5)
    result = value + custom
    assert any(arg is custom for arg in result.args)
    assert effects == []


@pytest.mark.parametrize("failure", [TypeError, ValueError, RuntimeError, MemoryError, s.EvaluationBudgetExceeded, s.EvaluationCancelled])
def test_native_arithmetic_failure_is_terminal(monkeypatch, failure):
    value = retained("0.25")
    def refused(*args, **kwargs):
        raise failure("retained arithmetic refused")
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", refused)
    with pytest.raises(failure, match="retained arithmetic refused"):
        value + 1


def test_no_retained_operand_keeps_the_existing_float_profile():
    value = s.Float(0.125)
    assert type(value + s.Float(0.25)) is s.Float
    assert type(value * 2) is s.Float
    assert float(value + s.Float(0.25)) == 0.375


def test_declared_precision_and_power_bounds_refuse_instead_of_lowering():
    with pytest.raises(ValueError, match="precision"):
        retained("0.25", 1001) + 1
    with pytest.raises(ValueError, match="exponent|limit"):
        retained("1.25") ** 10000


def test_float_precision_equality_is_transitive_across_all_exact_owners():
    low, high, stored = s.Float(1, 15), s.Float(1, 50), retained("1", 50)
    assert low != high and high != low
    assert high == stored and stored == high
    assert low != stored and stored != low
    assert s.Float(1, 50) != 1.0
    assert s.Float(1, 15) == 1.0
    assert hash(high) == hash(stored)
    assert {high: "same value and precision"}[stored] == "same value and precision"


@pytest.mark.parametrize("exponent", [-2, 0, 1, 2])
def test_integer_valued_float_exponents_are_exactly_admitted(exponent):
    value = retained("1.25")
    with localcontext() as context:
        context.prec = 100
        expected = Decimal("1.25") ** exponent
    for power in (s.Float(exponent), retained(str(exponent), 60)):
        result = value ** power
        assert type(result) is _ExactDecimalFloat
        assert result.dps == max(value.dps, power.dps)
        assert decimal(result) == rounded(expected, result.dps)
    assert decimal(2 ** retained("3")) == 8


def composite(kind, coefficient):
    x = s.Symbol("retained_identity_x")
    if kind == "add":
        return s.Add(coefficient, x, evaluate=False)
    if kind == "mul":
        return s.Mul(coefficient, x, evaluate=False)
    if kind == "pow":
        return s.Pow(coefficient, 2, evaluate=False)
    if kind == "sin":
        return s.sin(coefficient, evaluate=False)
    function = s.Function("RetainedFloatIdentityFunction")
    return function(coefficient, evaluate=False)


@pytest.mark.parametrize("kind", ["add", "mul", "pow", "sin", "opaque_function"])
@pytest.mark.parametrize("left, right", [
    (retained("1.00", 15), s.Float(1)),
    (retained("0", 15), s.Float(-0.0)),
    (retained("-0.0", 15), s.Float(0.0)),
    (retained("1e400"), retained("10e399")),
])
def test_equal_composites_hash_equally_across_retained_legacy_and_signed_zero(kind, left, right):
    a, b = composite(kind, left), composite(kind, right)
    assert a == b and b == a
    assert hash(a) == hash(b)
    assert {a: "same retained structure"}[b] == "same retained structure"


@pytest.mark.parametrize("kind", ["add", "mul", "pow", "sin", "opaque_function"])
def test_composite_equality_retains_digits_and_float_precision(kind):
    a = retained("1.0000000000000000000000000000000000000000000000001")
    b = retained("1.0000000000000000000000000000000000000000000000002")
    left, right = composite(kind, a), composite(kind, b)
    assert left != right and right != left
    assert len({left, right}) == 2
    assert composite(kind, s.Float(1, 15)) != composite(kind, s.Float(1, 50))
    assert composite(kind, retained("1e-400")) != composite(kind, s.Float(0, 50))


def test_nested_arithmetic_and_registered_functions_preserve_child_identity():
    x = s.Symbol("nested_retained_identity_x")
    a = retained("1.0000000000000000000000000000000000000000000000001")
    b = retained("1.0000000000000000000000000000000000000000000000002")
    left = s.Add(x, s.sin(a, evaluate=False), evaluate=False)
    right = s.Add(x, s.sin(b, evaluate=False), evaluate=False)
    assert left != right
    assert left - right != s.S.Zero
    assert left + left != right + right


def test_sign_changes_and_absolute_values_keep_extreme_decimal_payloads():
    value = retained("-1.0000000000000000000000000000000000000000000000001e-400")
    assert str(-value) == str(value)[1:]
    assert str(abs(value)) == str(value)[1:]
    assert (-value).dps == abs(value).dps == 50
    assert (-value).is_positive and abs(value).is_nonzero


def test_original_undefined_base_is_not_erased_by_a_float_zero_exponent():
    pole = s.Pow(0, -1, evaluate=False)
    result = pole ** retained("0")
    assert result.args[0] is pole
    with pytest.raises(ValueError):
        s.N(result, 50)


def test_arithmetic_shares_its_private_ledger_and_preserves_spent_work(monkeypatch):
    from sympy.core import _decimal_arithmetic as lane
    budget = s.EvaluationBudget(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    monkeypatch.setattr(lane, "_budget", lambda: budget)
    original = _native.evalf_complex_decimal_native
    calls = []
    def observed(source, digits, *, budget):
        calls.append((budget, budget.remaining_steps))
        return original(source, digits, budget=budget)
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", observed)
    value = retained("0.125")
    assert decimal(value + 1) == Decimal("1.125")
    first = budget.remaining_steps
    assert decimal(value * 2) == Decimal("0.25")
    assert budget.remaining_steps < first < 1 << 45
    assert len(calls) == 2 and all(used is budget for used, _ in calls)
    assert all(steps < 1 << 45 for _, steps in calls)


def test_output_cancellation_is_terminal_before_arithmetic_publication(monkeypatch):
    from sympy.core import _decimal_arithmetic as lane
    budget = s.EvaluationBudget(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    monkeypatch.setattr(lane, "_budget", lambda: budget)
    original = lane._format_decimal
    def cancel_after_format(text, digits):
        result = original(text, digits)
        budget.cancel()
        return result
    monkeypatch.setattr(lane, "_format_decimal", cancel_after_format)
    with pytest.raises(s.EvaluationCancelled):
        retained("0.125") + 1


def test_source_mutation_during_native_arithmetic_is_rejected(monkeypatch):
    value = retained("0.125")
    original = _native.evalf_complex_decimal_native
    slot = _ExactDecimalFloat.__dict__["_decimal"]
    def mutate_after_native(source, digits, *, budget):
        result = original(source, digits, budget=budget)
        slot.__set__(value, "0.25")
        return result
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mutate_after_native)
    with pytest.raises(RuntimeError, match="mutated"):
        value + 1


def test_native_only_function_tags_cannot_recover_an_opaque_python_owner():
    function = s.Function("DecimalIdentityNativeTag")
    actual = function(s.Float(1), evaluate=False)
    raw = object.__new__(s.Expr)
    s.Basic.__dict__["_value"].__set__(raw, _native.py_function(
        "DecimalIdentityNativeTag", s.Basic.__dict__["_value"].__get__(s.Float(1), s.Basic),
    ))
    assert raw != actual and actual != raw


@pytest.mark.parametrize("operation, method", [
    (operator.add, "add"), (operator.sub, "sub"), (operator.mul, "mul"),
    (operator.truediv, "div"), (operator.pow, "pow"),
])
def test_foreign_reflected_operators_receive_the_original_retained_owner(operation, method, monkeypatch):
    from sympy.core import _decimal_arithmetic as lane
    calls, forbidden_calls = [], []
    result = object()
    class Foreign:
        @property
        def args(self):
            forbidden_calls.append("args")
            raise AssertionError("foreign child getter was inspected")
        def __float__(self):
            forbidden_calls.append("float")
            raise AssertionError("foreign numeric conversion was inspected")
        def __str__(self):
            forbidden_calls.append("str")
            raise AssertionError("foreign printer was inspected")
        def reflect(self, name, owner):
            calls.append((name, owner))
            return result
        def __radd__(self, owner):
            return self.reflect("add", owner)
        def __rsub__(self, owner):
            return self.reflect("sub", owner)
        def __rmul__(self, owner):
            return self.reflect("mul", owner)
        def __rtruediv__(self, owner):
            return self.reflect("div", owner)
        def __rpow__(self, owner):
            return self.reflect("pow", owner)
    def no_numeric_admission():
        raise AssertionError("foreign reflection entered native arithmetic admission")
    monkeypatch.setattr(lane, "_budget", no_numeric_admission)
    value = retained("0.125")
    assert operation(value, Foreign()) is result
    assert len(calls) == 1 and calls[0][0] == method and calls[0][1] is value
    assert forbidden_calls == []


@pytest.mark.parametrize("constructor", [s.Add, s.Mul, s.Pow])
def test_constructors_do_not_publish_notimplemented_for_foreign_operands(constructor):
    with pytest.raises(TypeError):
        constructor(retained("0.125"), object())


@pytest.mark.parametrize("cls", [s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix])
@pytest.mark.parametrize("left", [False, True])
def test_matrix_scaling_uses_original_decimal_owners(cls, left):
    value = s.N(s.Rational(10**49 + 1, 10**49), 50)
    original = cls([[2, 0], [0, -1]])
    result = value * original if left else original * value
    assert type(result) is type(original)
    assert result.shape == (2, 2)
    assert result[0, 0].dps == 50
    assert decimal(result[0, 0]) == Decimal("2." + "0" * 48 + "2")
    assert decimal(result[1, 1]) == Decimal("-1." + "0" * 48 + "1")
    assert result[0, 1] == 0


@pytest.mark.parametrize("cls", [s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix])
def test_scaling_preserves_retained_matrix_cells(cls):
    original = cls([[retained("1e-400"), retained("1e400")]])
    result = original * 3
    assert decimal(result[0, 0]) == Decimal("3e-400")
    assert decimal(result[0, 1]) == Decimal("3e400")
    assert result[0, 0].is_nonzero is True
    assert result[0, 1].is_finite is True


def test_dense_decimal_division_rounds_each_quotient_once():
    assert decimal((s.Matrix([[2]]) / retained("3.00", 3))[0]) == Decimal("0.667")
    assert decimal((s.Matrix([[retained("2.00", 3)]]) / 3)[0]) == Decimal("0.667")


def test_decimal_scaling_preserves_implicit_zeros_and_native_sparse_input():
    original = s.SparseMatrix.eye(2)
    result = original * retained("1e-400")
    assert result.nnz == 2
    assert decimal(result[1, 1]) == Decimal("1e-400")
    assert result[0, 1] == 0
    wide = s.SparseMatrix(1, 10**9, {(0, 5): 2})
    scaled = retained("1e400") * wide
    assert scaled.nnz == 1
    assert decimal(scaled[0, 5]) == Decimal("2e400")


def test_scaling_held_decimal_cells_preserves_original_symbolic_children():
    coefficient = retained("1.0000000000000000000000000000000000000000000000001")
    held = s.Mul(coefficient, s.Symbol("x"), evaluate=False)
    for cls in (s.Matrix, s.SparseMatrix):
        result = (cls([[held]]) * 2)[0, 0]
        assert result.args[0] is held
        assert result.args[0].args[0] is coefficient


def test_sparse_implicit_decimal_products_must_be_representable():
    scalar = s.Mul(retained("0.1"), s.Symbol("x"), evaluate=False)
    with pytest.raises(NotImplementedError, match="implicit product"):
        s.SparseMatrix.eye(2) * scalar


@pytest.mark.parametrize("cls", [s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix])
def test_entrywise_matrix_sums_keep_decimal_residuals(cls):
    small = "0." + "0" * 48 + "1"
    near = retained("1." + "0" * 48 + "1")
    lhs, rhs = cls([[near, 2]]), cls([[1, -2]])
    assert decimal((lhs - rhs)[0, 0]) == Decimal(small)
    assert decimal((lhs + cls([[-1, 0]]))[0, 0]) == Decimal(small)
    assert (lhs + rhs)[0, 1] == 0


@pytest.mark.parametrize("left_cls,right_cls", [
    (s.Matrix, s.Matrix), (s.ImmutableMatrix, s.Matrix),
    (s.SparseMatrix, s.SparseMatrix), (s.ImmutableSparseMatrix, s.SparseMatrix),
    (s.Matrix, s.SparseMatrix), (s.SparseMatrix, s.Matrix),
])
@pytest.mark.parametrize("operation", [operator.mul, operator.matmul])
def test_matrix_dot_products_resolve_decimal_cancellation(left_cls, right_cls, operation):
    near = retained("1." + "0" * 48 + "1")
    lhs = left_cls([[near, -2]])
    rhs = right_cls([[2], [1]])
    result = operation(lhs, rhs)
    assert result.shape == (1, 1)
    assert result[0, 0].dps == 50
    assert decimal(result[0, 0]) == Decimal("2e-49")


@pytest.mark.parametrize("left_cls,right_cls", [
    (s.Matrix, s.Matrix), (s.Matrix, s.SparseMatrix),
    (s.SparseMatrix, s.Matrix), (s.SparseMatrix, s.SparseMatrix),
])
def test_matrix_dot_product_rounds_products_before_adding(left_cls, right_cls):
    # At 3 significant digits 1.23 * 1.23 rounds to 1.51.
    # A fused product/sum would instead return the nonzero 0.0029.
    lhs = left_cls([[retained("1.23", 3), -1]])
    rhs = right_cls([[retained("1.23", 3)], [retained("1.51", 3)]])
    assert (lhs @ rhs)[0, 0] == 0


@pytest.mark.parametrize("cls", [s.Matrix, s.SparseMatrix])
def test_sparse_diagonal_products_preserve_extreme_finite_decimal_values(cls):
    column = cls([[retained("1e400")], [retained("1e-400")]])
    diagonal = s.SparseMatrix([[2, 0], [0, -3]])
    result = diagonal @ column
    assert decimal(result[0, 0]) == Decimal("2e400")
    assert decimal(result[1, 0]) == Decimal("-3e-400")


def test_sparse_dot_product_refuses_an_unresolved_implicit_zero():
    held = s.Mul(retained("0.125"), s.Symbol("x"), evaluate=False)
    lhs = s.SparseMatrix([[held]])
    with pytest.raises(NotImplementedError, match="implicit product"):
        lhs @ s.SparseMatrix(1, 2, {(0, 0): 1})
    # Every potential term has a stored factor in this shape.
    assert (lhs @ s.SparseMatrix([[2]]))[0, 0].args[0] is held


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.matmul])
def test_decimal_matrix_algebra_rejects_mutated_sources(monkeypatch, operation):
    lhs = s.Matrix([[retained("0.125")]])
    rhs = s.Matrix([[2]])
    original = _native.evalf_complex_decimal_native
    def mutate_after_evaluation(*args, **kwargs):
        result = original(*args, **kwargs)
        rhs[0, 0] = 7
        return result
    monkeypatch.setattr(_native, "evalf_complex_decimal_native", mutate_after_evaluation)
    with pytest.raises(RuntimeError, match="Matrix mutated"):
        operation(lhs, rhs)


@pytest.mark.parametrize("left_cls,right_cls", [
    (s.Matrix, s.Matrix), (s.Matrix, s.SparseMatrix),
    (s.SparseMatrix, s.Matrix), (s.SparseMatrix, s.SparseMatrix),
])
def test_empty_matrix_decimal_product_has_no_implicit_terms(left_cls, right_cls):
    held = s.Mul(retained("0.125"), s.Symbol("x"), evaluate=False)
    assert (left_cls(0, 1, []) @ right_cls([[held]])).shape == (0, 1)
    assert (left_cls([[held]]) @ right_cls(1, 0, [])).shape == (1, 0)


def test_sparse_dot_cannot_hide_nonfinite_terms_with_other_decimal_entries():
    lhs = s.SparseMatrix([[s.oo, retained("1.25")]])
    rhs = s.SparseMatrix(2, 1, {(1, 0): 2})
    with pytest.raises(ValueError, match="implicit-zero"):
        lhs @ rhs


def test_sparse_decimal_products_allow_ordinary_finite_float_implicit_terms():
    lhs = s.SparseMatrix([[s.Float(2.5), retained("1e-400")]])
    rhs = s.SparseMatrix(2, 1, {(1, 0): 3})
    assert decimal((lhs @ rhs)[0, 0]) == Decimal("3e-400")
