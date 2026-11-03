"""Replacement and algebra keep the actual retained Python applications."""

from __future__ import annotations

from itertools import repeat

import pytest
import sympy as s

from sympy.core import Basic, Expr, FunctionClass, _native, _native_expr, supervised_callback_session
from sympy.core._opaque_arithmetic import native_transform


def _laws():
    first = FunctionClass("TransformSame", (s.Function,), {"__module__": __name__})
    second = FunctionClass("TransformSame", (s.Function,), {"__module__": __name__})
    x, y, z = s.symbols("tx ty tz")
    return first, second, x, y, z, first(x), second(x)


def test_xreplace_returns_the_actual_replacement_and_never_revisits_it():
    first, second, x, y, z, a, b = _laws()
    assert x.xreplace({x: a}) is a
    assert a.xreplace({a: b}) is b
    assert (a + 1).xreplace({b: z}) == a + 1
    assert (a + b).xreplace({a: b, b: z}) == b + z
    source = s.sin(a + b)
    assert source.xreplace({z: y}) is source
    assert source.xreplace({a: z}) == s.sin(b + z)


@pytest.mark.parametrize("replacement", [1, None, "raw", [1, 2]])
def test_exact_xreplace_returns_raw_values_without_coercion(replacement):
    x = s.Symbol("raw_replace")
    assert x.xreplace({x: replacement}) is replacement


def test_xreplace_does_not_conflate_exact_numbers_with_floats():
    assert s.Integer(1).xreplace({1.0: 7}) is s.S.One
    assert s.Integer(1).xreplace({1: 7}) == 7
    assert s.Float(1).xreplace({1: 7}) == s.Float(1)
    assert s.Float(1).xreplace({1.0: 7}) == 7
    assert s.Rational(1, 2).xreplace({0.5: 7}) == s.Rational(1, 2)


def test_xreplace_snapshots_a_bounded_mapping_without_subclass_iterators():
    x = s.Symbol("bounded_replace")

    class Deceptive(dict):
        def items(self):
            raise AssertionError("must inspect actual dictionary entries")

        def __len__(self):
            raise AssertionError("must count actual entries")

    assert x.xreplace(Deceptive({x: 2})) == 2
    with pytest.raises(ValueError, match="rule limit"):
        x.xreplace(repeat((x, 1)))
    with pytest.raises(TypeError, match="two-element"):
        x.xreplace([(x,)])


def test_substitution_matches_signed_sums_and_rational_offsets():
    first, second, x, y, z, a, b = _laws()
    assert (a + b + x).subs(a + b, z) == x + z
    assert (a - b + x).subs(b - a, z) == x - z
    assert (a + b + 2).subs(a + b + 3, z) == z - 1
    assert (a + b + 2).subs(-a - b - 3, z) == -z - 1
    assert (a + b).subs(x, y) == first(y) + second(y)


def test_substitution_matches_product_multiplicities_and_coefficients():
    first, second, x, y, z, a, b = _laws()
    assert (6 * a * b * x).subs(2 * a * b, z) == 3 * x * z
    assert (4 * a**2 * b**2).subs(2 * a * b, z) == z**2
    assert (3 * a).subs(2 * a, z) == 3 * a
    assert (s.Rational(3, 2) * a).subs(2 * a, z) == s.Rational(3, 2) * a


@pytest.mark.parametrize("coefficient,old_coefficient,power,expected", [
    (2, -2, 1, None), (4, -2, 2, None), (-2, 2, 1, (-1, 1)),
    (-4, -2, 2, (-1, 2)),
    (s.Rational(3, 4), s.Rational(1, 2), 1, (s.Rational(3, 2), 1)),
    (s.Rational(3, 4), s.Rational(1, 2), 2, (3, 2)),
    (s.Rational(2, 3), s.Rational(2, 9), 1, None),
    (s.Rational(6, 5), s.Rational(2, 3), 1, None),
])
def test_product_substitution_extracts_numerators_denominators_and_literal_signs(
        coefficient, old_coefficient, power, expected):
    first, second, x, y, z, a, b = _laws()
    source = coefficient * a**power
    result = source.subs(old_coefficient * a, z)
    assert result == (source if expected is None else expected[0] * z**expected[1])


@pytest.mark.parametrize("exponent,old_exponent,expected", [
    (6, 2, 3), (-2, 2, -1), (s.Rational(3, 2), s.Rational(1, 2), 3),
])
def test_power_substitution_uses_only_valid_integer_outer_powers(exponent, old_exponent, expected):
    first, second, x, y, z, a, b = _laws()
    assert (a**exponent).subs(a**old_exponent, z) == z**expected
    assert (a**s.Rational(1, 2)).subs(a**2, z) == s.sqrt(a)


def test_substitution_and_xreplace_canonicalize_only_changed_held_parents():
    first, second, x, y, z, a, b = _laws()
    held = s.Add(a, a, evaluate=False)
    assert held.subs(z, y) is held
    assert held.xreplace({z: y}) is held
    assert held.subs(x, y) == 2 * first(y)
    assert held.xreplace({x: y}) == 2 * first(y)
    assert held.args == (a, a)


def test_expansion_distributes_products_and_powers_without_erasing_classes():
    first, second, x, y, z, a, b = _laws()
    expanded = s.expand((a + x) * (b + y))
    assert expanded == x * y + x * b + y * a + a * b
    product = next(term for term in expanded.args if type(term) is s.Mul and a in term.args and b in term.args)
    assert product.args == (a, b)
    assert s.expand((a + x)**3) == a**3 + 3 * a**2 * x + 3 * a * x**2 + x**3
    assert s.expand(s.sin((a + x)**2)) == s.sin(a**2 + 2 * a * x + x**2)
    assert Basic.__dict__["_value"].__get__(expanded, Basic) is None
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(expanded)


def test_expansion_respects_depth_and_arithmetic_hints():
    first, second, x, y, z, a, b = _laws()
    nested = s.sin((a + x)**2)
    assert s.expand(nested, deep=False) is nested
    assert s.expand((a + x) * (b + y), mul=False) == (a + x) * (b + y)
    assert s.expand((a + x)**3, multinomial=False) == (a + x)**3
    assert s.expand(s.exp(a + b)) == s.exp(a) * s.exp(b)
    assert s.expand(s.Pow(2, a + b)) == 2**a * 2**b
    with pytest.raises(NotImplementedError, match="arithmetic"):
        s.expand(a + 1, trig=True)
    with pytest.raises(NotImplementedError, match="arithmetic"):
        s.expand(a + 1, modulus=2)
    with pytest.raises(ValueError, match="power envelope"):
        s.expand((a + x)**65)


def test_unchanged_transforms_never_read_custom_printers_or_children_properties():
    armed = False
    calls = []

    def guard(label):
        if armed:
            calls.append(label)
            raise AssertionError(label)

    class Guarded(s.Function):
        @property
        def args(self):
            guard("args")
            return super().args

        def __str__(self):
            guard("str")
            return super().__str__()

        def __setattr__(self, name, value):
            guard("setattr")
            object.__setattr__(self, name, value)

    x, y = s.symbols("guard_x guard_y")
    a = Guarded(x, evaluate=False)
    held = s.Add(a, a, evaluate=False)
    armed = True
    try:
        assert a.xreplace({y: 3}) is a
        assert a.subs(y, 3) is a
        assert s.simplify(a) is a
        result = s.expand(held)
        assert s.Add.__dict__["_args"].__get__(result, s.Add) == (a, a)
    finally:
        armed = False
    assert calls == []


def test_substitution_hook_cache_uses_typed_arguments_and_stable_values():
    x = s.Symbol("effect_x")
    positive = s.Symbol("effect_x", positive=True)
    calls = []

    class Law(s.Function):
        def _eval_subs(self, old, new):
            calls.append((old, new))
            return None

    a = Law(x)
    with supervised_callback_session():
        a.subs(x, 3)
        a.subs(x, 3)
        a.subs(positive, 3)
    assert calls == [(x, s.Integer(3)), (positive, s.Integer(3))]


def test_arithmetic_simplification_never_applies_builtin_identities_by_custom_name():
    sine = FunctionClass("sin", (s.Function,), {"__module__": __name__})
    cosine = FunctionClass("cos", (s.Function,), {"__module__": __name__})
    x = s.Symbol("spoof_x")
    a, b = sine(x), cosine(x)
    source = a**2 + b**2
    assert s.simplify(source) == source
    assert s.simplify(source) != 1
    assert s.simplify(s.Add(a, a, evaluate=False)) == 2 * a


def test_unknown_native_tags_cannot_recover_python_classes_during_transforms():
    calls = []

    class NativeOnlyLaw(s.Function):
        def __setattr__(self, name, value):
            calls.append(name)
            raise AssertionError("native-only input reconstructed a user class")

    class ActualLaw(s.Function):
        pass

    x = s.Symbol("native_unknown_x")
    unknown = Expr(_native.py_function("NativeOnlyLaw", _native_expr(x)))
    closed = Expr(_native.Mul(_native_expr(unknown), _native_expr(x)).as_expr())
    actual = ActualLaw(x)
    with pytest.raises(NotImplementedError, match="native-only"):
        actual + closed
    with pytest.raises(NotImplementedError, match="native-only"):
        closed.xreplace({x: actual})
    assert calls == []


def test_shared_codec_restores_every_result_and_preserves_real_slot_like_symbols():
    first, second, x, y, z, a, b = _laws()
    collision = s.Symbol("__fsym_operation_slot_00000000")
    result = native_transform(
        (a, b, collision),
        lambda values: (values[0], values[1], _native.Add(*values).as_expr()),
        multiple=True,
    )
    assert result[:2] == (a, b)
    assert result[0] is a and result[1] is b
    assert result[2] == a + b + collision
    assert result[2].free_symbols == {x, collision}
    with pytest.raises(TypeError, match="exact tuple"):
        native_transform((a,), lambda values: list(values), multiple=True)


def test_shared_codec_can_atomize_closed_intrinsics_without_invoking_them():
    first, second, x, y, z, a, b = _laws()
    builtin = s.sin(x)
    result = native_transform((builtin, a), lambda values: _native.Mul(*values).as_expr(),
                              atomize_functions=True)
    assert result == builtin * a
    assert any(child is builtin for child in result.args)
    assert any(child is a for child in result.args)


def test_all_transform_operands_are_cycle_checked_before_replacement():
    first, second, x, y, z, a, b = _laws()
    corrupt = object.__new__(first)
    Basic.__dict__["_value"].__set__(corrupt, None)
    s.Function.__dict__["_args"].__set__(corrupt, (corrupt,))
    for operation in (lambda: x.xreplace({x: corrupt}), lambda: x.subs(x, corrupt),
                      lambda: s.expand(corrupt), lambda: s.simplify(corrupt)):
        with pytest.raises(ValueError, match="cyclic"):
            operation()
