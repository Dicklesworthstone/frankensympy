"""Exact polynomial transforms retain actual custom-function identities."""

import math

import pytest
import sympy as s

from sympy.core import Basic, FunctionClass, _native, _native_expr
from sympy.polys.polyerrors import PolynomialError


class Law(s.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return 2 * self.args[0]


def _applications(expression):
    if isinstance(expression, s.Function):
        return [expression]
    return [application for child in expression.args for application in _applications(child)]


@pytest.mark.parametrize("coefficient,power", [(1, 2), (2, 3), (s.Rational(2, 3), 4)])
def test_native_factorization_retains_the_application_and_multiplicity(coefficient, power):
    x = s.Symbol("factor_x")
    law = Law(x)
    source = coefficient * s.expand((law - 1) ** power * (law + 2))
    scale, factors = s.factor_list(source)
    assert scale == coefficient
    assert factors == [(law + 2, 1), (law - 1, power)]
    result = s.factor(source)
    assert result == coefficient * (law - 1) ** power * (law + 2)
    assert all(application is law for application in _applications(result))
    assert Basic.__dict__["_value"].__get__(result, Basic) is None
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(result)


def test_multivariate_factorization_combines_closed_and_custom_function_generators():
    x = s.Symbol("factor_mix_x")
    law, sine = Law(x), s.sin(x)
    result = s.factor(law**2 + 2 * law * sine + sine**2)
    assert result == (law + sine)**2
    assert any(application is law for application in _applications(result))
    assert any(application is sine for application in _applications(result))
    assert s.factor_list(law**2 - x**2) == (-1, [(x - law, 1), (x + law, 1)])


def test_explicit_generators_keep_other_applications_in_the_coefficient():
    x, y = s.symbols("factor_parameter_x factor_parameter_y")
    law = Law(y)
    assert s.factor_list(law * (x**2 - 1), x) == (law, [(x - 1, 1), (x + 1, 1)])
    assert s.factor_list(law**2 - 1, law) == (1, [(law - 1, 1), (law + 1, 1)])
    with pytest.raises(PolynomialError, match="duplicated"):
        s.factor_list(law**2 - 1, law, law)


@pytest.mark.parametrize("transform", [s.factor, s.factor_list, s.cancel])
def test_function_coefficients_must_be_independent_of_explicit_generators(transform):
    x, y = s.symbols("coefficient_x coefficient_y")
    law = Law(x + y)
    with pytest.raises(PolynomialError, match="depends on"):
        transform(law * (x**2 - 1), x)


@pytest.mark.parametrize("transform", [s.factor, s.factor_list, s.cancel])
def test_nested_functions_cannot_be_coefficients_of_their_own_argument_generator(transform):
    x = s.Symbol("nested_generator_x")
    law = Law(x)
    outer = s.Function("OuterLaw")(law)
    with pytest.raises(PolynomialError, match="depends on"):
        transform(outer + law, law)


def test_explicit_application_and_symbol_generators_can_coexist():
    x = s.Symbol("joint_generator_x")
    law = Law(x)
    scale, factors = s.factor_list(law*(x**2 - 1), x, law)
    assert scale == 1
    assert set(factors) == {(law, 1), (x - 1, 1), (x + 1, 1)}


def test_closed_subexpressions_restore_builtin_applications_without_slot_leaks():
    x, y = s.symbols("mixed_closed_x mixed_closed_y")
    law, other, sine = Law(x+y), s.Function("SecondLaw")(x-y), s.sin(x)
    factored = s.factor((law+sine)**2 - (other+sine)**2)
    assert s.expand(factored - ((law+sine)**2 - (other+sine)**2)) == 0
    combined = s.together(1/law + 1/sine)
    assert combined == (law+sine)/(law*sine)
    assert "__fsym_operation_slot_" not in str(factored)
    assert "__fsym_operation_slot_" not in str(combined)


def test_custom_factor_hooks_are_not_silently_ignored_or_invoked():
    calls = []

    class Hooked(s.Function):
        def _eval_factor(self):
            calls.append("factor")
            return 99

    source = Hooked(s.Symbol("factor_hook_x")) + 1
    with pytest.raises(NotImplementedError, match="factor hooks"):
        s.factor(source)
    assert calls == []


def test_factor_list_preserves_oracle_order_across_mixed_generator_blocks():
    x = s.Symbol("x")
    law = Law(x)
    expected = (1, [(x - 1, 1), (law, 1), (x + 1, 1)])
    assert s.factor_list((x**2 - 1) * law) == expected
    assert s.factor_list(x**2 * law - law) == expected


@pytest.mark.parametrize("transform", [s.factor, s.cancel])
def test_rational_transforms_cancel_exact_polynomial_factors(transform):
    x = s.Symbol("rational_x")
    law = Law(x)
    assert transform((law**2 - 1)/(law + 1)) == law - 1
    assert transform((law**3 - law)/(law**2 + law)) == law - 1
    other = s.Function("OtherLaw")(x)
    if transform is s.cancel:
        assert transform((2 * law + 2)/(4 * other + 4)) == (law + 1)/(2 * other + 2)
        assert transform(1/law + 1/(law + 1)) == (2 * law + 1)/(law**2 + law)


def test_together_preserves_factored_denominators_and_actual_applications():
    x = s.Symbol("together_x")
    law = Law(x)
    other = s.Function("OtherLaw")(x)
    assert s.together(1/law + 1/other) == (law + other)/(law * other)
    assert s.together(1/law + 1/(law + 1)) == (2 * law + 1)/(law * (law + 1))
    assert s.together((law**2 - 1)/(law + 1)) == (law**2 - 1)/(law + 1)
    with pytest.raises(NotImplementedError, match="deep"):
        s.together(law, deep=True)
    with pytest.raises(NotImplementedError, match="fraction=True"):
        s.together(1/law + 1/other, fraction=False)


@pytest.mark.parametrize("operation", [s.factor, s.factor_list, s.cancel, s.together])
def test_operation_slots_never_enter_the_public_symbol_registry(operation):
    import sympy.core as core

    x = s.Symbol("slot_registry_x")
    law = Law(x)
    source = law**2 - 1 if operation is s.factor_list else 1/law + 1/(law + 1)
    before = dict(core._surface_symbols)
    operation(source)
    assert core._surface_symbols == before


def test_same_named_classes_never_cancel_or_share_callback_slots():
    first = FunctionClass("SamePolynomialLaw", (s.Function,), {"__module__": __name__})
    second = FunctionClass("SamePolynomialLaw", (s.Function,), {"__module__": __name__})
    x = s.Symbol("same_polynomial_x")
    a, b = first(x), second(x)
    source = (a**2 - 1)/(b + 1)
    result = s.cancel(source)
    assert result != a - 1
    applications = _applications(result)
    assert any(value is a for value in applications)
    assert any(value is b for value in applications)
    program = s.compile_numeric_hybrid(x, result, callbacks={first: lambda v: v, second: lambda v: 2*v})
    assert program(3) == pytest.approx(8/7)


def test_typed_symbols_and_slot_like_names_survive_factorization():
    plain = s.Symbol("parameter")
    positive = s.Symbol("parameter", positive=True)
    collision = s.Symbol("__fsym_operation_slot_00000000")
    law = Law(collision)
    result = s.factor((plain - positive)**2 * law**2 - (plain - positive)**2)
    assert result.free_symbols == {plain, positive, collision}
    evaluated = result.subs({plain: 3, positive: 1, law: 4})
    assert evaluated == 60
    assert all(application is law for application in _applications(result))


def test_polynomial_transforms_do_not_reconstruct_or_print_custom_applications():
    state = {"armed": False, "calls": []}

    def guard(name):
        if state["armed"]:
            state["calls"].append(name)
            raise AssertionError(name)

    class Guarded(s.Function):
        def __new__(cls, *args, **kwargs):
            guard("new")
            return super().__new__(cls, *args, **kwargs)

        def __setattr__(self, name, value):
            guard("setattr")
            object.__setattr__(self, name, value)

        @property
        def args(self):
            guard("args")
            return super().args

        def __str__(self):
            guard("str")
            return super().__str__()

    x = s.Symbol("guarded_polynomial_x")
    law = Guarded(x, evaluate=False)
    source = (law**2 - 1)/(law + 1)
    state["armed"] = True
    try:
        factored = s.factor(source)
        cancelled = s.cancel(source)
        combined = s.together(1/law + 1/(law + 1))
    finally:
        state["armed"] = False
    assert state["calls"] == []
    assert factored == cancelled == law - 1
    assert any(application is law for application in _applications(combined))


def test_custom_function_named_exp_does_not_receive_exponential_identities():
    custom = FunctionClass("exp", (s.Function,), {"__module__": __name__})
    x = s.Symbol("opaque_exp_x")
    a, b = custom(x), custom(2*x)
    result = s.factor(b - a**2)
    assert result != 0
    program = s.compile_numeric_hybrid(x, result, callbacks={custom: lambda value: value + 1})
    assert program(2) == -4


@pytest.mark.parametrize("operation", [s.factor, s.factor_list, s.cancel, s.together])
def test_degree_limits_precede_native_polynomial_algorithms(operation, monkeypatch):
    x = s.Symbol("bounded_polynomial_x")
    source = Law(x)**33 + 1
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("native algorithm called before degree admission")

    monkeypatch.setattr(_native, "poly_factor_list_native", forbidden)
    monkeypatch.setattr(_native, "poly_factor_multivariate_native", forbidden)
    monkeypatch.setattr(_native, "poly_multivariate_gcd_native", forbidden)
    with pytest.raises(ValueError, match="exponent|degree"):
        operation(source)
    assert calls == []


def test_generator_and_coefficient_amplification_are_bounded():
    x = s.Symbol("bounded_generators_x")
    laws = [s.Function("Generator" + str(i))(x) for i in range(9)]
    with pytest.raises(ValueError, match="eight generators"):
        s.factor(s.Add(*laws))
    with pytest.raises(ValueError, match="coefficient|literal limit"):
        s.cancel(s.Integer(1 << 4097) * laws[0] + 1)
    with pytest.raises(PolynomialError, match="integer powers"):
        s.factor(laws[0]**s.Rational(1, 2) + 1)
    with pytest.raises(PolynomialError, match="negative"):
        s.factor_list(1/laws[0] + 1)


def test_wrong_native_factor_product_is_not_published(monkeypatch):
    x = s.Symbol("factor_reconstruction_x")
    law = Law(x)

    def forged(source, binding):
        return _native.py_integer(1), [(source, 2)]

    monkeypatch.setattr(_native, "poly_factor_list_native", forged)
    with pytest.raises(ValueError, match="reconstruction"):
        s.factor_list(law**2 - 1)


def test_wrong_native_quotient_is_not_published(monkeypatch):
    x = s.Symbol("division_reconstruction_x")
    law = Law(x)

    def forged(*args):
        return _native.py_integer(0), _native.py_integer(0)

    monkeypatch.setattr(_native, "poly_multivariate_div_rem_native", forged)
    with pytest.raises(ValueError, match="reconstruction"):
        s.cancel((law**2 - 1)/(law + 1))


def test_factored_custom_residual_reaches_symbolic_jacobian_and_hybrid_solve():
    x = s.Symbol("factored_solver_x")
    application = Law(x)
    residual = s.factor(application**2 - 5*application + 4)
    matrix = s.Matrix([residual]).as_immutable()
    jacobian = matrix.jacobian([x])
    calls = []
    program = s.compile_hybrid_residual_system(
        matrix, [x], jacobian=jacobian,
        callbacks={Law: lambda value: calls.append(value) or value*value},
    )
    root = program.solve([2.2])
    assert float(root[0]) == pytest.approx(2.0, abs=1e-10)
    residuals, derivatives = program.evaluate(2.0)
    assert residuals == [0.0]
    assert derivatives == [[12.0]]
    assert calls and all(math.isfinite(value) for value in calls)
    assert program.evidence == "approximate_with_user_callbacks"
