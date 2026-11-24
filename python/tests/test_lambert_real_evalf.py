"""Real Lambert W values, precision, and complete-solver numeric consumers.

Decimal strings and binary64 anchors were obtained with isolated SymPy 1.14.0.
The comparator is exact displayed digits for evalf/N and relative error at the
shell's 15-digit conversion precision for float (with no absolute zero floor).
Held native source admission and explicit solveset results are separate native
contracts; upstream commonly retains ConditionSet for these equations.
"""

from decimal import Decimal
import math

import pytest
import sympy as s


NATIVE_ONLY = pytest.mark.skipif(
    not hasattr(s.Matrix([[1]]), "_native_storage"),
    reason="native real Lambert source and complete-solver extension",
)

# name, original exact argument, real branch, 15 digits, 40 digits, float
CASES = (
    (
        "principal_one", s.Integer(1), 0,
        "0.567143290409784",
        "0.5671432904097838729999686622103555497538",
        0.5671432904097838,
    ),
    (
        "negative_principal", -s.Rational(1, 4), 0,
        "-0.357402956181389",
        "-0.3574029561813889030688111040559047533166",
        -0.3574029561813889,
    ),
    (
        "negative_lower", -s.Rational(1, 4), -1,
        "-2.15329236411035",
        "-2.153292364110349649169099150092981375536",
        -2.15329236411035,
    ),
    (
        "large_principal", s.Integer(10)**20, 0,
        "42.3067550917384",
        "42.30675509173839385177115930518188403470",
        42.306755091738395,
    ),
    (
        "lower_tail", -s.Rational(1, 10**100), -1,
        "-235.721158875685",
        "-235.7211588756853136604606061305238190409",
        -235.72115887568532,
    ),
    (
        "tiny_positive", s.Rational(1, 10**80), 0,
        "1.00000000000000e-80",
        "1.000000000000000000000000000000000000000e-80",
        1e-80,
    ),
    (
        "tiny_negative", -s.Rational(1, 10**80), 0,
        "-1.00000000000000e-80",
        "-1.000000000000000000000000000000000000000e-80",
        -1e-80,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
@pytest.mark.parametrize("digits", [15, 40])
def test_evalf_and_N_preserve_requested_real_decimal_digits(case, digits):
    _, argument, branch, short, long, _ = case
    value = s.LambertW(argument, branch)
    expected = short if digits == 15 else long
    for result in (value.evalf(digits), s.N(value, digits)):
        assert isinstance(result, s.Float)
        assert result.is_real is True
        assert str(result) == expected


@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_float_conversion_is_finite_branch_correct_and_keeps_tiny_signs(case):
    _, argument, branch, _, _, reference = case
    value = float(s.LambertW(argument, branch))
    assert math.isfinite(value)
    assert value == pytest.approx(reference, rel=5e-15, abs=0.0)
    if branch == -1:
        assert value < -1
    elif argument < 0:
        assert -1 < value < 0
    else:
        assert value > 0


@pytest.mark.parametrize("argument,branch,expected", [
    (s.Integer(0), 0, 0),
    (s.E, 0, 1),
    (-1/s.E, 0, -1),
    (-1/s.E, -1, -1),
])
def test_exact_constructor_values_include_the_shared_real_branch_point(
    argument, branch, expected,
):
    value = s.LambertW(argument, branch)
    assert value == s.Integer(expected)
    assert float(value) == float(expected)
    assert Decimal(str(value.evalf(25))) == Decimal(expected)


@NATIVE_ONLY
@pytest.mark.parametrize("branch", [0, -1])
@pytest.mark.parametrize("form", ["exp", "power"])
def test_held_exact_branch_point_reaches_native_admission_before_exact_folding(branch, form):
    from sympy.core import _native, _native_expr

    reciprocal_e = (s.exp(-1, evaluate=False) if form == "exp" else
                    s.Pow(s.E, -1, evaluate=False))
    argument = s.Mul(-1, reciprocal_e, evaluate=False)
    value = s.LambertW(argument, branch, evaluate=False)
    decimal = _native.evalf_decimal_native(_native_expr(value), 25)
    assert Decimal(decimal) == Decimal(-1)


@NATIVE_ONLY
def test_defined_held_argument_reaches_native_numeric_evaluation():
    from sympy.core import _native, _native_expr

    argument = s.Add(1, s.Mul(0, 2, evaluate=False), evaluate=False)
    value = s.LambertW(argument, evaluate=False)
    decimal = _native.evalf_decimal_native(_native_expr(value), 40)
    assert Decimal(decimal) == Decimal(CASES[0][4])


@NATIVE_ONLY
@pytest.mark.parametrize("branch", [0, -1])
@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power", "nonreal_product"])
def test_original_invalid_children_cannot_be_folded_out_of_native_numeric_input(branch, wrapper):
    from sympy.core import _native, _native_expr

    pole = s.log(0, evaluate=False)
    if wrapper == "zero_product":
        argument = s.Mul(0, pole, evaluate=False)
    elif wrapper == "zero_power":
        argument = s.Pow(pole, 0, evaluate=False)
    else:
        argument = s.Mul(0, s.I, evaluate=False)
    value = s.LambertW(argument, branch, evaluate=False)
    with pytest.raises(ValueError):
        _native.evalf_decimal_native(_native_expr(value), 25)


@NATIVE_ONLY
@pytest.mark.parametrize("equation", ["positive", "negative", "affine"])
def test_complete_real_solver_roots_are_usable_by_evalf_and_float(equation):
    x = s.Symbol("lambert_numeric_solution")
    if equation == "positive":
        source = x*s.exp(x) - 1
        expected_roots = (s.LambertW(1),)
        expected_decimals = (CASES[0][4],)
        residual = lambda value: value*math.exp(value) - 1
    elif equation == "negative":
        source = x*s.exp(x) + s.Rational(1, 4)
        expected_roots = (s.LambertW(-s.Rational(1, 4)),
                          s.LambertW(-s.Rational(1, 4), -1))
        expected_decimals = (CASES[1][4], CASES[2][4])
        residual = lambda value: value*math.exp(value) + 0.25
    else:
        source = s.exp(2*x + 3) - 3*(2*x + 3)
        expected_roots = tuple(
            (-s.LambertW(-s.Rational(1, 3), branch) - 3)/2
            for branch in (0, -1)
        )
        expected_decimals = (
            "-1.190469356632027443923836502989538883349",
            "-0.7439327241710787630516301609639806476982",
        )
        residual = lambda value: math.exp(2*value + 3) - 3*(2*value + 3)
    result = s.solveset(source, x, s.S.Reals)
    assert result == s.FiniteSet(*expected_roots)
    assert sorted(str(root.evalf(40)) for root in result) == sorted(expected_decimals)
    for root in result:
        value = float(root)
        assert math.isfinite(value)
        # This is a binary64 end-to-end check, not a root certificate.
        assert abs(residual(value)) < 4e-13


@NATIVE_ONLY
@pytest.mark.parametrize("branch,point,expected", [
    (0, 1.0, CASES[0][5]),
    (0, -0.25, CASES[1][5]),
    (-1, -0.25, CASES[2][5]),
])
def test_compiled_real_branches_keep_value_chain_rule_and_second_derivative(
    branch, point, expected,
):
    x, y = s.symbols("compiled_lambert_x compiled_lambert_y")
    source = s.LambertW(2*x + y, branch, evaluate=False)
    coordinates = (point / 4, point / 2)
    slope = expected / (point * (1 + expected))
    curvature = -(expected**2 * (expected + 2)) / (point**2 * (1 + expected)**3)

    assert s.compile_numeric((x, y), source)(*coordinates) == pytest.approx(expected, rel=5e-14)
    system = s.compile_residual_system([source], (x, y))
    values, jacobian = system.evaluate(*coordinates)
    assert values == pytest.approx([expected], rel=5e-14)
    assert jacobian[0] == pytest.approx([2*slope, slope], rel=5e-13)
    assert system.jacobian_vector_product([2, 3], *coordinates) == pytest.approx([7*slope])
    assert system.jacobian_transpose_vector_product([5], *coordinates) == pytest.approx(
        [10*slope, 5*slope],
    )

    objective = s.compile_objective(source, (x, y))
    value, gradient, hessian = objective.evaluate(*coordinates)
    assert value == pytest.approx(expected, rel=5e-14)
    assert gradient == pytest.approx([2*slope, slope], rel=5e-13)
    assert hessian[0] == pytest.approx([4*curvature, 2*curvature], rel=5e-12)
    assert hessian[1] == pytest.approx([2*curvature, curvature], rel=5e-12)
    assert objective.hessian_vector_product([2, 3], *coordinates) == pytest.approx(
        [14*curvature, 7*curvature], rel=5e-12,
    )


@NATIVE_ONLY
def test_compiled_principal_zero_uses_removable_derivatives_and_typed_bindings():
    x = s.Dummy("lambert_same_name")
    other = s.Dummy("lambert_same_name")
    for source in (s.LambertW(x, evaluate=False), s.LambertW(x, 0, evaluate=False)):
        assert s.compile_numeric(x, source)(0) == 0.0
        assert s.compile_residual_system([source], x).evaluate(0) == ([0.0], [[1.0]])
        assert s.compile_objective(source, x).evaluate(0) == (0.0, [1.0], [[-2.0]])
        with pytest.raises(ValueError):
            s.compile_numeric(other, source)


@NATIVE_ONLY
@pytest.mark.parametrize("branch,point", [(0, -0.5), (-1, 0.0), (-1, 0.25)])
def test_compiled_lambert_original_domains_survive_zero_products_and_directions(branch, point):
    x = s.Symbol("compiled_lambert_domain")
    source = s.Mul(0, s.LambertW(x, branch, evaluate=False), evaluate=False)
    scalar = s.compile_numeric(x, source)
    residual = s.compile_residual_system([source], x)
    objective = s.compile_objective(source, x)
    for call in (
        lambda: scalar(point),
        lambda: residual.jacobian_vector_product([0], point),
        lambda: objective.hessian_vector_product([0], point),
    ):
        with pytest.raises(ValueError):
            call()


@NATIVE_ONLY
def test_lambert_numeric_admission_requires_literal_real_branch_metadata():
    from sympy.core import _native, _native_expr

    x = s.Symbol("compiled_lambert_branch")
    malformed = [
        s.Integer(1), s.Integer(-2), s.Integer(1 << 8192), s.Rational(1, 2), s.Float(0),
        s.Symbol("symbolic_lambert_branch"), s.Add(0, 0, evaluate=False),
    ]
    for branch in malformed:
        source = s.LambertW(x, branch, evaluate=False)
        raw = _native_expr(source)
        if type(branch) is s.Float or type(branch) is s.Add:
            assert not raw.args[1].is_integer
        for original in (source, raw):
            with pytest.raises(ValueError):
                s.compile_numeric(x, original)
            with pytest.raises(ValueError):
                s.compile_residual_system([original], x)
    for arguments in ((), (_native_expr(x), _native_expr(s.Integer(0)), _native_expr(s.Integer(0)))):
        with pytest.raises(ValueError):
            s.compile_numeric(x, _native.py_function("LambertW", *arguments))


@NATIVE_ONLY
def test_lambert_compilation_does_not_accept_custom_classes_or_changed_hooks(monkeypatch):
    from sympy.core import _native_expr

    x = s.Symbol("compiled_lambert_class")
    source = s.LambertW(x, evaluate=False)
    native = _native_expr(source)

    class CustomLambertW(s.LambertW):
        pass

    for custom in (s.Function("LambertW")(x), CustomLambertW(x, evaluate=False)):
        with pytest.raises(NotImplementedError):
            s.compile_numeric(x, custom)
    calls = []

    def changed(*args):
        calls.append(args)
        raise AssertionError("closed source admission cannot invoke a changed hook")

    monkeypatch.setattr(s.LambertW, "_eval_derivative", changed, raising=False)
    for original in (source, native):
        with pytest.raises(NotImplementedError):
            s.compile_numeric(x, original)
        with pytest.raises(NotImplementedError):
            s.compile_residual_system([original], x)
    assert calls == []


@NATIVE_ONLY
def test_lambert_saved_programs_replay_branch_bound_derivatives_and_reject_substitution():
    x = s.Symbol("compiled_lambert_transport")
    source = s.LambertW(x, -1, evaluate=False)
    system = s.compile_residual_system([source], x)
    encoded = system.to_bytes()
    restored = s.load_residual_system(encoded, residuals=[source], args=x)
    assert restored.evaluate(-0.25) == system.evaluate(-0.25)
    # Residual bundles store checked derivative programs, not value bytecode.
    assert b'"LambertWMinusOneDerivative"' in encoded
    forged = encoded.replace(b'"LambertWMinusOneDerivative"', b'"LambertWDerivative"')
    with pytest.raises(ValueError):
        s.load_residual_system(forged, residuals=[source], args=x)
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=[s.LambertW(x)], args=x)

    objective = s.compile_objective(source, x)
    loaded = s.load_objective(objective.to_bytes(), objective=source, args=x)
    assert loaded.evaluate(-0.25) == objective.evaluate(-0.25)


@NATIVE_ONLY
def test_lambert_native_work_retains_operational_budget_and_cancellation_errors():
    x = s.Symbol("compiled_lambert_budget")
    program = s.compile_numeric(x, s.LambertW(x))
    budget = s.EvaluationBudget(max_steps=32)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program(1.0, budget=budget)
    assert budget.remaining_steps < 32
    remaining = budget.remaining_steps
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program(1.0, budget=budget)
    assert budget.remaining_steps == remaining
    assert program(1.0) == pytest.approx(CASES[0][5], rel=5e-14)


@NATIVE_ONLY
def test_lambert_native_admission_does_not_change_the_math_backend():
    x = s.Symbol("compiled_lambert_math")
    with pytest.raises(NotImplementedError):
        s.lambdify(x, s.LambertW(x), modules="math")
