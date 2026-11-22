"""Scalar Newton uses checked jets without changing its value-only root shortcut."""

import importlib
import math

import pytest
import sympy as s


def ledger(enabled):
    return s.EvaluationBudget() if enabled else None


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("initial, expected", ((3, 2), (-4, -3)))
def test_absolute_chain_newton_preserves_equations_and_typed_solution_keys(budgeted, initial, expected):
    x = s.Dummy("scalar_absolute", real=True)
    source = s.Eq(s.Abs(2*x+1), 5, evaluate=False)
    result = s.nsolve(source, x, initial, dict=True, budget=ledger(budgeted))
    assert len(result) == 1 and len(result[0]) == 1
    variable, root = next(iter(result[0].items()))
    assert variable is x
    assert type(root) is s.Float and float(root) == expected


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("initial", (0.0, 1e-12, -1e-12))
def test_sinc_origin_newton_uses_stable_numeric_derivatives(budgeted, initial, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_sinc")
    source = s.sinc(x) + x - 1 - s.sin(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar checked jets must not use public symbolic differentiation")

    monkeypatch.setattr(numerical, "diff", forbidden)
    root = float(s.nsolve(source, x, initial, budget=ledger(budgeted)))
    assert root == pytest.approx(1, abs=1e-12)
    assert abs(math.sin(root)/root + root - 1 - math.sin(1)) <= 1e-12


@pytest.mark.parametrize("budgeted", (False, True))
def test_exact_zero_residuals_precede_cusp_and_singular_derivative_evaluation(budgeted):
    x = s.Symbol("scalar_zero_root")
    for source in (x*x, s.sqrt(x), s.Abs(x), s.sinc(x)-1):
        root = s.nsolve(source, x, 0, budget=ledger(budgeted))
        assert type(root) is s.Float and float(root) == 0
    # The same contract applies to a trial root reached after a smooth step.
    assert float(s.nsolve(s.Abs(x), x, -3, maxiter=1, budget=ledger(budgeted))) == 0


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("held", ("direct", "zero_product", "zero_power", "square"))
def test_nonroot_cusps_are_not_erased_by_zero_derivatives_or_held_arithmetic(budgeted, held):
    x = s.Symbol("scalar_cusp")
    absolute = s.Abs(x, evaluate=False)
    source = {
        "direct": absolute-2,
        "zero_product": s.Add(x, -1, s.Mul(0, absolute, evaluate=False), evaluate=False),
        "zero_power": s.Add(x, -2, s.Pow(absolute, 0, evaluate=False), evaluate=False),
        "square": s.Abs(x*x, evaluate=False)-1,
    }[held]
    with pytest.raises(ValueError):
        s.nsolve(source, x, 0, budget=ledger(budgeted))


@pytest.mark.parametrize("budgeted", (False, True))
def test_original_source_poles_cannot_become_apparent_exact_roots(budgeted):
    x = s.Symbol("scalar_source_pole")
    reciprocal = s.Pow(x, -1, evaluate=False)
    for source in (
        s.Add(x, s.Mul(0, reciprocal, evaluate=False), evaluate=False),
        s.Abs(s.Mul(0, reciprocal, evaluate=False), evaluate=False),
        s.sinc(reciprocal, evaluate=False),
    ):
        with pytest.raises(ValueError):
            s.nsolve(source, x, 0, budget=ledger(budgeted))


@pytest.mark.parametrize("name, argument, value", (
    ("floor", s.Rational(3, 2), 1),
    ("ceiling", s.Rational(3, 2), 2),
    ("gamma", 3, 2),
    ("factorial", 3, 6),
))
def test_math_only_constant_functions_keep_the_existing_scalar_path(name, argument, value, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_math_only")
    source = s.Add(x, getattr(s, name)(argument, evaluate=False), -value-2, evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("an explicit math-only source must not attempt a checked native Jacobian")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert float(s.nsolve(source, x, 0, maxiter=1)) == 2
    # A caller ledger still requests finite native value evaluation, which has
    # never supported these functions. It must not silently use unmetered math.
    with pytest.raises((ValueError, NotImplementedError)):
        s.nsolve(source, x, 0, budget=s.EvaluationBudget())


@pytest.mark.parametrize("budgeted", (False, True))
@pytest.mark.parametrize("sign", (-1, 1))
def test_piecewise_keeps_existing_lazy_symbolic_scalar_newton(budgeted, sign, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_piecewise")
    source = s.Piecewise((x-2, x >= 0), (x+2, True), evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Piecewise does not carry a checked smooth derivative certificate")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    root = s.nsolve(source, x, sign*3, budget=ledger(budgeted))
    assert float(root) == sign*2


def test_exact_integer_cancellation_keeps_math_intermediates_outside_binary64(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_integer_intermediates")

    def forbidden(*args, **kwargs):
        raise AssertionError("exact integer arithmetic must select the math path before compilation")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    huge = s.Pow(2, 1024, evaluate=False)
    source = s.Add(huge, s.Mul(-1, huge, evaluate=False), x, -2, evaluate=False)
    assert float(s.nsolve(source, x, 0)) == 2

    # Every literal fits in 53 bits, and the source is smaller than the checked
    # lane's source cap. Only the intermediate integer product exceeds binary64.
    huge = s.Mul(*(s.Integer(1 << 40) for _ in range(32)), evaluate=False)
    source = s.Add(huge, s.Mul(-1, huge, evaluate=False), x, s.sinc(x), -1,
                   s.Mul(-1, s.sin(1), evaluate=False), evaluate=False)
    assert float(s.nsolve(source, x, 2)) == pytest.approx(1, abs=1e-12)


def test_integer_precision_is_retained_before_a_sinc_and_logarithm_chain(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_integer_precision")
    base = 1 << 27
    square = s.Mul(base+1, base+1, evaluate=False)
    neighbor = s.Add(s.Mul(base, base, evaluate=False), 2*base, evaluate=False)
    # The exact constant difference is one; binary64 rounds the two products
    # to the same value. Losing it changes the logarithm's original domain.
    argument = s.Add(square, s.Mul(-1, neighbor, evaluate=False), x, evaluate=False)
    source = s.Add(s.log(argument, evaluate=False), s.sinc(x), -1, evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("integer cancellation must not round before checking the source domain")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert float(s.nsolve(source, x, 0.1)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("exponent", (2**53+1, -(2**53-1), s.Rational(2**80+1, 2**80)))
def test_large_exact_exponents_keep_scalar_early_roots(exponent, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_exact_exponent")
    power = s.Pow(x, exponent, evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible source exponents must not reach checked compilation")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert float(s.nsolve(power-1, x, 1)) == 1
    with_sinc = s.Add(power, -1, s.sinc(x), s.Mul(-1, s.sin(1), evaluate=False),
                      evaluate=False)
    assert float(s.nsolve(with_sinc, x, 1)) == 1


def test_computed_integer_exponents_keep_math_negative_base_semantics(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_computed_exponent")
    exponent = s.Add(1, 1, evaluate=False)
    source = s.Add(s.Pow(x, exponent, evaluate=False), s.sinc(x), -1,
                   s.Mul(-1, s.sin(1), evaluate=False), evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("a computed integer exponent is not a native literal exponent")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert float(s.nsolve(source, x, -2)) == pytest.approx(-1, abs=1e-12)


def test_large_math_sources_do_not_inherit_native_derivative_program_limits(monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_large_math_source")
    source = s.Add(*(s.Integer(0) for _ in range(8193)), x, s.sinc(x), -1,
                   s.Mul(-1, s.sin(1), evaluate=False), evaluate=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("a large admitted math source must retain its established scalar path")

    monkeypatch.setattr(numerical, "compile_residual_system", forbidden)
    assert float(s.nsolve(source, x, 2)) == pytest.approx(1, abs=1e-12)


def test_unbound_typed_coordinates_and_opaque_functions_refuse_before_derivative_hooks():
    x = s.Symbol("scalar_closed")
    other = s.Symbol("scalar_closed", positive=True)
    with pytest.raises(ValueError, match="unbound"):
        s.nsolve(s.Abs(other)-2, x, 3)

    class OpaqueScalarJet(s.Function):
        def _eval_derivative(self, variable):
            raise AssertionError("closed scalar admission must precede custom derivative hooks")

    with pytest.raises(NotImplementedError):
        s.nsolve(OpaqueScalarJet(x, evaluate=False), x, 1)


@pytest.mark.parametrize("exception", (ValueError, RuntimeError))
def test_checked_compilation_failures_never_retry_symbolic_derivatives(exception, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_compile_refusal")
    refusal = exception("checked derivative compilation refused")

    def refused(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(numerical, "compile_residual_system", refused)
    with pytest.raises(exception) as caught:
        s.nsolve(s.Abs(x)-2, x, 3)
    assert caught.value is refusal


def test_absolute_newton_keeps_one_cumulative_evaluation_budget():
    x = s.Symbol("scalar_cumulative")
    source = s.Abs(x)-2
    measured = s.EvaluationBudget(max_steps=1_000_000)
    assert float(s.nsolve(source, x, -3, budget=measured)) == -2
    consumed = 1_000_000 - measured.remaining_steps
    assert consumed > 4
    exact = s.EvaluationBudget(max_steps=consumed)
    assert float(s.nsolve(source, x, -3, budget=exact)) == -2
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(source, x, -3, budget=exact)
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.nsolve(source, x, -3, budget=s.EvaluationBudget(max_steps=consumed-1))


@pytest.mark.parametrize("mapping", (False, True))
def test_scalar_exact_root_checks_cancellation_after_result_assembly(mapping, monkeypatch):
    numerical = importlib.import_module("sympy.solvers.numerical")
    x = s.Symbol("scalar_cancel_publication")
    budget = s.EvaluationBudget()
    original_float = numerical.Float

    def cancelled_float(value):
        result = original_float(value)
        budget.cancel()
        return result

    monkeypatch.setattr(numerical, "Float", cancelled_float)
    with pytest.raises(s.EvaluationCancelled):
        s.nsolve(s.Abs(x), x, 0, budget=budget, dict=mapping)


@pytest.mark.parametrize("options, exception", (
    ({"maxiter": 0}, ValueError),
    ({"maxiter": True}, TypeError),
    ({"tol": 0}, ValueError),
    ({"dict": 1}, TypeError),
    ({"max_step_norm": 1}, ValueError),
    ({"backend": "unknown"}, ValueError),
    ({"budget": object()}, TypeError),
))
def test_scalar_checked_jets_keep_existing_solver_control_admission(options, exception):
    x = s.Symbol("scalar_controls")
    with pytest.raises(exception):
        s.nsolve(s.Abs(x)-2, x, 3, **options)
