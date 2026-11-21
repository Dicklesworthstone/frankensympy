"""Midpoint Newton proposals cannot narrow or replace an exact parameter family."""

from fractions import Fraction
import importlib

import pytest
import sympy as s


LIMIT = 1 << 40


def ledger(**changes):
    limits = dict(max_steps=LIMIT, max_memory_bytes=LIMIT, max_allocations=LIMIT)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def exact(value):
    return Fraction(value.p, value.q)


def identities(bindings):
    return tuple((binding.name, binding.plain,
                  None if binding.plain else binding.identity_hex) for binding in bindings)


def affine_program():
    x, p = s.symbols("certified_parameter_unknown certified_parameter_value")
    return s.compile_residual_system([x-p], (x, p)), (x, p)


def test_nonlinear_solve_certifies_the_whole_family_and_its_sensitivity():
    x, p = s.symbols("certified_parameter_sqrt_x certified_parameter_sqrt_p")
    source = x*x-p
    program = s.compile_residual_system([source], (x, p))
    budget = ledger()
    receipt = program.solve_parametric_certified(
        1, [(2, 3)], unknown_radii=[Fraction(1, 2)], precision=16, budget=budget,
    )
    assert type(receipt) is s.ParametricRootBoxReceipt
    assert receipt.residuals == (source,)
    assert receipt.parameter_domain == ((2, 3),)
    assert receipt.domain == receipt.unknown_domain + receipt.parameter_domain
    assert receipt.precision_digits == 16
    center = exact(receipt.center[0])
    assert tuple(map(exact, receipt.unknown_domain[0])) == (center-Fraction(1, 2), center+Fraction(1, 2))
    lo, hi = map(exact, receipt.root_enclosure[0])
    assert 0 < lo and lo*lo <= 2 and hi*hi >= 3
    assert 0 <= receipt.contraction_bound < 1
    before = budget.remaining_steps
    sensitivity = receipt.enclose_sensitivity(budget=budget)
    derivative_lo, derivative_hi = map(exact, sensitivity.solution_jacobian[0][0])
    assert 0 < derivative_lo <= derivative_hi
    assert derivative_lo*derivative_lo <= Fraction(1, 12)
    assert derivative_hi*derivative_hi >= Fraction(1, 8)
    assert budget.remaining_steps < before < LIMIT


@pytest.mark.parametrize("parameter,radius", [
    (Fraction(1, 10), Fraction(1, 100)),
    (Fraction(1, 10**400), Fraction(1, 10**399)),
])
def test_exact_fractional_point_parameters_survive_rounded_or_underflowed_midpoints(parameter, radius):
    program, _ = affine_program()
    receipt = program.solve_parametric_certified(
        0, [(parameter, parameter)], unknown_radii=[radius],
    )
    assert tuple(map(exact, receipt.parameter_domain[0])) == (parameter, parameter)
    assert tuple(map(exact, receipt.root_enclosure[0])) == (parameter, parameter)
    center = exact(receipt.center[0])
    assert center == Fraction.from_float(float(parameter))
    assert center != parameter
    assert tuple(map(exact, receipt.unknown_domain[0])) == (center-radius, center+radius)
    assert receipt.contraction_bound == 0


@pytest.mark.parametrize("transport", [None, "legacy", "portable"])
def test_captured_source_typed_order_assumptions_and_nested_layout_are_preserved(transport):
    y = s.Symbol("certified_parameter_same_name", positive=True)
    x = s.Symbol("certified_parameter_same_name")
    b, a = s.Dummy("certified_parameter_same_name"), s.Dummy("certified_parameter_same_name")
    source = s.Matrix([x+y-a, x-2*y-b])
    layout = ((y, x), (b, a))
    program = s.compile_residual_system(source, layout)
    original = (x+y-a, x-2*y-b)
    if transport == "legacy":
        program = s.load_residual_system(program.to_bytes(), residuals=original, args=layout)
    elif transport == "portable":
        program = s.load_portable_residual_system(
            program.to_portable_bytes(), residuals=original, args=layout,
        )
    expected_bindings = identities(program.native_system.variables)
    source[0, 0] = x+y-999
    facts = dict(y._assumptions)
    try:
        y._assumptions.clear()
        y._assumptions["negative"] = True
        # Starting outside the captured positive domain is allowed. Only the
        # final full unknown box is a mathematical claim about those symbols.
        receipt = program.solve_parametric_certified(
            [-10, -10], [(-2, 0), (4, 6)], unknown_radii=[1, 2], tol=0,
        )
        assert receipt.residuals == original
        assert identities(receipt.variables) == expected_bindings
        assert identities(receipt.unknown_variables) == expected_bindings[:2]
        assert identities(receipt.parameters) == expected_bindings[2:]
        assert receipt.parameter_domain == ((-2, 0), (4, 6))
        assert receipt.root_enclosure == ((s.Rational(4, 3), s.Rational(8, 3)), (2, 4))
        with pytest.raises(s.RootSearchError, match="positive") as rejected:
            program.solve_parametric_certified(
                [0, 0], [(-2, 0), (4, 6)], unknown_radii=[3, 2], tol=0,
            )
        assert rejected.value.stage == "certification"
    finally:
        y._assumptions.clear()
        y._assumptions.update(facts)


def test_exact_domains_and_radii_are_snapshotted_before_the_numerical_proposal(monkeypatch):
    program, _ = affine_program()
    parameter_box, radii = [[1, 2]], [1]
    original = type(program).solve_parametric_candidate

    def mutate_after_snapshot(self, *args, **kwargs):
        parameter_box[0][:] = [100, 200]
        radii[0] = 100
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(program), "solve_parametric_candidate", mutate_after_snapshot)
    receipt = program.solve_parametric_certified(0, parameter_box, unknown_radii=radii)
    assert receipt.parameter_domain == ((1, 2),)
    assert receipt.unknown_domain == ((s.Rational(1, 2), s.Rational(5, 2)),)
    assert receipt.root_enclosure == ((1, 2),)


@pytest.mark.parametrize("loose_numeric_proposal", [False, True])
def test_single_explicit_box_rejection_never_widens_retries_or_returns_a_candidate(
        monkeypatch, loose_numeric_proposal):
    module = importlib.import_module("sympy.utilities._parametric_root")
    original, calls = module.certify_parametric_root_box, []

    def observe(*args, **kwargs):
        calls.append((args[1], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "certify_parametric_root_box", observe)
    program, (x, p) = affine_program()
    if loose_numeric_proposal:
        parameters, options = [(1, 1)], dict(tol=2, maxiter=0)
    else:
        program = s.compile_residual_system([x*x-p], (x, p))
        parameters, options = [(2, 3)], {}
    with pytest.raises(s.RootSearchError) as rejected:
        program.solve_parametric_certified(
            0 if loose_numeric_proposal else 1, parameters,
            unknown_radii=[Fraction(1, 100)], **options,
        )
    assert rejected.value.stage == "certification"
    assert rejected.value.attempts == 1 and len(rejected.value.failures) == 1
    assert len(calls) == 1
    assert calls[0][0][0][1] - calls[0][0][0][0] == Fraction(1, 50)
    assert calls[0][1] == tuple(tuple(map(Fraction, pair)) for pair in parameters)


@pytest.mark.parametrize("hidden", ["zero_product", "zero_power", "parameter_derivative"])
def test_valid_midpoint_cannot_erase_parameter_poles_or_derivative_domains(hidden):
    x, p = s.symbols("certified_parameter_pole_x certified_parameter_pole_p")
    if hidden == "parameter_derivative":
        source = x-s.sqrt(p)
    else:
        inverse = s.Pow(s.Add(p, -1, evaluate=False), -1, evaluate=False)
        term = (s.Mul(0, inverse, evaluate=False) if hidden == "zero_product" else
                s.Add(s.Pow(inverse, 0, evaluate=False), -1, evaluate=False))
        source = s.Add(x, term, evaluate=False)
    program = s.compile_residual_system([source], (x, p))
    assert program.solve_parametric_candidate(0, 0.5).residual_norm <= 1e-12
    with pytest.raises(s.RootSearchError) as rejected:
        program.solve_parametric_certified(0, [(0, 1)], unknown_radii=[1])
    assert rejected.value.stage == "certification"


@pytest.mark.parametrize("invalid", [
    "floating_source", "unsupported_unknown_assumption", "unsupported_parameter_assumption",
    "nonreal_unknown_assumption", "zero_unknown_assumption",
    "parameter_assumption", "reversed_bounds", "floating_bound", "floating_radius",
    "zero_radius", "negative_radius", "oversized_radius", "radius_shape", "parameter_shape",
    "midpoint_overflow", "precision", "square", "empty", "too_many_unknowns", "too_many_coordinates",
])
def test_static_source_and_exact_input_refusals_precede_all_numerical_work(monkeypatch, invalid):
    program, (x, p) = affine_program()
    parameters, radii, options = [(1, 2)], [1], {}
    if invalid == "floating_source":
        program = s.compile_residual_system([x-p+s.Float(0.5)], (x, p))
    elif invalid in ("unsupported_unknown_assumption", "unsupported_parameter_assumption"):
        if invalid == "unsupported_unknown_assumption":
            x = s.Symbol("certified_integer_unknown", integer=True)
        else:
            p = s.Symbol("certified_integer_parameter", integer=True)
        program = s.compile_residual_system([x-p], (x, p))
    elif invalid == "nonreal_unknown_assumption":
        x = s.Symbol("certified_nonreal_unknown", real=False)
        program = s.compile_residual_system([x-p], (x, p))
    elif invalid == "zero_unknown_assumption":
        x = s.Symbol("certified_zero_unknown", zero=True)
        program = s.compile_residual_system([x-p], (x, p))
    elif invalid == "parameter_assumption":
        p = s.Symbol("certified_positive_parameter", positive=True)
        program = s.compile_residual_system([x-p], (x, p))
        parameters = [(-1, 2)]
    elif invalid == "reversed_bounds":
        parameters = [(2, 1)]
    elif invalid == "floating_bound":
        parameters = [(1.0, 2)]
    elif invalid == "floating_radius":
        radii = [0.5]
    elif invalid == "zero_radius":
        radii = [0]
    elif invalid == "negative_radius":
        radii = [-1]
    elif invalid == "oversized_radius":
        radii = [1 << 4096]
    elif invalid == "radius_shape":
        radii = []
    elif invalid == "parameter_shape":
        parameters = []
    elif invalid == "midpoint_overflow":
        parameters = [(1 << 1025, 1 << 1025)]
    elif invalid == "precision":
        options["precision"] = True
    elif invalid == "square":
        program = s.compile_residual_system([x], x)
    elif invalid == "empty":
        program = s.compile_residual_system([], (x, p))
    elif invalid == "too_many_unknowns":
        program = s.compile_residual_system([x, x*x], x)
    else:
        variables = (x, p) + tuple(s.Symbol(f"certified_extra_parameter_{i}") for i in range(15))
        program = s.compile_residual_system([x-p], variables)

    def forbidden(*args, **kwargs):
        raise AssertionError("static certificate/input refusal must precede numerical solving")

    monkeypatch.setattr(type(program), "solve_parametric_candidate", forbidden)
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        program.solve_parametric_certified(object(), parameters, unknown_radii=radii, **options)


def test_exact_input_snapshots_never_run_numeric_or_container_overrides(monkeypatch):
    program, _ = affine_program()

    class Number:
        def __float__(self):
            raise AssertionError("exact inputs must not invoke float conversion hooks")

        def __int__(self):
            raise AssertionError("exact inputs must not invoke integer conversion hooks")

    class Container(list):
        def __len__(self):
            raise AssertionError("exact snapshots must not invoke custom container hooks")

        def __iter__(self):
            raise AssertionError("exact snapshots must not invoke custom container hooks")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid exact inputs must refuse before numerical work")

    monkeypatch.setattr(type(program), "solve_parametric_candidate", forbidden)
    for parameters, radii in (
        ([(Number(), 2)], [1]), ([(1, 2)], [Number()]),
        (Container([(1, 2)]), [1]), ([(1, 2)], Container([1])),
        ([Container([1, 2])], [1]),
    ):
        with pytest.raises(TypeError):
            program.solve_parametric_certified(object(), parameters, unknown_radii=radii)


@pytest.mark.parametrize("supplied", [False, True])
def test_one_cumulative_ledger_covers_exact_midpoint_newton_and_full_family_check(monkeypatch, supplied):
    module = importlib.import_module("sympy.utilities._parametric_root")
    program, _ = affine_program()
    original_candidate = type(program).solve_parametric_candidate
    original_certificate = module.certify_parametric_root_box
    original_fallback = module._fallback_budget
    created, observations = [], []

    def fallback():
        budget = original_fallback()
        created.append(budget)
        return budget

    def candidate(self, *args, budget, **kwargs):
        observations.append(("candidate", budget, budget.remaining_steps))
        return original_candidate(self, *args, budget=budget, **kwargs)

    def certificate(*args, budget, **kwargs):
        observations.append(("certificate", budget, budget.remaining_steps))
        return original_certificate(*args, budget=budget, **kwargs)

    monkeypatch.setattr(module, "_fallback_budget", fallback)
    monkeypatch.setattr(type(program), "solve_parametric_candidate", candidate)
    monkeypatch.setattr(module, "certify_parametric_root_box", certificate)
    budget = ledger() if supplied else None
    receipt = program.solve_parametric_certified(
        0, [(1, 2)], unknown_radii=[1], budget=budget,
    )
    assert receipt.root_enclosure == ((1, 2),)
    assert len(created) == (0 if supplied else 1)
    actual_budget = budget if supplied else created[0]
    assert [stage for stage, _, _ in observations] == ["candidate", "certificate"]
    assert all(observed is actual_budget for _, observed, _ in observations)
    assert actual_budget.remaining_steps < observations[1][2] < observations[0][2]


@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_terminal_budget_refusals_do_not_publish_or_convert_exception_types(limit):
    program, _ = affine_program()
    budget = ledger(**{limit: 0})
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=budget)
    assert program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1]).root_enclosure == ((1, 2),)


def test_total_measured_work_cannot_reset_between_phases_or_repeated_calls():
    program, _ = affine_program()
    probe = ledger()
    program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=probe)
    cost = LIMIT-probe.remaining_steps
    assert cost > 0
    budget = ledger(max_steps=2*cost)
    for _ in range(2):
        assert program.solve_parametric_certified(
            0, [(1, 2)], unknown_radii=[1], budget=budget,
        ).root_enclosure == ((1, 2),)
    assert budget.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=budget)
    late = ledger(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=late)
    assert late.remaining_steps < cost-1


def test_pre_cancelled_ledger_refuses_before_any_input_hook():
    program, _ = affine_program()
    budget = ledger()
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.solve_parametric_certified(object(), object(), unknown_radii=object(), budget=budget)
    assert budget.remaining_steps == LIMIT


def test_cancellation_after_newton_prevents_the_certificate_phase(monkeypatch):
    module = importlib.import_module("sympy.utilities._parametric_root")
    program, _ = affine_program()
    original = type(program).solve_parametric_candidate
    budget = ledger()

    def candidate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        budget.cancel()
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled proposal must not reach full-family certification")

    monkeypatch.setattr(type(program), "solve_parametric_candidate", candidate)
    monkeypatch.setattr(module, "certify_parametric_root_box", forbidden)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=budget)


def test_final_cancellation_discards_a_privately_assembled_accepted_receipt(monkeypatch):
    module = importlib.import_module("sympy.utilities._parametric_root")
    program, _ = affine_program()
    original = module.ParametricRootBoxReceipt
    budget, assembled = ledger(), []

    def cancel(receipt):
        result = original(receipt)
        assembled.append(result)
        budget.cancel()
        return result

    monkeypatch.setattr(module, "ParametricRootBoxReceipt", cancel)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1], budget=budget)
    assert len(assembled) == 1
    assert assembled[0].root_enclosure == ((1, 2),)


def test_allocation_failure_keeps_its_operational_type(monkeypatch):
    program, _ = affine_program()

    def exhausted(*args, **kwargs):
        raise MemoryError("test native allocation refusal")

    monkeypatch.setattr(type(program), "solve_parametric_candidate", exhausted)
    with pytest.raises(MemoryError, match="allocation refusal"):
        program.solve_parametric_certified(0, [(1, 2)], unknown_radii=[1])
