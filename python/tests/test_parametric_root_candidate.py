"""Fixed-parameter Newton proposals retain the complete original typed source."""

from fractions import Fraction
import importlib
import math

import pytest
import sympy as s

from sympy.core import _native, _symbol_binding
from sympy.utilities.lambdify import NumericParametricRootCandidate


LIMIT = 1 << 40


def ledger(**changes):
    limits = dict(max_steps=LIMIT, max_memory_bytes=LIMIT, max_allocations=LIMIT)
    limits.update(changes)
    return s.EvaluationBudget(**limits)


def remaining(budget):
    return (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)


def affine_program():
    x, p = s.symbols("fixed_parameter_x fixed_parameter_p")
    return s.compile_residual_system([2*x-p], (x, p)), (x, p)


def test_one_compiled_program_solves_multiple_fixed_parameter_values_without_recompilation(monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x, p = s.symbols("sweep_x sweep_p")
    program = s.compile_residual_system([x*x-p], (x, p))
    original_bytes = program.to_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed-parameter solving must reuse the original compiled system")

    monkeypatch.setattr(module, "compile_residual_system", forbidden)
    monkeypatch.setattr(module, "_numeric_source", forbidden)
    budget = ledger()
    candidates = []
    for parameter, root in ((4, 2), (9, 3), (16, 4)):
        before = remaining(budget)
        candidate = program.solve_parametric_candidate(1, parameter, budget=budget)
        assert type(candidate) is NumericParametricRootCandidate
        assert candidate.native_system is program.native_system
        assert candidate.root == pytest.approx((root,), abs=1e-11)
        assert candidate.parameter_values == (float(parameter),)
        actual = program.native_system.residuals(list(candidate.root + candidate.parameter_values))
        assert candidate.residual_norm == max(map(abs, actual))
        assert candidate.residual_norm <= 1e-12
        assert all(after < old for after, old in zip(remaining(budget), before))
        candidates.append(candidate)
    assert tuple(candidate.parameter_values for candidate in candidates) == ((4.0,), (9.0,), (16.0,))
    assert program.to_bytes() == original_bytes


@pytest.mark.parametrize("transport", [None, "legacy", "portable"])
def test_typed_unknown_and_parameter_orders_survive_nested_arguments_and_transport(transport):
    x = s.Symbol("parameter_same_name")
    y = s.Symbol("parameter_same_name", positive=True)
    a, b = s.Dummy("parameter_same_name"), s.Dummy("parameter_same_name")
    variables = (y, x, b, a)
    residuals = [x+y-a, x-2*y-b]
    layout = ((y, x), (b, a))
    program = s.compile_residual_system(residuals, layout)
    if transport == "legacy":
        program = s.load_residual_system(program.to_bytes(), residuals=residuals, args=layout)
    elif transport == "portable":
        program = s.load_portable_residual_system(
            program.to_portable_bytes(), residuals=residuals, args=layout,
        )
    # The native snapshot must retain semantic identities, not merely names.
    bindings = program.native_system.variables
    assert len(bindings) == len(variables)
    for actual, axis in zip(bindings, variables):
        expected = _symbol_binding(axis)
        assert actual.name == expected.name
        assert actual.plain == expected.plain
        if not actual.plain:
            assert actual.identity_hex == expected.identity_hex
    snapshot = tuple((axis.name, axis.plain, axis.identity_hex) for axis in bindings)
    with pytest.raises(AttributeError):
        bindings[0].identity_hex = "00" * 32
    bindings.clear()
    assert tuple((axis.name, axis.plain, axis.identity_hex)
                 for axis in program.native_system.variables) == snapshot
    with pytest.raises(AttributeError):
        program.native_system.variables = []
    with pytest.raises(AttributeError):
        del program.native_system.variables
    candidate = program.solve_parametric_candidate([0, 0], [-1, 5], tol=0)
    assert candidate.root == (2.0, 3.0)
    assert all(actual is expected for actual, expected in zip(candidate.variables, variables))
    assert candidate.unknown_variables == (y, x)
    assert candidate.parameters == (b, a)
    assert candidate.parameter_values == (-1.0, 5.0)
    assert program.residuals(candidate.root, candidate.parameter_values) == [0.0, 0.0]


def test_candidate_retains_native_source_and_snapshots_mutable_inputs():
    x, p = s.symbols("snapshot_unknown snapshot_parameter")
    source = s.Matrix([p*x-2])
    program = s.compile_residual_system(source, (x, p))
    source[0, 0] = x-999
    initial, parameters = [0], [2]
    candidate = program.solve_parametric_candidate(initial, parameters)
    initial[0], parameters[0] = 900, 900
    assert candidate.root == (1.0,)
    assert candidate.parameter_values == (2.0,)
    assert candidate.native_system is program.native_system
    assert program.solve_parametric_candidate(0, 4).root == (0.5,)
    assert candidate.root == (1.0,)


def test_parameter_values_report_the_binary64_numbers_actually_used():
    program, variables = affine_program()
    candidate = program.solve_parametric_candidate(0, s.Rational(1, 10))
    assert candidate.parameter_values == (0.1,)
    assert Fraction.from_float(candidate.parameter_values[0]) != Fraction(1, 10)
    assert candidate.unknown_variables == variables[:1]
    assert candidate.parameters == variables[1:]
    assert type(candidate.parameter_values) is tuple
    assert all(type(value) is float and math.isfinite(value) for value in candidate.root)
    assert candidate.backend == "frankennumpy" and candidate.evidence == "approximate"
    assert not hasattr(candidate, "root_enclosure")


def test_candidate_source_binding_and_immutability_cannot_be_replaced_by_a_lookalike():
    from sympy.utilities import NumericParametricRootCandidate as UtilityCandidate

    assert s.NumericParametricRootCandidate is UtilityCandidate is NumericParametricRootCandidate
    program, variables = affine_program()
    candidate = program.solve_parametric_candidate(0, 2)
    proposal = program.native_system.solve_parametric_candidate([0], [2])
    same_source = s.compile_residual_system([2*variables[0]-variables[1]], variables)
    with pytest.raises(ValueError, match="different residual system"):
        NumericParametricRootCandidate(proposal, same_source.native_system, variables)
    with pytest.raises(ValueError, match="typed coordinate order"):
        NumericParametricRootCandidate(proposal, program.native_system, variables[::-1])
    with pytest.raises(TypeError, match="native parametric root proposal"):
        NumericParametricRootCandidate(object(), program.native_system, variables)
    with pytest.raises(TypeError):
        _native.NumericParametricRootProposal()
    for name in ("root", "parameter_values", "variables", "unknown_variables", "parameters",
                 "native_system", "residual_norm", "iterations", "backend", "evidence", "_proposal"):
        with pytest.raises(AttributeError):
            setattr(candidate, name, None)
        with pytest.raises(AttributeError):
            delattr(candidate, name)
    with pytest.raises(AttributeError):
        candidate.__init__(proposal, program.native_system, variables)
    with pytest.raises(AttributeError):
        proposal.parameter_values = (100.0,)
    assert "NumericParametricRootCandidate" in repr(candidate)


@pytest.mark.parametrize("unknown_count,parameter_count", [(1, 1), (1, 15), (15, 1)])
def test_bounded_coordinate_groups_include_unused_fixed_parameters(unknown_count, parameter_count):
    unknowns = tuple(s.Symbol(f"bounded_unknown_{index}") for index in range(unknown_count))
    parameters = tuple(s.Symbol(f"bounded_parameter_{index}") for index in range(parameter_count))
    sources = [unknown-(index+1)*parameters[-1] for index, unknown in enumerate(unknowns)]
    program = s.compile_residual_system(sources, unknowns+parameters)
    values = list(range(parameter_count-1))+[2]
    candidate = program.solve_parametric_candidate([0]*unknown_count, values, tol=0, maxiter=1)
    assert candidate.root == tuple(float(2*(index+1)) for index in range(unknown_count))
    assert candidate.parameter_values == tuple(map(float, values))
    assert candidate.residual_norm == 0 and candidate.iterations == 1


@pytest.mark.parametrize("hidden", ["zero_product", "zero_power", "parameter_derivative"])
def test_original_parameter_domains_and_every_derivative_remain_active(hidden):
    x, p = s.symbols("domain_unknown domain_parameter")
    if hidden == "zero_product":
        source = s.Add(x-p, s.Mul(0, s.log(p), evaluate=False), evaluate=False)
    elif hidden == "zero_power":
        source = s.Add(x-p, s.Pow(s.log(p), 0, evaluate=False), -1, evaluate=False)
    else:
        # The residual and unknown derivative are finite at (0,0), but the
        # retained parameter derivative of sqrt(p) is undefined there.
        source = x-s.sqrt(p)
    program = s.compile_residual_system([source], (x, p))
    budget = ledger()
    before = budget.remaining_steps
    with pytest.raises(ValueError):
        program.solve_parametric_candidate(0, 0, maxiter=0, budget=budget)
    assert budget.remaining_steps < before
    assert program.solve_parametric_candidate(1, 1).residual_norm == 0.0


@pytest.mark.parametrize("operation,endpoint,interior", [
    ("sqrt", 0, 1),
    ("asin", -1, 0),
    ("asin", 1, 0),
    ("acos", -1, 0),
    ("acos", 1, 0),
    ("acosh", 1, 2),
    ("Abs", 0, -1),
    ("fractional_power", 0, 1),
])
@pytest.mark.parametrize("hidden", ["zero_product", "zero_power"])
def test_retained_parameter_endpoints_are_scalar_values_but_not_smooth_root_proposals(
        operation, endpoint, interior, hidden):
    x, p = s.symbols("smooth_endpoint_unknown smooth_endpoint_parameter")
    inner = (s.Pow(p, s.Rational(3, 2), evaluate=False) if operation == "fractional_power"
             else getattr(s, operation)(p))
    term = (s.Mul(0, inner, evaluate=False) if hidden == "zero_product" else
            s.Add(s.Pow(inner, 0, evaluate=False), -1, evaluate=False))
    source = s.Add(x, -1, term, evaluate=False)
    scalar = s.compile_numeric((x, p), source)
    assert scalar(1, endpoint) == 0.0
    program = s.compile_residual_system([source], (x, p))
    # F=0 with a unit formal unknown derivative still cannot bypass a retained
    # source operation lacking a common smooth neighborhood in the full axes.
    with pytest.raises(ValueError):
        program.solve_parametric_candidate(1, endpoint, tol=0, maxiter=0)
    accepted = program.solve_parametric_candidate(1, interior, tol=0, maxiter=0)
    assert accepted.root == (1.0,) and accepted.residual_norm == 0.0
    assert accepted.parameter_values == (float(interior),)


def test_integer_zero_base_remains_smooth_and_unrepresentable_literal_exponents_refuse():
    x, p = s.symbols("smooth_integer_unknown smooth_integer_parameter")
    source = s.Add(x, -1, s.Mul(0, s.Pow(p, 2, evaluate=False), evaluate=False), evaluate=False)
    program = s.compile_residual_system([source], (x, p))
    assert program.solve_parametric_candidate(1, 0, tol=0, maxiter=0).root == (1.0,)
    # A noninteger exponent that binary64 would round to 1 cannot enter the
    # checked program: exact literal components must fit within 53 bits.
    exponent = s.Rational((1 << 60) + 1, 1 << 60)
    wide = s.Add(x, -1, s.Mul(0, s.Pow(p, exponent, evaluate=False), evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.compile_residual_system([wide], (x, p))


def test_only_unknown_jacobian_columns_drive_newton_and_small_residuals_remain_proposals():
    x, p = s.symbols("singular_unknown fixed_parameter")
    program = s.compile_residual_system([x*x-p], (x, p))
    with pytest.raises(ValueError):
        program.solve_parametric_candidate(0, 1)
    candidate = program.solve_parametric_candidate(0, 0, maxiter=0, tol=0)
    assert candidate.root == (0.0,) and candidate.iterations == 0
    assert candidate.evidence == "approximate"
    with pytest.raises(ValueError):
        program.certify_parametric_root_box([(-1, 1)], [(0, 0)])


def test_newton_controls_are_effective_and_never_change_fixed_parameters():
    program, _ = affine_program()
    assert program.solve_parametric_candidate(1, 2, maxiter=0, tol=0).iterations == 0
    with pytest.raises(ValueError, match="no convergence"):
        program.solve_parametric_candidate(0, 2, maxiter=0)
    with pytest.raises(ValueError, match="no convergence"):
        program.solve_parametric_candidate(0, 2, max_step_norm=0.1)
    result = program.solve_parametric_candidate(0, 2, maxiter=1, tol=0, max_step_norm=2)
    assert result.root == (1.0,) and result.parameter_values == (2.0,)


def test_approximate_proposal_composes_with_independent_uniform_root_and_sensitivity_checks():
    program, _ = affine_program()
    budget = ledger()
    candidate = program.solve_parametric_candidate(0, 2, budget=budget)
    before = budget.remaining_steps
    center = [Fraction.from_float(value) for value in candidate.root]
    family = program.certify_parametric_root_box(
        [(Fraction(1, 2), Fraction(3, 2))], [(Fraction(3, 2), Fraction(5, 2))],
        center=center, budget=budget,
    )
    assert family.root_enclosure == ((s.Rational(3, 4), s.Rational(5, 4)),)
    assert family.parameter_domain == ((s.Rational(3, 2), s.Rational(5, 2)),)
    sensitivity = family.enclose_sensitivity(budget=budget)
    assert sensitivity.solution_jacobian == (((s.Rational(1, 2), s.Rational(1, 2)),),)
    assert budget.remaining_steps < before


@pytest.mark.parametrize("limit", ["max_steps", "max_memory_bytes", "max_allocations"])
def test_all_resource_dimensions_refuse_without_poisoning_the_compiled_program(limit):
    program, _ = affine_program()
    for target, initial, parameters in (
        (program, 0, 2), (program.native_system, [0], [2]),
    ):
        budget = ledger(**{limit: 0})
        with pytest.raises(s.EvaluationBudgetExceeded):
            target.solve_parametric_candidate(initial, parameters, budget=budget)
    assert program.solve_parametric_candidate(0, 2).root == (1.0,)


def test_repeated_solves_and_final_publication_share_the_measured_budget():
    program, _ = affine_program()
    probe = ledger()
    program.solve_parametric_candidate(0, 2, budget=probe)
    cost = LIMIT-probe.remaining_steps
    assert cost > 0
    budget = ledger(max_steps=2*cost)
    for parameter in (2, 4):
        result = program.solve_parametric_candidate(0, parameter, budget=budget)
        assert result.root == (parameter/2,)
    assert budget.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_parametric_candidate(0, 2, budget=budget)
    late = ledger(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_parametric_candidate(0, 2, budget=late)
    assert late.remaining_steps < cost-1


def test_cancelled_calls_refuse_before_inputs_and_final_publication_checks_cancellation(monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    program, _ = affine_program()
    budget = ledger()
    budget.cancel()
    for target in (program, program.native_system):
        with pytest.raises(s.EvaluationCancelled):
            target.solve_parametric_candidate(object(), object(), budget=budget)
    assert remaining(budget) == (LIMIT, LIMIT, LIMIT)
    active, constructed = ledger(), []
    original = module.NumericParametricRootCandidate

    def cancel(*args):
        result = original(*args)
        constructed.append(result)
        active.cancel()
        return result

    monkeypatch.setattr(module, "NumericParametricRootCandidate", cancel)
    with pytest.raises(s.EvaluationCancelled):
        program.solve_parametric_candidate(0, 2, budget=active)
    assert len(constructed) == 1 and constructed[0].root == (1.0,)


@pytest.mark.parametrize("options", [
    {"tol": -1}, {"tol": float("nan")}, {"tol": True}, {"maxiter": -1},
    {"maxiter": 1025}, {"maxiter": True}, {"max_step_norm": 0},
    {"max_step_norm": float("inf")}, {"max_step_norm": True},
])
def test_invalid_controls_refuse_before_numeric_input_hooks(options):
    program, _ = affine_program()

    class Number:
        def __float__(self):
            raise AssertionError("invalid controls must not invoke numeric conversion hooks")

    for target, initial, parameters in (
        (program, Number(), Number()), (program.native_system, [Number()], [Number()]),
    ):
        with pytest.raises((TypeError, ValueError)):
            target.solve_parametric_candidate(initial, parameters, **options)


@pytest.mark.parametrize("initial,parameters", [
    ([], [2]), ([0, 1], [2]), ([0], []), ([0], [2, 3]),
    ([float("nan")], [2]), ([0], [float("inf")]), ([[0]], [2]), ([0], [[2]]),
])
def test_coordinate_shape_and_nonfinite_values_are_explicit_refusals(initial, parameters):
    program, _ = affine_program()
    for target in (program, program.native_system):
        with pytest.raises((TypeError, ValueError)):
            target.solve_parametric_candidate(initial, parameters)


def test_custom_numeric_and_container_hooks_never_run():
    program, _ = affine_program()

    class Number:
        def __float__(self):
            raise AssertionError("numeric proposals must not invoke custom conversion hooks")

    class Container(list):
        def __len__(self):
            raise AssertionError("numeric proposals must not invoke custom container hooks")

        def __iter__(self):
            raise AssertionError("numeric proposals must not invoke custom container hooks")

    for initial, parameters in ((Number(), 2), (0, Number()), (Container([0]), [2]), ([0], Container([2]))):
        with pytest.raises((TypeError, ValueError)):
            program.solve_parametric_candidate(initial, parameters)
    with pytest.raises((TypeError, ValueError)):
        program.native_system.solve_parametric_candidate([Number()], [2])


@pytest.mark.parametrize("kind", ["square", "empty", "too_many_residuals", "too_many_variables"])
def test_static_parametric_shape_is_admitted_before_input_conversion(kind):
    x, p = s.symbols("shape_unknown shape_parameter")
    if kind == "square":
        program = s.compile_residual_system([x], x)
    elif kind == "empty":
        program = s.compile_residual_system([], (x, p))
    elif kind == "too_many_residuals":
        program = s.compile_residual_system([x, x*x], x)
    else:
        variables = (x, p) + tuple(s.Symbol(f"shape_extra_{i}") for i in range(15))
        program = s.compile_residual_system([x-p], variables)
    for target in (program, program.native_system):
        with pytest.raises(ValueError):
            target.solve_parametric_candidate(object(), object())


@pytest.mark.parametrize("shape", ["square", "overdetermined", "underdetermined"])
def test_sparse_parametric_roots_separate_unknown_count_from_residual_count(shape, monkeypatch):
    x, y, unused, p = s.symbols("rectangular_x rectangular_y rectangular_unused rectangular_p")
    if shape == "overdetermined":
        sources, axes, initial = [x*x-p, 2*x*x-2*p], (x, p), [1]
    elif shape == "underdetermined":
        sources, axes, initial = [x+y-p], (x, y, unused, p), [0, 0, 17]
    else:
        sources, axes, initial = [x*x-p], (x, p), [1]
    program = s.compile_residual_system(sources, axes)
    original = program.to_bytes()
    module = importlib.import_module("sympy.utilities.lambdify")

    def forbidden(*args, **kwargs):
        raise AssertionError("parametric sparse solving recompiled its frozen source")

    monkeypatch.setattr(module, "compile_residual_system", forbidden)
    monkeypatch.setattr(module, "_numeric_source", forbidden)
    for parameter in (4, 9):
        result = program.solve_parametric_candidate(initial, [parameter], backend="sparse")
        assert type(result) is NumericParametricRootCandidate
        assert result.backend == "sparse" and result.evidence == "approximate"
        assert result.native_system is program.native_system
        assert result.unknown_variables == axes[:-1]
        assert result.parameters == (p,)
        assert result.parameter_values == (float(parameter),)
        assert result.residual_norm <= 1e-12
        if shape == "underdetermined":
            assert result.root[0] + result.root[1] == pytest.approx(parameter, abs=1e-11)
            assert result.root[2] == 17
        else:
            assert result.root == pytest.approx((math.sqrt(parameter),), abs=1e-11)
    assert program.to_bytes() == original


def test_sparse_parametric_root_never_uses_fixed_values_to_repair_inconsistent_rows():
    x, p, q = s.symbols("inconsistent_unknown inconsistent_parameter another_parameter")
    program = s.compile_residual_system([x-p, x-q], (x, p, q))
    with pytest.raises(ValueError, match="no convergence"):
        program.solve_parametric_candidate(0, [1, 3], backend="sparse")
    assert program.solve_parametric_candidate(0, [2, 2], backend="sparse").root == pytest.approx((2,))


@pytest.mark.parametrize("hidden", ["zero_product", "parameter_derivative"])
def test_sparse_parametric_root_keeps_all_fixed_parameter_domain_obligations(hidden):
    x, p = s.symbols("sparse_guard_unknown sparse_guard_parameter")
    expression = (x-s.sqrt(p) if hidden == "parameter_derivative" else
                  s.Add(x, s.Mul(0, s.log(p), evaluate=False), evaluate=False))
    program = s.compile_residual_system([expression], (x, p))
    with pytest.raises(ValueError):
        program.solve_parametric_candidate(0, 0, backend="sparse", maxiter=0)


def test_sparse_parametric_root_supports_maximum_parameter_suffix_and_retains_signed_zero():
    x = s.Symbol("sparse_many_fixed_unknown")
    parameters = s.symbols("sparse_many_fixed_0:255")
    program = s.compile_residual_system([x-parameters[-1]], (x,) + parameters)
    values = [-0.0] * 254 + [3.0]
    result = program.solve_parametric_candidate(0, values, backend="sparse")
    assert result.root == pytest.approx((3.0,))
    assert len(result.parameters) == 255
    assert math.copysign(1, result.parameter_values[0]) == -1.0
    values[0], values[-1] = 500, 500
    assert math.copysign(1, result.parameter_values[0]) == -1.0
    assert result.parameter_values[-1] == 3.0
    with pytest.raises(ValueError, match="16 total variables"):
        program.solve_parametric_candidate(0, values)


@pytest.mark.parametrize("backend", ["auto", "frankennumpy", "SPARSE"])
def test_parametric_root_rejects_unknown_backend_without_silent_fallback(backend):
    program, _ = affine_program()
    for target in (program, program.native_system):
        with pytest.raises(ValueError, match="backend"):
            target.solve_parametric_candidate([0], [1], backend=backend)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("options", [{"tol": True}, {"tol": False}, {"max_step_norm": True}])
def test_parametric_root_rejects_boolean_controls_with_valid_coordinates(backend, options):
    program, _ = affine_program()
    for target in (program, program.native_system):
        with pytest.raises(TypeError):
            target.solve_parametric_candidate([0], [1], backend=backend, **options)
