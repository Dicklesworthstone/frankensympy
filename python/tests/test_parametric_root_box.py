"""Uniform root claims retain every declared parameter value, not one sample."""

from fractions import Fraction
import importlib

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _symbol_binding


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


def rational(value):
    return Fraction(value.p, value.q)


def identities(variables):
    # The digest describes assumptions, not the whole coordinate. Plain
    # bindings have no semantic digest; Dummy identity lives in its native name.
    return tuple((variable.name, variable.plain,
                  None if variable.plain else variable.identity_hex)
                 for variable in variables)


def assert_uniform(receipt, unknown_count, parameter_count):
    assert type(receipt) is s.ParametricRootBoxReceipt
    assert len(receipt.residuals) == unknown_count
    assert len(receipt.unknown_variables) == unknown_count
    assert len(receipt.parameters) == parameter_count
    assert len(receipt.center) == len(receipt.root_enclosure) == unknown_count
    assert len(receipt.preconditioner) == unknown_count
    assert all(len(row) == unknown_count for row in receipt.preconditioner)
    assert receipt.domain == receipt.unknown_domain + receipt.parameter_domain
    assert identities(receipt.variables) == identities(receipt.unknown_variables + receipt.parameters)
    assert 0 <= receipt.contraction_bound < 1
    for (lo, hi), center, (root_lo, root_hi) in zip(
            receipt.unknown_domain, receipt.center, receipt.root_enclosure):
        assert lo <= center <= hi
        assert lo < root_lo <= root_hi < hi
    assert "every" in receipt.claim and "parameter" in receipt.claim


def test_nonlinear_square_root_family_is_certified_for_the_entire_parameter_interval():
    x, p = s.symbols("parametric_sqrt_x parametric_sqrt_p")
    source = x*x - p
    result = s.compile_residual_system([source], (x, p)).certify_parametric_root_box(
        [(1, 2)], [(2, 3)], precision=16,
    )
    assert_uniform(result, 1, 1)
    assert result.residuals == (source,)
    assert result.unknown_domain == ((1, 2),)
    assert result.parameter_domain == ((2, 3),)
    assert result.precision_digits == 16
    assert identities(result.unknown_variables) == identities((_symbol_binding(x),))
    assert identities(result.parameters) == identities((_symbol_binding(p),))
    lo, hi = map(rational, result.root_enclosure[0])
    # The family's exact extreme roots are sqrt(2) and sqrt(3), not merely
    # the root at the parameter midpoint 5/2.
    assert 0 < lo and lo*lo <= 2
    assert hi*hi >= 3


def test_coupled_affine_family_preserves_explicit_roles_and_nested_layout():
    x, y, a, b = s.symbols("parametric_x parametric_y parametric_a parametric_b")
    source = s.Matrix([x+y-a, x-y-b])
    program = s.compile_residual_system(source, [(x, y), (a, b)])
    source[0, 0] = x+y-999
    result = program.certify_parametric_root_box([(-1, 3), (-1, 3)], [(1, 2), (-1, 1)])
    assert_uniform(result, 2, 2)
    assert result.residuals == (x+y-a, x-y-b)
    assert result.root_enclosure == ((0, s.Rational(3, 2)),) * 2
    assert result.contraction_bound == 0
    assert result.preconditioner == (
        (s.Rational(1, 2), s.Rational(1, 2)),
        (s.Rational(1, 2), s.Rational(-1, 2)),
    )
    assert identities(result.unknown_variables) == identities(_symbol_binding(v) for v in (x, y))
    assert identities(result.parameters) == identities(_symbol_binding(v) for v in (a, b))


@pytest.mark.parametrize("manual_preconditioner", [False, True])
def test_a_valid_parameter_midpoint_cannot_certify_a_family_that_leaves_the_unknown_box(
        manual_preconditioner):
    x, p = s.symbols("parametric_midpoint_x parametric_midpoint_p")
    unknown_box = [(Fraction(-1, 4), Fraction(1, 4))]
    # At p=0 the local problem is x=0 and certifies immediately. At p=1
    # the root is x=1, outside this box. Sampling p's midpoint is unsound.
    local = s.compile_residual_system([x], x).certify_root_box(unknown_box)
    assert local.root_enclosure == ((0, 0),)
    options = dict(center=[0], preconditioner=[[1]]) if manual_preconditioner else {}
    with pytest.raises(ValueError):
        s.compile_residual_system([x-p*p], (x, p)).certify_parametric_root_box(
            unknown_box, [(-1, 1)], **options,
        )


def test_uniform_contraction_refuses_a_parameter_bifurcation_even_with_a_finite_proposal():
    x, p = s.symbols("parametric_bifurcation_x parametric_bifurcation_p")
    # An explicit nonzero center avoids a merely singular midpoint inverse.
    # At p=0 the root is singular; positive p admits two roots in X.
    with pytest.raises(ValueError):
        s.compile_residual_system([x*x-p], (x, p)).certify_parametric_root_box(
            [(-1, 1)], [(0, Fraction(1, 4))],
            center=[Fraction(1, 2)], preconditioner=[[1]],
        )


def test_roots_on_the_outer_unknown_boundary_cannot_pass_strict_uniform_inclusion():
    x, p = s.symbols("parametric_boundary_x parametric_boundary_p")
    with pytest.raises(ValueError):
        s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
            [(-1, 1)], [(-1, 1)], center=[0], preconditioner=[[1]],
        )


def test_point_parameters_and_interval_parameters_can_share_one_family():
    x, p, q = s.symbols("parametric_fixed_x parametric_fixed_p parametric_fixed_q")
    program = s.compile_residual_system([x-p-q], (x, p, q))
    result = program.certify_parametric_root_box([(0, 4)], [(1, 1), (1, 2)])
    assert_uniform(result, 1, 2)
    assert result.parameter_domain == ((1, 1), (1, 2))
    assert result.root_enclosure == ((2, 3),)
    fixed = program.certify_parametric_root_box([(0, 4)], [(1, 1), (1, 1)])
    assert fixed.root_enclosure == ((2, 2),)
    assert fixed.contraction_bound == 0


def test_maximum_total_coordinate_dimension_preserves_every_unknown_root():
    unknowns = tuple(s.Symbol(f"parametric_dimension_{i}") for i in range(15))
    p = s.Symbol("parametric_dimension_p")
    residuals = [x-p-i for i, x in enumerate(unknowns)]
    result = s.compile_residual_system(residuals, unknowns + (p,)).certify_parametric_root_box(
        [(i-1, i+2) for i in range(15)], [(0, 1)],
    )
    assert_uniform(result, 15, 1)
    assert result.root_enclosure == tuple((i, i+1) for i in range(15))
    assert result.contraction_bound == 0


def test_exact_tiny_parameter_family_never_collapses_through_binary64():
    x, p = s.symbols("parametric_tiny_x parametric_tiny_p")
    tiny = Fraction(1, 10**400)
    result = s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
        [(0, 4*tiny)], [(tiny, 3*tiny)],
    )
    exact = s.Rational(tiny.numerator, tiny.denominator)
    assert_uniform(result, 1, 1)
    assert result.parameter_domain == ((exact, 3*exact),)
    assert result.root_enclosure == ((exact, 3*exact),)
    assert result.contraction_bound == 0


def test_captured_unknown_and_parameter_assumptions_cannot_be_relabelled():
    x = s.Symbol("parametric_positive_x", positive=True)
    p = s.Symbol("parametric_positive_p", positive=True)
    bindings = identities(_symbol_binding(v) for v in (x, p))
    facts = [dict(v._assumptions) for v in (x, p)]
    program = s.compile_residual_system([x-p], (x, p))
    try:
        for variable in (x, p):
            variable._assumptions.clear()
            variable._assumptions["negative"] = True
        result = program.certify_parametric_root_box([(Fraction(1, 2), 3)], [(1, 2)])
        assert identities(result.variables) == bindings
        assert result.root_enclosure == ((1, 2),)
        with pytest.raises(ValueError, match="positive"):
            program.certify_parametric_root_box([(-1, 3)], [(1, 2)])
        with pytest.raises(ValueError, match="positive"):
            program.certify_parametric_root_box([(Fraction(1, 2), 3)], [(-1, 2)])
    finally:
        for variable, original in zip((x, p), facts):
            variable._assumptions.clear()
            variable._assumptions.update(original)


def test_equal_display_names_do_not_merge_unknown_and_parameter_identities():
    x, p = s.Dummy("same"), s.Dummy("same")
    result = s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
        [(-2, 2)], [(-1, 1)],
    )
    assert_uniform(result, 1, 1)
    assert identities(result.unknown_variables) == identities((_symbol_binding(x),))
    assert identities(result.parameters) == identities((_symbol_binding(p),))
    assert identities(result.unknown_variables) != identities(result.parameters)
    assert result.unknown_variables[0].name != result.parameters[0].name
    assert result.root_enclosure == ((-1, 1),)


@pytest.mark.parametrize("hidden", ["zero_product", "zero_exponent"])
def test_original_parameter_pole_is_not_erased_by_a_held_vanishing_term(hidden):
    x, p = s.symbols("parametric_pole_x parametric_pole_p")
    inverse = s.Pow(s.Add(p, -1, evaluate=False), -1, evaluate=False)
    if hidden == "zero_product":
        term = s.Mul(0, inverse, evaluate=False)
    else:
        term = s.Add(s.Pow(inverse, 0, evaluate=False), -1, evaluate=False)
    source = s.Add(x, term, evaluate=False)
    # The parameter midpoint 1/2 is valid; the pole is on P's far endpoint.
    with pytest.raises(ValueError):
        s.compile_residual_system([source], (x, p)).certify_parametric_root_box(
            [(-1, 1)], [(0, 1)], center=[0], preconditioner=[[1]],
        )


@pytest.mark.parametrize("value", [True, 0.0, "0", s.Float(0.0)])
def test_unknown_parameter_center_and_inverse_inputs_are_explicitly_exact(value):
    x, p = s.symbols("parametric_exact_x parametric_exact_p")
    program = s.compile_residual_system([x-p], (x, p))
    with pytest.raises(TypeError, match="exact"):
        program.certify_parametric_root_box([(value, 2)], [(-1, 1)])
    with pytest.raises(TypeError, match="exact"):
        program.certify_parametric_root_box([(-2, 2)], [(value, 1)])
    with pytest.raises(TypeError, match="exact"):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], center=[value])
    with pytest.raises(TypeError, match="exact"):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], preconditioner=[[value]])


def test_shapes_precision_inexact_source_and_scalar_height_are_checked_before_verification():
    x, p = s.symbols("parametric_bad_x parametric_bad_p")
    program = s.compile_residual_system([x-p], (x, p))
    for precision in (7, 129, True, 16.0):
        with pytest.raises(ValueError, match="precision"):
            program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], precision=precision)
    with pytest.raises(ValueError, match="4096-bit"):
        program.certify_parametric_root_box([(-2, 2)], [(Fraction(1, 1 << 4096), 1)])
    with pytest.raises(ValueError):
        program.certify_parametric_root_box([(0, 0)], [(-1, 1)])
    with pytest.raises(ValueError, match="lower bound"):
        program.certify_parametric_root_box([(-2, 2)], [(1, -1)])
    for unknown_box, parameter_box in [([], [(-1, 1)]), ([(-2, 2)], [])]:
        with pytest.raises(TypeError, match="entries"):
            program.certify_parametric_root_box(unknown_box, parameter_box)
    with pytest.raises(TypeError, match="entries"):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], center=[])
    with pytest.raises(TypeError, match="entries"):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], preconditioner=[[1, 0]])
    with pytest.raises(ValueError):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], center=[3])
    with pytest.raises(ValueError):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], preconditioner=[[0]])
    with pytest.raises(ValueError):
        s.compile_residual_system([x], x).certify_parametric_root_box([(-1, 1)], [])
    source = s.Add(x, -p, s.Float(0.5), evaluate=False)
    with pytest.raises(ValueError, match="exact source"):
        s.compile_residual_system([source], (x, p)).certify_parametric_root_box([(-2, 2)], [(-1, 1)])
    variables = tuple(s.Symbol(f"parametric_excessive_{i}") for i in range(17))
    with pytest.raises(ValueError):
        s.compile_residual_system([variables[0]-variables[-1]], variables).certify_parametric_root_box(
            [(-2, 2)], [(-1, 1)]*16,
        )


def test_admission_does_not_call_custom_numeric_or_sequence_conversion_hooks():
    x, p = s.symbols("parametric_callbacks_x parametric_callbacks_p")
    program = s.compile_residual_system([x-p], (x, p))
    calls = []

    class Numeric:
        def __float__(self):
            calls.append("float")
            return 0.0

        def __int__(self):
            calls.append("int")
            return 0

    class CustomList(list):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

    class CustomInt(int):
        def __int__(self):
            calls.append("custom_int")
            return super().__int__()

    for value in (Numeric(), CustomInt(0)):
        with pytest.raises(TypeError):
            program.certify_parametric_root_box([(-2, 2)], [(value, 1)])
        with pytest.raises(TypeError):
            program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], center=[value])
        with pytest.raises(TypeError):
            program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], preconditioner=[[value]])
    with pytest.raises(TypeError):
        program.certify_parametric_root_box(CustomList([(-2, 2)]), [(-1, 1)])
    with pytest.raises(TypeError):
        program.certify_parametric_root_box([(-2, 2)], CustomList([(-1, 1)]))
    with pytest.raises(TypeError):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], preconditioner=[CustomList([1])])
    assert calls == []


def test_receipt_owns_both_domain_groups_and_has_no_unchecked_mutation_path():
    x, p = s.symbols("parametric_receipt_x parametric_receipt_p")
    unknown_box, parameter_box = [[-2, 2]], [[-1, 1]]
    center, preconditioner = [0], [[1]]
    result = s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
        unknown_box, parameter_box, center=center, preconditioner=preconditioner,
    )
    unknown_box[0][:] = [100, 200]
    parameter_box[0][:] = [100, 200]
    center[0] = 100
    preconditioner[0][0] = 100
    assert result.unknown_domain == ((-2, 2),)
    assert result.parameter_domain == ((-1, 1),)
    assert result.center == (0,) and result.preconditioner == ((1,),)
    assert result.root_enclosure == ((-1, 1),)
    assert type(result.unknown_variables) is type(result.parameters) is tuple
    with pytest.raises(AttributeError):
        result.parameter_domain = ((100, 200),)
    with pytest.raises(AttributeError):
        del result._receipt
    with pytest.raises(AttributeError):
        result.__init__(result._receipt)
    with pytest.raises(TypeError):
        s.ParametricRootBoxReceipt(object())
    with pytest.raises(TypeError):
        _native.ParametricRootBoxReceipt()
    with pytest.raises(AttributeError):
        result._receipt.parameter_domain = ()
    from sympy.utilities import ParametricRootBoxReceipt
    assert ParametricRootBoxReceipt is s.ParametricRootBoxReceipt


def test_native_uniform_checker_preserves_roles_and_does_not_trust_a_preconditioner():
    x, p = s.symbols("parametric_native_x parametric_native_p")
    native = s.compile_residual_system([x-p], (x, p)).native_system
    unknown_box = [(_native_expr(-2), _native_expr(2))]
    parameter_box = [(_native_expr(-1), _native_expr(1))]
    with pytest.raises(TypeError):
        native.certify_parametric_root_box(unknown_box, [(-1.0, 1.0)], [_native_expr(0)], [[_native_expr(1)]])
    with pytest.raises(ValueError):
        native.certify_parametric_root_box(unknown_box, parameter_box, [_native_expr(0)], [[_native_expr(-1)]])
    result = native.certify_parametric_root_box(
        unknown_box, parameter_box, [_native_expr(0)], [[_native_expr(1)]], 16,
    )
    assert type(result) is _native.ParametricRootBoxReceipt
    assert len(result.unknown_variables) == len(result.parameters) == 1
    assert result.precision_digits == 16


def test_automatic_preconditioner_and_uniform_verifier_use_one_supplied_budget():
    x, p = s.symbols("parametric_budget_x parametric_budget_p")
    program = s.compile_residual_system([x-p], (x, p))
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.certify_parametric_root_box(
            [(-2, 2)], [(-1, 1)], budget=ample_budget(max_memory_bytes=100*1024*1024),
        )
    # The explicit inverse avoids the first native enclosure phase; it still
    # receives the same independent uniform check and consumes its ledger.
    manual = ample_budget(max_memory_bytes=100*1024*1024)
    result = program.certify_parametric_root_box(
        [(-2, 2)], [(-1, 1)], center=[0], preconditioner=[[1]], budget=manual,
    )
    assert result.root_enclosure == ((-1, 1),)
    assert manual.remaining_memory_bytes < 100*1024*1024 - 64*1024*1024
    probe = ample_budget()
    program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=probe)
    cost = (1 << 40) - probe.remaining_memory_bytes
    assert cost >= 128*1024*1024
    exact = ample_budget(max_memory_bytes=cost)
    program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=exact)
    assert exact.remaining_memory_bytes == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=exact)


def test_one_private_fallback_ledger_spans_preparation_and_both_native_phases(monkeypatch):
    x, p = s.symbols("parametric_fallback_x parametric_fallback_p")
    module = importlib.import_module("sympy.utilities._parametric_root")
    original = module._call_budgeted
    created, observed = [], []

    def fallback():
        budget = ample_budget()
        created.append(budget)
        return budget

    def observe(function, *args, budget):
        observed.append((function.__name__, budget, budget.remaining_steps))
        return original(function, *args, budget=budget)

    monkeypatch.setattr(module, "_fallback_budget", fallback)
    monkeypatch.setattr(module, "_call_budgeted", observe)
    program = s.compile_residual_system([x-p], (x, p))
    program.certify_parametric_root_box([(-2, 2)], [(-1, 1)])
    assert len(created) == 1
    assert [name for name, _, _ in observed] == ["enclose", "certify_parametric_root_box"]
    assert all(budget is created[0] for _, budget, _ in observed)
    assert observed[0][2] < 1 << 45
    assert observed[1][2] < observed[0][2]
    supplied = ample_budget()
    observed.clear()
    program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=supplied)
    assert len(created) == 1
    assert all(budget is supplied for _, budget, _ in observed)


def test_cancellation_and_exhaustion_are_not_converted_into_mathematical_refusals(monkeypatch):
    x, p = s.symbols("parametric_cancel_x parametric_cancel_p")
    program = s.compile_residual_system([x-p], (x, p))
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=cancelled)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=ample_budget(max_steps=0))
    module = importlib.import_module("sympy.utilities._parametric_root")
    original = module._call_budgeted
    budget = ample_budget()

    def cancel_after_uniform_acceptance(function, *args, **kwargs):
        result = original(function, *args, **kwargs)
        if function.__name__ == "certify_parametric_root_box":
            budget.cancel()
        return result

    monkeypatch.setattr(module, "_call_budgeted", cancel_after_uniform_acceptance)
    with pytest.raises(s.EvaluationCancelled):
        program.certify_parametric_root_box([(-2, 2)], [(-1, 1)], budget=budget)


def test_unchecked_native_value_cannot_authorize_a_uniform_receipt(monkeypatch):
    x, p = s.symbols("parametric_forged_x parametric_forged_p")
    module = importlib.import_module("sympy.utilities._parametric_root")
    monkeypatch.setattr(module, "_call_budgeted", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="accepted native"):
        s.compile_residual_system([x-p], (x, p)).certify_parametric_root_box(
            [(-2, 2)], [(-1, 1)], center=[0], preconditioner=[[1]],
        )
