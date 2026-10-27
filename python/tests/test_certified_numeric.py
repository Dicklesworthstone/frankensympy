"""Independent public regressions for exact source-bound numerical claims."""

from fractions import Fraction

import pytest

from sympy import (
    Add, Dummy, Float, Mul, Pow, Rational, Symbol, compile_residual_system,
    exp, log, sin, sqrt,
)
from sympy.core import _native, _symbol_binding
from sympy.utilities import ResidualJacobianEnclosure, RootBoxReceipt


def test_rectangular_residual_and_jacobian_enclosures_cover_exact_ranges():
    x, y = Symbol("enclosure_x"), Symbol("enclosure_y")
    program = compile_residual_system([x**2 + y, x*y, x-y], [x, y])
    result = program.enclose([(1, 2), (3, 4)])
    assert type(result) is ResidualJacobianEnclosure
    assert result.residual_values[0] == (4, 8)
    # RealBall multiplication need not produce the smallest interval. Its
    # independently known exact range on this positive rectangle is [3, 8].
    lower, upper = result.residual_values[1]
    assert lower <= 3 <= 8 <= upper
    assert result.residual_values[2] == (-3, -1)
    assert result.jacobian == (((2, 4), (1, 1)), ((3, 4), (1, 2)), ((1, 1), (-1, -1)))
    assert result.domain == ((1, 2), (3, 4))
    assert result.precision_digits == 32
    assert result.residuals == (x**2+y, x*y, x-y)
    assert [v.name for v in result.variables] == [x.name, y.name]
    with pytest.raises(ValueError, match="square"):
        program.certify_root_box([(1, 2), (3, 4)])


def test_exact_point_boxes_and_coordinates_beyond_binary64():
    x = Symbol("large_coordinate_x")
    program = compile_residual_system([x], [x])
    huge = 10**400
    result = program.enclose([(huge, huge)])
    assert result.residual_values == ((huge, huge),)
    assert result.jacobian == (((1, 1),),)
    assert program.enclose([(Fraction(1, 3), Rational(1, 3))]).residual_values == ((Rational(1, 3), Rational(1, 3)),)
    with pytest.raises(ValueError, match="positive width"):
        program.certify_root_box([(0, 0)])


def test_scalar_irrational_root_receipt_has_exact_enclosure():
    x = Symbol("sqrt_two_root_x")
    result = compile_residual_system([x*x-2], [x]).certify_root_box([(Rational(7, 5), Rational(3, 2))])
    assert type(result) is RootBoxReceipt
    lo, hi = result.root_enclosure[0]
    assert Rational(7, 5) < lo <= hi < Rational(3, 2)
    assert lo**2 <= 2 <= hi**2
    assert 0 <= result.contraction_bound < 1
    assert result.center == (Rational(29, 20),)
    assert result.preconditioner == ((Rational(10, 29),),)
    assert "declared closed box" in result.claim


@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_all_supported_system_dimensions_have_exact_affine_roots(dimension):
    variables = [Symbol(f"affine_{dimension}_{i}") for i in range(dimension)]
    program = compile_residual_system([x-(i+1) for i, x in enumerate(variables)], variables)
    result = program.certify_root_box([(i, i+2) for i in range(dimension)])
    assert result.root_enclosure == tuple((i+1, i+1) for i in range(dimension))
    assert result.contraction_bound == 0


def test_coupled_nonlinear_root_has_a_checked_local_claim():
    x, y = Symbol("coupled_root_x"), Symbol("coupled_root_y")
    program = compile_residual_system([x*x+y*y-1, x-y], [x, y])
    result = program.certify_root_box([(Rational(7, 10), Rational(71, 100))]*2)
    for lo, hi in result.root_enclosure:
        assert lo**2 <= Rational(1, 2) <= hi**2
    assert result.contraction_bound < 1


def test_nonsymmetric_inverse_and_nested_argument_layout():
    x, y, z = Symbol("nonsymmetric_x"), Symbol("nonsymmetric_y"), Symbol("nonsymmetric_z")
    program = compile_residual_system([2*x+y-4, 3*y+z-9, x+4*z-13], [(x, y), z])
    assert program.evaluate((1, 2), 3)[0] == [0.0, 0.0, 0.0]
    assert program.variables == (x, y, z)
    result = program.certify_root_box([(0, 2), (1, 3), (2, 4)])
    assert result.root_enclosure == ((1, 1), (2, 2), (3, 3))
    assert result.contraction_bound == 0


def test_transcendental_residuals_reach_independent_interval_checker():
    x, y = Symbol("transcendental_root_x"), Symbol("transcendental_root_y")
    program = compile_residual_system([exp(x)-2, y-sin(x)], [x, y])
    domain = [(Rational(69, 100), Rational(7, 10)), (Rational(63, 100), Rational(13, 20))]
    result = program.certify_root_box(domain, precision=32)
    assert result.domain == tuple(domain)
    assert result.contraction_bound < 1
    assert all(a < lo <= hi < b for (a, b), (lo, hi) in zip(domain, result.root_enclosure))


def test_rational_power_and_whole_box_smooth_domain():
    x = Symbol("fractional_power_root_x")
    program = compile_residual_system([sqrt(x)-2], [x])
    result = program.certify_root_box([(Rational(39, 10), Rational(41, 10))])
    lo, hi = result.root_enclosure[0]
    assert lo <= 4 <= hi
    with pytest.raises(ValueError):
        program.enclose([(0, 1)])


def test_manual_witness_inputs_are_snapshotted_and_rechecked():
    x = Symbol("manual_root_x")
    program = compile_residual_system([x-1], [x])
    domain, center, preconditioner = [[0, 2]], [1], [[1]]
    result = program.certify_root_box(domain, center=center, preconditioner=preconditioner)
    domain[0][0], center[0], preconditioner[0][0] = 100, 100, 0
    assert result.domain == ((0, 2),)
    assert result.center == (1,)
    assert result.preconditioner == ((1,),)
    with pytest.raises(ValueError):
        program.certify_root_box([(0, 2)], preconditioner=[[0]])
    with pytest.raises(ValueError, match="inside"):
        program.certify_root_box([(0, 2)], center=[3])


@pytest.mark.parametrize("case", ["absent", "multiple", "singular", "boundary"])
def test_invalid_root_claims_refuse(case):
    x = Symbol("invalid_root_"+case)
    expression, domain = {
        "absent": (x*x+1, [(1, 2)]),
        "multiple": (x*x-1, [(-2, 2)]),
        "singular": (x*x, [(-1, 1)]),
        "boundary": (x, [(0, 1)]),
    }[case]
    with pytest.raises(ValueError):
        compile_residual_system([expression], [x]).certify_root_box(domain)


@pytest.mark.parametrize("hidden", ["zero_product", "zero_power"])
def test_original_held_domains_cannot_be_erased(hidden):
    x = Symbol("held_domain_"+hidden)
    expression = (Mul(0, Pow(x, -1, evaluate=False), evaluate=False)
                  if hidden == "zero_product" else Pow(log(x), 0, evaluate=False))
    with pytest.raises(ValueError):
        compile_residual_system([expression], [x]).enclose([(-1, 1)])


@pytest.mark.parametrize("value", [0.5, True, "1/2", Float(0.5)])
def test_inexact_or_non_numeric_bounds_refuse(value):
    x = Symbol("invalid_bound_x")
    program = compile_residual_system([x], [x])
    with pytest.raises(TypeError):
        program.enclose([(value, 1)])


def test_conversion_hooks_are_never_called_for_box_inputs():
    class Hook:
        def __float__(self):
            raise AssertionError("foreign conversion hook called")

        def __int__(self):
            raise AssertionError("foreign conversion hook called")

    x = Symbol("hook_bound_x")
    with pytest.raises(TypeError):
        compile_residual_system([x], [x]).enclose([(Hook(), 1)])


def test_floating_source_remains_approximate_only():
    x = Symbol("inexact_source_x")
    source = Add(Mul(Float(0.5), x, evaluate=False), -1, evaluate=False)
    program = compile_residual_system([source], [x])
    assert program.evaluate(2)[0] == [0.0]
    with pytest.raises(ValueError, match="exact source"):
        program.enclose([(1, 3)])
    with pytest.raises(ValueError, match="exact source"):
        program.certify_root_box([(1, 3)])


def test_typed_variables_and_dummies_remain_distinct():
    x = Symbol("typed_box_x")
    positive = Symbol("typed_box_x", positive=True)
    dummy_a, dummy_b = Dummy("same"), Dummy("same")
    variables = [x, positive, dummy_a, dummy_b]
    program = compile_residual_system([v-(i+1) for i, v in enumerate(variables)], variables)
    result = program.certify_root_box([(Rational(1, 2), Rational(3, 2)), (1, 3), (2, 4), (3, 5)])
    assert result.variables[0].plain
    assert not result.variables[1].plain
    assert result.variables[2].name != result.variables[3].name
    assert result.root_enclosure == ((1, 1), (2, 2), (3, 3), (4, 4))


def test_assumptions_are_snapshots_and_apply_to_the_whole_box():
    x = Symbol("positive_snapshot_x", positive=True)
    original = dict(x._assumptions)
    binding = _symbol_binding(x)
    program = compile_residual_system([x-1], [x])
    try:
        x._assumptions.clear()
        result = program.certify_root_box([(Rational(1, 2), Rational(3, 2))])
        assert result.variables[0].identity_hex == binding.identity_hex
        with pytest.raises(ValueError, match="positive"):
            program.enclose([(-1, 2)])
    finally:
        x._assumptions.update(original)


def test_mutation_during_native_constructor_cannot_relabel_captured_assumptions(monkeypatch):
    x = Symbol("constructor_race_x", positive=True)
    original_facts = dict(x._assumptions)
    binding = _symbol_binding(x)
    constructor = _native.VerifiedResidualSystem

    def mutate_after_construction(sources, variables):
        result = constructor(sources, variables)
        x._assumptions.clear()
        x._assumptions["negative"] = True
        return result

    monkeypatch.setattr(_native, "VerifiedResidualSystem", mutate_after_construction)
    try:
        program = compile_residual_system([x-1], [x])
        assert program.enclose([(1, 2)]).variables[0].identity_hex == binding.identity_hex
        with pytest.raises(ValueError, match="positive"):
            program.enclose([(-2, -1)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)


def test_inconsistent_assumption_capture_disables_only_certification(monkeypatch):
    import importlib
    module = importlib.import_module("sympy.utilities.lambdify")
    x = Symbol("capture_race_x", positive=True)
    source = x-1
    original_facts = dict(x._assumptions)
    binding_function = module._symbol_binding

    def mutate_after_binding(variable):
        binding = binding_function(variable)
        variable._assumptions.clear()
        variable._assumptions["negative"] = True
        return binding

    monkeypatch.setattr(module, "_symbol_binding", mutate_after_binding)
    try:
        program = compile_residual_system([source], [x])
        assert program.evaluate(1)[0] == [0.0]
        with pytest.raises(ValueError, match="captured symbol bindings"):
            program.enclose([(-2, -1)])
    finally:
        x._assumptions.clear()
        x._assumptions.update(original_facts)


def test_unsupported_assumptions_refuse_without_affecting_approximation():
    x = Symbol("integer_box_x", integer=True)
    program = compile_residual_system([x-1], [x])
    assert program.evaluate(1)[0] == [0.0]
    with pytest.raises(NotImplementedError, match="assumptions"):
        program.certify_root_box([(0, 2)])


def test_receipts_and_programs_cannot_be_relabelled_or_forged():
    x = Symbol("immutable_box_x")
    program = compile_residual_system([x-1], [x])
    result = program.certify_root_box([(0, 2)])
    with pytest.raises(AttributeError):
        result.domain = ((100, 200),)
    with pytest.raises(AttributeError):
        program.native_system = None
    with pytest.raises(AttributeError):
        result.__init__(result._receipt)
    for cls in (ResidualJacobianEnclosure, RootBoxReceipt, _native.ResidualJacobianEnclosure, _native.RootBoxReceipt):
        with pytest.raises(TypeError):
            cls({"verified": True})


def test_bound_and_dimension_refusals_are_explicit():
    x = Symbol("bounded_box_x")
    program = compile_residual_system([x], [x])
    for precision in (7, 129, True, 32.0):
        with pytest.raises(ValueError, match="precision"):
            program.enclose([(0, 1)], precision=precision)
    with pytest.raises(ValueError, match="bit limit"):
        program.enclose([(0, 1 << 4097)])
    variables = [Symbol(f"too_many_box_{i}") for i in range(17)]
    oversized = compile_residual_system(variables, variables)
    with pytest.raises(ValueError, match="1..16"):
        oversized.enclose([(0, 1)]*17)
