"""Fixed real Lambert W branches through compiled and certified consumers.

Approximate reference values are fixed independently of the candidate engine.
Exact zero jets, chain factors, source binding, branch domains and refused
forgeries are separate contracts from binary64 numerical agreement.
"""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr


def _source(argument, branch):
    return (s.LambertW(argument, evaluate=False) if branch is None else
            s.LambertW(argument, branch, evaluate=False))


@pytest.mark.parametrize("branch,point,expected", [
    (None, 1.0, 0.5671432904097838),
    (0, -0.25, -0.3574029561813889),
    (-1, -0.25, -2.15329236411035),
    (0, 1e20, 42.306755091738395),
    (-1, -1e-100, -235.72115887568532),
    (0, 1e-80, 1e-80),
    (0, -1e-80, -1e-80),
])
@pytest.mark.parametrize("form", ["shell", "native", "generic"])
def test_real_branches_reach_native_scalar_callables(branch, point, expected, form):
    x = s.Symbol("lambert_value")
    source = _source(x, branch)
    if form == "native":
        source = _native_expr(source)
    elif form == "generic":
        source = s.Expr(_native_expr(source))
    scalar = s.compile_numeric(x, source)
    assert scalar(point) == pytest.approx(expected, rel=8e-14, abs=0)
    assert s.lambdify(x, source, modules="frankensympy")(point) == pytest.approx(
        expected, rel=8e-14, abs=0,
    )


def test_branch_selection_survives_matrix_outputs_and_batch_evaluation():
    x = s.Symbol("lambert_batch")
    scalar = s.compile_numeric(x, s.Matrix([[_source(x, 0), _source(x, -1)]]))
    expected = [[-0.3574029561813889, -2.15329236411035]]
    assert scalar(-0.25)[0] == pytest.approx(expected[0])
    assert scalar.evaluate_batch([[-0.25], [-0.25]]) == [scalar(-0.25), scalar(-0.25)]
    with pytest.raises(ValueError):
        scalar.evaluate_batch([[-0.25], [0.0]])


@pytest.mark.parametrize("branch", [None, 0])
def test_principal_origin_jet_and_directional_products_are_regular(branch):
    x = s.Symbol("lambert_origin")
    source = _source(x, branch)
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    for point in (-0.0, 0.0):
        assert residual.evaluate(point) == ([0.0], [[1.0]])
        assert objective.evaluate(point) == (0.0, [1.0], [[-2.0]])
        assert residual.jacobian_vector_product([3], point) == [3.0]
        assert residual.jacobian_transpose_vector_product([4], point) == [4.0]
        assert objective.hessian_vector_product([3], point) == [-6.0]


@pytest.mark.parametrize("branch,w", [(0, -0.3574029561813889), (-1, -2.15329236411035)])
def test_checked_nonzero_jets_preserve_nonlinear_chain_terms(branch, w):
    x, y = s.symbols("lambert_chain_x lambert_chain_y")
    source = _source(s.Rational(-1, 4) + 2*x + y + 3*x*x, branch)
    point = -0.25
    slope = w / (point*(1+w))
    curvature = -(w*w)*(w+2) / (point*point*(1+w)**3)
    residual = s.compile_residual_system(source, (x, y))
    objective = s.compile_objective(source, (x, y))
    values, jacobian = residual.evaluate(0, 0)
    value, gradient, hessian = objective.evaluate(0, 0)
    assert values == pytest.approx([w], rel=2e-13)
    assert value == pytest.approx(w, rel=2e-13)
    assert jacobian[0] == pytest.approx([2*slope, slope], rel=2e-13)
    assert gradient == pytest.approx([2*slope, slope], rel=2e-13)
    assert hessian[0] == pytest.approx([4*curvature+6*slope, 2*curvature], rel=2e-12)
    assert hessian[1] == pytest.approx([2*curvature, curvature], rel=2e-12)
    assert objective.hessian_vector_product([2, 3], 0, 0) == pytest.approx(
        [14*curvature+12*slope, 7*curvature], rel=2e-12,
    )


def test_direct_source_certification_has_exact_origin_derivatives_without_compilation(monkeypatch):
    x = s.Symbol("lambert_exact_source")
    source = _source(2*x+3*x*x, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("source certificates cannot compile derivative programs")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    assert residual.residual_values == ((0, 0),)
    assert residual.jacobian == (((2, 2),),)
    assert objective.value == (0, 0)
    assert objective.gradient == ((2, 2),)
    # W''(0)*(2)^2 + W'(0)*6 = -8 + 6.
    assert objective.hessian == (((-2, -2),),)
    assert residual.residuals == (source,) and objective.objective == source


@pytest.mark.parametrize("branch,w", [(0, -0.3574029561813889), (-1, -2.15329236411035)])
def test_direct_and_compiled_interval_jets_retain_the_original_branch(branch, w):
    x = s.Symbol("lambert_enclosure")
    source = _source(x, branch)
    box = [(s.Rational(-1, 4), s.Rational(-1, 4))]
    direct = s.enclose_objective(source, x, box, precision=24)
    compiled = s.compile_objective(source, x).enclose(box, precision=24)
    assert (compiled.value, compiled.gradient, compiled.hessian) == (
        direct.value, direct.gradient, direct.hessian,
    )
    slope = w / (-0.25*(1+w))
    curvature = -w*w*(w+2) / (0.0625*(1+w)**3)
    for interval, expected in ((direct.value, w), (direct.gradient[0], slope),
                               (direct.hessian[0][0], curvature)):
        lower, upper = interval
        assert lower <= upper
        assert upper-lower < s.Rational(1, 10**18)
        assert float(lower) == pytest.approx(expected, rel=2e-12)
        assert float(upper) == pytest.approx(expected, rel=2e-12)


def test_principal_root_and_stationary_claims_cross_zero_smoothly():
    x = s.Symbol("lambert_zero_claims")
    source = _source(x, 0)
    box = [(s.Rational(-1, 100), s.Rational(1, 100))]
    for receipt in (
        s.certify_root_box(source, x, box),
        s.compile_residual_system(source, x).certify_root_box(box),
    ):
        assert receipt.root_enclosure[0][0] <= 0 <= receipt.root_enclosure[0][1]
        assert receipt.contraction_bound < 1
        assert receipt.residuals == (source,)
    objective = s.Pow(source, 2, evaluate=False)
    receipt = s.compile_objective(objective, x).certify_stationary_point(box)
    assert receipt.kind == "strict_minimum"
    assert receipt.stationary_enclosure[0][0] <= 0 <= receipt.stationary_enclosure[0][1]
    assert receipt.hessian_enclosure[0][0][0] > 0


def test_lower_branch_root_certificate_binds_its_source():
    x = s.Symbol("lambert_lower_root")
    source = _source(x, -1) + 2
    box = [(s.Rational(-28, 100), s.Rational(-26, 100))]
    receipt = s.certify_root_box(source, x, box)
    lower, upper = receipt.root_enclosure[0]
    assert box[0][0] < lower <= upper < box[0][1]
    assert float(lower) <= -2/math.e**2 <= float(upper)
    assert receipt.contraction_bound < 1 and receipt.residuals == (source,)


@pytest.mark.parametrize("branch,point", [(0, -0.5), (-1, -0.5), (-1, 0), (-1, 0.25)])
def test_nonreal_or_infinite_selected_branches_never_publish_finite_outputs(branch, point):
    x = s.Symbol("lambert_domain")
    source = s.Mul(0, _source(x, branch), evaluate=False)
    scalar = s.compile_numeric(x, source)
    with pytest.raises(ValueError):
        scalar(point)
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    with pytest.raises(ValueError):
        residual.jacobian_vector_product([0], point)
    with pytest.raises(ValueError):
        objective.hessian_vector_product([0], point)
    for enclose in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            enclose(source, x, [(s.Rational(point), s.Rational(point))])


@pytest.mark.parametrize("branch", [0, -1])
def test_real_branchpoint_is_not_a_smooth_source_certificate(branch):
    x = s.Symbol("lambert_branchpoint")
    source = _source(x - 1/s.E, branch)
    for enclose in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            enclose(source, x, [(0, 0)])


@pytest.mark.parametrize("hidden", ["zero_product", "zero_power"])
def test_original_argument_poles_survive_lambert_zero_special_values(hidden):
    x = s.Symbol("lambert_original_argument")
    argument = (s.Mul(0, s.Pow(x, -1, evaluate=False), evaluate=False)
                if hidden == "zero_product" else
                s.Pow(s.log(x, evaluate=False), 0, evaluate=False))
    source = _source(argument, 0)
    for construct in (s.compile_residual_system, s.compile_objective):
        program = construct(source, x)
        with pytest.raises(ValueError):
            program.evaluate(0)
    with pytest.raises(ValueError):
        s.compile_numeric(x, source)(0)
    for enclose in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            enclose(source, x, [(0, 0)])


@pytest.mark.parametrize("branch", [0, -1])
def test_residual_and_objective_replay_keep_fixed_branches_and_reject_jet_forgery(branch):
    x = s.Symbol("lambert_transport", real=True)
    source = _source(x, branch)
    residual = s.compile_residual_system(source, x)
    encoded = residual.to_bytes()
    for restored in (
        s.load_residual_system(encoded, residuals=source, args=x),
        s.load_portable_residual_system(residual.to_portable_bytes(), residuals=source, args=x),
    ):
        assert restored.evaluate(-0.25) == residual.evaluate(-0.25)
    other = _source(x, -1 if branch == 0 else 0)
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=other, args=x)
    prefix = b"LambertW" if branch == 0 else b"LambertWMinusOne"
    first, second = b'"'+prefix+b'Derivative"', b'"'+prefix+b'SecondDerivative"'
    assert first in encoded
    with pytest.raises(ValueError):
        s.load_residual_system(encoded.replace(first, second), residuals=source, args=x)
    objective = s.compile_objective(source, x)
    encoded = objective.to_bytes()
    assert s.load_objective(encoded, objective=source, args=x).evaluate(-0.25) == objective.evaluate(-0.25)
    assert second in encoded
    with pytest.raises(ValueError):
        s.load_objective(encoded.replace(second, first), objective=source, args=x)


def test_same_printed_typed_axes_are_not_branch_indices_or_shared_coordinates():
    plain = s.Symbol("lambert_typed")
    negative = s.Symbol("lambert_typed", negative=True)
    dummy = s.Dummy("lambert_typed")
    axes = (negative, dummy, plain)
    source = _source(plain, 0) + _source(negative, -1) + dummy
    residual = s.compile_residual_system(source, axes)
    assert all(left is right for left, right in zip(residual.variables, axes))
    assert residual.evaluate(-0.25, 3, 1)[0] == pytest.approx(
        [0.5671432904097838 - 2.15329236411035 + 3],
    )


@pytest.mark.parametrize("kind", ["positive", "lower", "fraction", "float", "variable", "held"])
@pytest.mark.parametrize("form", ["shell", "native"])
def test_dynamic_nonliteral_and_nonreal_branch_indices_refuse(kind, form):
    x = s.Symbol("lambert_bad_branch")
    branch = {
        "positive": s.Integer(1), "lower": s.Integer(-2),
        "fraction": s.Rational(-1, 2), "float": s.Float(-1),
        "variable": s.Symbol("lambert_index", integer=True),
        "held": s.Add(-1, 0, evaluate=False),
    }[kind]
    source = _source(x, branch)
    if form == "native":
        source = _native.py_function("LambertW", _native_expr(x), _native_expr(branch))
    for operation in (
        lambda: s.compile_numeric(x, source),
        lambda: s.compile_residual_system(source, x),
        lambda: s.compile_objective(source, x),
        lambda: s.enclose_objective(source, x, [(1, 1)]),
    ):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()


@pytest.mark.parametrize("arity", [0, 3])
def test_malformed_native_function_arity_is_rejected_before_compilation(arity):
    x = s.Symbol("lambert_bad_arity")
    source = _native.py_function("LambertW", *([_native_expr(x)]*arity))
    with pytest.raises(ValueError, match="LambertW"):
        s.compile_numeric(x, source)


@pytest.mark.parametrize("name", [
    "LambertW", "lambertw", "_fsym_lambert_w0_d1", "_fsym_lambert_w0_d2",
    "_fsym_lambert_wm1_d1", "_fsym_lambert_wm1_d2",
])
def test_opaque_names_cannot_acquire_lambert_or_private_jet_semantics(name):
    x = s.Symbol("lambert_opaque")
    source = s.Function(name)(x)
    for operation in (
        lambda: s.compile_numeric(x, source),
        lambda: s.compile_residual_system(source, x),
        lambda: s.enclose_objective(source, x, [(1, 1)]),
    ):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()
    if name != "LambertW":
        # Private derivative tags are authorized only inside independently
        # admitted native derivative programs, never by raw public syntax.
        raw = _native.py_function(name, _native_expr(x))
        with pytest.raises(NotImplementedError):
            s.compile_numeric(x, raw)
        with pytest.raises(NotImplementedError):
            s.enclose_objective(raw, x, [(1, 1)])


def test_subclasses_and_changed_hooks_never_enter_closed_lambert_semantics(monkeypatch):
    x = s.Symbol("lambert_hooks")

    class CustomLambert(s.LambertW):
        pass

    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, CustomLambert(x, evaluate=False))
    original = _source(x, -1)
    raw = _native_expr(original)
    effects = []

    def forbidden(*args):
        effects.append(args)
        raise AssertionError("closed admission cannot call a modified function hook")

    monkeypatch.setattr(s.LambertW, "_eval_derivative", forbidden, raising=False)
    for source in (original, raw):
        for operation in (
            lambda: s.compile_numeric(x, source),
            lambda: s.enclose_objective(source, x, [(s.Rational(-1, 4), s.Rational(-1, 4))]),
        ):
            with pytest.raises(NotImplementedError):
                operation()
    assert effects == []


@pytest.mark.parametrize("raw", [False, True])
def test_math_backend_does_not_silently_choose_a_lambert_implementation(raw):
    x = s.Symbol("lambert_math")
    source = _source(x, -1)
    if raw:
        source = _native_expr(source)
    with pytest.raises(NotImplementedError):
        s.lambdify(x, source, modules="math")


def test_lambert_evaluation_and_certification_keep_budget_and_cancellation_failures():
    x = s.Symbol("lambert_budget")
    source = _source(x, -1)
    compiled = s.compile_numeric(x, source)
    objective = s.compile_objective(source, x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        compiled(-0.25, budget=s.EvaluationBudget(max_steps=1))
    with pytest.raises(s.EvaluationBudgetExceeded):
        objective.evaluate(-0.25, budget=s.EvaluationBudget(max_steps=1))
    with pytest.raises(s.EvaluationBudgetExceeded):
        s.enclose_objective(source, x, [(s.Rational(-1, 4), s.Rational(-1, 4))],
                            budget=s.EvaluationBudget(max_steps=1))
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    for operation in (
        lambda: compiled(-0.25, budget=cancelled),
        lambda: s.compile_numeric(x, source, budget=cancelled),
        lambda: objective.evaluate(-0.25, budget=cancelled),
        lambda: s.enclose_objective(source, x, [(s.Rational(-1, 4), s.Rational(-1, 4))],
                                   budget=cancelled),
    ):
        with pytest.raises(s.EvaluationCancelled):
            operation()
