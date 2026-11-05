"""The analytic SciPy donor consumes the captured checked Jacobian."""

import pytest
import sympy as s


BACKEND = "frankenscipy_analytic"


@pytest.mark.parametrize("entrypoint", ["compiled", "native", "certified"])
def test_analytic_backend_consumes_singular_derivatives_of_finite_original_residuals(entrypoint):
    x = s.Symbol("analytic_donor_singular")
    program = s.compile_residual_system([s.sqrt(x)-1], x)
    # The original residual is finite here. Only consuming its symbolic
    # derivative exposes the singularity, so a residual-only donor is not enough.
    assert program.residuals(0) == [-1]
    assert program.jacobian_pattern == ((0, 0),)
    if entrypoint == "certified":
        with pytest.raises(s.RootSearchError) as failed:
            program.solve_certified(0, backend=BACKEND)
        assert failed.value.stage == "candidate"
        assert isinstance(failed.value.__cause__, ValueError)
    else:
        callable_ = program if entrypoint == "compiled" else program.native_system
        initial = 0 if entrypoint == "compiled" else [0]
        with pytest.raises(ValueError):
            callable_.solve_candidate(initial, backend=BACKEND)
    # A valid derivative point remains usable after the refused request.
    candidate = program.solve_candidate(1, backend=BACKEND)
    assert candidate.root == (1.0,) and candidate.residual_norm == 0
    assert candidate.backend == BACKEND and candidate.evidence == "approximate"


def test_off_diagonal_sparse_derivatives_drive_the_same_snapshot_and_restored_certificate():
    x, y = s.symbols("analytic_donor_snapshot_x analytic_donor_snapshot_y")
    source = s.SparseMatrix([3*y-6, 2*x-2])
    captured_source = source.as_immutable()
    layout = [(x, y)]
    program = s.compile_residual_system(source, layout)
    encoded = program.to_bytes()
    source[0, 0], source[1, 0] = 999, 999
    layout[0] = (y, x)
    restored = s.load_residual_system(encoded, residuals=captured_source, args=[(x, y)])
    for captured in (program, restored):
        assert captured.variables[0] is x and captured.variables[1] is y
        assert captured.jacobian_pattern == ((0, 1), (1, 0))
        assert captured.evaluate_sparse((0, 0)) == ([-6, -2], [3, 2])
        candidate = captured.solve_candidate([0, 0], backend=BACKEND)
        assert candidate.root == (1.0, 2.0)
        assert candidate.residual_norm == 0 and candidate.iterations > 0
        assert captured.residuals(candidate.root) == [0, 0]
        receipt = captured.solve_certified([0, 0], backend=BACKEND)
        assert type(receipt) is s.RootBoxReceipt
        assert receipt.root_enclosure == ((1, 1), (2, 2))
        assert receipt.residuals == (3*y-6, 2*x-2)
        assert receipt.contraction_bound == 0
    assert program.to_bytes() == restored.to_bytes() == encoded


@pytest.mark.parametrize("option,value", [("tol", 1e-10), ("maxiter", 200), ("max_step_norm", 1e6)])
def test_analytic_fixed_controls_refuse_before_conversion_candidate_or_certificate_work(option, value):
    x = s.Symbol("analytic_donor_controls")
    program = s.compile_residual_system([x-1], x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("invalid controls must refuse before initial conversion")

    initial = Initial()
    budget = s.EvaluationBudget(max_steps=1000)
    options = dict(backend=BACKEND, budget=budget)
    options[option] = value
    for call in (
        lambda: program.solve_candidate(initial, **options),
        lambda: program.native_system.solve_candidate([initial], **options),
        lambda: program.solve_certified(initial, **options),
        lambda: s.nsolve_certified(x-1, x, initial, **options),
    ):
        with pytest.raises(ValueError, match="fixed solver controls"):
            call()
    assert effects == [] and budget.remaining_steps == 1000
