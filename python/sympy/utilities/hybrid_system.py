"""Explicit residual/Jacobian assertions with supervised numerical evaluation.

These programs retain caller-supplied derivatives as approximate numerical
inputs. They do not create symbolic derivative proofs or root certificates.
"""

from ..core import Basic, Eq, Relational, _native
from .hybrid import (
    _BASIC_VALUE, _HybridCallable, _callback_bindings, _compile_graph, _evaluation_budget,
    _evaluation_session, _native_operator, _output_snapshot, _session_ready, _snapshot_source,
)
from .lambdify import (
    _MAX_OUTPUTS, _argument_layout, _call_arguments, _charge, _check_evaluation_budget,
    _functions, _outputs,
)

__all__ = ["compile_hybrid_residual_system"]
_RELATIONAL_SIDES = Relational.__dict__["_relational_sides"]


def _is_sequence(value):
    return type(value) is list or type(value) is tuple


def _arithmetic_equation_side(source, count, depth):
    """Admit native-only arithmetic without guessing any function's class."""
    _charge(count, depth)
    if source.is_number or source.is_symbol:
        return source
    if not (source.is_add or source.is_mul or source.is_pow):
        raise NotImplementedError(
            "native-only equations require retained Python sides for function expressions"
        )
    for child in source.args:
        _arithmetic_equation_side(child, count, depth + 1)
    return source


def _snapshot_equation(value, bindings, functions, count):
    """Lower an exact Eq from captured sides, without surface reconstruction."""
    _charge(count, 0)
    try:
        sides = _RELATIONAL_SIDES.__get__(value, Relational)
    except AttributeError:
        original = _BASIC_VALUE.__get__(value, Basic)
        if type(original) is not _native.Expr:
            raise TypeError("a native-only equation requires an exact native expression")
        # Check the full borrowed tree before .args clones either side. In
        # this fallback even an intrinsic-looking function name is ambiguous:
        # no original class survives to distinguish a custom alias from it.
        _native.CompiledFunction._validate_source(original)
        if original.func_name not in ("Equality", "Eq"):
            raise ValueError("native-only equation payload must be an equality")
        sides = original.args
        if len(sides) != 2:
            raise ValueError("native-only equality must have exactly two sides")
        left = ("closed", None, (), _arithmetic_equation_side(sides[0], count, 1))
        right = ("closed", None, (), _arithmetic_equation_side(sides[1], count, 2))
    else:
        if type(sides) is not tuple or len(sides) != 2:
            raise ValueError("equation source requires an exact captured pair of sides")
        left = _snapshot_source(sides[0], bindings, functions, count, 1,
                                native_admission=_arithmetic_equation_side)
        right = _snapshot_source(sides[1], bindings, functions, count, 2,
                                 native_admission=_arithmetic_equation_side)

    def combine(kind, children):
        if all(child[3] is not None for child in children):
            return ("closed", None, (), _native_operator(kind, None, [child[3] for child in children]))
        return (kind, None, children, None)

    _charge(count, 1)
    negative_one = _snapshot_source(-1, bindings, functions, count, 2)
    # Retain lhs-before-rhs occurrence order and both original domains. The
    # checked snapshots precede every newly constructed native Add/Mul node.
    return combine("add", (left, combine("mul", (negative_one, right))))


def _source_vectors(residuals, jacobian, columns):
    """Snapshot both inputs before inspecting their admitted vector shapes."""
    from ..matrices import ImmutableDenseMatrix, ImmutableSparseMatrix, Matrix, SparseMatrix
    from ..solvers.numerical import _MAX_JACOBIAN_ENTRIES, _MAX_SYSTEM_RESIDUALS

    cls = type(residuals)
    residual_is_matrix = any(cls is kind for kind in (
        Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix,
    ))
    captured_residuals, captured_jacobian = _output_snapshot((residuals, jacobian))
    if residual_is_matrix:
        if len(captured_residuals) == 1:
            values = list(captured_residuals[0])
        elif captured_residuals and all(len(row) == 1 for row in captured_residuals):
            values = [row[0] for row in captured_residuals]
        else:
            raise ValueError("hybrid residual matrix must be a nonempty row or column vector")
    elif _is_sequence(captured_residuals):
        values = list(captured_residuals)
    else:
        values = [captured_residuals]
    rows = len(values)
    if not 1 <= rows <= _MAX_SYSTEM_RESIDUALS:
        raise ValueError("hybrid residuals must contain 1 through 256 entries")
    if any(_is_sequence(value) for value in values):
        raise TypeError("hybrid residuals must be a scalar or a flat vector")
    if rows * columns > _MAX_JACOBIAN_ENTRIES or rows * (columns + 1) > _MAX_OUTPUTS:
        raise ValueError("hybrid residual/Jacobian exceeds the dense entry or combined output limit")
    if not _is_sequence(captured_jacobian) or len(captured_jacobian) != rows:
        raise TypeError("hybrid Jacobian must have one explicit row per residual")
    if any(not _is_sequence(row) or len(row) != columns for row in captured_jacobian):
        raise TypeError("hybrid Jacobian rows must match the flat variable order")
    if any(_is_sequence(value) for row in captured_jacobian for value in row):
        raise TypeError("hybrid Jacobian entries must be scalar expressions")
    return values, [list(row) for row in captured_jacobian]


def _initial_point(initial, columns):
    from ..solvers.numerical import _finite_real

    if _is_sequence(initial):
        if len(initial) != columns:
            raise TypeError("hybrid solve initial vector must match the number of variables")
        try:
            captured = tuple(initial[index] for index in range(columns))
        except IndexError:
            raise ValueError("hybrid solve initial vector changed during its snapshot") from None
        if len(initial) != columns:
            raise ValueError("hybrid solve initial vector changed during its snapshot")
    elif columns == 1:
        captured = (initial,)
    else:
        raise TypeError("hybrid solve initial values must be an exact list or tuple")
    return [_finite_real(value, "initial value") for value in captured]


class _HybridResidualSystem:
    """One immutable residual/Jacobian graph with explicit user provenance."""

    __slots__ = ("_program", "_residual_program", "_rows", "_columns")

    def __init__(self, program, residual_program, rows, columns):
        if hasattr(self, "_program"):
            raise AttributeError("hybrid residual systems are immutable")
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_residual_program", residual_program)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_columns", columns)

    def __setattr__(self, name, value):
        raise AttributeError("hybrid residual systems are immutable")

    def __delattr__(self, name):
        raise AttributeError("hybrid residual systems are immutable")

    def __reduce_ex__(self, protocol):
        raise TypeError("hybrid residual systems have no portable or pickle transport")

    @property
    def variables(self):
        """Captured argument objects, in flat numerical coordinate order."""
        return self._program.variables

    @property
    def variable_count(self):
        return self._columns

    @property
    def residual_count(self):
        return self._rows

    @property
    def callback_slots(self):
        return self._program.callback_slots

    @property
    def operation_count(self):
        return self._program.operation_count

    @property
    def evidence(self):
        return "approximate_with_user_callbacks"

    @property
    def jacobian_evidence(self):
        return "user_supplied"

    def evaluate(self, *values, budget=None, session=None):
        """Return ``(residual_values, Jacobian_rows)`` in captured source order."""
        return self._program(*values, budget=budget, session=session)

    __call__ = evaluate

    def residuals(self, *values, budget=None, session=None):
        """Evaluate the residual prefix without invoking derivative callbacks."""
        return self._residual_program(*values, budget=budget, session=session)

    def jacobian(self, *values, budget=None, session=None):
        """Check every original residual before returning the asserted Jacobian.

        This evaluates residual callback occurrences too: an asserted derivative
        must not erase the original source's domain checks or observable effects.
        """
        _check_evaluation_budget(budget)
        with _evaluation_session(session) as active, _evaluation_budget(budget) as budget:
            inputs = _call_arguments(self._program._layout, values, native=True)
            result = self._program._evaluate(inputs, budget, active)[1]
            _session_ready(active)
            _check_evaluation_budget(budget)
            return result

    def evaluate_batch(self, rows, *, budget=None, session=None):
        """Evaluate complete residual/Jacobian pairs under one ledger/session.

        Every row is admitted before effects start. A later failure publishes
        no partial result list; already completed Python effects remain visible.
        """
        return self._program.evaluate_batch(rows, budget=budget, session=session)

    def solve(self, initial, *, tol=1e-12, maxiter=200, budget=None, session=None):
        """Return an approximate column Matrix using bounded pivoted-QR Newton.

        The explicit Jacobian is a caller assertion, never verified derivative
        evidence. Square and overdetermined systems use the same scaling, rank,
        residual/step tolerances and fixed work bounds as ordinary ``nsolve``.
        A zero binary64 residual may terminate before derivative evaluation.
        Initial values are a flat exact list/tuple in ``variables`` order,
        or a scalar for one variable, even if ordinary arguments are nested.

        One callback session and EvaluationBudget span all iterations, native
        fragments, logical dense work and backtracking. An omitted ledger is
        inherited from an enclosing hybrid request. Each evaluation executes
        its callback occurrences afresh; no purity or state consistency is
        assumed. Only a native domain failure before any callback occurrence
        in that trial may backtrack. A failure after an effect, a callback
        exception, cancellation or resource exhaustion propagates immediately.
        Completed effects cannot be rolled back. No root claim is granted.
        """
        from ..core import Float
        from ..matrices import Matrix
        from ..solvers.numerical import _iterate_system_functions, _newton_options

        _check_evaluation_budget(budget)
        with _evaluation_session(session) as active, _evaluation_budget(budget) as budget:
            tolerance = _newton_options(tol, maxiter)
            if self._rows < self._columns:
                raise ValueError("hybrid solve needs at least as many equations as variables")
            values = _initial_point(initial, self._columns)

            def residuals(point, *, budget=None):
                return self._residual_program._evaluate(point, budget, active)

            def jacobian(point, *, budget=None):
                return self._program._evaluate(point, budget, active)[1]

            def trial_residuals(point, *, budget=None):
                return self._residual_program._evaluate(
                    point, budget, active, reject_native_domain=True,
                )

            def charge_work(units):
                _session_ready(active)
                active.charge_nested_steps(units)

            root = _iterate_system_functions(
                values, tolerance, maxiter, rows=self._rows, columns=self._columns,
                residuals=residuals, jacobian=jacobian, trial_residuals=trial_residuals,
                budget=budget, charge_work=charge_work,
            )
            _session_ready(active)
            _check_evaluation_budget(budget)
            result = Matrix([Float(value) for value in root])
            _session_ready(active)
            _check_evaluation_budget(budget)
            return result


def compile_hybrid_residual_system(residuals, args, *, jacobian, callbacks):
    """Compile explicit residual and Jacobian expressions with callback slots.

    ``residuals`` is a scalar, flat list/tuple, or row/column Matrix. An ``Eq``
    uses its originally captured Python sides, preserving custom class identity.
    Native-only equations without those sides admit arithmetic/atoms only;
    function-bearing sides refuse instead of recovering a class by name.
    The same rule applies to native-only subtrees of retained sides: use held
    Add/Mul/Pow forms to preserve function operand classes through arithmetic.
    ``args`` uses the ordinary numerical argument layout, including nested lists/tuples.
    ``jacobian`` is a Matrix or exact list/tuple of rows, with one column per
    flattened argument. It is required: compilation neither invokes derivative
    hooks nor checks that these expressions are derivatives of the residuals.
    A caller may prepare it separately with the existing symbolic APIs.

    ``callbacks`` follows ``compile_numeric_hybrid``: an exact dict from exact
    custom Function classes to callables returning finite built-in real values.
    Mapping, layout, both sources and occurrence order are captured once.
    Residual occurrences precede Jacobian occurrences, and repeated occurrences
    are never deduplicated. Use held forms to retain custom surface identity.

    The immutable result exposes ``evaluate``, ``residuals``, ``jacobian``,
    ``evaluate_batch`` and ``solve``. Numerical calls accept ``budget`` and
    ``session``. Batches and solves retain one owning session and cumulative
    ledger, including nested hybrid requests. Arbitrary Python callback work,
    CPython object allocation and compilation remain outside native accounting;
    callbacks have no forced cancellation guarantee.

    Up to 64 variables, 256 residuals and 4096 dense Jacobian entries are
    admitted, subject also to the existing 4096 combined output/action and
    65536 native instruction caps. Results carry explicit approximate/user
    provenance. This API exposes no checked native system, derivative proof,
    interval/root certificate, or portable/pickle artifact.
    """
    from ..solvers.numerical import _MAX_SYSTEM_VARIABLES

    functions = _functions()
    bindings = _callback_bindings(callbacks, functions)
    layout, variables = _argument_layout(args)
    columns = len(variables)
    if not 1 <= columns <= _MAX_SYSTEM_VARIABLES:
        raise ValueError("hybrid systems require 1 through 64 variables")
    residuals, jacobian = _source_vectors(residuals, jacobian, columns)
    rows = len(residuals)
    count = [0]
    leaf_index = 0

    def snapshot(value):
        nonlocal leaf_index
        is_residual = leaf_index < rows
        leaf_index += 1
        if is_residual and type(value) is Eq:
            return _snapshot_equation(value, bindings, functions, count)
        return _snapshot_source(value, bindings, functions, count)

    shape, roots = _outputs(
        (residuals, jacobian), snapshot, native=True,
    )
    actions, outputs, slots, operations = _compile_graph(roots, variables)
    program = _HybridCallable(layout, shape, variables, actions, outputs, slots, operations)
    # Emission is postorder with every residual root before every Jacobian
    # root. This view shares the same captured instructions and callbacks;
    # it does not recompile or reread a mutable caller source or mapping.
    prefix = actions[:outputs[rows - 1] + 1]
    callback_count = sum(kind == "callback" for kind, _, _ in prefix)
    residual_operations = sum(payload.operation_count for kind, payload, _ in prefix
                              if kind != "callback")
    residual_program = _HybridCallable(
        layout, shape[1][0], variables, prefix, outputs[:rows],
        slots[:callback_count], residual_operations,
    )
    return _HybridResidualSystem(program, residual_program, rows, columns)
