"""Typed numerical callables without printing or executing expression source.

``lambdify(..., modules="math")`` preserves Python scalar arithmetic types.
``compile_numeric`` and ``modules="frankensympy"`` select the bounded native
finite-real binary64 engine; matrix outputs in that preview API are nested
lists. ``compile_residual_system`` additionally checks symbolic Jacobian proofs.
Binary64 results are approximate, not certified interval enclosures.
"""

from __future__ import annotations

import math
import operator

from ..core import (
    Abs, Add, Basic, Dummy, Expr, Float, Function, Integer, Mul, Pow, Rational, Symbol,
    _ExactDecimalFloat, _FLOAT_INTERN, _admitted_python_float, _bits_to_float,
    _exact_ratio, _exact_surface_types,
    _native, _native_expr, _require_closed_function_class,
    _require_closed_native_function_tag, _require_closed_native_functions,
    _require_symbol, _symbol_binding, _wrap,
)

__all__ = [
    "EvaluationBudget", "EvaluationBudgetExceeded", "EvaluationCancelled", "NumericRootCandidate",
    "compile_numeric", "compile_residual_system", "load_residual_system", "lambdify",
]

EvaluationBudget = _native.EvaluationBudget
EvaluationBudgetExceeded = _native.EvaluationBudgetExceeded
EvaluationCancelled = _native.EvaluationCancelled

_MAX_VARIABLES = 256
_MAX_OUTPUTS = 4096
_MAX_NODES = 65_536
_MAX_DEPTH = 64
_MAX_CONTAINER_DEPTH = 32
_MAX_LITERAL_BITS = 4096
_MAX_INTEGER_RESULT_BITS = 65_536
_MAX_BATCH_ROWS = 4096
_MAX_BATCH_VALUES = 262_144
_MAX_VERIFIED_PROGRAM_BYTES = 16 * 1024 * 1024
_BASIC_VALUE = Basic.__dict__["_value"]
_SURFACE_ARGS = Basic.__dict__["_surface_args"]
_FUNCTION_ARGS = Function.__dict__["_args"]
# Native source tags already emitted by the admitted shell classes. These are
# an IR vocabulary, never names used to recover a Python class. In particular,
# ln/abs/sqrt are not additional source aliases: opaque classes can use them.
_CLOSED_NATIVE_FUNCTIONS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "asinh": "asinh", "acosh": "acosh", "atanh": "atanh",
    "exp": "exp", "log": "log", "Abs": "abs",
    "floor": "floor", "ceiling": "ceiling", "gamma": "gamma",
    "erf": "erf", "erfc": "erfc", "factorial": "factorial",
}


def _check_evaluation_budget(budget):
    if budget is not None:
        if type(budget) is not EvaluationBudget:
            raise TypeError("budget must be an EvaluationBudget or None")
        budget._checkpoint()
    return budget


def _batch_arguments(layout, rows, variable_count, output_count):
    if type(rows) not in (list, tuple):
        raise TypeError("numeric batch must be an exact built-in list or tuple")
    count = len(rows)
    if (count > _MAX_BATCH_ROWS
            or count * max(variable_count, output_count, 1) > _MAX_BATCH_VALUES):
        raise ValueError("numeric batch exceeds the row or total input/output value limit")
    # Index a fixed admitted range: another caller growing a mutable input
    # list cannot turn the shell's conversion into an unbounded iterator.
    result = [_call_arguments(layout, rows[index], native=True) for index in range(count)]
    if len(rows) != count:
        raise ValueError("numeric batch length changed during input conversion")
    return result


class _FunctionTable:
    """An exact-identity map that cannot invoke a changed metaclass hash."""

    __slots__ = ("_entries",)

    def __init__(self, entries):
        self._entries = entries

    def items(self):
        return self._entries

    def __iter__(self):
        return (entry[0] for entry in self._entries)

    def __contains__(self, cls):
        return any(cls is key for key, _ in self._entries)

    def __getitem__(self, cls):
        for key, value in self._entries:
            if cls is key:
                return value
        raise KeyError("unregistered numerical function class")


def _functions():
    # Exact classes, never class names supplied by a user subclass.
    from ..functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan
    from ..functions.elementary.hyperbolic import sinh, cosh, tanh, asinh, acosh, atanh
    from ..functions.elementary.exponential import exp, log
    from ..functions.elementary.integers import floor, ceiling
    from ..functions.special.gamma_functions import gamma
    from ..functions.special.error_functions import erf, erfc
    from ..functions.combinatorial.factorials import factorial

    return _FunctionTable((
        (sin, ("sin", math.sin)), (cos, ("cos", math.cos)), (tan, ("tan", math.tan)),
        (asin, ("asin", math.asin)), (acos, ("acos", math.acos)), (atan, ("atan", math.atan)),
        (sinh, ("sinh", math.sinh)), (cosh, ("cosh", math.cosh)), (tanh, ("tanh", math.tanh)),
        (asinh, ("asinh", math.asinh)), (acosh, ("acosh", math.acosh)),
        (atanh, ("atanh", math.atanh)), (exp, ("exp", math.exp)), (log, ("log", math.log)),
        (Abs, ("abs", abs)), (floor, ("floor", math.floor)), (ceiling, ("ceiling", math.ceil)),
        (gamma, ("gamma", math.gamma)), (erf, ("erf", math.erf)), (erfc, ("erfc", math.erfc)),
        (factorial, ("factorial", math.factorial)),
    ))


def _charge(count, depth, *, container=False):
    count[0] += 1
    if count[0] > _MAX_NODES or depth > (_MAX_CONTAINER_DEPTH if container else _MAX_DEPTH):
        raise ValueError("numeric expression or container exceeds the node/depth limit")


def _argument_layout(args):
    variables, count = [], [0]

    def visit(value, depth):
        _charge(count, depth, container=True)
        if type(value) in (list, tuple):
            if len(value) > _MAX_VARIABLES:
                raise ValueError("numeric argument list exceeds the variable limit")
            return tuple(visit(child, depth + 1) for child in value)
        symbol = _require_symbol(value)
        if len(variables) >= _MAX_VARIABLES:
            raise ValueError("numeric argument list exceeds the variable limit")
        if symbol in variables:
            raise ValueError("duplicate typed numerical variable")
        variables.append(symbol)
        return None

    root = args if type(args) in (list, tuple) else (args,)
    return visit(root, 0), variables


def _call_arguments(layout, supplied, *, native):
    flat = []

    def visit(shape, value):
        if shape is not None:
            if type(value) not in (tuple, list) or len(value) != len(shape):
                raise TypeError("numerical arguments do not match the declared argument structure")
            for child, item in zip(shape, value):
                visit(child, item)
            return
        if native:
            if type(value) is str:
                raise TypeError("native numerical inputs must be real numbers, not strings")
            flat.append(value._as_python_float() if type(value) is _ExactDecimalFloat
                        else _admitted_python_float(value))
        elif type(value) in (bool, int, float, complex):
            if type(value) is int and value.bit_length() > _MAX_LITERAL_BITS:
                raise ValueError("numeric input integer exceeds the bit limit")
            flat.append(value)
        elif type(value) in _exact_surface_types() and isinstance(value, Integer):
            flat.append(value.p)
        elif type(value) in _exact_surface_types() and isinstance(value, Rational):
            flat.append(value.p / value.q)
        elif type(value) in (Float, _ExactDecimalFloat):
            flat.append(value._as_python_float())
        else:
            raise TypeError("math numerical inputs must be built-in numbers or admitted numeric atoms")

    visit(layout, supplied)
    return flat


def _numeric_matrix_snapshot(value):
    """Capture exact matrix cells without entering Python class-name lifting."""
    from ..matrices import Matrix, SparseMatrix

    if type(value) is SparseMatrix:
        matrix = object.__getattribute__(value, "__dict__")["_native"]
        _native.CompiledFunction._validate_matrix_source(matrix)
        shape = matrix.shape
        surface, generation = None, None
    else:
        # Read the owning slots rather than _native: shell-only matrices have
        # real retained cells, but deliberately have no native algebra handle.
        storage_slot = Matrix.__dict__["_native_storage"]
        shape_slot = Matrix.__dict__["_matrix_shape"]
        surface_slot = Matrix.__dict__["_surface"]
        generation_slot = Matrix.__dict__["_mutation_generation"]
        matrix = storage_slot.__get__(value, Matrix)
        shape = shape_slot.__get__(value, Matrix)
        surface = surface_slot.__get__(value, Matrix)
        generation = generation_slot.__get__(value, Matrix)
        if type(generation) is not int or generation < 0:
            raise ValueError("numeric matrix has an invalid mutation generation")
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)):
        raise ValueError("numeric matrix has an invalid retained shape")
    rows, columns = shape
    size = rows * columns
    if max(rows, columns) > _MAX_OUTPUTS or size > _MAX_OUTPUTS:
        raise ValueError("numeric matrix exceeds the output limit")
    if matrix is not None:
        # Borrow the complete native source before indexing can clone cells.
        # A retained surface is not permission to skip admission of a native
        # payload which the matrix also claims to carry.
        _native.CompiledFunction._validate_matrix_source(matrix)
        if matrix.shape != shape:
            raise ValueError("numeric matrix native and retained shapes differ")
    if surface is not None:
        if (type(surface) is not list and type(surface) is not tuple) or len(surface) != size:
            raise ValueError("numeric matrix has an invalid retained source snapshot")
        try:
            entries = tuple(surface[index] for index in range(size))
        except IndexError:
            raise ValueError("numeric matrix changed during its source snapshot") from None
        if len(surface) != size:
            raise ValueError("numeric matrix changed during its source snapshot")
    else:
        if matrix is None:
            raise ValueError("numeric matrix has neither native nor retained cells")
        entries = tuple(matrix[(row, column)] for row in range(rows) for column in range(columns))
    if type(value) is SparseMatrix:
        changed = (object.__getattribute__(value, "__dict__").get("_native") is not matrix
                   or matrix.shape != shape)
    else:
        current_generation = generation_slot.__get__(value, Matrix)
        changed = (storage_slot.__get__(value, Matrix) is not matrix
                   or shape_slot.__get__(value, Matrix) is not shape
                   or surface_slot.__get__(value, Matrix) is not surface
                   or type(current_generation) is not int
                   or current_generation != generation
                   or (matrix is not None and matrix.shape != shape))
    if changed:
        raise ValueError("numeric matrix changed during its source snapshot")
    return rows, columns, entries


def _outputs(expression, convert, *, native):
    from ..matrices import Matrix, ImmutableDenseMatrix, SparseMatrix

    leaves, count = [], [0]

    def visit(value, depth):
        _charge(count, depth, container=True)
        cls = type(value)
        if cls is list or cls is tuple:
            if len(value) > _MAX_OUTPUTS:
                raise ValueError("numeric output container exceeds the output limit")
            return (type(value), tuple(visit(child, depth + 1) for child in value))
        if cls is Matrix or cls is ImmutableDenseMatrix or cls is SparseMatrix:
            if not native:
                raise NotImplementedError("matrix numerical outputs require modules='frankensympy'")
            rows, columns, entries = _numeric_matrix_snapshot(value)
            return (list, tuple(
                (list, tuple(visit(entries[row * columns + column], depth + 2)
                             for column in range(columns)))
                for row in range(rows)
            ))
        if len(leaves) >= _MAX_OUTPUTS:
            raise ValueError("numeric program exceeds the output limit")
        index = len(leaves)
        leaves.append(convert(value))
        return index

    return visit(expression, 0), leaves


def _restore_outputs(shape, values):
    if type(shape) is int:
        return values[shape]
    container, children = shape
    return container(_restore_outputs(child, values) for child in children)


def _literal(value):
    if type(value) is bool:
        return value
    if type(value) in (int, float, complex):
        return value
    if type(value) in (Float, _ExactDecimalFloat):
        return value._as_python_float()
    if type(value) in _exact_surface_types() and isinstance(value, Integer):
        return value.p
    if type(value) in _exact_surface_types() and isinstance(value, Rational):
        return value.p / value.q
    return None


def _constant(value):
    native = _native_expr(value)
    for name, result in (("pi", math.pi), ("E", math.e), ("I", 1j),
                         ("oo", math.inf), ("-oo", -math.inf), ("nan", math.nan)):
        # Fixed constant tokens establish typed constants; expression printers
        # and user-controlled source strings never enter this path.
        if native == _native.Expr(name):
            return result
    raise NotImplementedError("unsupported typed numerical constant")


def _native_numeric_source(source, count, depth, *, inexact=None):
    """Normalize bounded native syntax without reconstructing Python objects.

    The caller has counted the root. Borrowed admission precedes every .args,
    function-name or integer-payload clone; descendants then share the original
    shell count/depth ledger. No public args getter or function registry runs.
    """
    if type(source) is not _native.Expr:
        raise TypeError("native numerical source must be an exact native expression")
    _native.CompiledFunction._validate_source(source)

    def visit(node, level, *, counted=False):
        if not counted:
            _charge(count, level)
        if node.is_number or node.is_symbol:
            return node
        if node.is_add:
            kind, name = "add", None
        elif node.is_mul:
            kind, name = "mul", None
        elif node.is_pow:
            kind, name = "pow", None
        else:
            tag = node.func_name
            if tag == _FLOAT_INTERN:
                children = node.args
                if len(children) != 1 or not children[0].is_integer:
                    raise ValueError("malformed floating literal in numerical source")
                _charge(count, level + 1)
                bits = children[0].exact_numerator()
                if type(bits) is not int or bits < 0 or bits.bit_length() > 64:
                    raise ValueError("malformed floating literal in numerical source")
                value = _bits_to_float(bits)
                if not math.isfinite(value):
                    raise ValueError("native numerical literals must be finite")
                if inexact is not None:
                    inexact[0] = True
                return _native.py_rational(*value.as_integer_ratio())
            name = _CLOSED_NATIVE_FUNCTIONS.get(tag)
            if name is None:
                raise NotImplementedError("custom or opaque objects have no closed numerical lowering")
            _require_closed_native_function_tag(tag)
            kind = "function"
        children = node.args
        if kind == "function" and len(children) != 1:
            raise ValueError("native numerical functions require exactly one argument")
        lowered = [visit(child, level + 1) for child in children]
        if kind == "add":
            return _native.Add(*lowered, evaluate=False).as_expr()
        if kind == "mul":
            return _native.Mul(*lowered, evaluate=False).as_expr()
        if kind == "pow":
            return _native.Pow(*lowered, evaluate=False).as_expr()
        return _native.py_function(name, *lowered)

    return visit(source, depth, counted=True)


def _numeric_source(value, functions, count, depth=0, *, inexact=None):
    _charge(count, depth)
    cls = type(value)
    if inexact is not None and any(cls is builtin for builtin in (float, Float, _ExactDecimalFloat)):
        inexact[0] = True
    if any(cls is builtin for builtin in (int, float, bool, Float, _ExactDecimalFloat)):
        if cls is _ExactDecimalFloat:
            ratio = value._as_python_float().as_integer_ratio()
        else:
            ratio = _exact_ratio(value)
        if ratio is None:
            raise ValueError("native numerical literals must be finite")
        p, q = ratio
        if max(abs(p).bit_length(), q.bit_length()) > _MAX_LITERAL_BITS:
            raise ValueError("numeric literal exceeds the bit limit")
        return _native.py_rational(p, q)
    if cls is _native.Expr:
        return _native_numeric_source(value, count, depth, inexact=inexact)
    function = next((spec for builtin, spec in functions.items() if cls is builtin), None)
    if function is not None:
        _require_closed_function_class(cls)
    if not any(cls is builtin for builtin in _exact_surface_types()) and function is None:
        raise NotImplementedError("custom or opaque objects have no closed numerical lowering")
    if cls is Add or cls is Mul or cls is Pow or function is not None:
        descriptor = cls.__dict__.get("_args") if function is None else _FUNCTION_ARGS
        try:
            arguments = _SURFACE_ARGS.__get__(value, Basic)
        except AttributeError:
            arguments = None
        if arguments is None:
            try:
                arguments = descriptor.__get__(value, cls) if descriptor is not None else None
            except AttributeError:
                arguments = None
        if arguments is None:
            return _native_numeric_source(_BASIC_VALUE.__get__(value, Basic), count, depth, inexact=inexact)
        if type(arguments) is not tuple or len(arguments) > _MAX_NODES:
            raise ValueError("numeric source requires bounded exact stored arguments")
        children = [_numeric_source(arg, functions, count, depth + 1, inexact=inexact)
                    for arg in arguments]
        if cls is Add:
            return _native.Add(*children, evaluate=False).as_expr()
        if cls is Mul:
            return _native.Mul(*children, evaluate=False).as_expr()
        if cls is Pow:
            return _native.Pow(*children, evaluate=False).as_expr()
        return _native.py_function(function[0], *children)
    # Exact numeric atoms and constants already carry typed native payloads.
    # Unsupported operators are refused by the native compiler.
    native = _BASIC_VALUE.__get__(value, Basic)
    if native is None:
        raise NotImplementedError("custom or opaque objects have no closed numerical lowering")
    return _native_numeric_source(native, count, depth, inexact=inexact)


def compile_numeric(args, expression):
    """Compile reusable finite-real binary64 scalar, list, tuple or matrix output.

    Arguments are typed Symbols/Dummies, optionally nested in lists/tuples.
    Invocation inputs are built-in real numbers or admitted numeric atoms.
    Every invocation returns ordinary Python floats; matrices return nested
    lists. Unknown symbols, opaque hooks, unsupported native functions,
    nonfinite intermediates and excessive resources refuse explicitly.

    Pass ``budget=EvaluationBudget(...)`` when calling the result to share
    cumulative native evaluation fuel and buffer limits across calls and
    programs. Failures retain consumed work; cancellation is irreversible.
    ``evaluate_batch(rows, budget=...)`` accepts a list/tuple of positional
    argument lists/tuples and returns all rows together, or raises without
    publishing partial rows. Batches have at most 4096 rows and 262144 flat
    input/output values. Compilation, symbolic proof admission and CPython
    allocation overhead are outside the numerical evaluation ledger. The same
    budget can also meter compiled residual enclosure and root certification.
    """
    layout, variables = _argument_layout(args)
    functions, count = _functions(), [0]
    shape, sources = _outputs(
        expression, lambda value: _numeric_source(value, functions, count), native=True,
    )
    program = _native.CompiledFunction(sources, [_symbol_binding(v) for v in variables])

    def compiled(*values, budget=None):
        _check_evaluation_budget(budget)
        inputs = _call_arguments(layout, values, native=True)
        result = _restore_outputs(shape, program.evaluate(inputs, budget=budget))
        _check_evaluation_budget(budget)
        return result

    def evaluate_batch(rows, *, budget=None):
        _check_evaluation_budget(budget)
        inputs = _batch_arguments(layout, rows, program.variable_count, program.output_count)
        outputs = program.evaluate_batch(inputs, budget=budget)
        result = [_restore_outputs(shape, values) for values in outputs]
        _check_evaluation_budget(budget)
        return result

    compiled.__doc__ = "Typed finite-real binary64 callable; outputs are approximate."
    compiled.native_program = program
    compiled.evaluate_batch = evaluate_batch
    return compiled


class NumericRootCandidate:
    """Immutable approximate result from a pinned numerical consumer.

    ``root`` contains finite binary64 coordinates in ``variables`` order.
    ``residual_norm`` is the re-evaluated infinity norm of the original
    residual program. Neither convergence nor this diagnostic proves a root
    exists, is unique, or satisfies the variables' declared assumptions.
    """

    __slots__ = ("_root", "_variables", "_residual_norm", "_iterations", "_backend")

    def __init__(self, root, variables, residual_norm, iterations, backend):
        if hasattr(self, "_root"):
            raise AttributeError("numerical root candidates are immutable")
        object.__setattr__(self, "_root", tuple(root))
        object.__setattr__(self, "_variables", tuple(variables))
        object.__setattr__(self, "_residual_norm", residual_norm)
        object.__setattr__(self, "_iterations", iterations)
        object.__setattr__(self, "_backend", backend)

    def __setattr__(self, name, value):
        raise AttributeError("numerical root candidates are immutable")

    def __delattr__(self, name):
        raise AttributeError("numerical root candidates are immutable")

    @property
    def root(self):
        return self._root

    @property
    def variables(self):
        """The caller's captured Symbol/Dummy objects, in coordinate order."""
        return self._variables

    @property
    def residual_norm(self):
        return self._residual_norm

    @property
    def iterations(self):
        return self._iterations

    @property
    def backend(self):
        return self._backend

    @property
    def evidence(self):
        return "approximate"

    def __repr__(self):
        return (f"NumericRootCandidate(root={self.root!r}, residual_norm={self.residual_norm!r}, "
                f"iterations={self.iterations!r}, backend={self.backend!r})")


class _ResidualCallable:
    """Python argument layout around an immutable checked native system."""

    __slots__ = ("_layout", "_native_system", "_variables", "_has_inexact_source", "_assumptions")

    def __init__(self, layout, native, variables, inexact, assumptions):
        if hasattr(self, "_native_system"):
            raise AttributeError("compiled residual systems are immutable")
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_native_system", native)
        object.__setattr__(self, "_variables", tuple(variables))
        object.__setattr__(self, "_has_inexact_source", inexact)
        object.__setattr__(self, "_assumptions", assumptions)

    def __setattr__(self, name, value):
        raise AttributeError("compiled residual systems are immutable")

    @property
    def native_system(self):
        return self._native_system

    @property
    def variables(self):
        """Original argument objects, in the flat order used by box coordinates."""
        return self._variables

    def to_bytes(self):
        """Return a bounded, versioned bundle for :func:`load_residual_system`.

        The immutable bytes retain exact native residuals, typed variables,
        symbolic derivative proofs and checked derivative bytecode. Loading
        replays proof admission and requires the caller's expected residuals
        and argument declarations. Python objects, argument layout, root
        receipts and evaluation budgets are not stored. Encoding and replay
        are bounded admission work outside `EvaluationBudget`.
        """
        return self.native_system.to_bytes()

    def evaluate(self, *values, budget=None):
        """Return residuals and Jacobian rows, sharing any supplied evaluation budget."""
        _check_evaluation_budget(budget)
        return self.native_system.evaluate(
            _call_arguments(self._layout, values, native=True), budget=budget,
        )

    def residuals(self, *values, budget=None):
        _check_evaluation_budget(budget)
        return self.native_system.residuals(
            _call_arguments(self._layout, values, native=True), budget=budget,
        )

    def jacobian(self, *values, budget=None):
        _check_evaluation_budget(budget)
        return self.native_system.jacobian(
            _call_arguments(self._layout, values, native=True), budget=budget,
        )

    def evaluate_batch(self, rows, *, budget=None):
        """Return complete residual/Jacobian pairs for a bounded batch atomically.

        Each row is a list/tuple of positional arguments with the declared
        nesting. A supplied budget is cumulative across every row and both
        derivative stages; a later failure publishes no earlier row.
        """
        _check_evaluation_budget(budget)
        size = self.native_system.variable_count
        outputs = self.native_system.residual_count * (size + 1)
        inputs = _batch_arguments(self._layout, rows, size, outputs)
        return self.native_system.evaluate_batch(inputs, budget=budget)

    def solve_candidate(self, initial, *, backend="frankennumpy", tol=None,
                        maxiter=None, max_step_norm=None, budget=None):
        """Generate an approximate root with a real pinned numerical consumer.

        Systems must be square with 1..16 variables. ``initial`` is a finite
        real scalar for one variable or a flat vector in ``variables`` order,
        including programs whose ordinary evaluation arguments are nested.

        ``frankennumpy`` uses the checked symbolic Jacobian and the pinned
        dense linear solver. Its defaults are ``tol=1e-12``, ``maxiter=64`` and
        ``max_step_norm=1e6``; the iteration limit may be 0..1024. A zero limit
        only checks the initial point. ``frankenscipy`` runs the pinned fsolve
        with its finite-difference Jacobian, at most 200 iterations and a
        residual tolerance of 1e-10. It refuses every explicit non-None tol,
        maxiter or max_step_norm override, because that donor API has no such
        controls. Neither backend silently falls back to another algorithm.

        The immutable ``NumericRootCandidate`` remains approximate. Use
        ``solve_certified`` for an independently checked local-root claim.
        One supplied EvaluationBudget spans native input buffers, all source
        evaluations and conservative donor work allowances. Cancellation is
        cooperative between synchronous donor kernels, without a wall-time
        promise. Python containers and compilation remain outside that ledger.
        """
        from ..solvers.numerical import _candidate_initial, _candidate_options

        _check_evaluation_budget(budget)
        tol, maxiter, max_step_norm = _candidate_options(
            backend, tol, maxiter, max_step_norm, allow_python=False,
        )
        values = _candidate_initial(self, initial)
        root, norm, iterations = self.native_system.solve_candidate(
            values, backend=backend, tol=tol, maxiter=maxiter,
            max_step_norm=max_step_norm, budget=budget,
        )
        result = NumericRootCandidate(root, self.variables, norm, iterations, backend)
        _check_evaluation_budget(budget)
        return result

    def enclose(self, box, *, precision=32, budget=None):
        """Independently enclose residuals and their Jacobian on a closed real box.

        Bounds are exact int, Rational or fractions.Fraction values; float
        bounds and floating source literals refuse. ``box`` lists lower/upper
        pairs in flat variable order, even for nested ordinary arguments.
        Supported declared assumptions must hold throughout the box. The core
        checks the original source's C1 domain independently of compiled and
        symbolic derivative instructions. Up to 16 variables, 64 residuals and
        precision 8..128 digits are supported. No root claim is made.
        A supplied ``EvaluationBudget`` counts native interval work and exact
        input preparation cumulatively and permits cooperative cancellation.
        """
        from ._certified_numeric import enclose
        return enclose(self, box, precision, budget=budget)

    def certify_root_box(self, box, *, center=None, preconditioner=None, precision=32, budget=None):
        """Prove one unique nonsingular real root in this closed coordinate box.

        Exact input rules match ``enclose``. The system must be square with
        1..16 variables and every coordinate must have positive width. Center
        defaults to the exact midpoint. A manual exact preconditioner is an
        untrusted proposal; otherwise bounded rational elimination proposes an
        inverse of a midpoint Jacobian enclosure. Acceptance depends only on
        independent whole-box C1 checks, strict Krawczyk inclusion and an exact
        contraction bound below one. This makes no claim about other boxes.
        ``budget`` is shared by preconditioner preparation and independent
        interval verification; a rejected witness does not refund work.
        """
        from ._certified_numeric import certify_root_box
        return certify_root_box(self, box, center, preconditioner, precision, budget=budget)

    def certify_parametric_root_box(
            self, unknown_box, parameter_box, *, center=None, preconditioner=None,
            precision=32, budget=None):
        """Certify a unique root for every value in an exact parameter box.

        For m residuals, the first m compiled variables are the unknowns;
        all remaining variables are parameters. At least one of each and at
        most 16 total variables are required. Unknown coordinates have
        positive width; parameter coordinates may be fixed point intervals.
        Both boxes use exact rational bounds and captured typed assumptions.

        An omitted center uses the unknown-box midpoint. An omitted inverse
        preconditioner is proposed by exact rational elimination of the
        unknown Jacobian at that center and the parameter midpoint. The
        independent original-source verifier retains the full parameter box
        in its residuals and derivatives and requires uniform strict
        Krawczyk inclusion and contraction. The returned immutable receipt
        encloses every parameter-dependent root inside unknown_box; it makes
        no claim outside that box or of an exact symbolic solution function.

        One supplied EvaluationBudget, or one bounded private ledger when
        omitted, covers preparation, the inverse proposal, full-family
        verification and final cancellation. Inconclusive verification,
        unsupported domains and operational failures return no receipt.
        """
        from ._parametric_root import certify_parametric_root_box
        return certify_parametric_root_box(
            self, unknown_box, parameter_box, center, preconditioner, precision, budget=budget,
        )

    def isolate_roots_certified(self, box, *, precision=32, max_leaves=512, budget=None):
        """Discover and certify every nonsingular real root in a finite box.

        No initial guess is required. Exact bounds and captured assumptions
        follow enclose; the system must be square with 1..16 variables and
        every coordinate must have positive width. The immutable result
        contains a tuple of RootBoxReceipt objects and an exact root_count.
        Zero roots proves the entire declared domain is root-free.

        Native search proposes rational root boxes and a complete partition.
        The independent verifier checks every root against the original
        source, proves the root boxes distinct, and checks every partition
        leaf is either root-free or contained in an accepted root box.
        Root boxes can cross internal subdivision faces. Singular roots,
        roots on the outer boundary, or unresolved regions prevent a complete
        receipt; a partial collection is never returned as complete.

        Precision is fixed at 8..128 digits and max_leaves is 1..1024. Search
        and independent verification share one supplied EvaluationBudget,
        including failed cells and final cancellation checks. The returned
        root enclosures carry no requested coordinate tolerance, exact
        symbolic-coordinate claim, or assertion outside the declared box.
        """
        from ._root_isolation import isolate_roots_certified
        return isolate_roots_certified(self, box, precision, max_leaves, budget=budget)

    def solve_certified(self, initial, *, tol=None, maxiter=None,
                        backend="python", max_step_norm=None,
                        precision=32, max_attempts=12, budget=None):
        """Search for and independently certify one nonsingular real root.

        ``initial`` is a finite-real scalar for one variable or a flat vector
        in ``variables`` order, even when ordinary arguments are nested. The
        captured exact source, typed symbols and assumptions stay fixed across
        candidate generation and certification. The system must be square
        with 1..16 variables. A bounded numerical candidate only proposes exact
        rational boxes; native domain, inclusion and contraction checks alone
        authorize the returned ``RootBoxReceipt``. Failed searches raise
        ``RootSearchError`` and never return an uncertified candidate.

        ``backend="python"`` preserves the scaled QR/Newton search with
        default ``tol=1e-12`` and ``maxiter=200``. ``frankennumpy`` and
        ``frankenscipy`` select the real pinned consumers with the controls
        documented by ``solve_candidate``. None selects each backend's
        defaults; the Python backend rejects ``max_step_norm`` and SciPy
        rejects all non-None solver-control overrides. No backend fallback
        occurs. Source eligibility and captured assumptions are checked before
        candidate work, and every backend uses the same root-box verifier.

        Certification tries at most 1..24 boxes (default 12), increasing
        precision from the requested 8..128 digits up to 128. This is a local
        root claim, without global completeness. See ``nsolve_certified`` for
        the search contract. A supplied ``EvaluationBudget`` spans candidate
        iterations, rejected boxes and precision escalation without resetting
        its counters. Donor cancellation remains cooperative between kernels.
        """
        from ..solvers.numerical import _solve_compiled_certified
        return _solve_compiled_certified(
            self, initial, tol=tol, maxiter=maxiter, precision=precision,
            max_attempts=max_attempts, backend=backend,
            max_step_norm=max_step_norm, budget=budget,
        )


def compile_residual_system(residuals, args):
    """Compile a residual system with independently checked symbolic derivatives.

    Residual containers flatten in row-major order. Derivative proofs and
    bytecode provenance are checked once; numerical values remain binary64
    approximations. Unsupported derivatives and functions refuse at compile time.
    Separate ``enclose`` and ``certify_root_box`` methods use the original exact
    source and independent interval arithmetic. Floating source literals remain
    usable for approximate evaluation but cannot enter those certified methods.
    Approximate ``evaluate``, ``residuals``, ``jacobian`` and ``evaluate_batch``
    accept the same reusable ``EvaluationBudget`` as ``compile_numeric``.
    Compilation and symbolic proof admission are outside that ledger; the
    certified enclosure and root-box methods accept the same budget. Python
    object/result-container allocation remains outside native-buffer accounting.
    """
    prepared = _residual_source(residuals, args)
    layout, variables, bindings, assumptions, sources, inexact = prepared
    native = _native.VerifiedResidualSystem(sources, bindings)
    return _ResidualCallable(layout, native, variables, inexact, assumptions)


def _residual_source(residuals, args):
    """Capture caller-owned layout, typed declarations and exact source once."""
    layout, variables = _argument_layout(args)
    from ._certified_numeric import bound_assumptions
    bindings = [_symbol_binding(v) for v in variables]
    assumptions = bound_assumptions(variables, bindings)
    functions, count, inexact = _functions(), [0], [False]
    _, sources = _outputs(
        residuals, lambda value: _numeric_source(value, functions, count, inexact=inexact), native=True,
    )
    return layout, variables, bindings, assumptions, sources, inexact[0]


def load_residual_system(data, *, residuals, args):
    """Replay a saved residual program and bind it to explicitly expected source.

    ``data`` must be exact built-in ``bytes`` of at most 16 MiB, as returned
    by a compiled residual program's ``to_bytes()``. The versioned native
    decoder independently rechecks retained symbolic derivative proofs,
    omitted zero derivatives, sparsity and bytecode before publication.
    Corrupt data, incompatible schemas and mismatched source raise ValueError.

    The caller must supply both ``residuals`` and ``args``. Their exact
    lowered source and ordered typed symbols must match the bundle, including
    held operations and assumption identities. No Python symbols are created
    from serialized names. Recreate declared Symbols explicitly when loading in a
    fresh process; Dummy identities must already match and are never remapped.
    Argument nesting is taken from the supplied ``args``.

    Assumptions and the floating-source guard are recaptured from the caller's
    objects. Floating residual literals remain approximate-only even if their
    native rational representation matches the file. An explicitly supplied
    exact Rational source is treated as that exact source. No serialized
    boolean can grant proof acceptance or enable root certification.

    The result supports the same evaluation, batching, enclosure and certified
    root search methods as ``compile_residual_system``. Encoding, decoding
    and symbolic proof admission are outside ``EvaluationBudget``; later
    numerical calls can share an ordinary caller budget. This restores a
    checked program, with no stored root receipt or running execution state.
    """
    if type(data) is not bytes:
        raise TypeError("verified program data must be exact built-in bytes")
    if len(data) > _MAX_VERIFIED_PROGRAM_BYTES:
        raise ValueError("verified program data exceeds the 16 MiB byte limit")
    prepared = _residual_source(residuals, args)
    layout, variables, bindings, assumptions, sources, inexact = prepared
    native = _native.VerifiedResidualSystem.from_bytes(
        data, residuals=sources, variables=bindings,
    )
    return _ResidualCallable(layout, native, variables, inexact, assumptions)


def _math_program(value, positions, functions, count, program, depth=0):
    _charge(count, depth)
    cls = type(value)
    if cls is _native.Expr:
        _require_closed_native_functions(value)
        value = _wrap(value)
        cls = type(value)
    if cls in functions:
        _require_closed_function_class(cls)
    if any(cls is builtin for builtin in _exact_surface_types()) or cls in functions:
        try:
            native = _BASIC_VALUE.__get__(value, Basic)
        except AttributeError:
            native = None
        if native is not None:
            _require_closed_native_functions(native)
    literal = _literal(value)
    if literal is not None:
        if type(literal) is int and literal.bit_length() > _MAX_LITERAL_BITS:
            raise ValueError("numeric literal exceeds the bit limit")
        program.append(("constant", literal))
        return
    if cls not in _exact_surface_types() and cls not in functions:
        raise NotImplementedError("custom or opaque objects require an explicit numerical callback contract")
    if cls in (Symbol, Dummy):
        if value not in positions:
            raise ValueError("numerical expression contains an unbound typed symbol")
        program.append(("variable", positions[value]))
        return
    if cls is Expr and not value.args:
        program.append(("constant", _constant(value)))
        return
    if cls not in (Add, Mul, Pow) and cls not in functions:
        raise NotImplementedError("unsupported typed expression for the math numerical backend")
    for child in value.args:
        _math_program(child, positions, functions, count, program, depth + 1)
    if cls is Add:
        program.append(("add", len(value.args)))
    elif cls is Mul:
        program.append(("multiply", len(value.args)))
    elif cls is Pow:
        program.append(("power", 2))
    else:
        program.append(("function", functions[cls][1], len(value.args)))


def _evaluate_math(program, inputs):
    stack = []
    for instruction in program:
        kind = instruction[0]
        if kind == "constant":
            stack.append(instruction[1])
        elif kind == "variable":
            stack.append(inputs[instruction[1]])
        else:
            arity = instruction[-1]
            values = stack[-arity:] if arity else []
            if arity:
                del stack[-arity:]
            if kind == "function":
                if instruction[1] is math.factorial and values[0] > _MAX_LITERAL_BITS:
                    raise ValueError("numeric factorial exceeds the integer work limit")
                stack.append(instruction[1](*values))
            elif kind == "power":
                base, exponent = values
                if (type(base) in (int, bool) and type(exponent) in (int, bool)
                        and exponent >= 0 and abs(base) > 1
                        and (abs(base).bit_length() - 1) * exponent > _MAX_INTEGER_RESULT_BITS):
                    raise ValueError("numeric power exceeds the integer result limit")
                stack.append(operator.pow(*values))
            else:
                result = 0 if kind == "add" else 1
                operation = operator.add if kind == "add" else operator.mul
                for value in values:
                    if (kind == "multiply" and type(result) is int and type(value) is int
                            and result and value
                            and result.bit_length() + value.bit_length() > _MAX_INTEGER_RESULT_BITS):
                        raise ValueError("numeric product exceeds the integer result limit")
                    result = operation(result, value)
                stack.append(result)
            if type(stack[-1]) is int and stack[-1].bit_length() > _MAX_INTEGER_RESULT_BITS:
                raise ValueError("numeric result exceeds the integer bit limit")
    return stack[0]


def lambdify(args, expression, modules="math"):
    """Build a typed callable for ``math`` or the explicit native preview backend.

    The math backend preserves Python scalar arithmetic, including integer
    results and the usual math exceptions. It supports scalar/list/tuple output.
    The ``frankensympy`` backend is :func:`compile_numeric` and always uses
    finite real binary64 arithmetic. Module dictionaries, arbitrary callbacks,
    and other numerical backends require separate contracts and are refused.
    """
    if type(modules) in (list, tuple):
        if len(modules) != 1:
            raise NotImplementedError("exactly one supported numerical backend is required")
        modules = modules[0]
    if modules is math:
        modules = "math"
    if modules is None:
        modules = "math"
    if type(modules) is not str or modules not in ("math", "frankensympy"):
        raise NotImplementedError("supported numerical backends are 'math' and 'frankensympy'")
    if modules == "frankensympy":
        return compile_numeric(args, expression)
    layout, variables = _argument_layout(args)
    positions = {symbol: index for index, symbol in enumerate(variables)}
    functions, count = _functions(), [0]

    def compile_one(value):
        program = []
        _math_program(value, positions, functions, count, program)
        return tuple(program)

    shape, programs = _outputs(expression, compile_one, native=False)

    def evaluate(*values):
        inputs = _call_arguments(layout, values, native=False)
        outputs = [_evaluate_math(program, inputs) for program in programs]
        return _restore_outputs(shape, outputs)

    evaluate.__doc__ = "Typed math callable; no expression source code is executed."
    return evaluate
