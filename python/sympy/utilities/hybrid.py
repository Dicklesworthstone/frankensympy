"""Finite-real native fragments with explicit supervised Python callback slots.

The captured Python callables are user assertions. They do not enter symbolic
proof replay, certified numerical programs, or portable program transport.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import threading

from ..core import (
    Add, Basic, Float, Function, FunctionClass, Mul, Pow, SupervisedCallbackSession,
    _ExactDecimalFloat, _FLOAT_INTERN, _active_callback_session, _bits_to_float,
    _exact_surface_types, _known_functions, _native, _require_closed_function_class,
    _require_closed_native_function_tag, _symbol_binding, supervised_callback_session,
)
from .lambdify import (
    _MAX_LITERAL_BITS, _MAX_NODES, _MAX_OUTPUTS, _MAX_VARIABLES,
    _argument_layout, _batch_arguments, _call_arguments, _charge,
    _check_evaluation_budget, _functions, _numeric_matrix_snapshot, _numeric_source,
    _outputs, _restore_outputs,
)

__all__ = ["compile_numeric_hybrid"]

_MAX_CALLBACK_CLASSES = 128
_MAX_GRAPH_ACTIONS = 4096
_FUNCTION_ARGS = Function.__dict__["_args"]
_BASIC_VALUE = Basic.__dict__["_value"]
_SURFACE_ARGS = Basic.__dict__["_surface_args"]
_NATIVE_FUNCTION_NAMES = ("sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "ln", "log", "sqrt", "abs")
_request_state = threading.local()


def _callback_bindings(callbacks, functions):
    if type(callbacks) is not dict:
        raise TypeError("callbacks must be an exact built-in dict of Function classes to callables")
    count = len(callbacks)
    if count > _MAX_CALLBACK_CLASSES:
        raise ValueError("hybrid callback mapping exceeds the class limit")
    # Bound the snapshot even if another thread grows the caller's dictionary
    # after the length check. Keys are later matched by identity, never hashing
    # or equality supplied by an extension metaclass.
    items = tuple(itertools.islice(dict.items(callbacks), _MAX_CALLBACK_CLASSES + 1))
    if len(items) != count or len(callbacks) != count:
        raise ValueError("hybrid callback mapping changed during its snapshot")
    for cls, callback in items:
        if type(cls) is not FunctionClass or not issubclass(cls, Function):
            raise TypeError("callback keys must be exact FunctionClass classes deriving from Function")
        if (cls is Function or any(cls is builtin for builtin in functions)
                or any(cls is builtin for builtin in _known_functions.values())):
            raise ValueError("callbacks cannot override registered built-in numerical classes")
        if not callable(callback):
            raise TypeError("each explicit numerical callback must be callable")
    return items


def _output_snapshot(expression):
    """Freeze containers without lifting native-only cells into custom classes."""
    from ..matrices import ImmutableDenseMatrix, ImmutableSparseMatrix, Matrix, SparseMatrix

    count = [0]

    def visit(value, depth):
        _charge(count, depth, container=True)
        cls = type(value)
        if cls is list or cls is tuple:
            size = len(value)
            if size > _MAX_OUTPUTS:
                raise ValueError("numeric output container exceeds the output limit")
            children = tuple(value[index] for index in range(size))
            if len(value) != size:
                raise ValueError("hybrid output container changed during its snapshot")
            result = tuple(visit(child, depth + 1) for child in children)
            return list(result) if cls is list else result
        if any(cls is kind for kind in (Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix)):
            rows, columns, entries = _numeric_matrix_snapshot(value)
            return [[entries[row * columns + col] for col in range(columns)] for row in range(rows)]
        return value

    return visit(expression, 0)


def _native_operator(kind, payload, children):
    if kind == "add":
        return _native.Add(*children, evaluate=False).as_expr()
    if kind == "mul":
        return _native.Mul(*children, evaluate=False).as_expr()
    if kind == "pow":
        return _native.Pow(*children, evaluate=False).as_expr()
    return _native.py_function(payload, *children)


def _snapshot_source(value, bindings, functions, count, depth=0, *, native_admission=None):
    """Capture the actual surface tree before arranging any execution actions."""
    _charge(count, depth)
    cls = type(value)
    intrinsic = next((spec[0] for builtin, spec in functions.items() if cls is builtin), None)
    if intrinsic is not None:
        _require_closed_function_class(cls)
    custom = next(((key, callback) for key, callback in bindings if cls is key), None)
    if custom is not None:
        try:
            children = _FUNCTION_ARGS.__get__(value, Function)
        except AttributeError:
            raise ValueError("custom numerical functions require retained surface arguments") from None
        if type(children) is not tuple or len(children) > _MAX_VARIABLES:
            raise ValueError("custom numerical function arguments must be a bounded exact tuple")
        return ("callback", custom, tuple(
            _snapshot_source(child, bindings, functions, count, depth + 1,
                             native_admission=native_admission) for child in children
        ), None)

    operator = "add" if cls is Add else "mul" if cls is Mul else "pow" if cls is Pow else None
    if operator is not None or intrinsic is not None:
        descriptor = cls.__dict__.get("_args") if operator is not None else _FUNCTION_ARGS
        try:
            children = descriptor.__get__(value, cls) if descriptor is not None else None
        except AttributeError:
            children = None
        if children is None and operator is not None:
            try:
                children = _SURFACE_ARGS.__get__(value, Basic)
            except AttributeError:
                pass
        if children is None:
            # Canonical native storage may have lost opaque Python class
            # identity. Admit closed native tags only; never reconstruct a
            # custom class from a name or call its properties while compiling.
            original = _BASIC_VALUE.__get__(value, Basic)
            _native.CompiledFunction._validate_source(original)
            source = (native_admission(original, count, depth) if native_admission is not None
                      else _native_source(original, bindings, functions, count, depth))
            return ("closed", None, (), source)
        if type(children) is not tuple or len(children) > _MAX_NODES:
            raise ValueError("hybrid source requires bounded exact stored argument tuples")
        children = tuple(
            _snapshot_source(child, bindings, functions, count, depth + 1,
                             native_admission=native_admission) for child in children
        )
        kind = operator or "function"
        if all(child[3] is not None for child in children):
            return ("closed", None, (), _native_operator(kind, intrinsic, [child[3] for child in children]))
        if len(children) > _MAX_VARIABLES:
            raise ValueError("mixed numerical operators support at most 256 children")
        return (kind, intrinsic, children, None)

    if cls is _native.Expr:
        _native.CompiledFunction._validate_source(value)
        source = (native_admission(value, count, depth) if native_admission is not None
                  else _native_source(value, bindings, functions, count, depth))
        return ("closed", None, (), source)
    if (any(cls is builtin for builtin in _exact_surface_types())
            or cls is int or cls is float or cls is bool or cls is _ExactDecimalFloat):
        if (native_admission is not None
                and not any(cls is literal for literal in (int, float, bool, Float, _ExactDecimalFloat))):
            # Check provenance before normalization can turn an ambiguous
            # native Float-intern function into an ordinary exact rational.
            # Actual retained literal objects have their own exact-class lane.
            original = _BASIC_VALUE.__get__(value, Basic)
            _native.CompiledFunction._validate_source(original)
            native_admission(original, count, depth)
        source = _numeric_source(value, functions, count, depth)
        _native.CompiledFunction._validate_source(source)
        return ("closed", None, (), source)
    raise NotImplementedError("custom numerical nodes require an explicit exact-class callback binding")


def _native_source(source, bindings, functions, count, depth):
    """Normalize closed native syntax without entering Python class lifting."""
    _charge(count, depth)
    if source.is_number or source.is_symbol:
        return source
    children = source.args
    if len(children) > _MAX_NODES:
        raise ValueError("hybrid native source exceeds the argument limit")
    if source.is_add:
        kind, name = "add", None
    elif source.is_mul:
        kind, name = "mul", None
    elif source.is_pow:
        kind, name = "pow", None
    else:
        name = source.func_name
        if any(name == type.__getattribute__(cls, "__name__") for cls, _ in bindings):
            raise NotImplementedError(
                "native-only custom function source has no retained exact Python class; use held surface forms"
            )
        if name == _FLOAT_INTERN:
            if len(children) != 1 or not children[0].is_integer:
                raise ValueError("malformed floating literal in hybrid native source")
            # The borrowed root preflight checked literal bit limits before
            # any native child clone or Python integer conversion.
            bits = children[0].exact_numerator()
            if type(bits) is not int or bits < 0 or bits.bit_length() > 64:
                raise ValueError("malformed floating literal in hybrid native source")
            value = _bits_to_float(bits)
            if not math.isfinite(value):
                raise ValueError("hybrid numerical literals must be finite")
            return _native.py_rational(*value.as_integer_ratio())
        if name not in _NATIVE_FUNCTION_NAMES:
            raise NotImplementedError(
                "native-only custom function source has no retained exact Python class; use held surface forms"
            )
        _require_closed_native_function_tag(name)
        kind = "function"
    lowered = [_native_source(child, bindings, functions, count, depth + 1) for child in children]
    return _native_operator(kind, name, lowered)


def _compile_graph(roots, variables):
    actions, callbacks = [], []
    bindings = [_symbol_binding(variable) for variable in variables]
    operations = 0

    def native(source, coordinates):
        nonlocal operations
        program = _native.CompiledFunction([source], coordinates)
        operations += program.operation_count
        if operations > _MAX_NODES:
            raise ValueError("hybrid program exceeds the total native operation limit")
        return program

    def emit(node):
        kind, payload, children, closed = node
        if closed is not None:
            action = ("native_input", native(closed, bindings), ())
        elif kind == "callback":
            arguments = tuple(emit(child) for child in children)
            slot = len(callbacks)
            callbacks.append((slot, payload[0], payload[1]))
            action = ("callback", (slot, payload[1]), arguments)
        else:
            # A literal exponent stays syntax in its native combiner. Turning
            # it into a variable would change Pow into the positive-base-only
            # dynamic power lane and incorrectly reject negative bases.
            literal_power = kind == "pow" and children[1][3] is not None and (
                children[1][3].is_integer or children[1][3].is_rational
            )
            evaluated = children[:1] if literal_power else children
            arguments = tuple(emit(child) for child in evaluated)
            names = [f"__fsym_hybrid_argument_{index}" for index in range(len(arguments))]
            coordinates = [_native.SymbolBinding(name) for name in names]
            placeholders = [_native.py_symbol(name) for name in names]
            if literal_power:
                placeholders.append(children[1][3])
            action = ("native_slots", native(_native_operator(kind, payload, placeholders), coordinates), arguments)
        if len(actions) >= _MAX_GRAPH_ACTIONS:
            raise ValueError("hybrid program exceeds the 4096-action limit")
        result = len(actions)
        actions.append(action)
        return result

    outputs = tuple(emit(root) for root in roots)
    return tuple(actions), outputs, tuple(callbacks), operations


def _session_ready(session):
    session.verify_ownership()
    if session.unknown_effect_tainted:
        raise RuntimeError("callback session has an unknown effect outcome and cannot retry")
    if session.cancelled:
        raise RuntimeError("callback session cancelled before hybrid evaluation")


@contextlib.contextmanager
def _evaluation_session(explicit):
    active = _active_callback_session()
    enclosing = getattr(_request_state, "session", None)
    if active is not None and type(active) is not SupervisedCallbackSession:
        raise TypeError("active callback session must be an exact SupervisedCallbackSession")
    if explicit is not None and type(explicit) is not SupervisedCallbackSession:
        raise TypeError("session must be a SupervisedCallbackSession or None")
    if explicit is not None and active is not None and explicit is not active:
        raise ValueError("explicit callback session conflicts with the active request")
    if enclosing is not None and active is not enclosing:
        raise ValueError("nested hybrid requests must retain their owning callback session")
    with supervised_callback_session(explicit if explicit is not None else active) as session:
        _session_ready(session)
        _request_state.session = session
        try:
            yield session
        finally:
            _request_state.session = enclosing


@contextlib.contextmanager
def _evaluation_budget(explicit):
    enclosing = getattr(_request_state, "budget", None)
    if enclosing is not None and explicit is not None and explicit is not enclosing:
        raise ValueError("nested hybrid requests must retain the active EvaluationBudget")
    budget = enclosing if explicit is None else explicit
    _check_evaluation_budget(budget)
    _request_state.budget = budget
    try:
        yield budget
    finally:
        _request_state.budget = enclosing


def _steps(budget, count):
    if budget is not None:
        budget._charge(count)


def _callback_value(callback, arguments, budget):
    # This complete wrapper runs inside invoke_fresh. Invalid returns and
    # cancellation observed after an effect therefore taint the same session.
    _check_evaluation_budget(budget)
    _steps(budget, len(arguments) + 1)
    value = callback(*arguments)
    _check_evaluation_budget(budget)
    cls = type(value)
    if cls is not int and cls is not float and cls is not bool:
        raise TypeError("numeric callbacks must return an exact built-in int, float or bool")
    bits = value.bit_length() if cls is int else 1
    if bits > _MAX_LITERAL_BITS:
        raise ValueError("numeric callback integer exceeds the 4096-bit limit")
    _steps(budget, max(1, (bits + 31) // 32))
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("numeric callback result must be finite")
    _check_evaluation_budget(budget)
    return converted


class _HybridCallable:
    """Immutable occurrence graph; opaque callbacks are never proof evidence."""

    __slots__ = ("_layout", "_shape", "_variables", "_actions", "_outputs", "_callbacks", "_operations")

    def __init__(self, layout, shape, variables, actions, outputs, callbacks, operations):
        if hasattr(self, "_actions"):
            raise AttributeError("hybrid numerical programs are immutable")
        for name, value in (
            ("_layout", layout), ("_shape", shape), ("_variables", tuple(variables)),
            ("_actions", actions), ("_outputs", outputs), ("_callbacks", callbacks),
            ("_operations", operations),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("hybrid numerical programs are immutable")

    def __delattr__(self, name):
        raise AttributeError("hybrid numerical programs are immutable")

    @property
    def variables(self):
        return self._variables

    @property
    def callback_slots(self):
        """Ordered (occurrence, exact Python class, captured callable) tuples."""
        return self._callbacks

    @property
    def evidence(self):
        return "approximate_with_user_callbacks"

    @property
    def operation_count(self):
        """Native instruction count, excluding opaque Python execution."""
        return self._operations

    @property
    def native_fragment_count(self):
        return sum(kind != "callback" for kind, _, _ in self._actions)

    def __reduce_ex__(self, protocol):
        raise TypeError("hybrid callback programs have no portable or pickle transport")

    def _evaluate(self, inputs, budget, session, *, reject_native_domain=False):
        _session_ready(session)
        _check_evaluation_budget(budget)
        invocations_before = session.invocations if reject_native_domain else None
        if any(not math.isfinite(value) for value in inputs):
            raise ValueError("hybrid numerical inputs must be finite")
        _steps(budget, len(self._actions) + len(inputs) + 1)
        values = [None] * len(self._actions)
        for index, (kind, payload, arguments) in enumerate(self._actions):
            _session_ready(session)
            _check_evaluation_budget(budget)
            _steps(budget, len(arguments) + 1)
            if kind == "callback":
                slot, callback = payload
                supplied = tuple(values[position] for position in arguments)
                values[index] = session.invoke_fresh(
                    "numeric_callback", f"slot-{slot}", len(supplied) + 1,
                    lambda *args: _callback_value(callback, args, budget), supplied,
                )
            else:
                supplied = inputs if kind == "native_input" else [values[position] for position in arguments]
                session.charge_nested_steps(payload.operation_count + len(supplied) + 1)
                try:
                    evaluated = payload.evaluate(supplied, budget=budget)
                except ValueError:
                    # Only a closed native fragment can reject a solver trial
                    # that has executed no callbacks. Admission and callback exceptions
                    # stay outside this handler, including their ValueErrors.
                    if (not reject_native_domain or session.invocations != invocations_before
                            or session.unknown_effect_tainted or session.cancelled):
                        raise
                    _session_ready(session)
                    _check_evaluation_budget(budget)
                    return None
                values[index] = evaluated[0]
        _session_ready(session)
        _check_evaluation_budget(budget)
        _steps(budget, len(self._outputs) + 1)
        result = _restore_outputs(self._shape, [values[index] for index in self._outputs])
        _session_ready(session)
        _check_evaluation_budget(budget)
        return result

    def __call__(self, *values, budget=None, session=None):
        _check_evaluation_budget(budget)
        with _evaluation_session(session) as active, _evaluation_budget(budget) as budget:
            inputs = _call_arguments(self._layout, values, native=True)
            return self._evaluate(inputs, budget, active)

    def evaluate_batch(self, rows, *, budget=None, session=None):
        """Evaluate rows in order under one ledger and callback session.

        A later refusal publishes no partial result list. Already completed
        Python effects remain observable and are never rolled back or retried.
        """
        _check_evaluation_budget(budget)
        with _evaluation_session(session) as active, _evaluation_budget(budget) as budget:
            inputs = _batch_arguments(self._layout, rows, len(self._variables), len(self._outputs))
            if any(not math.isfinite(value) for row in inputs for value in row):
                raise ValueError("hybrid numerical inputs must be finite")
            _steps(budget, len(inputs) + 1)
            result = [self._evaluate(values, budget, active) for values in inputs]
            _session_ready(active)
            _check_evaluation_budget(budget)
            return result


def compile_numeric_hybrid(args, expression, *, callbacks):
    """Compile finite-real native fragments around explicit Python callbacks.

    ``callbacks`` is an exact dict mapping exact custom ``Function`` classes
    to callables. Each occurrence receives the finite float values of its
    arguments and must return an exact built-in int, float or bool that
    converts to finite binary64. Class keys match by identity; built-in
    numerical classes cannot be overridden. The mapping and callable identities
    are captured at compilation. No ``_imp_`` or other ambient hook is selected.

    The result accepts ``program(*values, budget=None, session=None)`` and
    ``evaluate_batch(rows, budget=None, session=None)``. Repeated occurrences
    and repeated invocations run their effects independently in stored source
    order. Native domain failures before a slot prevent that slot from running.
    Custom nodes must retain their original surface children: use explicit
    held Add/Mul/Pow forms when composing them. Native-only custom names cannot
    recover exact Python class identity and refuse. Matrices are snapshotted.

    An explicit or active ``SupervisedCallbackSession`` owns every callback,
    including nested calls; otherwise one session is created per invocation
    or complete batch. Invocation counts and nested native work accumulate;
    each occurrence observes input, depth and post-return wall-time caps.
    A conflicting active session refuses. Failed
    callback result admission taints the session; there is no automatic retry.
    Python execution remains in-process and has no forced cancellation bound.

    ``EvaluationBudget`` charges native work and hybrid dispatch cumulatively.
    Reentrant hybrid calls inherit an omitted budget and refuse a different
    explicit ledger or owning session while their parent request is active.
    It does not measure arbitrary callback internals, compilation or CPython
    object allocation. Callback return checks and final publication observe
    cancellation. Batches preserve completed effects but return no partial rows.

    Inputs retain the ordinary native numeric argument layout and output shape.
    Source/node limits match ``compile_numeric``; the graph has at most 4096
    actions, 65536 native instructions, 128 registered callback classes and
    256 children per mixed operator or callback. Pure and mixed results remain
    approximate, with ``evidence == 'approximate_with_user_callbacks'``. No
    derivative certificate, interval/root claim or portable artifact is granted.
    """
    functions = _functions()
    bindings = _callback_bindings(callbacks, functions)
    layout, variables = _argument_layout(args)
    count = [0]
    shape, roots = _outputs(
        _output_snapshot(expression),
        lambda value: _snapshot_source(value, bindings, functions, count), native=True,
    )
    actions, outputs, slots, operations = _compile_graph(roots, variables)
    return _HybridCallable(layout, shape, variables, actions, outputs, slots, operations)
