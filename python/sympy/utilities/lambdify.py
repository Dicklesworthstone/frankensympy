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
    Abs, Add, Basic, Dummy, Expr, Float, Integer, Mul, Pow, Rational, Symbol,
    _ExactDecimalFloat, _admitted_python_float, _exact_ratio, _exact_surface_types,
    _native, _native_expr, _require_symbol, _symbol_binding, _wrap,
)

__all__ = ["compile_numeric", "compile_residual_system", "lambdify"]

_MAX_VARIABLES = 256
_MAX_OUTPUTS = 4096
_MAX_NODES = 65_536
_MAX_DEPTH = 64
_MAX_CONTAINER_DEPTH = 32
_MAX_LITERAL_BITS = 4096
_MAX_INTEGER_RESULT_BITS = 65_536


def _functions():
    # Exact classes, never class names supplied by a user subclass.
    from ..functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan
    from ..functions.elementary.hyperbolic import sinh, cosh, tanh, asinh, acosh, atanh
    from ..functions.elementary.exponential import exp, log
    from ..functions.elementary.integers import floor, ceiling
    from ..functions.special.gamma_functions import gamma
    from ..functions.special.error_functions import erf, erfc
    from ..functions.combinatorial.factorials import factorial

    return {
        sin: ("sin", math.sin), cos: ("cos", math.cos), tan: ("tan", math.tan),
        asin: ("asin", math.asin), acos: ("acos", math.acos), atan: ("atan", math.atan),
        sinh: ("sinh", math.sinh), cosh: ("cosh", math.cosh), tanh: ("tanh", math.tanh),
        asinh: ("asinh", math.asinh), acosh: ("acosh", math.acosh),
        atanh: ("atanh", math.atanh), exp: ("exp", math.exp), log: ("log", math.log),
        Abs: ("abs", abs), floor: ("floor", math.floor), ceiling: ("ceiling", math.ceil),
        gamma: ("gamma", math.gamma), erf: ("erf", math.erf), erfc: ("erfc", math.erfc),
        factorial: ("factorial", math.factorial),
    }


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


def _outputs(expression, convert, *, native):
    from ..matrices import Matrix, ImmutableDenseMatrix, SparseMatrix

    leaves, count = [], [0]

    def visit(value, depth):
        _charge(count, depth, container=True)
        if type(value) in (list, tuple):
            if len(value) > _MAX_OUTPUTS:
                raise ValueError("numeric output container exceeds the output limit")
            return (type(value), tuple(visit(child, depth + 1) for child in value))
        if type(value) in (Matrix, ImmutableDenseMatrix, SparseMatrix):
            if not native:
                raise NotImplementedError("matrix numerical outputs require modules='frankensympy'")
            rows, columns = value.rows, value.cols
            if max(rows, columns) > _MAX_OUTPUTS or rows * columns > _MAX_OUTPUTS:
                raise ValueError("numeric matrix exceeds the output limit")
            # Compile an immutable snapshot; subsequent shell mutation cannot
            # alter the already checked program.
            return (list, tuple(
                (list, tuple(visit(value[row, column], depth + 2) for column in range(columns)))
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


def _numeric_source(value, functions, count, depth=0):
    _charge(count, depth)
    cls = type(value)
    if cls in (int, float, bool, Float, _ExactDecimalFloat):
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
        value = _wrap(value)
        cls = type(value)
    if cls not in _exact_surface_types() and cls not in functions:
        raise NotImplementedError("custom or opaque objects have no closed numerical lowering")
    if cls in (Symbol, Dummy):
        return _native_expr(value)
    if cls in (Add, Mul, Pow) or cls in functions:
        children = [_numeric_source(arg, functions, count, depth + 1) for arg in value.args]
        if cls is Add:
            return _native.Add(*children, evaluate=False).as_expr()
        if cls is Mul:
            return _native.Mul(*children, evaluate=False).as_expr()
        if cls is Pow:
            return _native.Pow(*children, evaluate=False).as_expr()
        return _native.py_function(functions[cls][0], *children)
    # Exact numeric atoms and constants already carry typed native payloads.
    # Unsupported operators are refused by the native compiler.
    return _native_expr(value)


def compile_numeric(args, expression):
    """Compile reusable finite-real binary64 scalar, list, tuple or matrix output.

    Arguments are typed Symbols/Dummies, optionally nested in lists/tuples.
    Invocation inputs are built-in real numbers or admitted numeric atoms.
    Every invocation returns ordinary Python floats; matrices return nested
    lists. Unknown symbols, opaque hooks, unsupported native functions,
    nonfinite intermediates and excessive resources refuse explicitly.
    """
    layout, variables = _argument_layout(args)
    functions, count = _functions(), [0]
    shape, sources = _outputs(
        expression, lambda value: _numeric_source(value, functions, count), native=True,
    )
    program = _native.CompiledFunction(sources, [_symbol_binding(v) for v in variables])

    def compiled(*values):
        inputs = _call_arguments(layout, values, native=True)
        return _restore_outputs(shape, program.evaluate(inputs))

    compiled.__doc__ = "Typed finite-real binary64 callable; outputs are approximate."
    compiled.native_program = program
    return compiled


class _ResidualCallable:
    """Python argument layout around an immutable checked native system."""

    def __init__(self, layout, native):
        self._layout = layout
        self.native_system = native

    def evaluate(self, *values):
        """Return the approximate residual vector and nested Jacobian rows."""
        return self.native_system.evaluate(_call_arguments(self._layout, values, native=True))

    def residuals(self, *values):
        return self.native_system.residuals(_call_arguments(self._layout, values, native=True))

    def jacobian(self, *values):
        return self.native_system.jacobian(_call_arguments(self._layout, values, native=True))


def compile_residual_system(residuals, args):
    """Compile a residual system with independently checked symbolic derivatives.

    Residual containers flatten in row-major order. Derivative proofs and
    bytecode provenance are checked once; numerical values remain binary64
    approximations. Unsupported derivatives and functions refuse at compile time.
    """
    layout, variables = _argument_layout(args)
    functions, count = _functions(), [0]
    _, sources = _outputs(
        residuals, lambda value: _numeric_source(value, functions, count), native=True,
    )
    native = _native.VerifiedResidualSystem(sources, [_symbol_binding(v) for v in variables])
    return _ResidualCallable(layout, native)


def _math_program(value, positions, functions, count, program, depth=0):
    _charge(count, depth)
    cls = type(value)
    if cls is _native.Expr:
        value = _wrap(value)
        cls = type(value)
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
