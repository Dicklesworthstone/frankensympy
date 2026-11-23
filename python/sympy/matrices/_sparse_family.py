"""Publish checked sparse least-squares families with retained scalar owners."""

from ..core import (
    Add, Atom, AtomicExpr, Basic, ComplexInfinity, Dummy, Expr, Float, Function, Half, Integer,
    Mul, NegativeOne, Number, One, Pow, Rational, Symbol, Zero,
    _charge_nested_if_active, _native_expr, _require_closed_function_class, _wrap,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from . import sparse as _s
from .dense import _least_squares_stored_children

_MAX_NODES = 4096
_MAX_PRODUCTS = 1_000_000
_MAX_RENDER_NODES = 131_072
_ONE = One()
_NATIVE_ONE = _s._REDUCTION_VALUE_SLOT.__get__(_ONE, Basic)
_CLASSES = (Basic, Atom, AtomicExpr, Expr, Number, Integer, Rational, Float,
            Zero, One, NegativeOne, Half, ComplexInfinity, Symbol, Dummy, Add, Mul, Pow)
_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_ARITHMETIC = ("__add__", "__radd__", "__mul__", "__rmul__", "__neg__")
_OWNERS = tuple(dict.fromkeys(owner for cls in _CLASSES for owner in _MRO.__get__(cls)))
_RECORDS = tuple((owner, _owner_record(owner), _MRO.__get__(owner),
                  tuple(_implementation(_DICT.__get__(owner).get(name))
                        for name in _ARITHMETIC)) for owner in _OWNERS)
_NATIVE_CONSTRUCTION = tuple((name, _s._reduction_binding(_s._NativeSparseMatrix, name))
                             for name in ("__new__", "__init__"))


def _original_scalar_interfaces():
    # In-place code replacement is a semantic change too. Do not call a
    # supplied arithmetic/constructor/accessor override while rendering W.
    for owner, record, parents, arithmetic in _RECORDS:
        current = _MRO.__get__(owner)
        if (len(current) != len(parents)
                or any(a is not b for a, b in zip(current, parents))
                or not _owner_matches(owner, record)):
            raise NotImplementedError("least-squares parameters require original scalar interfaces")
        for name, expected in zip(_ARITHMETIC, arithmetic):
            tokens = _implementation(_DICT.__get__(owner).get(name))
            if len(tokens) != len(expected) or any(a is not b for a, b in zip(tokens, expected)):
                raise NotImplementedError("least-squares parameters require original scalar arithmetic")


def _parameter_forest(entries):
    """Admit all retained owners before reading any native scalar payload."""
    _original_scalar_interfaces()
    if len(entries) > _MAX_NODES:
        raise ValueError("least-squares parameters exceed their bounded expression envelope")
    pending = [(entry, 0) for entry in entries]
    count = 0
    while pending:
        entry, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > 64:
            raise ValueError("least-squares parameters exceed their bounded expression envelope")
        kind = type(entry)
        if any(base is Function for base in _MRO.__get__(kind)):
            _require_closed_function_class(kind)
            if any(_s._reduction_binding(kind, name) is not _s._reduction_binding(Expr, name)
                   for name in _ARITHMETIC):
                raise NotImplementedError("least-squares parameters require original scalar arithmetic")
        elif not any(kind is cls for cls in _CLASSES):
            raise NotImplementedError("least-squares parameters require built-in scalar owners")
        children = _least_squares_stored_children(entry)
        if children:
            if len(children) > _MAX_NODES - count - len(pending):
                raise ValueError("least-squares parameters exceed their bounded expression envelope")
            pending.extend((child, depth + 1) for child in children)
    # This existing guard is independent of the transform catalog: it retains
    # exact scalar slots, typed symbol declarations and raw argument tuples.
    # Its native record checker is also shared by the Mellin boundary.
    from ..integrals.transforms import _FourierSource

    return _FourierSource(*entries)


class _ParameterMatrix:
    """A bounded sparse view of W, bound to the original public storage."""

    def __init__(self, matrix, shape, *, session=None):
        accepted = (_s.Matrix, _s.ImmutableDenseMatrix, _s.SparseMatrix, _s.ImmutableSparseMatrix)
        if not any(type(matrix) is cls for cls in accepted):
            raise TypeError("arbitrary_matrix must be a built-in Matrix")
        layout = next(item for item in _s._REDUCTION_RESULT_LAYOUTS if item[0] is type(matrix))
        _, base, slots, _ = layout
        if (not _s._reduction_default_access(type(matrix))
                or any(_s._reduction_binding(type(matrix), name) is not slot
                       for name, slot in zip(_s._REDUCTION_STORAGE_NAMES, slots))):
            raise NotImplementedError("least-squares parameters require original matrix accessors")
        native, actual_shape, surface, generation = (slot.__get__(matrix, base) for slot in slots)
        if (type(actual_shape) is not tuple or len(actual_shape) != 2
                or any(type(size) is not int or size < 0 for size in actual_shape)
                or type(generation) is not int or generation < 0):
            raise ValueError("arbitrary_matrix has invalid retained storage")
        if actual_shape != shape:
            raise ValueError("arbitrary_matrix must have shape (A.cols, B.cols)")
        dense = base is _s.Matrix
        members = _s._SOLVE_DENSE_MEMBERS if dense else _s._REDUCTION_NATIVE_MEMBERS
        native_type = _s._SOLVE_NATIVE_DENSE if dense else _s._NativeSparseMatrix
        namespace = _DICT.__get__(native_type)
        if (not _s._reduction_default_access(native_type)
                or any(namespace.get(name) is not member for name, member in members.items())):
            raise NotImplementedError("least-squares parameters require original native interfaces")
        if native is None:
            raise NotImplementedError("opaque least-squares parameters require a supervised override lane")
        if type(native) is not native_type or members["shape"].__get__(native, native_type) != shape:
            raise ValueError("arbitrary_matrix native and retained shapes differ")
        count = shape[0]*shape[1] if dense else members["nnz"].__get__(native, native_type)
        if surface is not None and type(surface) is not (tuple if dense else dict):
            raise ValueError("arbitrary_matrix has invalid retained cells")
        if count > _MAX_NODES or surface is not None and len(surface) > _MAX_NODES:
            raise ValueError("least-squares parameters exceed their bounded expression envelope")
        if dense and surface is not None and len(surface) != count:
            raise ValueError("arbitrary_matrix retained cells do not match its shape")
        self.matrix, self.base, self.slots = matrix, base, slots
        self.state = (native, actual_shape, surface, generation)
        self.retained = None if surface is None else (tuple(surface) if dense else dict(surface))
        if surface is None:
            entries = None
        elif dense:
            entries = {divmod(index, shape[1]): entry for index, entry in enumerate(surface)}
        else:
            entries = dict(surface)
        for coordinate in entries or ():
            if (type(coordinate) is not tuple or len(coordinate) != 2
                    or any(type(index) is not int for index in coordinate)
                    or not (0 <= coordinate[0] < shape[0] and 0 <= coordinate[1] < shape[1])):
                raise ValueError("arbitrary_matrix has an invalid retained coordinate")
        self.forest = None if entries is None else _parameter_forest(tuple(entries.values()))
        # Admission borrows original roots before todok/flat/to_sparse can copy
        # an arbitrarily deep or tall expression. Check every supplied domain
        # even if A has full column rank and the projector will erase W.
        validate = (members["validate_sparse_least_squares_parameters"] if dense else
                    members["validate_least_squares_parameters"])
        sizes = validate(native, None if self.forest is None else self.forest.records(),
                         session=session)
        native = members["to_sparse"](native) if dense else native
        raw = _s._REDUCTION_NATIVE_MEMBERS["todok"](native)
        if len(sizes) != (count if dense else len(raw)):
            raise RuntimeError("native parameter admission returned invalid source sizes")
        self.sizes = ({divmod(index, shape[1]): size for index, size in enumerate(sizes)}
                      if dense else dict(zip(sorted(raw), sizes)))
        if entries is None:
            # Native-only cells still need the closed-function admission used
            # by sparse construction; a string tag cannot invent a Python owner.
            for value in raw.values():
                _s._admit_native_cell(value)
            entries = {coordinate: _wrap(value) for coordinate, value in raw.items()}
            self.forest = _parameter_forest(tuple(entries.values()))
        for coordinate, entry in entries.items():
            if _s._REDUCTION_NATIVE_EQUAL(
                    _native_expr(entry), raw.get(coordinate, _s._REDUCTION_NATIVE_ZERO)) is not True:
                raise RuntimeError("arbitrary_matrix retained cells differ from the native snapshot")
        if any(coordinate not in entries for coordinate in raw):
            raise RuntimeError("arbitrary_matrix retained cells differ from the native snapshot")
        self.entries = {coordinate: entry for coordinate, entry in entries.items() if entry is not _s._ZERO}
        self.unchanged()

    def unchanged(self):
        native, shape, surface, generation = self.state
        current = tuple(slot.__get__(self.matrix, self.base) for slot in self.slots)
        if (current[0] is not native or current[1] is not shape or current[2] is not surface
                or type(current[3]) is not int or current[3] != generation):
            raise RuntimeError("arbitrary_matrix mutated during sparse linear algebra")
        if self.retained is not None:
            changed = (any(a is not b for a, b in zip(surface, self.retained))
                       if self.base is _s.Matrix else
                       any(surface.get(key) is not entry for key, entry in self.retained.items()))
            if len(surface) != len(self.retained) or changed:
                raise RuntimeError("arbitrary_matrix retained cells mutated during sparse linear algebra")
        self.forest.unchanged()
        _original_scalar_interfaces()


def pinv_solve(matrix, rhs, arbitrary_matrix=None):
    """Render X0 + N*W only after original-source complete-family acceptance."""
    if not any(type(matrix) is cls for cls in (_s.SparseMatrix, _s.ImmutableSparseMatrix)):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    accepted = (_s.Matrix, _s.ImmutableDenseMatrix, _s.SparseMatrix, _s.ImmutableSparseMatrix)
    if not any(type(rhs) is cls for cls in accepted):
        raise TypeError("pinv_solve requires a built-in Matrix right-hand side")
    dense_rhs = type(rhs) is _s.Matrix or type(rhs) is _s.ImmutableDenseMatrix
    left = _s._rational_source_snapshot(matrix, context="sparse least-squares family")
    right = _s._rational_source_snapshot(rhs, dense=dense_rhs, context="sparse least-squares family")
    if left is None or right is None:
        raise ValueError("sparse least-squares family supports exact rational entries only")
    coefficients, source = left
    rhs_native, rhs_source = right
    rows, unknowns = source[4]
    rhs_rows, columns = rhs_source[4]
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")
    if max(rows, unknowns, columns) > 8192:
        raise ValueError("sparse least-squares dimensions exceed 8192")
    _original_scalar_interfaces()
    supplied = None if arbitrary_matrix is None else _ParameterMatrix(arbitrary_matrix, (unknowns, columns))
    kinds = (type(matrix), type(rhs), type(arbitrary_matrix))
    result_type = (_s.ImmutableSparseMatrix if _s.ImmutableSparseMatrix in kinds else
                   _s.ImmutableDenseMatrix if _s.ImmutableDenseMatrix in kinds else _s.Matrix)
    if result_type is not _s.ImmutableSparseMatrix:
        _s._matrix_dimensions(unknowns, columns)
    if supplied is None and unknowns*columns > _MAX_NODES:
        raise ValueError("least-squares fresh parameter count exceeds 4096")
    if dense_rhs:
        rhs_native = _s._SOLVE_DENSE_MEMBERS["to_sparse"](rhs_native)
    _s._check_rational_source(source)
    _s._check_rational_source(rhs_source)
    minimum, projector = _s._REDUCTION_NATIVE_MEMBERS["least_squares_family"](coefficients, rhs_native)

    def unchanged():
        _s._check_rational_source(source)
        _s._check_rational_source(rhs_source)

    return _render_family(minimum, projector, (unknowns, columns), result_type, supplied, unchanged)


def _render_family(minimum, projector, shape, result_type, supplied, unchanged):
    """Render an accepted family while retaining every original parameter owner.

    Dense Q(i) and sparse QQ generators share this bounded publication path.
    The matrices supplied here are already checked; this layer preserves the
    public parameter forest and rechecks sources before returning the result.
    """
    unknowns, columns = shape
    unchanged()
    _original_scalar_interfaces()
    _charge_nested_if_active(unknowns * max(columns, 1) + 1)
    shape_slot = _s._REDUCTION_NATIVE_MEMBERS["shape"]
    if (shape_slot.__get__(minimum, _s._NativeSparseMatrix) != (unknowns, columns)
            or shape_slot.__get__(projector, _s._NativeSparseMatrix) != (unknowns, unknowns)):
        raise RuntimeError("native least-squares family has invalid output shapes")
    if supplied is None:
        # SymPy 1.14.0 constructs a (B.cols, A.cols) w-grid then transposes it.
        parameters = [Dummy(f"w{row}_{column}")
                      for row in range(unknowns) for column in range(columns)]
        entries = {(row, column): parameters[column*unknowns + row]
                   for row in range(unknowns) for column in range(columns)}
        forest = _parameter_forest(tuple(entries.values()))
    else:
        entries, forest = supplied.entries, supplied.forest
    by_row = {}
    for (row, column), value in entries.items():
        by_row.setdefault(row, []).append((column, value))
    raw_projector = _s._REDUCTION_NATIVE_MEMBERS["todok"](projector)
    products = sum(len(by_row.get(inner, ())) for _, inner in raw_projector)
    if products > _MAX_PRODUCTS:
        raise ValueError("least-squares parameter projection exceeds the sparse product limit")
    raw_minimum = _s._REDUCTION_NATIVE_MEMBERS["todok"](minimum)
    sizes = {} if supplied is None else supplied.sizes
    output_sizes = {coordinate: 1 for coordinate in raw_minimum}
    render_nodes = len(output_sizes)
    # Charge cumulative symbolic copying, including each growing partial sum,
    # before the first arithmetic operation. A product count alone would allow
    # repeatedly copying a deep parameter forest into an unbounded output.
    for (row, inner) in raw_projector:
        for column, _ in by_row.get(inner, ()):
            coordinate = (row, column)
            size = output_sizes.get(coordinate, 0) + sizes.get((inner, column), 1) + 3
            render_nodes += size
            if render_nodes > _MAX_RENDER_NODES:
                raise ValueError("least-squares parameter projection exceeds the rendered expression limit")
            output_sizes[coordinate] = size
    if render_nodes > _MAX_RENDER_NODES:
        raise ValueError("least-squares family exceeds the rendered expression limit")
    # Include the entire bounded output forest in the active request rather
    # than restarting its accounting after the native checker has returned.
    _charge_nested_if_active(render_nodes + products + 1)
    unchanged()
    output = {coordinate: _wrap(value) for coordinate, value in raw_minimum.items()}
    for (row, inner), raw_coefficient in raw_projector.items():
        coefficient = _wrap(raw_coefficient)
        identity = _s._REDUCTION_NATIVE_EQUAL(raw_coefficient, _NATIVE_ONE) is True
        for column, value in by_row.get(inner, ()):
            _charge_nested_if_active()
            coordinate = (row, column)
            term = value if identity else coefficient*value
            previous = output.get(coordinate)
            output[coordinate] = term if previous is None else previous + term
            if len(output) > _s._MAX_SPARSE_ENTRIES:
                raise ValueError("least-squares family exceeds the stored-entry limit")
    output = {coordinate: value for coordinate, value in output.items() if value is not _s._ZERO}
    if any(_s._reduction_binding(_s._NativeSparseMatrix, name) is not original
           for name, original in _NATIVE_CONSTRUCTION):
        raise NotImplementedError("least-squares results require original native construction interfaces")
    native_output = _s._NativeSparseMatrix(unknowns, columns, {
        coordinate: _native_expr(value) for coordinate, value in output.items()
    })
    # Retain both the consistent closed native snapshot and original scalar
    # objects. A subsequent native operation must not misclassify this closed
    # family as opaque, and a pure parameter keeps its exact public owner.
    result = _s._checked_sparse_lift(native_output, result_type)
    _, base, slots, _ = next(item for item in _s._REDUCTION_RESULT_LAYOUTS if item[0] is result_type)
    if base is _s.Matrix:
        surface = tuple(output.get(divmod(index, columns), _s._ZERO)
                        for index in range(unknowns*columns))
    else:
        surface = output
    slots[2].__set__(result, surface)
    unchanged()
    forest.unchanged()
    if supplied is not None:
        supplied.unchanged()
    _original_scalar_interfaces()
    _charge_nested_if_active()
    return result
