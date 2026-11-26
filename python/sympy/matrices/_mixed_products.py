"""Bounded products and comparisons without expanding sparse operands."""

from types import WrapperDescriptorType

from . import sparse as _sparse
from .dense import Matrix, ImmutableDenseMatrix, _cells_have_decimal, _matrix_dimensions


def _layout(matrix):
    layout = next((item for item in _sparse._REDUCTION_RESULT_LAYOUTS
                   if item[0] is type(matrix)), None)
    if layout is None:
        raise NotImplementedError("mixed matrix operations require built-in matrix classes")
    return layout


def _interfaces(layout, *, result=False):
    cls, _, slots, hooks = layout
    if (not _sparse._reduction_default_access(cls)
            or any(_sparse._reduction_binding(cls, name) is not slot
                   for name, slot in zip(_sparse._REDUCTION_STORAGE_NAMES, slots))
            or result and any(_sparse._reduction_binding(cls, name) is not original
                              for name, original in hooks)):
        raise NotImplementedError("mixed matrix operations require original matrix interfaces")


def _snapshot(matrix):
    layout = _layout(matrix)
    _interfaces(layout)
    _, base, slots, _ = layout
    native, shape, surface, generation = (slot.__get__(matrix, base) for slot in slots)
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("invalid mixed matrix source snapshot")
    sparse = base is _sparse.SparseMatrix
    if sparse:
        _sparse._sparse_dimensions(*shape)
    else:
        _matrix_dimensions(*shape)
    retained = None
    if surface is not None:
        if type(surface) is not (dict if sparse else tuple):
            raise ValueError("mixed matrix operations require original retained storage")
        _sparse._stored_entries(surface)
        if sparse:
            for coordinate in surface:
                if (type(coordinate) is not tuple or len(coordinate) != 2
                        or any(type(index) is not int for index in coordinate)
                        or not (0 <= coordinate[0] < shape[0]
                                and 0 <= coordinate[1] < shape[1])):
                    raise ValueError("invalid mixed matrix stored coordinate")
            retained = dict(surface)
        elif len(surface) != shape[0]*shape[1]:
            raise ValueError("dense retained cells do not match the matrix shape")
        else:
            retained = surface
    elif native is None:
        raise ValueError("mixed matrix source has no cell storage")
    if native is not None:
        native_type = _sparse._NativeSparseMatrix if sparse else _sparse._SOLVE_NATIVE_DENSE
        members = (_sparse._REDUCTION_NATIVE_MEMBERS if sparse
                   else _sparse._SOLVE_DENSE_MEMBERS)
        namespace = _sparse._REDUCTION_DICT.__get__(native_type)
        if (type(native) is not native_type
                or not _sparse._reduction_default_access(native_type)
                or any(namespace.get(name) is not member for name, member in members.items())):
            raise NotImplementedError("mixed matrix operations require original native interfaces")
        if members["shape"].__get__(native, native_type) != shape:
            raise ValueError("native and retained matrix shapes differ")
    return matrix, layout, native, shape, surface, generation, retained


def _check(snapshots):
    for matrix, layout, native, shape, surface, generation, retained in snapshots:
        _interfaces(layout)
        _, base, slots, _ = layout
        current = tuple(slot.__get__(matrix, base) for slot in slots)
        if (current[0] is not native or current[1] is not shape
                or current[2] is not surface or type(current[3]) is not int
                or current[3] != generation):
            raise RuntimeError("Matrix mutated during mixed matrix operation")
        if type(retained) is dict and (len(surface) != len(retained)
                or any(surface.get(key) is not value for key, value in retained.items())):
            raise RuntimeError("Matrix retained cells mutated during mixed matrix operation")


def _entries(snapshot, *, native_only=False):
    _, layout, native, shape, _, _, retained = snapshot
    sparse = layout[1] is _sparse.SparseMatrix
    if retained is not None:
        return (dict(retained) if sparse else
                {divmod(index, shape[1]): value for index, value in enumerate(retained)})
    entries = (_sparse._REDUCTION_NATIVE_MEMBERS["todok"](native) if sparse else
               {divmod(index, shape[1]): value for index, value in enumerate(
                   _sparse._SOLVE_DENSE_MEMBERS["flat"](native))})
    _sparse._stored_entries(entries)
    return entries if native_only else _sparse._lift_structural_entries(entries)


def _result_layout(snapshots):
    classes = tuple(snapshot[1][0] for snapshot in snapshots)
    cls = (_sparse.ImmutableSparseMatrix if _sparse.ImmutableSparseMatrix in classes else
           ImmutableDenseMatrix if ImmutableDenseMatrix in classes else Matrix)
    layout = next(item for item in _sparse._REDUCTION_RESULT_LAYOUTS if item[0] is cls)
    _interfaces(layout, result=True)
    return layout


def _require_zero_product(value, *, raw, zero_left):
    """Omitted coordinates may disappear only when zero annihilates the cell."""
    native_type = _sparse._REDUCTION_NATIVE_EXPR
    if not raw:
        from ..core._opaque_arithmetic import _admit_source, contains_opaque

        from ..core import Float

        if type(value) is Float:
            # Ordinary Floats override the scalar method, but their original
            # finite IEEE value still has an exact-zero product. Admission
            # validates the class and retained intern before reading it.
            from ..core._decimal_arithmetic import _budget, _capture

            _capture_state, lowered, _ = _capture((value,), _budget())
            if lowered is None:
                raise ValueError("mixed matrix product cannot omit a nonfinite implicit-zero product")
            return
        method = "__rmul__" if zero_left else "__mul__"
        if (_sparse._reduction_binding(type(value), method)
                is not _sparse._reduction_binding(_sparse.Expr, method)):
            raise NotImplementedError(
                "effectful implicit-zero products require a supervised multiplication lane"
            )
        if _cells_have_decimal((value,)):
            _sparse._require_decimal_zero_product(value, zero_left=zero_left)
            return
        if contains_opaque(value):
            _admit_source((value,), check_identity=True)
            return
        value = _sparse._native_expr(value)
    operation = _sparse._REDUCTION_DICT.__get__(native_type).get("__mul__")
    if (type(operation) is not WrapperDescriptorType
            or operation.__objclass__ is not native_type or operation.__name__ != "__mul__"):
        raise NotImplementedError("mixed matrix product requires original native scalar arithmetic")
    zero = _sparse._REDUCTION_NATIVE_ZERO
    product = operation(zero, value) if zero_left else operation(value, zero)
    if _sparse._REDUCTION_NATIVE_EQUAL(product, zero) is not True:
        raise ValueError("mixed matrix product cannot omit a nonzero implicit-zero product")


def _values_meeting_sparse_zeros(shape, coordinates, dense_entries, *, left_sparse):
    """Yield dense cells with a missing sparse factor at their inner index."""
    sparse_axis = 1 if left_sparse else 0
    dense_axis = 1 - sparse_axis
    extent = shape[dense_axis]
    occupancy = {}
    for coordinate in coordinates:
        inner = coordinate[sparse_axis]
        occupancy[inner] = occupancy.get(inner, 0) + 1
    for coordinate, value in dense_entries.items():
        if occupancy.get(coordinate[dense_axis], 0) < extent:
            yield value


def _raw_zero_preflight(snapshots):
    left_sparse = snapshots[0][1][1] is _sparse.SparseMatrix
    sparse, dense = snapshots if left_sparse else snapshots[::-1]
    sparse_entries = _entries(sparse, native_only=True)
    dense_entries = _entries(dense, native_only=True)
    if len(sparse_entries) < sparse[3][0]*sparse[3][1]:
        for value in _values_meeting_sparse_zeros(
            sparse[3], sparse_entries, dense_entries, left_sparse=left_sparse,
        ):
            _require_zero_product(value, raw=True, zero_left=left_sparse)
    # The native product converts the dense operand to sparse too. Its exact
    # zeros can omit products only at their own inner-axis coordinate.
    zero = _sparse._REDUCTION_NATIVE_ZERO
    sparse_axis = 1 if left_sparse else 0
    zero_inner = {
        coordinate[1 - sparse_axis] for coordinate, value in dense_entries.items()
        if _sparse._REDUCTION_NATIVE_EQUAL(value, zero) is True
    }
    if zero_inner:
        for coordinate, value in sparse_entries.items():
            if coordinate[sparse_axis] in zero_inner:
                _require_zero_product(value, raw=True, zero_left=not left_sparse)
    _check(snapshots)


def _native_product(snapshots, layout):
    native_type = _sparse._NativeSparseMatrix
    operation = _sparse._REDUCTION_DICT.__get__(native_type).get("__matmul__")
    if (type(operation) is not WrapperDescriptorType
            or operation.__objclass__ is not native_type or operation.__name__ != "__matmul__"):
        raise NotImplementedError("mixed matrix product requires the original native multiplication")
    operands = tuple(
        source[2] if source[1][1] is _sparse.SparseMatrix else
        _sparse._SOLVE_DENSE_MEMBERS["to_sparse"](source[2])
        for source in snapshots
    )
    _check(snapshots)
    native = operation(*operands)
    _check(snapshots)
    result = _sparse._checked_sparse_lift(native, layout[0])
    _check(snapshots)
    return result


def matmul(left, right):
    """Multiply stored sparse cells into a bounded dense or sparse result."""
    snapshots = (_snapshot(left), _snapshot(right))
    lhs, rhs = snapshots
    if lhs[3][1] != rhs[3][0]:
        raise ValueError(f"Shape mismatch: {lhs[3]} vs {rhs[3]}")
    rows, cols = lhs[3][0], rhs[3][1]
    layout = _result_layout(snapshots)
    if layout[1] is Matrix:
        _matrix_dimensions(rows, cols)

    # Literal rational admission also verifies each retained cell against the
    # native snapshot. Held expressions may not borrow their normalized mirror.
    literal = all(snapshot[6] is None or all(
        any(type(value) is atom for atom in _sparse._REDUCTION_ATOMS)
        for value in (snapshot[6].values() if type(snapshot[6]) is dict else snapshot[6])
    ) for snapshot in snapshots)
    rational = (tuple(_sparse._rational_source_snapshot(
        matrix, dense=snapshot[1][1] is Matrix, context="mixed matrix product",
    ) for matrix, snapshot in zip((left, right), snapshots)) if literal else (None, None))
    raw = all(snapshot[4] is None for snapshot in snapshots)
    if raw or all(source is not None for source in rational):
        if raw and not all(source is not None for source in rational) and rows and cols:
            _raw_zero_preflight(snapshots)
        result = _native_product(snapshots, layout)
        for source in rational:
            if source is not None:
                _sparse._check_rational_source(source[1])
        return result

    left_sparse = lhs[1][1] is _sparse.SparseMatrix
    sparse = lhs if left_sparse else rhs
    stored = (len(sparse[6]) if sparse[6] is not None else
              _sparse._STRUCTURAL_NATIVE_NNZ.__get__(sparse[2], _sparse._NativeSparseMatrix))
    width = cols if left_sparse else rows
    if stored*width > _sparse._MAX_SPARSE_PRODUCTS:
        raise ValueError("mixed matrix multiplication exceeds the term-product limit")
    a, b = _entries(lhs), _entries(rhs)
    _check(snapshots)
    coordinates = a if left_sparse else b
    occupied = {coordinate[0 if left_sparse else 1] for coordinate in coordinates}
    if len(occupied)*width > _sparse._MAX_SPARSE_ENTRIES:
        raise ValueError("mixed matrix multiplication exceeds the stored-entry limit")
    if stored < sparse[3][0]*sparse[3][1] and width:
        for value in _values_meeting_sparse_zeros(
            sparse[3], coordinates, b if left_sparse else a, left_sparse=left_sparse,
        ):
            _require_zero_product(value, raw=False, zero_left=left_sparse)
            _check(snapshots)
    terms = {}

    def checked(value):
        _check(snapshots)
        result = _sparse._sparse_entry(value)
        _check(snapshots)
        return result

    for (row, col), value in coordinates.items():
        for index in range(width):
            coordinate = (row, index) if left_sparse else (index, col)
            product = value*b[col, index] if left_sparse else a[index, row]*value
            terms.setdefault(coordinate, []).append(checked(product))
    from ..core import Add

    entries = {coordinate: checked(Add(*values)) for coordinate, values in terms.items()}
    _interfaces(layout, result=True)
    _check(snapshots)
    if layout[1] is Matrix:
        flat = [_sparse._ZERO]*(rows*cols)
        for (row, col), value in entries.items():
            flat[row*cols + col] = value
        result = layout[0](rows, cols, flat)
    else:
        result = layout[0](rows, cols, entries)
    _check(snapshots)
    return result


def equal(left, right):
    """Compare matching bounded shapes without converting sparse storage."""
    snapshots = (_snapshot(left), _snapshot(right))
    if snapshots[0][3] != snapshots[1][3]:
        return False
    native_only = all(snapshot[4] is None for snapshot in snapshots)
    a, b = (_entries(source, native_only=native_only) for source in snapshots)
    _check(snapshots)
    zero = _sparse._REDUCTION_NATIVE_ZERO if native_only else _sparse._ZERO
    rows, cols = snapshots[0][3]
    for row in range(rows):
        for col in range(cols):
            same = bool(a.get((row, col), zero) == b.get((row, col), zero))
            _check(snapshots)
            if not same:
                return False
    return True
