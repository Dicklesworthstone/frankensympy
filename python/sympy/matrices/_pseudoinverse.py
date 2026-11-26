"""Source-bound exact linear systems and pseudoinverses over Q(i).

The complex lanes share the original Gaussian scalar admission used by matrix
definiteness. Native results are checked against the original complex sources,
using Hermitian identities rather than a transpose-only normal equation.
The bounded complex lanes and active callback requests support at most 32 rows
and columns. Sparse complex sources enter a bounded dense native view only
after admission. Ordinary literal-rational calls retain their existing bounds.
"""

from ..core import _active_callback_session, _charge_nested_if_active, _wrap
from . import _definiteness as _source


_NATIVE = _source._NATIVE[_source.Matrix]
_NAMES = ("shape", "is_exact_rational", "validate_gaussian_pseudoinverse_source",
          "pinv", "solve_least_squares", "least_squares_family", "to_sparse",
          "solve", "inv", "has_gaussian_rational_syntax", "gaussian_qr",
          "gaussian_rref", "gaussian_nullspace", "gaussian_gauss_jordan_solve",
          "gaussian_lu", "lu", "gaussian_det", "gaussian_ldl", "gaussian_cholesky", "ldl",
          "gaussian_power", "has_polynomial_rhs_syntax", "validate_polynomial_rhs_source")
_METHODS = tuple(_source._DICT.__get__(_NATIVE)[name] for name in _NAMES)
_CLASSES = (_source.Matrix, _source.ImmutableDenseMatrix)
_SPARSE_CLASSES = (_source.SparseMatrix, _source.ImmutableSparseMatrix)
_SPARSE_NATIVE = _source._NATIVE[_source.SparseMatrix]
_SPARSE_NAMES = ("shape", "is_exact_rational", "validate_gaussian_pseudoinverse_source",
                 "to_dense", "has_gaussian_rational_syntax",
                 "has_polynomial_rhs_syntax", "validate_polynomial_rhs_source")
_SPARSE_METHODS = tuple(_source._DICT.__get__(_SPARSE_NATIVE)[name] for name in _SPARSE_NAMES)


def _interfaces(operation, context, base):
    _source._interfaces()
    namespace = _source._DICT.__get__(_NATIVE)
    required = ((0, 1, 2, operation, 6) if operation in (3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20)
                else (0, 1, 2, operation))
    if operation in (8, 11, 12, 13, 16, 17, 18, 20):
        required += (9,)
    if operation == 7:
        required += (9, 21, 22)
    if any(namespace.get(_NAMES[index]) is not _METHODS[index]
           for index in required):
        raise NotImplementedError("%s requires original native interfaces" % context)
    if base is _source.SparseMatrix:
        namespace = _source._DICT.__get__(_SPARSE_NATIVE)
        if any(namespace.get(name) is not method
               for name, method in zip(_SPARSE_NAMES, _SPARSE_METHODS)):
            raise NotImplementedError("%s requires original sparse native interfaces" % context)


def _needs_gaussian(matrix):
    """Inspect original borrowed storage to select one lane for both inputs."""
    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        return True
    base = _source.Matrix if dense else _source.SparseMatrix
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    slots = _source._SLOTS[base]
    native = slots[0].__get__(matrix, base)
    surface = slots[2].__get__(matrix, base)
    if type(native) is not native_type or (
            surface is not None and type(surface) is not (tuple if dense else dict)):
        return True
    cells = surface if dense or surface is None else dict.values(surface)
    return not methods[1].__get__(native, native_type) or (
        surface is not None and any(
            not any(type(entry) is atom for atom in _source._ATOMS) for entry in cells))


def _ingress_snapshot(matrix, context):
    """Retain built-in storage before the first reentrant request charge.

    The established rational lane admits larger sparse matrices. Reuse its
    stored-entry bound here; polynomial/Gaussian source admission still applies
    its stricter dimension and expression limits before any scalar lift.
    """
    from .sparse import _MAX_SPARSE_ENTRIES

    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        # Legacy rational subclasses retain their existing supervised/view
        # checks; capturing their custom accessors here would change that lane.
        return None
    base = _source.Matrix if dense else _source.SparseMatrix
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    state = tuple(slot.__get__(matrix, base) for slot in _source._SLOTS[base])
    native, shape, surface, generation = state
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("%s has an invalid source snapshot" % context)
    if native is not None and (
            type(native) is not native_type
            or methods[0].__get__(native, native_type) != shape):
        raise ValueError("%s requires intact native matrix storage" % context)
    if surface is not None:
        if (type(surface) is not (tuple if dense else dict)
                or len(surface) > _MAX_SPARSE_ENTRIES):
            raise ValueError("%s has an invalid stored-cell snapshot" % context)
        if native is not None and (
                len(surface) != shape[0]*shape[1] if dense
                else len(surface) > shape[0]*shape[1]):
            raise ValueError("%s retained cells do not match its shape" % context)
        if sparse and any(
                type(key) is not tuple or len(key) != 2
                or any(type(index) is not int for index in key)
                or not (0 <= key[0] < shape[0] and 0 <= key[1] < shape[1])
                for key in surface):
            # Reject foreign keys before copying or the unchanged check can
            # invoke their hash/equality methods.
            raise ValueError("%s has an invalid retained sparse coordinate" % context)
    retained = dict(surface) if sparse and surface is not None else surface
    return matrix, base, state, retained


def _check_ingress(snapshot):
    if snapshot is not None:
        _source._unchanged(*snapshot)


def _polynomial_rhs(matrix, session, *, ingress=None):
    """Select the polynomial grammar before solving, without copying cells.

    A false syntax result selects the established exact numeric path. Every
    resource, cancellation, malformed-storage or later domain error is terminal.
    """
    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        return False
    base = _source.Matrix if dense else _source.SparseMatrix
    ingress = _ingress_snapshot(matrix, "exact linear solve") if ingress is None else ingress
    if ingress[0] is not matrix:
        raise RuntimeError("exact linear solve received a different ingress owner")
    _interfaces(7, "exact linear solve", base)
    _check_ingress(ingress)
    state, retained = ingress[2:]
    native, _shape, _surface, _generation = state
    if native is None:
        return False
    gaussian = _METHODS[9] if dense else _SPARSE_METHODS[4]
    polynomial = _METHODS[21] if dense else _SPARSE_METHODS[5]
    selected = not gaussian(native, session=session) and polynomial(native, session=session)
    if _active_callback_session() is not session:
        raise RuntimeError("exact linear solve request changed during selection")
    _source._unchanged(matrix, base, state, retained)
    return selected


def _capture(matrix, session, operation, context, *, allow_sparse=False,
             force_gaussian=False, polynomial_rhs=False, ingress=None):
    """Bind one original source and retain its publication-time check."""
    ingress = _ingress_snapshot(matrix, context) if ingress is None else ingress
    if ingress is not None and ingress[0] is not matrix:
        raise RuntimeError("%s received a different ingress owner" % context)
    _charge_nested_if_active()
    _check_ingress(ingress)
    dense = any(type(matrix) is cls for cls in _CLASSES)
    sparse = allow_sparse and any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    if not dense and not sparse:
        raise NotImplementedError("%s requires a built-in dense Matrix" % context)
    base = _source.Matrix if dense else _source.SparseMatrix
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    _interfaces(operation, context, base)
    state = ingress[2]
    native, shape, surface, generation = state
    if (type(shape) is not tuple or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(generation) is not int or generation < 0):
        raise ValueError("%s has an invalid source snapshot" % context)
    if native is None and operation == 7:
        raise NotImplementedError(
            "matrix contains opaque Python expressions; this operation requires closed native cells")
    if (type(native) is not native_type
            or methods[0].__get__(native, native_type) != shape):
        raise ValueError("%s requires intact native matrix storage" % context)
    rows, columns = shape
    storage_type = tuple if dense else dict
    if surface is not None and (type(surface) is not storage_type
            or (len(surface) != rows*columns if dense else len(surface) > rows*columns)):
        raise ValueError("%s retained cells do not match its shape" % context)
    if sparse and surface is not None and any(
            type(key) is not tuple or len(key) != 2
            or any(type(index) is not int for index in key)
            or not (0 <= key[0] < rows and 0 <= key[1] < columns)
            for key in surface):
        raise ValueError("%s has an invalid retained sparse coordinate" % context)

    # Select from the original source grammar. A held rational value may still
    # contain an undefined pole and cannot enter the literal-rational shortcut.
    preflighted = session is not None or force_gaussian or polynomial_rhs
    preflight = ((_METHODS[22] if dense else _SPARSE_METHODS[6])
                 if polynomial_rhs else methods[2])
    if preflighted:
        preflight(native, session=session)
        _check_ingress(ingress)
    cells = surface if dense or surface is None else dict.values(surface)
    literal = methods[1].__get__(native, native_type) and (
        surface is None or all(any(type(entry) is atom for atom in _source._ATOMS)
                               for entry in cells)
    )
    rational_snapshot = None
    symbol_records = ()
    scalars = ()
    retained = ingress[3]
    if literal:
        _charge_nested_if_active(rows*columns + 1)
        _check_ingress(ingress)
        captured = _source._rational_source_snapshot(matrix, dense=dense, context=context)
        if captured is None:
            raise RuntimeError("%s lost its original rational source" % context)
        native, rational_snapshot = captured
        if sparse:
            retained = rational_snapshot[7]
    else:
        # Bound the complete original native graph before copying its cells.
        if not preflighted:
            preflight(native, session=session)
            _check_ingress(ingress)
        if polynomial_rhs:
            from ..solvers import _sparse_linear as symbols

            # Validate typed-binding descriptors before lifting or asking the
            # retained forest to read any symbol declarations.
            symbols._interfaces()
        if surface is not None or polynomial_rhs:
            if surface is None:
                # Native-only polynomial values still need their registered
                # Symbol/Dummy owners and declarations. Admission above bounds
                # this copy before any expression children can be lifted.
                _charge_nested_if_active(rows*columns + 1)
                _check_ingress(ingress)
                entries = _source._NATIVE_CELLS[base][1](native)
                retained = (tuple(_wrap(entry) for entry in entries) if dense else
                            {key: _wrap(entry) for key, entry in entries.items()})
            scalars = _source._closed(
                retained if dense else dict.values(retained), gaussian=True,
                polynomial=polynomial_rhs)
            if scalars is None:
                raise NotImplementedError(
                    "%s requires closed exact commuting polynomial cells" % context
                    if polynomial_rhs else "%s requires closed exact Gaussian-rational cells" % context)
            if polynomial_rhs:
                # Reuse the exact linear-system symbol boundary: it binds
                # declarations, registry owners and both Dummy identity slots.
                # Check each supplied owner, even repeated equal Dummies.
                records = []
                for scalar, *_ in scalars:
                    if type(scalar) not in (_source.Symbol, _source.Dummy):
                        continue
                    if type(scalar) is _source.Dummy:
                        name, number = (slot.__get__(scalar, _source.Dummy)
                                        for slot in symbols._DUMMY_SLOTS)
                        if (type(name) is not str or len(name) > 1024
                                or type(number) not in (int, bool)
                                or int.bit_length(number) > 4096):
                            raise ValueError("polynomial RHS Dummy identity exceeds its bounds")
                    records.extend(symbols._coordinates((scalar,), 1))
                symbol_records = tuple(records)
                _source._bind_gaussian(native, base, retained, scalars, polynomial=True)
            else:
                _source._bind_gaussian(native, base, retained, scalars)

    def unchanged():
        _interfaces(operation, context, base)
        if _active_callback_session() is not session:
            raise RuntimeError("%s request changed during evaluation" % context)
        if rational_snapshot is not None:
            _source._check_rational_source(rational_snapshot)
        if symbol_records:
            symbols._interfaces()
            symbols._check_coordinates(symbol_records)
        _source._unchanged(matrix, base, state, retained, scalars)

    if sparse:
        unchanged()
        _charge_nested_if_active(rows*columns + 1)
        converted = _SPARSE_METHODS[3](native)
        unchanged()
        if (type(converted) is not _NATIVE
                or _METHODS[0].__get__(converted, _NATIVE) != shape):
            raise RuntimeError("%s returned an invalid dense RHS snapshot" % context)
        return converted, shape, unchanged
    return native, shape, unchanged


def _publish(native, shape, cls, unchanged, context):
    if (type(native) is not _NATIVE
            or _METHODS[0].__get__(native, _NATIVE) != shape):
        raise RuntimeError("%s returned invalid output storage or shape" % context)
    _charge_nested_if_active()
    # Move accepted native storage through original slots, without invoking
    # replaced constructors or _new callbacks during result publication.
    result = object.__new__(cls)
    for slot, value in zip(_source._SLOTS[_source.Matrix], (native, shape, None, 0)):
        slot.__set__(result, value)
    _charge_nested_if_active()
    unchanged()
    return result


def _legacy_rational_view(matrix, operation, context):
    """Preserve inherited standalone literal-rational subclass operations.

    A plain subclass keeps its original result constructor. Borrowed original
    slots are exposed through a built-in view solely for the existing rational
    snapshot checks. Overridden input accessors and the new complex/held
    subclass cases cannot enter this preselected legacy lane.
    """
    if any(type(matrix) is cls for cls in _CLASSES + _SPARSE_CLASSES):
        return matrix, lambda: None
    if _active_callback_session() is not None:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    cls = type(matrix)
    parents = _source._MRO.__get__(cls)
    dense = any(parent is _source.Matrix for parent in parents)
    sparse = any(parent is _source.SparseMatrix for parent in parents)
    base = _source.Matrix if dense else _source.SparseMatrix
    if not (dense or sparse) or type(cls) is not type(base):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    _interfaces(operation, context, base)
    namespace = _source._DICT.__get__(base)
    for name in (*_source._NAMES, "_native", "shape", "rows", "cols"):
        if _source._reduction_binding(cls, name) is not namespace[name]:
            raise NotImplementedError("custom matrix subclasses require original input accessors")
    for name in ("__getattribute__", "__getattr__"):
        if (_source._reduction_binding(cls, name)
                is not _source._reduction_binding(base, name)):
            raise NotImplementedError("custom matrix subclasses require original input accessors")
    accesses = tuple((name, _source._reduction_binding(cls, name)) for name in
                     (*_source._NAMES, "_native", "shape", "rows", "cols",
                      "__getattribute__", "__getattr__"))
    constructor = _source._implementation(_source._reduction_binding(cls, "_new"))
    slots = _source._SLOTS[base]
    state = tuple(slot.__get__(matrix, base) for slot in slots)
    native, _shape, surface, _generation = state
    native_type = _NATIVE if dense else _SPARSE_NATIVE
    methods = _METHODS if dense else _SPARSE_METHODS
    if type(native) is not native_type:
        raise NotImplementedError("custom matrix subclasses require intact native matrix storage")
    native_rows, native_columns = methods[0].__get__(native, native_type)
    source_cells = native_rows * native_columns
    if surface is not None and (
            type(surface) is not (tuple if dense else dict)
            or (len(surface) != source_cells if dense else len(surface) > source_cells)):
        raise ValueError("custom matrix subclass retained cells do not match its native shape")
    retained = dict(surface) if sparse and type(surface) is dict else surface
    owners = []
    for owner in parents:
        if owner is base:
            break
        owners.append((owner, _source._owner_record(owner)))

    def unchanged():
        current = _source._MRO.__get__(cls)
        current_constructor = _source._implementation(_source._reduction_binding(cls, "_new"))
        if (len(current) != len(parents)
                or any(a is not b for a, b in zip(current, parents))
                or any(_source._reduction_binding(cls, name) is not original
                       for name, original in accesses)
                or len(current_constructor) != len(constructor)
                or any(a is not b for a, b in zip(current_constructor, constructor))
                or any(not _source._owner_matches(owner, recorded) for owner, recorded in owners)):
            raise RuntimeError("Matrix subclass changed during exact evaluation")
        _source._unchanged(matrix, base, state, retained)

    cells = dict.values(surface) if sparse and type(surface) is dict else surface
    literal = (type(native) is native_type
               and methods[1].__get__(native, native_type)
               and (surface is None or (type(surface) is (tuple if dense else dict) and all(
                   any(type(value) is atom for atom in _source._ATOMS) for value in cells))))
    if not literal:
        # Preserve previous symbolic inverse/reduction lanes for a plain subclass;
        # this syntactic decision precedes all exact algorithm execution.
        if operation in (8, 11, 12, 16, 17, 18, 20) and dense and type(native) is _NATIVE:
            exact = _METHODS[9](native, session=_active_callback_session())
            unchanged()
            if not exact:
                return matrix, unchanged
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    view = object.__new__(base)
    for slot, value in zip(slots, state):
        slot.__set__(view, value)
    return view, unchanged


def pseudoinverse(matrix):
    """Return a pseudoinverse only for the retained original matrix."""
    session = _active_callback_session()
    native, (rows, columns), unchanged = _capture(matrix, session, 3, "pseudoinverse")
    unchanged()
    inverse = _METHODS[3](native, session=session)
    unchanged()
    return _publish(inverse, (columns, rows), type(matrix), unchanged, "pseudoinverse")


def sparse_pseudoinverse(matrix, method):
    """Publish a checked Q(i) inverse while preserving the sparse class.

    Literal rational inputs keep the separate sparse rank-decomposition lane.
    This extension preflights all original complex cells before conversion;
    the native checker establishes all four Hermitian Penrose identities.
    """
    from ._row_reduction import _publish as publish_matrix

    if type(method) is not str:
        raise ValueError("invalid pinv method")
    if method == "ED":
        raise NotImplementedError("sparse pseudoinverse currently supports the RD method")
    if method != "RD":
        raise ValueError(f"invalid pinv method {method!r}")
    session = _active_callback_session()
    context = "sparse complex pseudoinverse"
    native, (rows, columns), unchanged = _capture(
        matrix, session, 3, context, allow_sparse=True, force_gaussian=True)
    unchanged()
    inverse = _METHODS[3](native, session=session)
    unchanged()
    return publish_matrix(inverse, (columns, rows), type(matrix), unchanged, context)


def solve_least_squares(matrix, rhs):
    """Return the unique Hermitian least-squares solution for original A/B."""
    _charge_nested_if_active()
    session = _active_callback_session()
    gaussian = session is not None or _needs_gaussian(matrix) or _needs_gaussian(rhs)
    left, (_, columns), left_unchanged = _capture(
        matrix, session, 4, "least squares", allow_sparse=True, force_gaussian=gaussian)
    right, (_, rhs_columns), right_unchanged = _capture(
        rhs, session, 4, "least squares", allow_sparse=True, force_gaussian=gaussian)

    def unchanged():
        left_unchanged()
        right_unchanged()

    unchanged()
    solution = _METHODS[4](left, right, session=session)
    unchanged()
    if any(type(matrix) is cls for cls in _SPARSE_CLASSES):
        from ._row_reduction import _publish as publish_matrix

        result_class = (_source.ImmutableSparseMatrix
                        if type(matrix) is _source.ImmutableSparseMatrix else type(rhs))
        return publish_matrix(solution, (columns, rhs_columns), result_class,
                              unchanged, "least squares")
    return _publish(solution, (columns, rhs_columns), type(matrix), unchanged, "least squares")


def solve_exact(matrix, rhs, *, lu=False):
    """Return the unique exact solution of both retained original sources.

    The Q(i) and polynomial-RHS lanes check a left inverse and the full
    original residual. Polynomial coefficients remain exact throughout.
    Failure is terminal, including inconsistency and request exhaustion.
    """
    original_matrix, original_rhs = matrix, rhs
    if not any(parent is _source.Matrix or parent is _source.SparseMatrix
               for parent in _source._MRO.__get__(type(rhs))):
        raise TypeError("solve requires a built-in Matrix right-hand side")
    left_ingress = _ingress_snapshot(matrix, "exact linear solve")
    right_ingress = _ingress_snapshot(rhs, "exact linear solve")
    _charge_nested_if_active()
    _check_ingress(left_ingress)
    _check_ingress(right_ingress)
    session = _active_callback_session()
    matrix, matrix_owner_unchanged = _legacy_rational_view(matrix, 7, "exact linear solve")
    rhs, rhs_owner_unchanged = _legacy_rational_view(rhs, 7, "exact linear solve")
    gaussian = session is not None or _needs_gaussian(matrix) or _needs_gaussian(rhs)
    polynomial = gaussian and _polynomial_rhs(rhs, session, ingress=right_ingress)
    if gaussian and (original_matrix is not matrix or original_rhs is not rhs):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    if polynomial and any(type(matrix) is cls for cls in _SPARSE_CLASSES):
        # Preserve the established sparse refusal for symbolic coefficient
        # matrices. This borrowed grammar selection precedes admission; a
        # budget/cancellation failure is propagated without catching it.
        _interfaces(7, "exact linear solve", _source.SparseMatrix)
        _check_ingress(left_ingress)
        coefficient_source = left_ingress[2][0]
        supported = (coefficient_source is not None
                     and _SPARSE_METHODS[4](coefficient_source, session=session))
        _check_ingress(left_ingress)
        if not supported:
            raise NotImplementedError(
                "sparse polynomial RHS solving requires exact rational or Gaussian-rational coefficients")
    left, (rows, columns), left_unchanged = _capture(
        matrix, session, 7, "exact linear solve", allow_sparse=True, force_gaussian=gaussian,
        ingress=left_ingress)
    right, (rhs_rows, rhs_columns), right_unchanged = _capture(
        rhs, session, 7, "exact linear solve", allow_sparse=True, force_gaussian=gaussian,
        polynomial_rhs=polynomial, ingress=right_ingress)
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")
    if rows != columns:
        if not lu and any(type(original_matrix) is cls for cls in _SPARSE_CLASSES):
            if rows < columns:
                raise ValueError("Under-determined system.")
            raise ValueError(
                "For over-determined system, M, having more rows than columns, "
                "try M.solve_least_squares(rhs).")
        if lu and rows < columns:
            raise NotImplementedError("Underdetermined systems not supported.")
        if not lu and columns != 0 and rhs_columns == 0:
            raise ValueError("Matrix det == 0; not invertible.")

    def unchanged():
        matrix_owner_unchanged()
        rhs_owner_unchanged()
        left_unchanged()
        right_unchanged()

    unchanged()
    solution = _METHODS[7](left, right, session=session)
    unchanged()
    shape = (columns, rhs_columns)
    # Built-in LUsolve follows the RHS class for every shape/storage kind;
    # square legacy dense subclasses keep their established _new constructor.
    # Ordinary sparse solve promotes mutable coefficients to the RHS class,
    # while immutable sparse coefficients retain their own class.
    if any(type(original_matrix) is cls for cls in _SPARSE_CLASSES):
        owner = (original_rhs if lu or type(original_matrix) is _source.SparseMatrix
                 else original_matrix)
    else:
        owner = (original_rhs if lu
                 and (rows != columns or any(type(original_matrix) is known for known in _CLASSES))
                 and any(type(original_rhs) is known for known in _CLASSES + _SPARSE_CLASSES)
                 else original_matrix)
    cls = type(owner)
    if not any(cls is known for known in _CLASSES + _SPARSE_CLASSES):
        # Keep the legacy subclass's construction hook, then recheck every
        # original input in case that hook had effects during publication.
        result = owner._new(solution)
        _charge_nested_if_active()
        unchanged()
        return result
    if not any(cls is candidate for candidate in _SPARSE_CLASSES):
        return _publish(solution, shape, cls, unchanged, "exact linear solve")
    if (type(solution) is not _NATIVE
            or _METHODS[0].__get__(solution, _NATIVE) != shape):
        raise RuntimeError("exact linear solve returned invalid output storage or shape")
    _charge_nested_if_active(columns * rhs_columns + 1)
    sparse = _METHODS[6](solution)
    if (type(sparse) is not _SPARSE_NATIVE
            or _SPARSE_METHODS[0].__get__(sparse, _SPARSE_NATIVE) != shape):
        raise RuntimeError("exact linear solve returned invalid sparse output")
    result = object.__new__(cls)
    for slot, value in zip(_source._SLOTS[_source.SparseMatrix], (sparse, shape, None, 0)):
        slot.__set__(result, value)
    _charge_nested_if_active()
    unchanged()
    return result


def inverse_if_exact(matrix):
    """Preselect the closed exact inverse lane, preserving symbolic inverses.

    None means the original native syntax is outside Q(i), before any
    algorithm ran. Every error after exact selection propagates unchanged.
    """
    _charge_nested_if_active()
    session = _active_callback_session()
    context = "exact matrix inverse"
    original_matrix = matrix
    matrix, owner_unchanged = _legacy_rational_view(matrix, 8, context)
    if session is not None and original_matrix is not matrix:
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    sparse = any(type(matrix) is cls for cls in _SPARSE_CLASSES)
    base = _source.SparseMatrix if sparse else _source.Matrix
    native_type = _SPARSE_NATIVE if sparse else _NATIVE
    syntax = _SPARSE_METHODS[4] if sparse else _METHODS[9]
    _interfaces(8, context, base)
    slots = _source._SLOTS[base]
    state = tuple(slot.__get__(matrix, base) for slot in slots)
    native, _shape, surface, _generation = state
    if native is None:
        return None
    if type(native) is not native_type:
        raise ValueError("exact matrix inverse requires intact native matrix storage")
    # Literal rational storage keeps the existing larger source bounds. A
    # Gaussian syntax walk is needed only for the newly admitted grammar.
    exact = not _needs_gaussian(matrix) or syntax(native, session=session)
    owner_unchanged()
    _source._unchanged(matrix, base, state, surface)
    if not exact:
        if session is not None:
            raise NotImplementedError("active matrix inverse requests require closed exact Q(i) cells")
        return None
    source, shape, source_unchanged = _capture(
        matrix, session, 8, context, allow_sparse=True)

    def unchanged():
        owner_unchanged()
        source_unchanged()

    unchanged()
    inverse = _METHODS[8](source, session=session)
    unchanged()
    if original_matrix is not matrix:
        result = original_matrix._new(inverse)
        unchanged()
        _charge_nested_if_active()
        return result
    if sparse:
        from ._row_reduction import _publish as publish_matrix

        return publish_matrix(inverse, shape, type(matrix), unchanged, context)
    return _publish(inverse, shape, type(matrix), unchanged, context)


def pinv_solve(matrix, rhs, arbitrary_matrix=None):
    """Return the complete checked Q(i) family X + P*W for original A/B.

    X is the unique minimum-norm least-squares solution; P is the Hermitian
    projector onto the complete kernel of A. Both are independently checked
    natively before the common bounded parameter renderer sees them.
    """
    from ._sparse_family import _ParameterMatrix, _render_family

    if not any(type(matrix) is cls for cls in _CLASSES + _SPARSE_CLASSES):
        raise NotImplementedError("custom matrix subclasses require a supervised override lane")
    if not any(type(rhs) is cls for cls in _CLASSES + _SPARSE_CLASSES):
        raise TypeError("pinv_solve requires a built-in Matrix right-hand side")
    _charge_nested_if_active()
    session = _active_callback_session()
    context = "least-squares family"
    # The entire family lane, including literal rational sources, has the same
    # 32-axis bound. Admit each original before its sparse conversion, then
    # retain both snapshots through solving and publication.
    left, (rows, columns), left_unchanged = _capture(
        matrix, session, 5, context, allow_sparse=True, force_gaussian=True)
    right, (rhs_rows, rhs_columns), right_unchanged = _capture(
        rhs, session, 5, context, allow_sparse=True, force_gaussian=True)
    if rows != rhs_rows:
        raise ValueError("The matrices have incompatible number of rows")

    def unchanged():
        left_unchanged()
        right_unchanged()

    unchanged()
    shape = (columns, rhs_columns)
    supplied = (None if arbitrary_matrix is None else
                _ParameterMatrix(arbitrary_matrix, shape, session=session))
    if supplied is not None:
        _charge_nested_if_active(sum(supplied.sizes.values()) + 1)
    kinds = (type(matrix), type(rhs), type(arbitrary_matrix))
    result_type = (_source.ImmutableSparseMatrix if _source.ImmutableSparseMatrix in kinds else
                   _source.ImmutableDenseMatrix if _source.ImmutableDenseMatrix in kinds else
                   _source.Matrix)
    unchanged()
    minimum, projector = _METHODS[5](left, right, session=session)
    unchanged()
    if (type(minimum) is not _NATIVE or type(projector) is not _NATIVE
            or _METHODS[0].__get__(minimum, _NATIVE) != shape
            or _METHODS[0].__get__(projector, _NATIVE) != (columns, columns)):
        raise RuntimeError("native least-squares family has invalid output shapes")
    _charge_nested_if_active(columns * (columns + rhs_columns) + 1)
    minimum = _METHODS[6](minimum)
    projector = _METHODS[6](projector)
    unchanged()
    return _render_family(minimum, projector, shape, result_type, supplied, unchanged)
