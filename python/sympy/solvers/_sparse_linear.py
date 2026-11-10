"""Closed matrix-form linear families over the checked sparse engine."""

from collections import OrderedDict

from ..core import (
    Add, Basic, Dummy, Function, Half, Integer, Mul, NegativeOne, One,
    Rational, Symbol, Tuple as SymTuple, Zero,
    _DUMMY_PREFIX, _SYMBOL_CACHE, _surface_symbols, _native,
    _ZERO, _ONE, _NEGATIVE_ONE, _HALF,
)
from ..matrices.sparse import (
    Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix,
    _NativeSparseMatrix, _REDUCTION_DICT, _REDUCTION_NATIVE_EXPR,
    _REDUCTION_NATIVE_EQUAL, _REDUCTION_NATIVE_MEMBERS, _REDUCTION_VALUE_SLOT,
    _SOLVE_DENSE_MEMBERS, _rational_source_snapshot, _check_rational_source,
    _reduction_binding, _reduction_default_access, _affine_parameter_symbols,
    _check_affine_parameters,
)
from ..polys.polyerrors import GeneratorsError
from ..sets.sets import EmptySet, FiniteSet, Set, _EMPTY_SET


UNSUPPORTED = object()
_MATRIX_CLASSES = (Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix)
_NATIVE_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_NativeSparseMatrix)[name]
    for name in ("solve_affine_parameters", "solve_augmented_affine_parameters", "transpose")
}
_EXPR_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_REDUCTION_NATIVE_EXPR)[name]
    for name in ("func_name", "args", "symbol_binding", "__hash__",
                 "exact_numerator", "exact_denominator")
}
_BINDING_CLASS = _native.SymbolBinding
_BINDING_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_BINDING_CLASS)[name]
    for name in ("name", "plain", "identity_hex")
}
_NATIVE_SYMBOL = _native.py_symbol
_NATIVE_FUNCTION = _native.py_function
_NATIVE_SET = _native.SymSet
_NATIVE_SET_FINITE = _NATIVE_SET.finite
_NATIVE_SET_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_NATIVE_SET)[name]
    for name in ("finite", "kind")
}
_ASSUMPTIONS_SLOT = Symbol.__dict__["_assumptions"]
_DUMMY_SLOTS = tuple(Dummy.__dict__[name] for name in ("_dummy_name", "_dummy_number"))
_ATOM_AUXILIARY_SLOTS = tuple(
    Basic.__dict__[name] for name in ("_struct_args", "_surface_args")
)
_ARITHMETIC_ARGS_SLOTS = {cls: cls.__dict__["_args"] for cls in (Add, Mul)}
_FUNCTION_ARGS_SLOT = Function.__dict__["_args"]
_SET_SLOT = Set.__dict__["_native_set"]
_FINITE_SLOT = FiniteSet.__dict__["_surface_elements"]
_EMPTY_NATIVE = _SET_SLOT.__get__(_EMPTY_SET, Set)
_SCALAR_SINGLETONS = tuple(
    (atom, _REDUCTION_VALUE_SLOT.__get__(atom, Basic))
    for atom in (_ZERO, _ONE, _NEGATIVE_ONE, _HALF)
)
_CLASS_INTERFACES = tuple(
    (cls, type(cls), tuple((name, _reduction_binding(cls, name)) for name in names))
    for cls, names in (
        *((cls, ("__new__", "__init__", "__setattr__", "__delattr__", "_value",
                  "_struct_args", "_surface_args", "_args", "args"))
          for cls in (Integer, Rational, Zero, One, NegativeOne, Half, Add, Mul)),
        *((cls, ("__new__", "__init__", "__setattr__", "__delattr__", "_value",
                  "_assumptions", "_struct_args", "_surface_args", "__hash__", "__eq__",
                  "name", "_dummy_name", "_dummy_number", "dummy_index"))
          for cls in (Symbol, Dummy)),
        (SymTuple, ("__new__", "__init__", "__setattr__", "__delattr__", "_value", "_args", "args")),
        (FiniteSet, ("__new__", "__init__", "__setattr__", "__delattr__", "_native_set",
                     "_surface_elements", "__iter__", "args")),
        (EmptySet, ("__new__", "__init__", "__setattr__", "__delattr__", "_native_set")),
    )
)


def _interfaces():
    for cls, metaclass, members in _CLASS_INTERFACES:
        if (type(cls) is not metaclass or not _reduction_default_access(cls)
                or any(_reduction_binding(cls, name) is not original for name, original in members)):
            raise NotImplementedError("checked linear systems require original scalar and result interfaces")
    for cls, members in ((_NativeSparseMatrix, {**_REDUCTION_NATIVE_MEMBERS, **_NATIVE_MEMBERS}),
                         (_REDUCTION_NATIVE_EXPR, _EXPR_MEMBERS),
                         (_BINDING_CLASS, _BINDING_MEMBERS),
                         (_NATIVE_SET, _NATIVE_SET_MEMBERS)):
        namespace = _REDUCTION_DICT.__get__(cls)
        if (not _reduction_default_access(cls)
                or any(namespace.get(name) is not original for name, original in members.items())):
            raise NotImplementedError("checked linear systems require original native interfaces")
    for atom, _ in _SCALAR_SINGLETONS:
        _require_atomic_storage(atom)
    if (type(_SYMBOL_CACHE) is not dict or type(_surface_symbols) is not OrderedDict
            or _SET_SLOT.__get__(_EMPTY_SET, Set) is not _EMPTY_NATIVE
            or any(_REDUCTION_VALUE_SLOT.__get__(atom, Basic) is not value
                   for atom, value in _SCALAR_SINGLETONS)):
        raise RuntimeError("checked linear-system owners have changed")


def _binding_key(native):
    binding = _EXPR_MEMBERS["symbol_binding"].__get__(native, _REDUCTION_NATIVE_EXPR)
    if type(binding) is not _BINDING_CLASS:
        raise TypeError("linear-system coordinates must be Symbol or Dummy atoms")
    name = _BINDING_MEMBERS["name"].__get__(binding, _BINDING_CLASS)
    plain = _BINDING_MEMBERS["plain"].__get__(binding, _BINDING_CLASS)
    identity = None if plain else _BINDING_MEMBERS["identity_hex"].__get__(binding, _BINDING_CLASS)
    return name, identity


def _require_atomic_storage(atom):
    for slot in _ATOM_AUXILIARY_SLOTS:
        try:
            slot.__get__(atom, Basic)
        except AttributeError:
            continue
        raise RuntimeError("linear-system atom carries non-atomic surface storage")


def _symbol_snapshot(variable):
    if type(variable) is not Symbol and type(variable) is not Dummy:
        raise TypeError("linear-system coordinates must be exact Symbol or Dummy atoms")
    _require_atomic_storage(variable)
    value = _REDUCTION_VALUE_SLOT.__get__(variable, Basic)
    assumptions = _ASSUMPTIONS_SLOT.__get__(variable, Symbol)
    if type(value) is not _REDUCTION_NATIVE_EXPR or type(assumptions) is not dict:
        raise RuntimeError("linear-system coordinate has invalid original storage")
    if any(type(key) is not str or type(fact) is not bool for key, fact in assumptions.items()):
        raise NotImplementedError("linear-system coordinates require literal declared assumptions")
    facts = tuple(sorted(assumptions.items()))
    key = _binding_key(value)
    if len(key[0]) > 4096 or len(key[0].encode("utf-8")) > 4096:
        raise ValueError("linear-system coordinate names exceed their limit")
    expected = _NATIVE_SYMBOL(key[0], [(name, "True" if fact else "False") for name, fact in facts])
    if _REDUCTION_NATIVE_EQUAL(value, expected) is not True:
        raise RuntimeError("linear-system coordinate assumptions differ from its native identity")
    dummy = None
    if type(variable) is Dummy:
        name, number = (slot.__get__(variable, Dummy) for slot in _DUMMY_SLOTS)
        if type(name) is not str or type(number) not in (int, bool):
            raise NotImplementedError("checked Dummy coordinates require a built-in integer index")
        if key[0] != f"{_DUMMY_PREFIX}_{int(number)}_{name}":
            raise RuntimeError("Dummy coordinate fields differ from its native identity")
        dummy = (name, number)
    return variable, value, assumptions, facts, key, dummy


def _coordinates(variables, unknowns):
    if variables is None:
        return None
    if len(variables) != unknowns:
        raise ValueError("Number of symbols does not match the linear-system columns")
    if unknowns > 8192:
        raise ValueError("linear-system coordinate count exceeds its bound")
    records = []
    keys = set()
    name_bytes = 0
    for variable in variables:
        record = _symbol_snapshot(variable)
        _, value, _, facts, key, dummy = record
        encoded_bytes = len(key[0].encode("utf-8"))
        name_bytes += encoded_bytes
        if encoded_bytes > 4096 or name_bytes > 1_048_576:
            raise ValueError("linear-system coordinate names exceed their limit")
        if key in keys:
            raise GeneratorsError("duplicated linear-system generators")
        keys.add(key)
        if dummy is None and dict.get(_SYMBOL_CACHE, (key[0], facts)) is not variable:
            raise RuntimeError("linear-system coordinate cache owner differs from the supplied Symbol")
        registered = OrderedDict.get(_surface_symbols, key)
        registered_snapshot = record
        if registered is not variable:
            # Dummy reconstruction preserves an existing registry owner. Keep
            # that owner intact while retaining the supplied object locally.
            if dummy is None or type(registered) is not Dummy:
                raise RuntimeError("linear-system coordinate registry owner differs from the supplied atom")
            registered_snapshot = _symbol_snapshot(registered)
            if (registered_snapshot[3] != facts or registered_snapshot[4] != key
                    or _REDUCTION_NATIVE_EQUAL(registered_snapshot[1], value) is not True):
                raise RuntimeError("reconstructed Dummy differs from its registered identity")
        records.append((*record, registered_snapshot))
    return tuple(records)


def _check_symbol_snapshot(record):
    variable, value, assumptions, facts, _, dummy = record
    _require_atomic_storage(variable)
    if any(type(key) is not str or type(fact) is not bool for key, fact in assumptions.items()):
        raise RuntimeError("linear-system coordinate assumptions changed to nonliteral values")
    if (_REDUCTION_VALUE_SLOT.__get__(variable, Basic) is not value
            or _ASSUMPTIONS_SLOT.__get__(variable, Symbol) is not assumptions
            or tuple(sorted(assumptions.items())) != facts):
        raise RuntimeError("linear-system coordinates mutated during the solve")
    if dummy is not None:
        name, number = dummy
        if (_DUMMY_SLOTS[0].__get__(variable, Dummy) is not name
                or _DUMMY_SLOTS[1].__get__(variable, Dummy) is not number):
            raise RuntimeError("Dummy coordinates mutated during the solve")


def _check_coordinates(records):
    if records is None:
        return
    for record in records:
        variable, _, _, facts, key, dummy, registered_snapshot = record
        _check_symbol_snapshot(record[:6])
        _check_symbol_snapshot(registered_snapshot)
        if (OrderedDict.get(_surface_symbols, key) is not registered_snapshot[0]
                or (dummy is None and dict.get(_SYMBOL_CACHE, (key[0], facts)) is not variable)):
            raise RuntimeError("linear-system coordinate ownership changed during the solve")


def _lift_scalar(value, owners):
    """Lift only the native affine grammar without invoking constructors."""
    if type(value) is not _REDUCTION_NATIVE_EXPR:
        raise RuntimeError("native linear solution contains an invalid expression")
    kind = _EXPR_MEMBERS["func_name"].__get__(value, _REDUCTION_NATIVE_EXPR)
    if kind == "Symbol":
        owner = owners.get(_binding_key(value))
        if owner is None or _REDUCTION_NATIVE_EQUAL(value, owner[1]) is not True:
            raise RuntimeError("native linear solution contains an unowned parameter")
        return owner[0]
    if kind in ("Integer", "Rational"):
        for atom, original in _SCALAR_SINGLETONS:
            if _REDUCTION_NATIVE_EQUAL(value, original) is True:
                return atom
        cls = Integer if kind == "Integer" else Rational
    elif kind in ("Add", "Mul"):
        cls = Add if kind == "Add" else Mul
    else:
        raise RuntimeError("native linear solution is outside its affine grammar")
    result = object.__new__(cls)
    _REDUCTION_VALUE_SLOT.__set__(result, value)
    if kind in ("Add", "Mul"):
        children = _EXPR_MEMBERS["args"].__get__(value, _REDUCTION_NATIVE_EXPR)
        _ARITHMETIC_ARGS_SLOTS[cls].__set__(
            result, tuple(_lift_scalar(child, owners) for child in children),
        )
    return result


def _solution_set(values, native_values):
    native_tuple = _NATIVE_FUNCTION("Tuple", *native_values)
    row = object.__new__(SymTuple)
    _REDUCTION_VALUE_SLOT.__set__(row, native_tuple)
    _FUNCTION_ARGS_SLOT.__set__(row, values)
    result = object.__new__(FiniteSet)
    _SET_SLOT.__set__(result, _NATIVE_SET_FINITE([native_tuple]))
    _FINITE_SLOT.__set__(result, (row,))
    return result


def checked_linear_system(matrix, variables, *, rhs=None, mapping=False, empty_augmented=False):
    """Return UNSUPPORTED only before entering the exact rational engine."""
    if not any(type(matrix) is cls for cls in _MATRIX_CLASSES):
        raise NotImplementedError("custom linear-system matrix classes require a supervised override lane")
    if rhs is not None and not any(type(rhs) is cls for cls in _MATRIX_CLASSES):
        raise NotImplementedError("custom linear-system RHS classes require a supervised override lane")
    left = _rational_source_snapshot(
        matrix, dense=type(matrix) in (Matrix, ImmutableDenseMatrix), context="linear-system solve",
    )
    right = None if rhs is None else _rational_source_snapshot(
        rhs, dense=type(rhs) in (Matrix, ImmutableDenseMatrix), context="linear-system solve",
    )
    if left is None or (rhs is not None and right is None):
        return UNSUPPORTED
    _interfaces()
    native, snapshot = left
    snapshots = (snapshot,) if right is None else (snapshot, right[1])
    rows, cols = snapshot[4]
    if rhs is None and empty_augmented and (rows == 0 or cols == 0):
        for source in snapshots:
            _check_rational_source(source)
        return _EMPTY_SET
    unknowns = cols if rhs is not None else cols - 1
    if unknowns < 0:
        raise ValueError("augmented linear system requires a right-hand-side column")
    records = _coordinates(variables, unknowns)
    coordinates = None if records is None else tuple(record[1] for record in records)
    if snapshot[-1]:
        native = _SOLVE_DENSE_MEMBERS["to_sparse"](native)
    if right is None:
        operation = _NATIVE_MEMBERS["solve_augmented_affine_parameters"]
        args = (native, coordinates)
    else:
        rhs_native, rhs_snapshot = right
        rhs_rows, rhs_cols = rhs_snapshot[4]
        transpose = rhs_rows == 1 and rhs_cols == rows
        if not ((rhs_cols == 1 and rhs_rows == rows) or transpose):
            raise ValueError("Incompatible dimensions for (A, b) in linsolve")
        if rhs_snapshot[-1]:
            rhs_native = _SOLVE_DENSE_MEMBERS["to_sparse"](rhs_native)
        if transpose:
            rhs_native = _NATIVE_MEMBERS["transpose"](rhs_native)
        operation = _NATIVE_MEMBERS["solve_affine_parameters"]
        args = (native, rhs_native, coordinates)
    for source in snapshots:
        _check_rational_source(source)
    _check_coordinates(records)
    # All errors after this point are terminal: no dense or generic retry.
    family = operation(*args)
    for source in snapshots:
        _check_rational_source(source)
    _check_coordinates(records)
    if family is None:
        return None if mapping else _EMPTY_SET
    solution, parameters, free = family
    shape = _REDUCTION_NATIVE_MEMBERS["shape"]
    if (shape.__get__(solution, _NativeSparseMatrix) != (unknowns, 1)
            or shape.__get__(parameters, _NativeSparseMatrix) != (len(free), 1)
            or any(type(column) is not int or not 0 <= column < unknowns for column in free)
            or free != sorted(set(free))):
        raise RuntimeError("native linear solution has invalid shapes or free coordinates")
    automatic = _affine_parameter_symbols(parameters) if records is None else ()
    for _, symbol, _, _, _, _ in automatic:
        _require_atomic_storage(symbol)
    owners = ({_binding_key(value): (symbol, value) for _, symbol, value, _, _, _ in automatic}
              if records is None else {records[column][4]: (records[column][0], records[column][1])
                                       for column in free})
    parameter_entries = _REDUCTION_NATIVE_MEMBERS["todok"](parameters)
    if len(parameter_entries) != len(free):
        raise RuntimeError("native linear parameter storage is incomplete")
    if records is not None and any(
        _REDUCTION_NATIVE_EQUAL(parameter_entries.get((index, 0)), records[column][1]) is not True
        for index, column in enumerate(free)
    ):
        raise RuntimeError("native linear parameters differ from the supplied coordinates")
    entries = _REDUCTION_NATIVE_MEMBERS["todok"](solution)
    zero = _SCALAR_SINGLETONS[0][1]
    native_values = tuple(entries.get((column, 0), zero) for column in range(unknowns))
    values = tuple(_lift_scalar(value, owners) for value in native_values)
    if mapping:
        free_columns = set(free)
        result = {record[0]: values[column] for column, record in enumerate(records)
                  if column not in free_columns}
    else:
        result = _solution_set(values, native_values)
    _interfaces()
    for source in snapshots:
        _check_rational_source(source)
    _check_coordinates(records)
    _check_affine_parameters(automatic)
    for _, symbol, _, _, _, _ in automatic:
        _require_atomic_storage(symbol)
    return result
