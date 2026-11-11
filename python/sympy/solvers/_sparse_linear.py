"""Closed matrix and affine-equation families over the checked sparse engine."""

from collections import OrderedDict
from fractions import Fraction
from math import gcd, isfinite

from ..core import (
    Add, Basic, ComplexInfinity, Dummy, Eq, Expr, Float, Function, Half, Integer, Mul,
    NegativeOne, One, Pow, Rational, Relational, Symbol, Tuple as SymTuple, Zero,
    _DUMMY_PREFIX, _SYMBOL_CACHE, _surface_symbols, _native,
    _ZERO, _ONE, _NEGATIVE_ONE, _HALF, _FLOAT_INTERN,
    _is_closed_function_class, _lookup_known_function,
    pi, E, I, oo, zoo, nan,
)
from ..core._function_capabilities import _implementation, _owner_record, _owner_matches
from ..matrices.sparse import (
    Matrix, ImmutableDenseMatrix, SparseMatrix, ImmutableSparseMatrix,
    _NativeSparseMatrix, _REDUCTION_DICT, _REDUCTION_NATIVE_EXPR,
    _REDUCTION_NATIVE_EQUAL, _REDUCTION_NATIVE_MEMBERS, _REDUCTION_VALUE_SLOT,
    _SOLVE_DENSE_MEMBERS, _rational_source_snapshot, _check_rational_source,
    _reduction_binding, _reduction_default_access, _affine_parameter_symbols,
    _check_affine_parameters, _checked_sparse_lift,
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


# Equation sources are admitted before either algebraic cancellation or sparse
# construction. These are source-boundary limits, separate from the existing
# generator/verifier budget inside the sparse affine solver.
_EQUATION_MAX_ROWS = 8192
_EQUATION_MAX_NODES = 65_536
_EQUATION_MAX_DEPTH = 64
_EQUATION_MAX_ENTRIES = 262_144
_EQUATION_MAX_WORK = 1_000_000
_EQUATION_MAX_BITS = 4096
_EQUATION_MISSING = object()
_EQUATION_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_EQUATION_CLASSES = (*_EQUATION_ATOMS, Symbol, Dummy, Add, Mul, Pow, Eq, Expr, Float, ComplexInfinity)
_EQUATION_RELATION_OPERATOR = Eq.__dict__["rel_op"]
_EQUATION_OWNERS = tuple(
    (owner, _owner_record(owner))
    for owner in dict.fromkeys(
        owner for cls in (*_EQUATION_CLASSES, Fraction)
        for owner in type.__dict__["__mro__"].__get__(cls)
    )
)
# Local rational arithmetic is part of source admission, before the native
# witness sees the extracted matrix. Seal its operators and properties too;
# the generic symbolic-owner roster intentionally excludes __add__/__mul__.
_EQUATION_FRACTION_IMPLEMENTATIONS = tuple(
    (name, _implementation(member)) for name, member in Fraction.__dict__.items()
)
_EQUATION_SLOTS = {
    (owner, name): owner.__dict__[name]
    for owner, names in (
        (Basic, ("_value", "_struct_args", "_surface_args")),
        (Add, ("_args",)), (Mul, ("_args",)), (Pow, ("_args",)),
        (Function, ("_args",)), (Relational, ("_relational_sides",)),
        (Fraction, ("_numerator", "_denominator")),
    )
    for name in names
}
_EQUATION_NATIVE_MEMBERS = {
    name: _REDUCTION_DICT.__get__(_REDUCTION_NATIVE_EXPR)[name]
    for name in ("is_integer", "is_rational", "is_symbol", "is_add", "is_mul", "is_pow")
}
_EQUATION_COMPILED = _native.CompiledFunction
_EQUATION_PREFLIGHT_DESCRIPTOR = _REDUCTION_DICT.__get__(_EQUATION_COMPILED)["_validate_source"]
_EQUATION_PREFLIGHT = _EQUATION_COMPILED._validate_source
_EQUATION_SPARSE_CONSTRUCTORS = {
    name: _REDUCTION_DICT.__get__(_NativeSparseMatrix)[name]
    for name in ("__new__", "__init__") if name in _REDUCTION_DICT.__get__(_NativeSparseMatrix)
}
_EQUATION_RATIONAL = _native.py_rational_from_python
_EQUATION_FINITE_CONSTANTS = tuple(_REDUCTION_VALUE_SLOT.__get__(value, Basic) for value in (pi, E, I))
_EQUATION_NONFINITE_CONSTANTS = tuple(
    _REDUCTION_VALUE_SLOT.__get__(value, Basic) for value in (oo, zoo, nan)
)
# These entire built-ins may remain symbolic coefficients in the existing
# generic lane. Functions of a solved coordinate are still nonlinear. Other
# functions require their own original-domain admission, never a zero shortcut.
_EQUATION_ENTIRE = frozenset({"exp", "sin", "cos", "sinh", "cosh", "erf", "erfc"})


def _equation_interfaces():
    _interfaces()
    if (any(not _owner_matches(owner, record) for owner, record in _EQUATION_OWNERS)
            or _REDUCTION_DICT.__get__(Eq).get("rel_op") is not _EQUATION_RELATION_OPERATOR):
        raise NotImplementedError("linear equations require original built-in implementations")
    namespace = _REDUCTION_DICT.__get__(Fraction)
    if len(namespace) != len(_EQUATION_FRACTION_IMPLEMENTATIONS):
        raise NotImplementedError("linear equations require original Fraction arithmetic")
    for name, original in _EQUATION_FRACTION_IMPLEMENTATIONS:
        try:
            current = _implementation(namespace[name])
        except (KeyError, ValueError):
            raise NotImplementedError("linear equations require original Fraction arithmetic") from None
        if len(current) != len(original) or any(left is not right for left, right in zip(current, original)):
            raise NotImplementedError("linear equations require original Fraction arithmetic")
    for (owner, name), descriptor in _EQUATION_SLOTS.items():
        if _REDUCTION_DICT.__get__(owner).get(name) is not descriptor:
            raise NotImplementedError("linear equations require original storage descriptors")
    if (any(_REDUCTION_DICT.__get__(_REDUCTION_NATIVE_EXPR).get(name) is not descriptor
            for name, descriptor in _EQUATION_NATIVE_MEMBERS.items())
            or _REDUCTION_DICT.__get__(_EQUATION_COMPILED).get("_validate_source")
            is not _EQUATION_PREFLIGHT_DESCRIPTOR
            or any(_REDUCTION_DICT.__get__(_NativeSparseMatrix).get(name) is not descriptor
                   for name, descriptor in _EQUATION_SPARSE_CONSTRUCTORS.items())
            or _native.py_rational_from_python is not _EQUATION_RATIONAL):
        raise NotImplementedError("linear equations require original native interfaces")


class _AffineEquationSource:
    """Read original trees without public getters, substitutions, or printers.

    Values are (constant, sparse coefficients, dependent columns). A None
    constant marks a legitimate symbolic-coefficient input for the preexisting
    generic lane; it can never be published by the exact rational lane.
    """

    def __init__(self, equations, coordinates):
        self.equations = equations
        self.original = tuple(equations)
        self.coordinates = coordinates
        self.columns = {record[4]: index for index, record in enumerate(coordinates)}
        self.slots = []
        self.symbols = {}
        self.surface_symbols = []
        self.functions = []
        self.active = set()
        self.memo = {}
        self.nodes = 0
        self.work = 0

    def charge(self, amount=1, depth=0):
        self.work += amount
        if self.work > _EQUATION_MAX_WORK or depth > _EQUATION_MAX_DEPTH:
            raise ValueError("linear equation source exceeds its work or depth limit")

    def bounded(self, value):
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _EQUATION_MAX_BITS:
            raise ValueError("linear equation coefficient exceeds 4096 bits")
        return value

    def constant(self, value):
        return self.bounded(value), {}, frozenset()

    def retain(self, node, owner, name):
        descriptor = _EQUATION_SLOTS[owner, name]
        try:
            value = descriptor.__get__(node, owner)
        except AttributeError:
            value = _EQUATION_MISSING
        self.slots.append((node, owner, name, descriptor, value))
        return value

    def unchanged(self):
        _equation_interfaces()
        if (len(self.equations) != len(self.original)
                or any(current is not old for current, old in zip(self.equations, self.original))):
            raise RuntimeError("linear equations mutated during the solve")
        for node, owner, name, descriptor, original in self.slots:
            try:
                current = descriptor.__get__(node, owner)
            except AttributeError:
                current = _EQUATION_MISSING
            if current is not original:
                raise RuntimeError("linear equation source mutated during the solve")
        _check_coordinates(self.coordinates)
        for record in self.symbols.values():
            _check_coordinates(record)
        for record in self.surface_symbols:
            _check_coordinates(record)
        if any(not _is_closed_function_class(cls) for cls in self.functions):
            raise RuntimeError("linear equation coefficient implementation changed")

    def symbol(self, native):
        key = _binding_key(native)
        if key not in self.symbols:
            owner = (self.coordinates[self.columns[key]][0] if key in self.columns
                     else OrderedDict.get(_surface_symbols, key))
            if owner is None:
                raise NotImplementedError("linear equation source has an unowned native symbol")
            records = _coordinates((owner,), 1)
            record = records[0]
            if _REDUCTION_NATIVE_EQUAL(native, record[1]) is not True:
                raise RuntimeError("linear equation symbol differs from its retained owner")
            facts = dict(record[3])
            if (facts.get("commutative") is False or facts.get("finite") is False
                    or facts.get("infinite") is True):
                raise NotImplementedError("linear equations require finite commuting scalar symbols")
            self.symbols[key] = records
        if key in self.columns:
            column = self.columns[key]
            return Fraction(0), {column: Fraction(1)}, frozenset((column,))
        return None, {}, frozenset()

    def combine(self, kind, values):
        self.charge(sum(len(value[1]) + 1 for value in values))
        if kind == "add":
            constant = Fraction(0)
            coefficients = {}
            dependencies = set()
            for scalar, terms, dependent in values:
                dependencies.update(dependent)
                constant = None if scalar is None or constant is None else self.bounded(constant + scalar)
                for column, coefficient in terms.items():
                    value = self.bounded(coefficients.get(column, Fraction(0)) + coefficient)
                    if value:
                        coefficients[column] = value
                    else:
                        coefficients.pop(column, None)
            return constant, coefficients, frozenset(dependencies if constant is None else coefficients)
        if kind == "mul":
            dependent = [value for value in values if value[2]]
            if len(dependent) > 1:
                raise ValueError("linear equation contains a nonlinear product")
            scalar = Fraction(1)
            for value, _, columns in values:
                if not columns:
                    scalar = None if scalar is None or value is None else self.bounded(scalar * value)
            if not dependent:
                return scalar, {}, frozenset()
            constant, terms, columns = dependent[0]
            if scalar is None or constant is None:
                return None, {}, columns
            coefficients = ({column: self.bounded(scalar * value) for column, value in terms.items()}
                            if scalar else {})
            return self.bounded(scalar * constant), coefficients, frozenset(coefficients)
        if kind == "pow":
            base, exponent = values
            scalar, terms, columns = base
            power, _, power_columns = exponent
            if power_columns or power is None:
                raise NotImplementedError("linear equations require literal scalar exponents")
            if power.denominator != 1:
                if columns:
                    raise ValueError("linear equation contains a nonlinear power")
                if scalar is None:
                    raise NotImplementedError("linear equation symbolic powers require a domain proof")
                if scalar == 0 and power < 0:
                    raise ValueError("linear equation contains an undefined original power")
                return None, {}, frozenset()
            power = power.numerator
            if abs(power) > 64:
                raise ValueError("linear equation power exceeds its bound")
            if power < 0 and (columns or scalar is None):
                raise NotImplementedError("linear equation original denominator requires a domain proof")
            if scalar == 0 and not columns and power < 0:
                raise ValueError("linear equation contains an undefined original denominator")
            if power == 0:
                return self.constant(Fraction(1))
            if power == 1:
                return base
            if columns:
                raise ValueError("linear equation contains a nonlinear power")
            if scalar is None:
                return None, {}, frozenset()
            if scalar not in (0, 1, -1) and max(abs(scalar.numerator).bit_length(), scalar.denominator.bit_length()) * abs(power) > _EQUATION_MAX_BITS:
                raise ValueError("linear equation power coefficient exceeds its bit limit")
            return self.constant(scalar ** power)
        if any(value[2] for value in values):
            raise ValueError("linear equation contains a nonlinear function")
        return None, {}, frozenset()

    def native(self, node, depth=0):
        self.charge(depth=depth)
        self.nodes += 1
        if self.nodes > _EQUATION_MAX_NODES:
            raise ValueError("linear equation source exceeds its node limit")
        predicate = lambda name: _EQUATION_NATIVE_MEMBERS[name].__get__(node, _REDUCTION_NATIVE_EXPR)
        if predicate("is_integer") or predicate("is_rational"):
            return self.constant(Fraction(_EXPR_MEMBERS["exact_numerator"](node), _EXPR_MEMBERS["exact_denominator"](node)))
        if predicate("is_symbol"):
            return self.symbol(node)
        kind = next((tag for tag in ("add", "mul", "pow") if predicate("is_" + tag)), None)
        if kind is None:
            if any(_REDUCTION_NATIVE_EQUAL(node, value) is True for value in _EQUATION_NONFINITE_CONSTANTS):
                raise ValueError("linear equations require finite original sources")
            if any(_REDUCTION_NATIVE_EQUAL(node, value) is True for value in _EQUATION_FINITE_CONSTANTS):
                return None, {}, frozenset()
            name = _EXPR_MEMBERS["func_name"].__get__(node, _REDUCTION_NATIVE_EXPR)
            if name == "Constant":
                # All other native constants are nonfinite, including -oo.
                raise ValueError("linear equations require finite original sources")
            if name == _FLOAT_INTERN:
                payload = _EXPR_MEMBERS["args"].__get__(node, _REDUCTION_NATIVE_EXPR)
                if (len(payload) != 1 or not _EQUATION_NATIVE_MEMBERS["is_integer"].__get__(
                        payload[0], _REDUCTION_NATIVE_EXPR)):
                    raise ValueError("linear equation Float has malformed native storage")
                bits = _EXPR_MEMBERS["exact_numerator"](payload[0])
                if not 0 <= bits < 1 << 64:
                    raise ValueError("linear equation Float has malformed native storage")
                if (bits >> 52) & 0x7ff == 0x7ff:
                    raise ValueError("linear equations require finite original sources")
                return None, {}, frozenset()
            if name not in _EQUATION_ENTIRE:
                raise NotImplementedError("linear equation function requires an original domain proof")
            cls = _lookup_known_function(name)
            if cls is None or not _is_closed_function_class(cls):
                raise NotImplementedError("linear equation has no original built-in function owner")
            self.functions.append(cls)
            kind = "coefficient"
        children = _EXPR_MEMBERS["args"].__get__(node, _REDUCTION_NATIVE_EXPR)
        if len(children) > _EQUATION_MAX_NODES - self.nodes:
            raise ValueError("linear equation source exceeds its child limit")
        if kind == "coefficient" and len(children) != 1:
            raise ValueError("linear equation built-in function requires one original argument")
        return self.combine(kind, [self.native(child, depth + 1) for child in children])

    def surface(self, node, depth=0, equation=False):
        self.charge(depth=depth)
        kind = type(node)
        if kind is int:
            if node.bit_length() > _EQUATION_MAX_BITS:
                raise ValueError("linear equation coefficient exceeds 4096 bits")
            return self.constant(Fraction(node))
        if kind is Fraction:
            numerator = self.retain(node, Fraction, "_numerator")
            denominator = self.retain(node, Fraction, "_denominator")
            if type(numerator) is not int or type(denominator) is not int:
                raise ValueError("linear equation Fraction requires original integer storage")
            if max(numerator.bit_length(), denominator.bit_length()) > _EQUATION_MAX_BITS:
                raise ValueError("linear equation coefficient exceeds 4096 bits")
            if denominator <= 0 or gcd(numerator, denominator) != 1:
                raise ValueError("linear equation Fraction requires normalized original storage")
            return self.constant(Fraction(numerator, denominator))
        if kind is float:
            if not isfinite(node):
                raise ValueError("linear equations require finite original sources")
            # Float coefficients retain their existing generic semantics;
            # they never become exact rational admission by conversion.
            return None, {}, frozenset()
        builtin = any(kind is owner for owner in _EQUATION_CLASSES)
        function = not builtin and _is_closed_function_class(kind)
        if not builtin and not function:
            raise NotImplementedError("linear equations require exact built-in source owners")
        identity = id(node)
        if identity in self.active:
            raise ValueError("linear equation contains a cyclic retained source")
        if identity in self.memo:
            return self.memo[identity]
        self.nodes += 1
        if self.nodes > _EQUATION_MAX_NODES:
            raise ValueError("linear equation source exceeds its node limit")
        if kind is Eq and not equation:
            raise ValueError("linear equation cannot contain a nested equation")
        self.active.add(identity)
        native = self.retain(node, Basic, "_value")
        retained = [self.retain(node, Basic, name) for name in ("_struct_args", "_surface_args")]
        operator = next((name for owner, name in ((Add, "add"), (Mul, "mul"), (Pow, "pow"), (Eq, "equation")) if kind is owner), None)
        if operator:
            retained.append(self.retain(node, Relational if kind is Eq else kind,
                                        "_relational_sides" if kind is Eq else "_args"))
        elif function:
            self.functions.append(kind)
            retained.append(self.retain(node, Function, "_args"))
        groups = [value for value in retained if value is not _EQUATION_MISSING and value is not None]
        if any(type(value) is not tuple for value in groups):
            raise ValueError("linear equation requires original argument tuples")
        if groups and any(len(value) != len(groups[0]) or any(left is not right for left, right in zip(value, groups[0])) for value in groups[1:]):
            raise ValueError("linear equation has inconsistent retained arguments")
        if type(native) is not _REDUCTION_NATIVE_EXPR:
            raise NotImplementedError("linear equation requires a closed native source")
        _EQUATION_PREFLIGHT(native)
        if kind is ComplexInfinity:
            raise ValueError("linear equations require finite original sources")
        if operator and operator != "equation":
            if not _EQUATION_NATIVE_MEMBERS["is_" + operator].__get__(native, _REDUCTION_NATIVE_EXPR):
                raise ValueError("linear equation native and retained operators differ")
        if any(kind is owner for owner in _EQUATION_ATOMS):
            predicate = ("is_integer" if any(kind is owner for owner in (Integer, Zero, One, NegativeOne))
                         else "is_rational")
            if not _EQUATION_NATIVE_MEMBERS[predicate].__get__(native, _REDUCTION_NATIVE_EXPR):
                raise ValueError("linear equation rational atom has nonrational native storage")
        if kind is Float and _EXPR_MEMBERS["func_name"].__get__(native, _REDUCTION_NATIVE_EXPR) != _FLOAT_INTERN:
            raise ValueError("linear equation Float has inconsistent native storage")
        if kind is Symbol or kind is Dummy:
            _require_atomic_storage(node)
            # Also validate the supplied owner, not just a registry substitute.
            records = _coordinates((node,), 1)
            self.surface_symbols.append(records)
            result = self.symbol(native)
            _check_coordinates(records)
        elif operator or function:
            children = groups[0] if groups else None
            if children is None:
                if kind is Eq:
                    raise NotImplementedError("linear equations require retained equation sides")
                result = self.native(native, depth)
            else:
                if len(children) > _EQUATION_MAX_NODES - self.nodes:
                    raise ValueError("linear equation retained source exceeds its child limit")
                expected = _EXPR_MEMBERS["args"].__get__(native, _REDUCTION_NATIVE_EXPR)
                if len(expected) != len(children):
                    raise ValueError("linear equation native and retained arguments differ")
                if kind is Eq:
                    if _EXPR_MEMBERS["func_name"].__get__(native, _REDUCTION_NATIVE_EXPR) != "Equality" or len(children) != 2:
                        raise ValueError("linear equation native and retained equation sides differ")
                elif function:
                    name = _EXPR_MEMBERS["func_name"].__get__(native, _REDUCTION_NATIVE_EXPR)
                    if name not in _EQUATION_ENTIRE:
                        raise NotImplementedError("linear equation function requires an original domain proof")
                    if _lookup_known_function(name) is not kind:
                        raise ValueError("linear equation native and retained function owners differ")
                    if len(children) != 1:
                        raise ValueError("linear equation built-in function requires one original argument")
                values = []
                for child, original in zip(children, expected):
                    value = self.surface(child, depth + 1)
                    child_native = _REDUCTION_VALUE_SLOT.__get__(child, Basic)
                    if _REDUCTION_NATIVE_EQUAL(child_native, original) is not True:
                        raise ValueError("linear equation native and retained arguments differ")
                    values.append(value)
                if kind is Eq:
                    negated = self.combine("mul", [self.constant(Fraction(-1)), values[1]])
                    result = self.combine("add", [values[0], negated])
                else:
                    result = self.combine(operator or "coefficient", values)
        else:
            if groups:
                raise ValueError("linear equation atom has non-atomic retained storage")
            result = self.native(native, depth)
        self.active.remove(identity)
        self.memo[identity] = result
        return result


def checked_linear_equations(equations, variables):
    """Return UNSUPPORTED only for safely admitted symbolic coefficients.

    Original denominators involving symbols and unknown function domains are
    terminal refusals, even beneath held zeros or before a symbolic fallback.
    Exact affine extraction is sparse throughout; it never probes values at
    zero/one, expands an expression, or invokes public scalar constructors.
    """
    if type(equations) not in (list, tuple):
        return UNSUPPORTED
    if len(equations) > _EQUATION_MAX_ROWS:
        raise ValueError("linear equation row count exceeds its bound")
    _equation_interfaces()
    try:
        records = _coordinates(variables, len(variables))
    except GeneratorsError as error:
        raise ValueError("duplicate symbols given") from error
    source = _AffineEquationSource(equations, records)
    for record in records:
        source.symbol(record[1])
    rows = [source.surface(value, equation=True) for value in source.original]
    source.unchanged()
    if any(value[0] is None for value in rows):
        return UNSUPPORTED
    entries = {}
    count = sum(len(terms) + bool(constant) for constant, terms, _ in rows)
    if count > _EQUATION_MAX_ENTRIES:
        raise ValueError("linear equation coefficients exceed their stored-entry limit")
    for row, (constant, terms, _) in enumerate(rows):
        for column, value in terms.items():
            entries[row, column] = _EQUATION_RATIONAL(value.numerator, value.denominator)
        if constant:
            entries[row, len(variables)] = _EQUATION_RATIONAL(-constant.numerator, constant.denominator)
    source.unchanged()
    native = _NativeSparseMatrix(len(rows), len(variables) + 1, entries)
    matrix = _checked_sparse_lift(native, SparseMatrix)
    source.unchanged()
    result = checked_linear_system(matrix, variables)
    source.unchanged()
    if result is UNSUPPORTED:
        raise RuntimeError("exact affine equations did not enter the checked sparse solver")
    return result
