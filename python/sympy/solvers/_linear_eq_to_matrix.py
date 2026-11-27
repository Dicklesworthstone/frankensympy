"""Structural extraction of an affine system in an ordered set of coordinates."""

from fractions import Fraction

from ..core import Basic, Float, Rational, Tuple as SymTuple, _native, _wrap
from ..matrices import Matrix
from ..matrices.dense import _matrix_dimensions
from ..matrices.sparse import _capture_sources, _check_sources
from ..polys.polyerrors import GeneratorsError
from ._sparse_linear import (
    _AffineEquationSource, _EQUATION_ENTIRE, _EQUATION_MAX_ROWS,
    _MATRIX_CLASSES, _coordinates, _equation_interfaces,
)


class NonlinearError(ValueError):
    """An original term is nonlinear in the requested coordinates."""


def _scalar(value):
    if type(value) is Fraction:
        return Rational(value.numerator, value.denominator)
    if type(value) is _native.Expr:
        return _wrap(value)
    if type(value) is float:
        return Float(value)
    return value


class _MatrixCoefficients(_AffineEquationSource):
    # Independent parameters may appear in functions and denominators. This
    # API extracts coefficients; the solver retains its separate domain gate.
    functions_allowed = _EQUATION_ENTIRE | {"log", "tan", "cot", "sec", "csc", "tanh"}

    def symbolic_scalar(self, node):
        return _scalar(node), {}, frozenset()

    def add(self, left, right):
        if type(left) is Fraction and type(right) is Fraction:
            return self.bounded(left + right)
        return _scalar(left) + _scalar(right)

    def multiply(self, left, right):
        if type(left) is Fraction and type(right) is Fraction:
            return self.bounded(left * right)
        return _scalar(left) * _scalar(right)

    def combine(self, kind, values, node=None):
        self.charge(sum(len(terms) + 1 for _, terms, _ in values))
        if kind == "add":
            constant = Fraction(0)
            terms = {}
            for offset, coefficients, _ in values:
                constant = self.add(constant, offset)
                for index, value in coefficients.items():
                    terms[index] = self.add(terms[index], value) if index in terms else value
            return constant, terms, frozenset(terms)
        if kind == "mul":
            factor = Fraction(1)
            affine = None
            for offset, terms, _ in values:
                if terms:
                    if affine is not None:
                        raise NonlinearError("nonlinear cross-term in original linear equation")
                    affine = offset, terms
                else:
                    factor = self.multiply(factor, offset)
            if affine is None:
                return factor, {}, frozenset()
            offset, terms = affine
            return (self.multiply(factor, offset),
                    {index: self.multiply(factor, value) for index, value in terms.items()},
                    frozenset(terms))
        if any(columns for _, _, columns in values):
            raise NonlinearError("nonlinear term in original linear equation")
        # Keep the parameter expression itself, including symbolic divisors.
        # Never sample or cancel it to decide whether it involves a coordinate.
        if kind == "pow" and all(type(v[0]) is Fraction for v in values):
            exact = super().combine(kind, values, node)
            if exact[0] is not None:
                return exact
        return self.symbolic_scalar(node)


def linear_eq_to_matrix(equations, *symbols):
    """Return mutable matrices (A, b) for the equations A*X = b.

    The order of the supplied Symbol or Dummy coordinates determines the
    columns. A scalar equation, ordered list/tuple, or built-in matrix is
    accepted. Matrix cells supply equations in row-major order. Both sides
    of an equality must separately be affine in the requested coordinates.

    Extraction walks the original arithmetic, preserving symbolic parameter
    coefficients. Coordinate-dependent powers, products of dependent factors,
    and nonlinear function applications raise NonlinearError. The bounded
    shell lane admits finite commuting scalars and selected unary built-ins.
    """
    if not symbols:
        raise ValueError("Symbols must be given for linear coefficient extraction")
    if len(symbols) == 1 and type(symbols[0]) in (list, tuple, SymTuple):
        seq = symbols[0]
        symbols = seq.args if type(seq) is SymTuple else seq
    if symbols and type(symbols[0]) in (set, frozenset):
        raise TypeError("unordered coordinates are unsupported")

    matrices = (equations,) if type(equations) in _MATRIX_CLASSES else ()
    owners = _capture_sources(*matrices)
    if matrices:
        count = equations.rows * equations.cols
    elif type(equations) in (list, tuple, SymTuple):
        equations = equations.args if type(equations) is SymTuple else equations
        count = len(equations)
    elif isinstance(equations, Basic) or type(equations) in (int, float, Fraction):
        equations = (equations,)
        count = 1
    else:
        raise ValueError("equations must be expressions, equalities, or an ordered sequence")
    if count > _EQUATION_MAX_ROWS:
        raise ValueError("linear equation row count exceeds its bound")
    _matrix_dimensions(count, len(symbols))
    _equation_interfaces()
    try:
        records = _coordinates(symbols, len(symbols))
    except GeneratorsError as exc:
        raise ValueError("Symbols must be unique") from exc

    if matrices:
        equations = tuple(equations[divmod(i, equations.cols)] for i in range(count))
    source = _MatrixCoefficients(equations, records)
    for record in records:
        source.symbol(record[1])
    rows = [source.surface(eq, equation=True) for eq in source.original]
    source.unchanged()
    zero = Rational(0, 1)
    entries = [zero] * (count * len(records))
    rhs = []
    for row, (constant, terms, _) in enumerate(rows):
        for column, value in terms.items():
            entries[row * len(records) + column] = _scalar(value)
        rhs.append(-_scalar(constant))
    result = Matrix(count, len(records), entries), Matrix(count, 1, rhs)
    source.unchanged()
    _check_sources(owners)
    return result
