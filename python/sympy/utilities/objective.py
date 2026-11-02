"""Checked scalar objectives and sparse second derivatives for numerical use.

The native checker admits both differentiation stages and their compiled
instructions. Evaluation returns approximate finite-real binary64 values;
these are not interval certificates or proofs of convexity.
"""

from ..core import _native, _symbol_binding
from .lambdify import _argument_layout, _call_arguments, _functions, _numeric_source

__all__ = ["compile_objective"]


class _ObjectiveCallable:
    """Argument layout around an immutable, source-bound native objective."""

    __slots__ = ("_layout", "_native_objective", "_variables")

    def __init__(self, layout, native, variables):
        if hasattr(self, "_native_objective"):
            raise AttributeError("compiled objectives are immutable")
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_native_objective", native)
        object.__setattr__(self, "_variables", tuple(variables))

    def __setattr__(self, name, value):
        raise AttributeError("compiled objectives are immutable")

    def __delattr__(self, name):
        raise AttributeError("compiled objectives are immutable")

    @property
    def native_objective(self):
        return self._native_objective

    @property
    def variables(self):
        """Original argument objects in flat gradient/vector coordinate order."""
        return self._variables

    @property
    def hessian_pattern(self):
        """Immutable, ordered (row, column) coordinates of Hessian nonzeros."""
        return tuple(self.native_objective.hessian_pattern)

    def evaluate_sparse(self, *values):
        """Return (value, gradient, sparse Hessian values) after all checks pass.

        Sparse values follow ``hessian_pattern`` in row-major order, retaining
        both mixed-partial entries independently. No dense matrix is allocated.
        """
        inputs = _call_arguments(self._layout, values, native=True)
        return self.native_objective.evaluate(inputs)

    def evaluate(self, *values):
        """Return (value, gradient, dense Hessian rows), all approximate.

        Source and first-derivative domain checks run before any results return.
        The dense output is limited by the native maximum of 64 variables.
        """
        value, gradient, sparse = self.evaluate_sparse(*values)
        size = self.native_objective.variable_count
        hessian = [[0.0] * size for _ in range(size)]
        for (row, column), entry in zip(self.hessian_pattern, sparse):
            hessian[row][column] = entry
        return value, gradient, hessian

    def hessian_vector_product(self, direction, *values):
        """Evaluate H(x) times a flat direction without forming a dense Hessian.

        ``direction`` is an exact list/tuple with one real value per variable
        in ``variables`` order, even when ordinary arguments are nested.
        Invalid dimensions, nonfinite arithmetic and source-domain failures
        raise an exception without publishing partial values.
        """
        size = self.native_objective.variable_count
        if type(direction) not in (list, tuple) or len(direction) != size:
            raise TypeError("Hessian direction must be a list/tuple with one entry per variable")
        vector = _call_arguments((None,) * size, direction, native=True)
        inputs = _call_arguments(self._layout, values, native=True)
        return self.native_objective.hessian_vector_product(inputs, vector)


def compile_objective(objective, args):
    """Compile a scalar objective, checked gradient and checked sparse Hessian.

    ``args`` contains typed Symbols/Dummies, optionally nested in lists/tuples.
    Both differentiation stages are independently replayed, including omitted
    zero entries and compiled instruction provenance. Mixed partials are checked
    in both orders; symmetry is not assumed. The admitted program is immutable.

    ``evaluate`` returns the value, gradient and dense Hessian. Use
    ``evaluate_sparse`` or ``hessian_vector_product`` to avoid dense output.
    These are bounded synchronous finite-real binary64 programs, not certified
    numerical enclosures, positivity/convexity claims or supervised cancellation.
    Unsupported functions, opaque hooks, unmatched symbols and oversized work
    refuse explicitly. No upstream SymPy or Python callback executes the math.
    """
    layout, variables = _argument_layout(args)
    source = _numeric_source(objective, _functions(), [0])
    bindings = [_symbol_binding(variable) for variable in variables]
    native = _native.VerifiedObjective(source, bindings)
    return _ObjectiveCallable(layout, native, variables)
