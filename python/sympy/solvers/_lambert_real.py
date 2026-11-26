"""Complete bounded real roots of mixed affine/exponential equations.

The independent native verifier binds the entire original expression to
A*x+B+(C*x+D)*exp(E*x+F), with rational coefficients and one nonzero A/C.
It decides every real Lambert-W branch with exact or certified comparisons.
This boundary retains the original shell owners through result publication.
"""

from ..core import (
    Dummy, Eq, Integer, Symbol, _active_callback_session,
    _charge_nested_if_active, _lift_native_result, _native, _native_expr,
    _require_closed_function_class, _symbol_binding,
)
from ..functions.elementary.exponential import LambertW
from ..sets import FiniteSet, Intersection, Reals, Union
from ._elementary_real import _ElementaryCapture
from ._real_domain import _Domain, _SET_TYPES, _is_exact


_SOLVE = _native.solve_real_lambert_native
_ADD, _MUL, _INTEGER = _native.Add, _native.Mul, _native.py_integer
_CLASS_DICT = type.__dict__["__dict__"]
_CONSTRUCTORS = tuple(
    (name, cls, tuple(_CLASS_DICT.__get__(cls).get(member)
                      for member in ("__new__", "__init__", "as_expr")))
    for name, cls in (("Add", _ADD), ("Mul", _MUL))
)
_ADD_EXPR, _MUL_EXPR = (_CONSTRUCTORS[0][2][2], _CONSTRUCTORS[1][2][2])


def _interfaces():
    if (_native.solve_real_lambert_native is not _SOLVE
            or _native.py_integer is not _INTEGER):
        raise NotImplementedError("real Lambert solving requires original native interfaces")
    for name, cls, recorded in _CONSTRUCTORS:
        namespace = _CLASS_DICT.__get__(cls)
        if (getattr(_native, name) is not cls
                or any(namespace.get(member) is not expected
                       for member, expected in zip(("__new__", "__init__", "as_expr"), recorded))):
            raise NotImplementedError("real Lambert solving requires original native interfaces")


def solve_lambert_real(expression, variable, domain):
    """Select this source grammar before arithmetic; never mask its failures."""
    if not (_is_exact(domain, _SET_TYPES) or type(domain) is Reals):
        return None
    _interfaces()
    capture = _ElementaryCapture()
    sides = capture.sides(expression) if type(expression) is Eq else (expression, Integer(0))
    originals = tuple(_native_expr(side) for side in sides)
    if not any(native.has_function_bounded("exp") for native in originals):
        return None
    _charge_nested_if_active()
    for native in originals:
        _native.real_constraint_source_preflight_native(native)
    for side in sides:
        capture.read(side)
    if variable is None:
        if len(capture.symbols) != 1:
            return None
        variable = capture.symbols[0]
    if type(variable) is not Symbol and type(variable) is not Dummy:
        raise TypeError("real Lambert solving requires an exact Symbol or Dummy")
    capture.symbol(variable)
    if not capture.supported or any(symbol != variable for symbol in capture.symbols):
        return None
    snapshot = _Domain(domain)
    if not snapshot.real:
        return None
    session = _active_callback_session()

    def unchanged():
        capture.unchanged()
        snapshot.unchanged()
        _require_closed_function_class(LambertW)
        _interfaces()
        if _active_callback_session() is not session:
            raise RuntimeError("real Lambert solving request changed during evaluation")

    unchanged()
    # Preserve both original sides, including held zero products and powers.
    # The native verifier admits the complete tree before any normalization.
    source = _ADD_EXPR(_ADD(
        originals[0],
        _MUL_EXPR(_MUL(_INTEGER(-1), originals[1], evaluate=False)),
        evaluate=False,
    ))
    roots = _SOLVE(source, _symbol_binding(variable), session=session)
    unchanged()
    if roots is None:
        return None
    if (type(roots) is not list or len(roots) > 2
            or any(type(root) is not _native.Expr for root in roots)):
        raise RuntimeError("real Lambert solving returned invalid checked roots")
    _charge_nested_if_active(len(roots) + 1)
    known, uncertain = [], []
    for native in roots:
        root = _lift_native_result(native, *sides, variable)
        membership = True if type(domain) is Reals else snapshot.contains(snapshot.scalar(native))
        if membership is not False:
            (known if membership is True else uncertain).append(root)
    result = FiniteSet(*known)
    if uncertain:
        result = Union(result, Intersection(domain, FiniteSet(*uncertain)))
    unchanged()
    _charge_nested_if_active()
    return result
