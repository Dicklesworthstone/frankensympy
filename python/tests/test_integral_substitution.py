"""Integral substitution and primitive evaluation pinned to SymPy 1.14.0.

Public substitution intentionally preserves the selected Python profile's
capture behavior. Native capture-avoiding substitution is a separate API.
"""

import copy
import pickle

import pytest

from sympy import (
    Add, Dummy, Function, Integral, Mul, Pow, Rational, Symbol, diff, integrate, nan,
    oo, symbols, zoo,
)


def test_indefinite_renaming_reaches_only_the_free_coordinate():
    x, y, z = symbols("x y z")
    assert Integral(x*x, x).subs(x, z) == Integral(z*z, z)
    assert Integral(1, x).subs(x, z) == Integral(1, z)
    assert Integral(x*y, x, x).subs(x, z) == Integral(z*y, z, z)
    assert Integral(x*y, x, y).subs(x, z) == Integral(z*y, z, y)
    assert Integral(x*y, (x, 0, 1), x).subs(x, z) == Integral(x*y, (x, 0, 1), z)


def test_outer_definite_scopes_protect_inner_limits_and_body():
    x, y, z = symbols("x y z")
    value = Integral(x*y, (x, 0, y), (y, 0, z))
    for old, new in ((x, 2), (y, 2), (y, x)):
        assert value.subs(old, new) == value
    assert value.subs(z, 2) == Integral(x*y, (x, 0, y), (y, 0, 2))
    endpoint = Integral(x*x, (x, 0, x))
    assert endpoint.subs(x, y) == Integral(x*x, (x, 0, y))
    assert endpoint.subs(x, 2) == Integral(x*x, (x, 0, 2))


def test_public_substitution_preserves_profile_capture_behavior():
    x, y = symbols("x y")
    source = Integral(x*y, (x, 0, y))
    assert source.subs(y, x) == Integral(x*x, (x, 0, x))
    assert source.subs(y, 2) == Integral(2*x, (x, 0, 2))
    assert (source + y).subs(y, x) == Integral(x*x, (x, 0, x)) + x


def test_compound_matches_respect_integration_coordinates():
    x, y, z = symbols("x y z")
    for value in (Integral(x*y, x), Integral(x*y, (x, 0, 1))):
        assert value.subs(x*y, z) == value
    assert Integral(x*y, (x, 0, x*y)).subs(x*y, z) == Integral(x*y, (x, 0, z))
    assert Integral(x + y*y, (x, 0, 1)).subs(y*y, z) == Integral(x + z, (x, 0, 1))
    # A prefix of a flat limit list is not an actual source subtree.
    value = Integral(x, (x, 1, 3), (y, 1, 2))
    assert value.subs(Integral(x, (x, 1, 3)), z) == value


def test_unknown_function_replacement_retains_declared_dummy_dependencies():
    x, y = symbols("x y")
    f = Function("integrand_law")
    value = Integral(f(x) + y, (x, 0, 1))
    assert value.subs(f(x), 0) == Integral(y, (x, 0, 1))
    assert value.subs(f(x), x*x) == Integral(x*x + y, (x, 0, 1))
    for application in (f(y), f(x + 1)):
        with pytest.raises(ValueError, match="substitution cannot create dummy dependencies"):
            Integral(application, (x, 0, 1)).subs(application, x)


def test_redundant_evaluation_limits_simplify_beneath_unchanged_outer_scope():
    x, y, z = symbols("x y z")
    value = Integral(x*y, (x, x), (y, 0, 1))
    expected = Integral(x*y, x, (y, 0, 1))
    assert value.subs(y, z) == expected
    assert value.subs(z, 2) == expected


def test_replacing_a_coordinate_by_a_value_creates_an_evaluation_limit():
    x, y = symbols("x y")
    assert Integral(x*x, x).subs(x, 2) == Integral(x*x, (x, 2))
    assert Integral(x*y, x).subs(x, y + 1) == Integral(x*y, (x, y + 1))
    assert Integral(x*y, x, x).subs(x, 2) == Integral(x*y, x, (x, 2))
    assert Integral(x*y, x, (y, 0, x)).subs(x, 2) == Integral(x*y, (x, 2), (y, 0, 2))
    assert Integral(x*y, (x, y)).subs(y, x) == Integral(x*x, x)


@pytest.mark.parametrize("point_kind", ["constant", "parameter", "coordinate", "shifted"])
def test_evaluation_limits_compute_primitive_values(point_kind):
    x, y = symbols("x y")
    point, expected = {
        "constant": (2, 2*y),
        "parameter": (y, y**3/2),
        "coordinate": (x, x*x*y/2),
        "shifted": (x + 1, y*(x + 1)**2/2),
    }[point_kind]
    value = Integral(x*y, (x, point))
    assert value.bound_symbols == [x]
    assert value.free_symbols == ({y} if point_kind in ("constant", "parameter") else {x, y})
    assert value.doit() == expected
    assert integrate(x*y, (x, point)) == expected


def test_evaluation_limits_compose_with_definite_limits():
    x, y = symbols("x y")
    assert Integral(x*y, (x, 2), (y, 0, 3)).doit() == 9
    assert Integral(x*y, (x, 0, 2), (y, 3)).doit() == 9
    assert Integral(x*x, (x, 2)).doit() == Rational(8, 3)


def test_unknown_primitive_retains_its_evaluation_point():
    x, y = symbols("x y")
    f = Function("unevaluated_integrand")
    value = Integral(f(x), (x, y))
    assert value.doit() == value
    assert integrate(f(x), (x, y)) == value


def test_evaluated_primitives_follow_the_chain_and_parameter_rules():
    x, y = symbols("x y")
    assert diff(Integral(x*y, (x, 2)), x) == 0
    assert diff(Integral(x*y, (x, 2)), y) == Integral(x, (x, 2))
    assert diff(Integral(x*y, (x, y)), y) == y*y + Integral(x, (x, y))
    assert diff(Integral(x*y, (x, x)), x) == x*y
    assert diff(Integral(x*y, (x, 2*y)), y) == 4*y*y + Integral(x, (x, 2*y))


@pytest.mark.parametrize("typed", ["assumptions", "dummy"])
def test_same_name_symbols_keep_bound_and_free_ownership(typed):
    if typed == "assumptions":
        bound, free = Symbol("coordinate", positive=True), Symbol("coordinate")
    else:
        bound, free = Dummy("coordinate"), Dummy("coordinate")
    value = Integral(bound + free, (bound, 0, 1))
    assert value.subs(bound, 3) == value
    assert value.subs(free, 3) == Integral(bound + 3, (bound, 0, 1))
    renamed = Integral(bound + free, bound).subs(bound, free)
    assert renamed == Integral(2*free, free)


def test_custom_integrands_preserve_their_python_class_and_scope():
    x, y, z = symbols("x y z")

    class Law(Function):
        nargs = 1

    application = Law(x)
    value = Integral(application + y, (x, 0, y))
    assert value.subs(x, z) == value
    changed = value.subs(y, z)
    assert changed == Integral(application + z, (x, 0, z))
    assert next(iter(changed.function.atoms(Law))) is application
    assert Integral(application, x).subs(x, z) == Integral(Law(z), z)
    assert Integral(application, x).subs(x, 2) == Integral(application, (x, 2))


def test_held_integrand_shallow_copy_keeps_ordered_arguments():
    x, y = symbols("x y")
    held = Add(x, x, evaluate=False)
    value = Integral(held, (x, y))
    restored = copy.copy(value)
    assert restored.function.args == (x, x)
    assert restored.limits == ((x, y),)


def test_evaluation_limit_survives_copy_and_pickle():
    x, y = symbols("x y")
    value = Integral(x*y, (x, y))
    for restored in (copy.copy(value), copy.deepcopy(value), pickle.loads(pickle.dumps(value))):
        assert restored == value
        assert restored.free_symbols == {y}
        assert restored.bound_symbols == [x]


def test_simultaneous_substitution_preserves_primitive_evaluation():
    x, y, z = symbols("x y z")
    value = Integral(x*y, x)
    assert value.subs({x: y, y: x}, simultaneous=True) == Integral(x*x, (x, y))
    assert value.subs({x: 0, y: 0}, simultaneous=True) == Integral(0, (x, 0))
    assert value.subs({x: y + 1, y: z}, simultaneous=True) == Integral(x*z, (x, y + 1))
    assert Integral(x*y, x, x).subs({x: 2, y: x}, simultaneous=True) == Integral(x*x, x, (x, 2))
    nested = Integral(x*y, (x, 0, y), y)
    assert nested.subs({x: y, y: x}, simultaneous=True) == Integral(x*y, (x, 0, y), (y, x))


def test_simultaneous_arithmetic_does_not_collapse_before_restoration():
    x, y, z = symbols("x y z")
    assert (x/y).subs({x: 0, y: 0}, simultaneous=True) is nan
    assert (x/y).subs({x: y, y: x}, simultaneous=True) == y/x
    assert (x + y).subs({x: y + 1, y: z}, simultaneous=True) == y + z + 1
    assert (x*y).subs({x: y + 1, y: z}, simultaneous=True) == z*(y + 1)


def test_evaluated_literal_products_preserve_nan_and_held_source_identity():
    x = Symbol("literal_product_coordinate")
    for nonfinite in (oo, -oo, zoo, nan):
        assert Mul(0, nonfinite) is nan
        assert nonfinite*0 is nan
        assert 0*nonfinite is nan
    assert Mul(2, nan) is nan
    assert Mul(0, zoo, evaluate=False).args == (0, zoo)
    reciprocal = Pow(x, -1, evaluate=False)
    assert Mul(0, reciprocal, evaluate=False).args == (0, reciprocal)


def test_preview_native_literal_products_and_nan_lifting_keep_exact_identity():
    from sympy.core import _native, _wrap

    assert _wrap(_native.Expr("nan")) is nan
    # The native bridge retains a Rational variant for zero. Both literal
    # variants must fold without exporting arbitrary numerators to Python.
    raw_zero = _native.py_rational(0, 3)
    assert raw_zero.is_rational
    assert Mul(raw_zero, zoo) is nan
    assert _wrap(raw_zero)*zoo is nan


def test_preview_eager_integral_construction_uses_point_evaluation():
    # Existing FrankenSymPy convenience, deliberately outside the oracle API.
    x, y = symbols("x y")
    assert Integral(x*y, (x, 2), evaluate=True) == 2*y
