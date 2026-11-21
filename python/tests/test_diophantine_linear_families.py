"""Complete integer linear families retain independent original coordinates.

Ordinary parameter forms below were checked on explicit held cancellations
in isolated SymPy 1.14.0. FrankenSymPy additionally retains original axes
inside held zero products, as required by its original-source contract.
Coverage tests also use primitive lattice directions and planted finite-index
subfamilies; satisfying the original equation alone is insufficient.
"""

import importlib
from itertools import product
from unittest.mock import patch

import pytest
import sympy as s

from sympy.core import SupervisedCallbackSession, supervised_callback_session


linear = importlib.import_module("sympy.solvers.diophantine")


def _retained(residual, *variables):
    return s.Add(residual, s.Mul(0, s.Add(*variables), evaluate=False), evaluate=False)


def _family(source, variables, **kwargs):
    result = s.diophantine(source, syms=variables, **kwargs)
    assert len(result) == 1
    (family,) = result
    assert len(family) == len(variables)
    return family


@pytest.mark.parametrize("active", range(3))
def test_one_active_coordinate_keeps_two_independent_integer_parameters(active):
    variables = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(3*variables[active] - 6, *variables)
    family = _family(source, variables)
    inactive = [index for index in range(3) if index != active]
    assert family[active] == 2
    assert tuple(family[index] for index in inactive) == (
        s.Symbol("n1", integer=True), s.Symbol("n2", integer=True),
    )
    for first, second in product(range(-3, 4), repeat=2):
        point = tuple(value.subs({family[inactive[0]]: first, family[inactive[1]]: second})
                      for value in family)
        assert point[active] == 2
        assert point[inactive[0]] == first
        assert point[inactive[1]] == second


def test_two_active_coordinates_preserve_the_pinned_formula_and_one_free_axis():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    t = s.Symbol("t_0", integer=True)
    n = s.Symbol("n1", integer=True)
    assert _family(_retained(2*x + 4*y - 6, x, y, z), (x, y, z)) == (3 - 2*t, t, n)
    assert _family(_retained(2*x + 4*y - 6, x, y, z), (z, y, x)) == (n, t, 3 - 2*t)


def test_explicit_held_cancellations_match_the_pinned_parameter_form():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = s.Add(2*x, 4*y, -6, z, -z, evaluate=False)
    t, n = s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True)
    assert _family(source, (x, y, z)) == (3 - 2*t, t, n)


@pytest.mark.parametrize("a,b,c,offsets,directions", [
    (1, 2, -3, (3, 0), (-2, 1)),
    (-1, -2, 3, (3, 0), (-2, 1)),
    (3, 5, -7, (14, -7), (5, -3)),
    (3, 5, 7, (-14, 7), (5, -3)),
    (1, 2, 0, (0, 0), (-2, 1)),
    (-2, 4, -6, (-3, 0), (2, 1)),
    (2, -4, -6, (3, 0), (2, 1)),
    (-2, -4, -6, (-3, 0), (-2, 1)),
    (4, 2, -6, (0, 3), (1, -2)),
])
def test_signed_coefficient_presentations_and_reversed_axes_match_pinned_forms(a, b, c, offsets, directions):
    x, y, z, w = s.symbols("linear_a linear_b linear_c linear_d", integer=True)
    source = s.Add(a*x + b*y + c, z, -z, w, -w, evaluate=False)
    t = s.Symbol("t_0", integer=True)
    n1, n2 = s.symbols("n1 n2", integer=True)
    active = tuple(offset + direction*t for offset, direction in zip(offsets, directions))
    assert _family(source, (x, y, z, w)) == (*active, n1, n2)
    assert _family(source, (w, z, y, x)) == (n2, n1, active[1], active[0])


def test_three_active_coordinates_keep_the_existing_primitive_coefficient_chain():
    x, y, z, w = s.symbols("a b c d", integer=True)
    t0, t1 = s.symbols("t_0 t_1", integer=True)
    n = s.Symbol("n1", integer=True)
    source = _retained(2*x + 3*y + 5*z - 7, x, y, z, w)
    family = _family(source, (x, y, z, w))
    assert family == (t0, t0 + 5*t1 + 14, -t0 - 3*t1 - 7, n)
    # The displayed two active kernel directions have primitive 2x2 minors:
    # gcd(5, -3, -2) == 1. A scaled rational basis can miss such directions.
    for target in ((-1, 3, 0), (1, 0, 1), (2, 1, 0), (7, 1, -2)):
        assert 2*target[0] + 3*target[1] + 5*target[2] == 7
        first = target[0]
        assert (target[1] - first - 14) % 5 == 0
        second = (target[1] - first - 14) // 5
        point = tuple(value.subs({t0: first, t1: second, n: 11}) for value in family)
        assert point == (*target, 11)


@pytest.mark.parametrize("constant", [1, 3, 5, -1])
def test_integer_divisibility_obstruction_is_a_checked_empty_set(constant):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - constant, x, y, z)
    assert s.diophantine(source, syms=(x, y, z)) == set()


def test_explicit_extra_symbols_order_existing_axes_without_adding_unknowns():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = 2*x + 4*y - 6
    assert s.diophantine(source, syms=(x, y, z)) == s.diophantine(source, syms=(x, y))
    assert s.diophantine(source, syms=(z, y, x)) == s.diophantine(source, syms=(y, x))
    assert all(len(family) == 2 for family in s.diophantine(source, syms=(x, y, z)))


@pytest.mark.parametrize("requested", [("linear_x",), ("linear_z",)])
def test_omitted_actual_coordinates_raise_the_pinned_key_error(requested):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    by_name = {variable.name: variable for variable in (x, y, z)}
    with pytest.raises(KeyError):
        s.diophantine(2*x + 4*y - 6, syms=tuple(by_name[name] for name in requested))


def test_omitted_retained_inactive_coordinate_raises_the_pinned_key_error():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    with pytest.raises(KeyError) as caught:
        s.diophantine(_retained(2*x + 4*y - 6, x, y, z), syms=(x, y))
    assert caught.value.args == (z,)


def test_custom_parameter_prefix_is_used_in_the_new_complete_family():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    k = s.Symbol("k")
    k0 = s.Symbol("k_0", integer=True)
    assert _family(_retained(2*x + 4*y - 6, x, y, z), (x, y, z), param=k) == (
        3 - 2*k0, k0, s.Symbol("n1", integer=True),
    )


def test_conventional_parameter_names_cannot_capture_original_coordinates():
    x = s.Symbol("a", integer=True)
    y = s.Symbol("t_0", integer=True)
    z = s.Symbol("n1", integer=True)
    family = _family(_retained(2*x + 4*y - 6, x, y, z), (x, y, z))
    parameters = set().union(*(value.free_symbols for value in family))
    assert len(parameters) == 2
    assert not parameters.intersection((x, y, z))
    assert all(parameter.is_integer is True for parameter in parameters)
    assert type(family[1]) is s.Dummy
    assert type(family[2]) is s.Dummy
    assert family[0].subs(family[1], -3) == 9


def test_same_printed_name_dummies_remain_distinct_coordinates():
    x, y = s.Dummy("coordinate", integer=True), s.Dummy("coordinate", integer=True)
    z = s.Symbol("coordinate", integer=True)
    family = _family(_retained(2*x + 4*y - 6, x, y, z), (z, x, y))
    n, t = s.Symbol("n1", integer=True), s.Symbol("t_0", integer=True)
    assert family == (n, 3 - 2*t, t)


def test_always_nonzero_original_guards_admit_complete_linear_families():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    pole = s.Pow((x*x + 1)*(z*z + 1), -1, evaluate=False)
    source = s.Eq(pole, s.Add(pole, y - 1, evaluate=False), evaluate=False)
    assert _family(source, (x, y, z)) == (
        s.Symbol("n1", integer=True), 1, s.Symbol("n2", integer=True),
    )


def test_unresolved_original_poles_remain_terminal_for_infinite_families():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    pole = s.Pow(z, -1, evaluate=False)
    source = s.Eq(pole, s.Add(pole, 2*x + 4*y - 6, evaluate=False), evaluate=False)
    with pytest.raises(NotImplementedError, match="unresolved parameter conditions"):
        s.diophantine(source, syms=(x, y, z))


def test_residual_zero_index_two_subfamily_is_rejected_as_incomplete():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)

    def even_parameter_subfamily(_coeff, _variables, _const, parameters=None):
        t = parameters[0]
        # Every proposed point solves 2*x+4*y=6, but y is forced even.
        return {(3 - 4*t, 2*t)}

    with patch.object(linear, "_diop_linear", side_effect=even_parameter_subfamily):
        with pytest.raises(ArithmeticError, match="omits integer solutions"):
            s.diophantine(_retained(2*x + 4*y - 6, x, y, z), syms=(x, y, z))


def test_wrong_linear_residual_is_rejected_before_coverage_publication():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)

    def wrong_offset(_coeff, _variables, _const, parameters=None):
        t = parameters[0]
        return {(4 - 2*t, t)}

    with patch.object(linear, "_diop_linear", side_effect=wrong_offset):
        with pytest.raises(ArithmeticError, match="fails the original integer equation"):
            s.diophantine(_retained(2*x + 4*y - 6, x, y, z), syms=(x, y, z))


@pytest.mark.parametrize("singleton", [False, True])
def test_literal_integer_candidate_payload_must_match_its_exact_owner(singleton):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    candidate = object.__new__(type(s.Integer(1)) if singleton else s.Integer)
    payload = s.Integer(2) if singleton else s.Rational(2, 3)
    slot = s.Basic.__dict__["_value"]
    slot.__set__(candidate, slot.__get__(payload, s.Basic))
    with patch.object(linear, "_diop_linear", return_value={(candidate,)}):
        with pytest.raises(ValueError, match="literal integer"):
            s.diophantine(_retained(3*x - 6, x, y, z), syms=(x, y, z))


def test_retained_generator_expression_is_rebuilt_from_the_verified_rows():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._diop_linear
    candidates = []
    slot = s.Basic.__dict__["_value"]

    def stale_native_payload(*args, **kwargs):
        result = original(*args, **kwargs)
        candidate, = result
        assert type(candidate[0]) is s.Add
        first = s.Add(*candidate[0].args, evaluate=False)
        candidates.append(first)
        slot.__set__(first, slot.__get__(s.Integer(999), s.Basic))
        return {(first, *candidate[1:])}

    with patch.object(linear, "_diop_linear", side_effect=stale_native_payload):
        family = _family(source, (x, y, z))
    t, n = s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True)
    assert family == (3 - 2*t, t, n)
    assert family[0] is not candidates[0]
    assert s.expand(2*family[0] + 4*family[1] - 6) == 0


def test_integer_lattice_reads_do_not_call_public_matrix_accessors():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    t, n = s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True)

    def forbidden(*_args):
        raise AssertionError("public matrix accessor ran during checked lattice extraction")

    with patch.object(s.Matrix, "shape", property(forbidden)), patch.object(s.Matrix, "__getitem__", forbidden):
        assert _family(source, (x, y, z)) == (3 - 2*t, t, n)


@pytest.mark.parametrize("owner", [s.Integer, type(s.Integer(0)), type(s.Integer(1)), type(s.Integer(-1))])
def test_replaced_integer_conversion_is_refused_without_invoking_it(owner):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    called = []

    def forbidden(_self):
        called.append(True)
        return 0

    with patch.object(owner, "__int__", forbidden):
        with pytest.raises(NotImplementedError):
            s.diophantine(source, syms=(x, y, z))
    assert called == []


@pytest.mark.parametrize("slot_name", ["_args", "_value"])
def test_original_held_source_mutation_during_generation_is_terminal(slot_name):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._diop_linear
    owner = s.Add if slot_name == "_args" else s.Basic
    slot = owner.__dict__[slot_name]
    before = slot.__get__(source, owner)

    def mutate_source(*args, **kwargs):
        result = original(*args, **kwargs)
        slot.__set__(source, (s.Integer(1),) if slot_name == "_args" else s.Integer(1)._value)
        return result

    try:
        with patch.object(linear, "_diop_linear", side_effect=mutate_source):
            with pytest.raises(RuntimeError, match="original source mutated"):
                s.diophantine(source, syms=(x, y, z))
    finally:
        slot.__set__(source, before)


def test_original_typed_symbol_mutation_during_generation_is_terminal():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._diop_linear
    declarations = s.Symbol.__dict__["_assumptions"].__get__(z, s.Symbol)
    before = dict(declarations)

    def mutate_source(*args, **kwargs):
        result = original(*args, **kwargs)
        declarations["integer"] = False
        return result

    try:
        with patch.object(linear, "_diop_linear", side_effect=mutate_source):
            with pytest.raises(RuntimeError, match="original symbol assumptions mutated"):
                s.diophantine(source, syms=(x, y, z))
    finally:
        declarations.clear()
        declarations.update(before)


def test_generated_integer_parameter_mutation_cannot_publish_a_false_domain():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._diop_linear
    changed = []

    def mutate_parameter(*args, **kwargs):
        result = original(*args, **kwargs)
        parameter = args[3][0]
        declarations = s.Symbol.__dict__["_assumptions"].__get__(parameter, s.Symbol)
        changed.append((declarations, dict(declarations)))
        declarations["integer"] = False
        return result

    try:
        with patch.object(linear, "_diop_linear", side_effect=mutate_parameter):
            with pytest.raises(RuntimeError, match="symbol assumptions mutated"):
                s.diophantine(source, syms=(x, y, z))
    finally:
        for declarations, before in changed:
            declarations.clear()
            declarations.update(before)


@pytest.mark.parametrize("late", [False, True])
def test_checked_family_storage_cannot_change_before_publication(late):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    slot = s.Basic.__dict__["_value"]
    changes = []

    def corrupt(expression):
        changes.append((expression, slot.__get__(expression, s.Basic)))
        slot.__set__(expression, slot.__get__(s.Integer(999), s.Basic))

    if late:
        target = "_check_diophantine_domains"
        original = getattr(linear, target)

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            family, = result
            corrupt(family[0])
            return result
    else:
        target = "_linear_family_coefficients"
        original = getattr(linear, target)

        def changed(expression, *args, **kwargs):
            result = original(expression, *args, **kwargs)
            if type(expression) is s.Add:
                corrupt(expression)
            return result

    try:
        with patch.object(linear, target, side_effect=changed):
            with pytest.raises(RuntimeError, match="source mutated"):
                s.diophantine(source, syms=(x, y, z))
    finally:
        for expression, value in changes:
            slot.__set__(expression, value)


def test_replacing_the_active_request_during_generation_is_terminal():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._diop_linear
    replacement = supervised_callback_session(max_nested_steps=10_000_000)
    entered = []

    def change_request(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement.__enter__()
        entered.append(True)
        return result

    with supervised_callback_session(max_nested_steps=10_000_000):
        try:
            with patch.object(linear, "_diop_linear", side_effect=change_request):
                with pytest.raises(RuntimeError, match="request changed"):
                    s.diophantine(source, syms=(x, y, z))
        finally:
            if entered:
                replacement.__exit__(None, None, None)


def test_dimension_and_original_coefficient_limits_refuse_before_publication():
    variables = tuple(s.Symbol("linear_bound_%d" % index, integer=True) for index in range(65))
    with pytest.raises(NotImplementedError, match="bounded exact symbol coordinates"):
        s.diophantine(_retained(variables[0] - 1, *variables), syms=variables)
    x, y, z = variables[:3]
    with pytest.raises(NotImplementedError):
        s.diophantine(_retained((1 << 4096)*x + y - 1, x, y, z), syms=(x, y, z))


@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_active_requests_never_publish_an_integer_family(cancelled):
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    session = SupervisedCallbackSession(max_nested_steps=10_000_000 if cancelled else 0)
    if cancelled:
        session.cancel()
    with supervised_callback_session(session):
        with pytest.raises((ValueError, RuntimeError), match="budget|cancelled|tainted|limit"):
            s.diophantine(source, syms=(x, y, z))


def test_source_generation_verification_and_publication_share_the_request():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    with supervised_callback_session(max_nested_steps=10_000_000) as measured:
        _family(source, (x, y, z))
    cost = measured.consumed_nested_steps
    assert cost > 0
    with supervised_callback_session(max_nested_steps=cost - 1):
        with pytest.raises((ValueError, RuntimeError), match="budget|cancelled|tainted|limit"):
            s.diophantine(source, syms=(x, y, z))


def test_late_request_cancellation_stops_before_family_publication():
    x, y, z = s.symbols("linear_x linear_y linear_z", integer=True)
    source = _retained(2*x + 4*y - 6, x, y, z)
    original = linear._linear_family_coefficients
    with supervised_callback_session(max_nested_steps=10_000_000) as session:
        def cancelled_after_coefficients(*args):
            result = original(*args)
            session.cancel()
            return result

        with patch.object(linear, "_linear_family_coefficients", side_effect=cancelled_after_coefficients):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                s.diophantine(source, syms=(x, y, z))

