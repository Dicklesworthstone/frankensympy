"""CSE exactness regressions against the actual FrankenSymPy shell.

The polynomial reference below uses only rational coefficient arithmetic and
the retained surface children. It neither expands through the production CAS
nor uses CSE's matching, counters, graph representation or substitution code.
"""

from __future__ import annotations

from fractions import Fraction
import importlib
from itertools import repeat

import pytest
import sympy as s

cse_module = importlib.import_module("sympy.simplify.cse_main")


def _polynomial(expression, variables, replacements=()):
    definitions = dict(replacements)
    zero = (0,) * len(variables)
    positions = {value: index for index, value in enumerate(variables)}
    memo, active = {}, set()

    def add(left, right):
        result = dict(left)
        for powers, coefficient in right.items():
            result[powers] = result.get(powers, Fraction()) + coefficient
        return {powers: coefficient for powers, coefficient in result.items() if coefficient}

    def multiply(left, right):
        result = {}
        for a, ca in left.items():
            for b, cb in right.items():
                powers = tuple(x + y for x, y in zip(a, b))
                result[powers] = result.get(powers, Fraction()) + ca * cb
        return {powers: coefficient for powers, coefficient in result.items() if coefficient}

    def visit(value):
        identity = id(value)
        if identity in active:
            raise AssertionError("cyclic CSE definition")
        if identity in memo:
            return memo[identity][1]
        active.add(identity)
        if value in definitions:
            result = visit(definitions[value])
        elif value in positions:
            powers = [0] * len(variables)
            powers[positions[value]] = 1
            result = {tuple(powers): Fraction(1)}
        elif isinstance(value, s.Rational):
            coefficient = Fraction(int(value.p), int(value.q))
            result = {zero: coefficient} if coefficient else {}
        elif type(value) is s.Add:
            result = {}
            for child in value.args:
                result = add(result, visit(child))
        elif type(value) is s.Mul:
            result = {zero: Fraction(1)}
            for child in value.args:
                result = multiply(result, visit(child))
        elif type(value) is s.Pow:
            base, exponent = value.args
            assert isinstance(exponent, s.Integer) and 0 <= exponent.p <= 32
            result = {zero: Fraction(1)}
            factor = visit(base)
            for _ in range(int(exponent.p)):
                result = multiply(result, factor)
        else:
            raise AssertionError("reference expects an exact polynomial")
        active.remove(identity)
        memo[identity] = (value, result)
        return result

    return visit(expression)


def _check_polynomials(original, replacements, reduced, variables):
    assert len(original) == len(reduced)
    names = [symbol.name for symbol, _ in replacements]
    assert len(names) == len(set(names))
    remaining = {symbol for symbol, _ in replacements}
    for symbol, value in replacements:
        assert not (value.free_symbols & remaining), "definition refers to itself or a later temporary"
        remaining.remove(symbol)
    for before, after in zip(original, reduced):
        assert _polynomial(before, variables) == _polynomial(after, variables, replacements)


@pytest.mark.parametrize("operator", [s.Add, s.Mul])
def test_single_shared_operand_does_not_drop_held_multiplicity(operator):
    x, y, z = s.symbols("cse_multiplicity_x cse_multiplicity_y cse_multiplicity_z")
    original = [operator(x, x, y, evaluate=False), operator(x, z, evaluate=False)]
    replacements, reduced = s.cse(original)
    assert replacements == []
    assert all(before is after for before, after in zip(original, reduced))
    _check_polynomials(original, replacements, reduced, (x, y, z))


@pytest.mark.parametrize("operator", [s.Add, s.Mul])
@pytest.mark.parametrize("left_count", [1, 2, 3, 4])
@pytest.mark.parametrize("right_count", [1, 2, 3, 4])
def test_common_multiset_removes_exactly_the_shared_count(operator, left_count, right_count):
    x, y, z = s.symbols("cse_count_x cse_count_y cse_count_z")
    original = [
        operator(*([x] * left_count), y, evaluate=False),
        operator(*([x] * right_count), z, evaluate=False),
    ]
    replacements, reduced = s.cse(original)
    _check_polynomials(original, replacements, reduced, (x, y, z))


@pytest.mark.parametrize("operator", [s.Add, s.Mul])
def test_overlapping_multisets_preserve_all_residual_operands(operator):
    x, y, z, w = s.symbols("cse_overlap_x cse_overlap_y cse_overlap_z cse_overlap_w")
    original = [
        operator(x, x, y, z, evaluate=False),
        operator(x, y, y, z, evaluate=False),
        operator(x, y, z, w, evaluate=False),
    ]
    replacements, reduced = s.cse(original)
    assert replacements
    _check_polynomials(original, replacements, reduced, (x, y, z, w))


def test_common_held_denominator_is_never_folded_away():
    x, y, z = s.symbols("cse_domain_x cse_domain_y cse_domain_z")
    reciprocal = s.Pow(x, -1, evaluate=False)
    original = [
        s.Mul(0, reciprocal, y, evaluate=False),
        s.Mul(0, reciprocal, z, evaluate=False),
    ]
    replacements, reduced = s.cse(original)
    assert len(replacements) == 1
    symbol, value = replacements[0]
    assert type(value) is s.Mul
    assert value.args.count(s.Integer(0)) == 1
    assert value.args.count(reciprocal) == 1
    assert all(symbol in result.args for result in reduced)


def _unfold(replacements, reduced):
    values = reduced
    for symbol, definition in reversed(replacements):
        values = [value.xreplace({symbol: definition}) for value in values]
    return values


def test_exact_duplicate_and_nested_definitions_keep_existing_scalar_optimizations():
    x, y, z = s.symbols("x y z")
    cases = [
        [(x + y)**2 + s.sqrt(x + y)],
        [(x + y)*z, (x + y)*z + 1],
        [s.exp(x*y) + s.exp(x*y)**2*(x*y)],
    ]
    for original in cases:
        replacements, reduced = s.cse(original)
        assert replacements
        assert _unfold(replacements, reduced) == original


@pytest.mark.parametrize("constant", [s.pi, s.E, s.I, s.oo, -s.oo, s.Rational(2, 3)])
def test_standard_constants_and_rationals_remain_admitted(constant):
    x = s.Symbol("cse_constant_x")
    source = s.Add(x, constant, evaluate=False)
    replacements, reduced = s.cse([source, source])
    assert len(replacements) == 1
    assert _unfold(replacements, reduced) == [source, source]


def test_current_builtin_special_functions_share_exact_scalar_arguments():
    x, y = s.symbols("cse_special_x cse_special_y")
    shared = x + y
    replacements, reduced = s.cse([s.gamma(shared), s.erf(shared), s.besselj(2, shared)])
    assert len(replacements) == 1
    symbol, definition = replacements[0]
    assert _polynomial(definition, (x, y)) == _polynomial(shared, (x, y))
    assert reduced == [s.gamma(symbol), s.erf(symbol), s.besselj(2, symbol)]


def test_definitions_form_a_dag_and_preserve_nested_repeated_polynomials():
    x, y, z = s.symbols("cse_dag_x cse_dag_y cse_dag_z")
    shared = x + y
    nested = shared**2 + z
    original = [nested**2 + shared, nested**3 + shared**2]
    replacements, reduced = s.cse(original)
    assert len(replacements) >= 2
    _check_polynomials(original, replacements, reduced, (x, y, z))


def test_large_shared_surface_dag_does_not_expand_every_path(monkeypatch):
    x, y = s.symbols("cse_shared_x cse_shared_y")
    expression = x + y
    # Native Add owns children, so keep construction itself bounded. The
    # retained Python graph has only eleven distinct Add nodes, although its
    # unrolled paths exceed the optimizer's deliberately smaller work budget.
    for _ in range(10):
        expression = s.Add(expression, expression, evaluate=False)
    monkeypatch.setattr(cse_module, "_MAX_WORK", 1000)
    replacements, reduced = s.cse([expression, expression])
    assert len(replacements) <= 11
    _check_polynomials([expression, expression], replacements, reduced, (x, y))


def test_default_names_avoid_typed_input_name_collisions():
    source = s.Symbol("x0", positive=True)
    y = s.Symbol("cse_name_y")
    common = source + y
    replacements, reduced = s.cse([common**2, common**3])
    assert replacements[0][0].name == "x1"
    _check_polynomials([common**2, common**3], replacements, reduced, (source, y))


def test_supplied_names_skip_inputs_and_duplicate_temporaries():
    x, y, u, v, temporary, next_temporary = s.symbols(
        "cse_input_x cse_input_y cse_input_u cse_input_v cse_t cse_next_t"
    )
    original = [(x + y)**2, (x + y)**3, (u + v)**2, (u + v)**3]
    replacements, reduced = s.cse(
        original, symbols=iter([x, temporary, temporary, next_temporary]),
    )
    assert [symbol for symbol, _ in replacements] == [temporary, next_temporary]
    _check_polynomials(original, replacements, reduced, (x, y, u, v))


def test_exhausted_or_endlessly_colliding_symbol_streams_fail(monkeypatch):
    x, y = s.symbols("cse_stream_x cse_stream_y")
    original = [(x + y)**2, (x + y)**3]
    with pytest.raises(ValueError, match="ran out"):
        s.cse(original, symbols=iter([x, y]))
    monkeypatch.setattr(cse_module, "_MAX_SYMBOL_ATTEMPTS", 4)
    with pytest.raises(ValueError, match="temporary-symbol attempt"):
        s.cse(original, symbols=repeat(x))


@pytest.mark.parametrize("temporary", [
    s.Integer(1), s.Symbol("cse_noncommuting_t", commutative=False),
    s.Symbol("cse_positive_t", positive=True),
])
def test_temporary_cannot_change_the_scalar_domain_or_assumptions(temporary):
    x, y = s.symbols("cse_temp_x cse_temp_y")
    with pytest.raises((TypeError, ValueError)):
        s.cse([(x + y)**2, (x + y)**3], symbols=iter([temporary]))


def test_ignored_symbols_are_not_hidden_in_common_argument_placeholders():
    x, y, z = s.symbols("cse_ignore_x cse_ignore_y cse_ignore_z")
    original = [(x + y)*z, (x + y)*(z + 1)]
    replacements, reduced = s.cse(original, ignore=(x,))
    assert all(x not in expression.free_symbols for _, expression in replacements)
    _check_polynomials(original, replacements, reduced, (x, y, z))


def test_noncommutative_products_keep_their_original_order_and_identity():
    a, b, c = s.symbols("cse_A cse_B cse_C", commutative=False)
    original = [s.Mul(a, b, c, evaluate=False), s.Mul(a, c, b, evaluate=False)]
    replacements, reduced = s.cse(original)
    assert replacements == []
    assert all(before is after for before, after in zip(original, reduced))
    assert reduced[0].args == (a, b, c) and reduced[1].args == (a, c, b)


def test_unknown_symbolic_objects_do_not_run_observation_or_reconstruction_hooks():
    x = s.Symbol("cse_hostile_x")

    class Hostile(s.Basic):
        @property
        def args(self):
            raise AssertionError("CSE inspected custom args")

        @property
        def free_symbols(self):
            raise AssertionError("CSE inspected custom free symbols")

        def __hash__(self):
            raise AssertionError("CSE invoked custom hashing")

        def __eq__(self, other):
            raise AssertionError("CSE invoked custom equality")

    node = object.__new__(Hostile)
    s.Basic.__dict__["_struct_args"].__set__(node, (x,))
    replacements, reduced = s.cse([node, node])
    assert replacements == [] and reduced[0] is node and reduced[1] is node


def test_custom_subclass_with_builtin_spelling_stays_opaque():
    class Add(s.Add):
        __module__ = "sympy.core.add"

        @property
        def args(self):
            raise AssertionError("a spelling must not grant built-in authority")

    node = object.__new__(Add)
    replacements, reduced = s.cse([node, node])
    assert replacements == [] and reduced[0] is node and reduced[1] is node


@pytest.mark.parametrize("builder", [
    lambda x, y: s.Integral(x + y, (x, 0, 1)),
    lambda x, y: s.Sum(x + y, (x, 0, 3)),
    lambda x, y: s.Product(x + y, (x, 1, 3)),
    lambda x, y: s.Limit(x + y, x, 0),
    lambda x, y: s.Order(x + y),
    lambda x, y: s.Subs(x + y, x, 2),
    lambda x, y: s.Piecewise((x + y, x > 0), (x - y, True)),
    lambda x, y: s.Lambda(x, x + y),
    lambda x, y: s.Derivative(s.sin(x + y), x, evaluate=False),
])
def test_bound_expressions_are_not_hoisted_out_of_scope(builder):
    x, y = s.symbols("cse_bound_x cse_bound_y")
    bound = builder(x, y)
    original = [bound, x + y]
    replacements, reduced = s.cse(original)
    assert replacements == []
    assert reduced[0] is bound and reduced[1] is original[1]


def test_undefined_function_classes_are_not_merged_by_their_spelling():
    x = s.Symbol("cse_opaque_x")
    first = s.FunctionClass("Same", (s.Function,), {"__module__": __name__})
    second = s.FunctionClass("Same", (s.Function,), {"__module__": __name__})
    a, b = first(x), second(x)
    replacements, reduced = s.cse([a, b, a])
    assert replacements == []
    assert reduced[0] is a and reduced[1] is b and reduced[2] is a


def test_modified_builtin_accessors_refuse_before_executing_the_hook(monkeypatch):
    x, y = s.symbols("cse_patched_x cse_patched_y")
    source = x + y

    def forbidden(self):
        raise AssertionError("CSE executed a replaced built-in accessor")

    monkeypatch.setattr(s.Add, "args", property(forbidden))
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.cse([source, source])


@pytest.mark.parametrize("owner, attribute, builder", [
    (s.Symbol, "name", lambda: s.Symbol("cse_metadata_x")),
    (s.Symbol, "_assumptions", lambda: s.Symbol("cse_metadata_facts")),
    (s.Float, "_dps", lambda: s.Float(1.25)),
    (s.Basic, "__str__", lambda: s.Symbol("cse_metadata_print")),
    (cse_module.AssocOp, "args", lambda: s.Symbol("cse_inherited_x") + 1),
])
def test_modified_scalar_metadata_and_inherited_hooks_refuse(owner, attribute, builder, monkeypatch):
    source = builder()

    def forbidden(*args):
        raise AssertionError("CSE executed a replaced metadata hook")

    replacement = forbidden if attribute == "__str__" else property(forbidden)
    monkeypatch.setattr(owner, attribute, replacement, raising=False)
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.cse([source, source])


def test_replaced_symbol_name_function_body_refuses_before_the_hook(monkeypatch):
    source = s.Symbol("cse_metadata_body")
    original = s.Symbol.__dict__["name"].fget

    def forbidden(self):
        raise AssertionError("CSE executed a changed name property body")

    monkeypatch.setattr(original, "__code__", forbidden.__code__)
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.cse([source, source])


def test_unsupported_converter_is_not_executed():
    class Converter:
        def _sympy_(self):
            raise AssertionError("CSE executed an unsupervised converter")

    with pytest.raises(TypeError, match="built-in scalar literals"):
        s.cse(Converter())


def test_floating_operands_are_not_reassociated():
    x, y = s.symbols("cse_float_x cse_float_y")
    large, small = s.Float(1e16), s.Float(1)
    original = [
        s.Add(large, small, x, evaluate=False),
        s.Add(large, small, y, evaluate=False),
    ]
    replacements, reduced = s.cse(original)
    assert replacements == []
    assert all(before is after for before, after in zip(original, reduced))


def test_work_and_depth_limits_are_terminal(monkeypatch):
    x, y = s.symbols("cse_limit_x cse_limit_y")
    source = (x + y)**2 + (x + y)**3
    with monkeypatch.context() as patch:
        patch.setattr(cse_module, "_MAX_WORK", 4)
        with pytest.raises(ValueError, match="work/depth"):
            s.cse(source)
    with monkeypatch.context() as patch:
        patch.setattr(cse_module, "_MAX_DEPTH", 1)
        with pytest.raises(ValueError, match="work/depth"):
            s.cse(source)


def test_list_false_preserves_scalar_and_tuple_result_shapes():
    x, y = s.symbols("cse_shape_x cse_shape_y")
    source = (x + y)**2 + (x + y)**3
    replacements, scalar = s.cse(source, list=False)
    assert isinstance(scalar, s.Expr)
    _check_polynomials([source], replacements, [scalar], (x, y))
    replacements, pair = s.cse((source, source), list=False)
    assert type(pair) is tuple
    _check_polynomials([source, source], replacements, pair, (x, y))
