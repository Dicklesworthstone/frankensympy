"""SAT must preserve the entire typed atom universe and every source operand."""

from itertools import product

import pytest
import sympy as s

from sympy.core import Basic, supervised_callback_session
from sympy.logic.boolalg import BooleanFunction, _BooleanRequest, _from_native_bool
from sympy.logic.inference import pl_true, valid


def _typed_atoms():
    return (
        s.Symbol("boolean_same_name"),
        s.Symbol("boolean_same_name", positive=True),
        s.Symbol("boolean_same_name", negative=True),
        s.Dummy("boolean_same_name"),
        s.Dummy("boolean_same_name"),
    )


@pytest.mark.parametrize("left,right", [(i, j) for i in range(5) for j in range(5) if i != j])
def test_same_printed_name_never_turns_independent_atoms_into_unsat(left, right):
    atoms = _typed_atoms()
    positive, negative = atoms[left], atoms[right]
    source = s.And(positive, s.Not(negative))
    model = s.satisfiable(source)
    assert model is not False
    assert model == {positive: True, negative: False}
    assert any(key is positive for key in model)
    assert any(key is negative for key in model)
    assert pl_true(source, model) is True


@pytest.mark.parametrize("first,second", [(0, 1), (1, 2), (3, 4), (0, 3)])
def test_all_models_keep_original_distinct_atoms_and_complete_assignments(first, second):
    atoms = _typed_atoms()
    left, right = atoms[first], atoms[second]
    source = s.Or(left, right)
    models = list(s.satisfiable(source, all_models=True))
    assert {(model[left], model[right]) for model in models} == {
        (False, True), (True, False), (True, True),
    }
    assert len(models) == 3
    assert all(len(model) == 2 and pl_true(source, model) is True for model in models)
    assert all(any(key is left for key in model) and any(key is right for key in model)
               for model in models)


@pytest.mark.parametrize("damage", ["missing", "extra", "unknown", "non_boolean"])
def test_native_model_must_match_the_complete_retained_atom_universe(damage):
    left, right = s.symbols("boolean_model_left boolean_model_right")
    request = _BooleanRequest(s.Or(left, right))
    model = request.native.satisfiable()
    assert type(model) is dict
    assert set(request.lift_model(model)) == {left, right}
    token = next(iter(model))
    if damage == "missing":
        del model[token]
    elif damage == "extra":
        model["unadmitted_boolean_token"] = True
    elif damage == "unknown":
        # Preserve cardinality so length validation alone cannot accept it.
        model["unadmitted_boolean_token"] = model.pop(token)
    else:
        # Python equality treats 1 as True; publication requires actual bools.
        model[token] = 1
    with pytest.raises(RuntimeError, match="does not match the retained atom universe"):
        request.lift_model(model)


@pytest.mark.parametrize("name", ["fsym_boolean_atom_0", "fsym_boolean_atom_1", "a:b\x00c", "λ"])
def test_user_spelling_cannot_capture_a_request_token(name):
    left, right = s.Symbol(name), s.Symbol(name, integer=True)
    assert s.satisfiable(s.And(left, s.Not(right))) == {left: True, right: False}


def test_reconstructed_dummy_identity_is_one_atom_but_a_fresh_dummy_is_not():
    first = s.Dummy("boolean_reconstruction", dummy_index=872451)
    equal = s.Dummy("boolean_reconstruction", dummy_index=872451)
    different = s.Dummy("boolean_reconstruction", dummy_index=872452)
    assert first is not equal and first == equal
    assert s.satisfiable(s.And(first, s.Not(equal))) is False
    model = s.satisfiable(s.And(first, s.Not(different)))
    assert model == {first: True, different: False}
    assert any(key is first for key in model)


@pytest.mark.parametrize("operator", [s.Xor, s.Equivalent])
@pytest.mark.parametrize("count", [3, 4, 5])
def test_nary_native_lowering_matches_every_independent_truth_assignment(operator, count):
    atoms = [s.Dummy("nary_boolean") for _ in range(count)]
    source = operator(*atoms)
    expected = set()
    for values in product((False, True), repeat=count):
        expected_truth = (sum(values) % 2 == 1 if operator is s.Xor
                          else all(value == values[0] for value in values))
        constraints = [atom if value else s.Not(atom) for atom, value in zip(atoms, values)]
        model = s.satisfiable(s.And(source, *constraints))
        assert (model is not False) is expected_truth
        if expected_truth:
            assert tuple(model[atom] for atom in atoms) == values
            expected.add(values)
    models = list(s.satisfiable(source, all_models=True))
    assert len(models) == len(expected)
    assert {tuple(model[atom] for atom in atoms) for model in models} == expected


@pytest.mark.parametrize("operation", [
    s.simplify_logic, s.to_cnf, s.to_dnf,
    lambda expression: s.to_cnf(expression, simplify=True),
    lambda expression: s.to_dnf(expression, simplify=True),
])
def test_surface_normal_forms_preserve_typed_atoms_and_truth(operation):
    a, b, _, c, _ = _typed_atoms()
    source = s.Or(s.And(a, s.Not(b)), s.And(c, b))
    result = operation(source)
    assert result.free_symbols == {a, b, c}
    assert all(any(value is original for original in (a, b, c))
               for value in result.free_symbols)
    for bits in product((False, True), repeat=3):
        model = dict(zip((a, b, c), bits))
        assert pl_true(result, model) is pl_true(source, model)


@pytest.mark.parametrize("operation", ["simplify", "to_cnf", "to_dnf"])
def test_native_normal_form_lifting_uses_retained_source_owners(operation):
    a, b, _, c, _ = _typed_atoms()
    source = s.Or(s.And(a, s.Not(b)), c)
    request = _BooleanRequest(source)
    native = getattr(request.native, operation)()
    result = _from_native_bool(native, request)
    assert result.free_symbols == {a, b, c}
    assert all(any(value is original for original in (a, b, c))
               for value in result.free_symbols)
    for bits in product((False, True), repeat=3):
        model = dict(zip((a, b, c), bits))
        assert pl_true(result, model) is pl_true(source, model)


def test_mutating_a_yielded_model_cannot_change_enumeration():
    a, b = s.symbols("boolean_model_a boolean_model_b")
    generator = s.satisfiable(s.Or(a, b), all_models=True)
    first = next(generator)
    observed = [(first[a], first[b])]
    first.clear()
    first[s.Symbol("foreign_model_atom")] = True
    observed.extend((model[a], model[b]) for model in generator)
    assert len(observed) == 3
    assert set(observed) == {(False, True), (True, False), (True, True)}


def test_changed_composite_source_refuses_before_the_next_model():
    a, b = s.symbols("boolean_mutation_a boolean_mutation_b")
    source = s.Or(a, b)
    generator = s.satisfiable(source, all_models=True)
    next(generator)
    original = source._args
    try:
        source._args = (a, s.Not(a))
        with pytest.raises(ValueError, match="source changed"):
            next(generator)
    finally:
        source._args = original


@pytest.mark.parametrize("change", ["assumptions", "native", "dummy_name", "dummy_index"])
def test_changed_atom_refuses_before_the_next_model(change):
    atom = s.Dummy("boolean_atom_mutation", positive=True)
    other = s.Symbol("boolean_other_mutation")
    generator = s.satisfiable(s.Or(atom, other), all_models=True)
    next(generator)
    if change == "assumptions":
        slot, replacement = "_assumptions", {"negative": True}
    elif change == "native":
        slot, replacement = "_value", other._value
    elif change == "dummy_name":
        slot, replacement = "_dummy_name", "changed_dummy"
    else:
        slot, replacement = "_dummy_number", atom.dummy_index + 1
    original = getattr(atom, slot)
    try:
        object.__setattr__(atom, slot, replacement)
        with pytest.raises(ValueError, match="changed"):
            next(generator)
    finally:
        object.__setattr__(atom, slot, original)


def test_forged_assumptions_are_rejected_before_native_solving():
    atom = s.Dummy("boolean_forged_assumptions", positive=True)
    original = atom._assumptions
    try:
        atom._assumptions = {"negative": True}
        with pytest.raises(ValueError, match="disagree"):
            s.satisfiable(atom)
    finally:
        atom._assumptions = original


@pytest.mark.parametrize("slot", ["_struct_args", "_surface_args"])
def test_forged_atom_surface_cannot_collapse_published_model_keys(slot):
    atom = s.Dummy("boolean_forged_surface")
    object.__setattr__(atom, slot, ())
    with pytest.raises(ValueError, match="original scalar source storage"):
        s.satisfiable(atom)


def test_custom_symbol_is_not_reinterpreted_by_its_name_or_hooks():
    effects = []

    class CustomSymbol(s.Symbol):
        @property
        def name(self):
            effects.append("name")
            raise AssertionError("opaque name hook ran")

        def __hash__(self):
            effects.append("hash")
            raise AssertionError("opaque hash hook ran")

    opaque = object.__new__(CustomSymbol)
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        s.satisfiable(opaque)
    assert effects == []


def test_custom_boolean_operator_is_not_lowered_as_its_builtin_parent():
    effects = []

    class CustomAnd(s.And):
        @property
        def args(self):
            effects.append("args")
            raise AssertionError("opaque argument hook ran")

    opaque = object.__new__(CustomAnd)
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        s.satisfiable(opaque)
    assert effects == []


def test_valid_admits_before_negating_a_custom_not():
    effects = []

    class CustomNot(s.Not):
        @property
        def args(self):
            effects.append("args")
            raise AssertionError("custom Not arguments cannot be inspected")

    opaque = object.__new__(CustomNot)
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        valid(opaque)
    assert effects == []


@pytest.mark.parametrize("owner,name", [(Basic, "__hash__"), (s.Symbol, "__eq__"),
                                        (BooleanFunction, "args")])
def test_changed_builtin_implementations_refuse_without_invocation(monkeypatch, owner, name):
    atom = s.Symbol("boolean_modified_owner")
    effects = []

    def forbidden(*args):
        effects.append(name)
        raise AssertionError("modified builtin hook ran")

    monkeypatch.setattr(owner, name, property(forbidden) if name == "args" else forbidden)
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.satisfiable(atom)
    assert effects == []


def test_cyclic_and_overdeep_sources_refuse_before_native_allocation():
    cycle = object.__new__(s.Not)
    cycle._args = (cycle,)
    with pytest.raises(ValueError, match="cyclic"):
        s.satisfiable(cycle)
    source = s.Symbol("boolean_deep")
    for _ in range(140):
        node = object.__new__(s.Not)
        node._args = (source,)
        source = node
    with pytest.raises(ValueError, match="depth limit"):
        s.satisfiable(source)


def test_resource_refusal_cannot_become_false_unsat_or_empty_model_iterator():
    atoms = tuple(s.Dummy("boolean_too_many") for _ in range(257))
    source = object.__new__(s.And)
    source._args = atoms
    with pytest.raises(ValueError, match="atom limit"):
        s.satisfiable(source)
    with pytest.raises(ValueError, match="atom limit"):
        next(s.satisfiable(source, all_models=True))


def test_native_solver_limit_remains_an_error():
    # The source atom bound admits this, but the native Tseitin encoding also
    # needs one auxiliary variable and must refuse its 257th total variable.
    source = s.And(*(s.Dummy("boolean_native_limit") for _ in range(256)))
    with pytest.raises(ValueError, match="resource limit"):
        s.satisfiable(source)


def test_request_budget_failure_is_terminal():
    a, b = s.symbols("boolean_budget_a boolean_budget_b")
    with supervised_callback_session(max_nested_steps=1):
        with pytest.raises(Exception, match="(?i)budget|steps|exhaust"):
            s.satisfiable(s.Or(a, b))


def test_enumeration_cannot_resume_under_a_different_request():
    a, b = s.symbols("boolean_request_a boolean_request_b")
    with supervised_callback_session(max_nested_steps=100_000):
        models = s.satisfiable(s.Or(a, b), all_models=True)
        next(models)
    with pytest.raises(RuntimeError, match="request changed"):
        next(models)


@pytest.mark.parametrize("source,expected", [(s.true, True), (s.false, False),
                                            (True, True), (False, False)])
def test_valid_does_not_confuse_empty_satisfying_assignment_with_unsat(source, expected):
    assert valid(source) is expected
    assert s.satisfiable(source) == ({} if expected else False)
    assert list(s.satisfiable(source, all_models=True)) == ([{}] if expected else [])


def test_distinct_assumed_atoms_do_not_prove_a_tautology():
    first = s.Symbol("boolean_valid_same", positive=True)
    second = s.Symbol("boolean_valid_same", negative=True)
    assert valid(s.Or(first, s.Not(second))) is False
    assert valid(s.Or(first, s.Not(first))) is True
