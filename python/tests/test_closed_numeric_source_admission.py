"""Closed numerical source admission cannot reconstruct or execute custom classes."""

import importlib
import math

import pytest
import sympy as s

from sympy.core import Basic, FunctionClass, Relational, _FLOAT_INTERN, _native, _native_expr


COMPILERS = ("scalar", "residual", "objective", "load")


def compile_source(kind, source, variable):
    if kind == "scalar":
        return s.compile_numeric(variable, source)
    if kind == "residual":
        return s.compile_residual_system(source, variable)
    if kind == "objective":
        return s.compile_objective(source, variable)
    data = s.compile_residual_system(variable-1, variable).to_bytes()
    return s.load_residual_system(data, residuals=source, args=variable)


@pytest.mark.parametrize("compiler", COMPILERS)
@pytest.mark.parametrize("form", ["native", "generic", "add", "mul", "pow", "held"])
def test_unknown_native_and_canonical_functions_refuse_without_python_reconstruction(compiler, form):
    x = s.Symbol("closed_admission_x")
    effects = []

    def setter(self, name, value):
        effects.append(("setter", name))
        object.__setattr__(self, name, value)

    def getter(self):
        effects.append(("getter", "args"))
        raise AssertionError("closed compilation cannot inspect custom arguments")

    custom = FunctionClass("ClosedAdmissionOpaque", (s.Function,), {
        "__module__": __name__, "__setattr__": setter, "args": property(getter),
    })
    node = custom(x, evaluate=False)
    raw = _native.py_function("ClosedAdmissionOpaque", _native_expr(x))
    sources = {
        "native": lambda: raw, "generic": lambda: s.Expr(raw),
        "add": lambda: s.Add(node, 1), "mul": lambda: s.Mul(node, 2),
        "pow": lambda: s.Pow(node, 2), "held": lambda: s.Add(node, 1, evaluate=False),
    }
    source = sources[form]()
    effects.clear()
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        compile_source(compiler, source, x)
    assert effects == []


@pytest.mark.parametrize("compiler", COMPILERS)
def test_unknown_metaclass_hash_equality_and_conversion_never_run_during_source_admission(compiler):
    effects = []

    class HostileMeta(type):
        def __eq__(cls, other):
            effects.append("eq")
            raise AssertionError("source type admission must use identity")

        def __hash__(cls):
            effects.append("hash")
            raise AssertionError("source type admission must use identity")

    class Unknown(metaclass=HostileMeta):
        def __float__(self):
            effects.append("float")
            raise AssertionError("opaque sources cannot become numeric literals")

    x = s.Symbol("closed_metaclass_x")
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        compile_source(compiler, Unknown(), x)
    assert effects == []


@pytest.mark.parametrize("kind", ["dense", "immutable", "sparse", "immutable_sparse", "retained"])
@pytest.mark.parametrize("compiler", ["scalar", "residual", "load"])
def test_matrix_sources_never_lift_unknown_functions_before_refusing(kind, compiler):
    x = s.Symbol("closed_matrix_x")
    effects = []

    def setter(self, name, value):
        effects.append(name)
        object.__setattr__(self, name, value)

    custom = FunctionClass("ClosedMatrixOpaque", (s.Function,), {
        "__module__": __name__, "__setattr__": setter,
    })
    raw = _native.py_function("ClosedMatrixOpaque", _native_expr(x))
    if kind in ("sparse", "immutable_sparse"):
        cls = s.SparseMatrix if kind == "sparse" else s.ImmutableSparseMatrix
        source = cls(_native.SparseMatrix(1, 1, {(0, 0): raw}))
    elif kind == "retained":
        source = s.Matrix([[custom(x, evaluate=False)]])
    else:
        cls = s.Matrix if kind == "dense" else s.ImmutableMatrix
        source = cls(_native.Matrix(1, 1, [raw]))
    effects.clear()
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        compile_source(compiler, source, x)
    assert effects == []


@pytest.mark.parametrize("slot,value,match", [
    ("_surface", {(0, 2): s.Integer(1)}, "coordinate"),
    ("_surface", {(True, 0): s.Integer(1)}, "coordinate"),
    ("_surface", {"0,0": s.Integer(1)}, "coordinate"),
    ("_surface", [s.Integer(1)], "snapshot"),
    ("_mutation_generation", -1, "generation"),
    ("_mutation_generation", False, "generation"),
    ("_matrix_shape", (2, 1), "shapes differ"),
])
def test_sparse_numeric_snapshot_rejects_invalid_retained_state(slot, value, match):
    matrix = s.SparseMatrix([[1]])
    object.__setattr__(matrix, slot, value)
    with pytest.raises(ValueError, match=match):
        s.compile_numeric((), matrix)
    with pytest.raises(ValueError, match=match):
        s.compile_numeric_hybrid((), matrix, callbacks={})


def test_sparse_snapshot_refuses_custom_storage_without_executing_its_protocol():
    effects = []

    class ForeignCells(dict):
        def __len__(self):
            effects.append("len")
            raise AssertionError("foreign storage cannot participate in native admission")

        def items(self):
            effects.append("items")
            raise AssertionError("foreign storage cannot participate in native admission")

    matrix = s.SparseMatrix([[1]])
    object.__setattr__(matrix, "_surface", ForeignCells({(0, 0): s.Integer(1)}))
    with pytest.raises(ValueError, match="snapshot"):
        s.compile_numeric((), matrix)
    assert effects == []


@pytest.mark.parametrize("name", ["ln", "abs", "sqrt", "Add", "Mul", "Pow", "OtherNativeTag"])
def test_unadmitted_native_tags_are_not_new_aliases_for_intrinsics_or_operators(name):
    x = s.Symbol("closed_native_alias_x")
    raw = _native.py_function(name, _native_expr(x))
    for source in (raw, s.Expr(raw), s.Add(s.Expr(raw), 1, evaluate=False)):
        with pytest.raises(NotImplementedError, match="custom or opaque"):
            s.compile_numeric(x, source)


def test_existing_builtin_native_tags_and_typed_coordinates_keep_their_meaning():
    plain = s.Symbol("closed_typed_same")
    positive = s.Symbol("closed_typed_same", positive=True)
    dummy = s.Dummy("closed_typed_same")
    variables = (dummy, positive, plain)
    expression = plain + 2*positive + 3*dummy + s.sin(positive)
    raw = _native_expr(expression)
    for source in (expression, raw, s.Expr(raw)):
        scalar = s.compile_numeric(variables, source)
        assert scalar(2, 1, -2) == pytest.approx(6 + math.sin(1))
        system = s.compile_residual_system([source], variables)
        residuals, jacobian = system.evaluate(2, 1, -2)
        assert residuals == pytest.approx([6 + math.sin(1)])
        assert jacobian[0] == pytest.approx([3, 2 + math.cos(1), 1])
        assert all(left is right for left, right in zip(system.variables, variables))
    absolute = _native_expr(s.Abs(plain) + s.log(positive) + s.exp(dummy))
    assert s.compile_numeric(variables, absolute)(0, 1, -2) == 3


def test_explicit_abs_source_tag_normalizes_without_registering_a_class_by_name():
    x = s.Symbol("closed_abs_x")
    raw = _native.py_function("Abs", _native_expr(x))
    assert s.compile_numeric(x, raw)(-3) == 3
    with pytest.raises(ValueError, match="argument"):
        s.compile_numeric(x, _native.py_function("sin", _native_expr(x), _native_expr(x)))


@pytest.mark.parametrize("form", ["native", "generic", "canonical", "matrix", "retained_matrix"])
def test_native_float_origin_survives_normalization_loading_and_certification_admission(form):
    x = s.Symbol("closed_float_origin_x")
    expression = x - s.Float(1.25)
    raw = _native_expr(expression)
    source = {
        "native": raw, "generic": s.Expr(raw), "canonical": expression,
        "matrix": s.Matrix(_native.Matrix(1, 1, [raw])),
        "retained_matrix": s.Matrix([expression]),
    }[form]
    program = s.compile_residual_system(source, x)
    assert program.residuals(2) == [0.75]
    restored = s.load_residual_system(program.to_bytes(), residuals=source, args=x)
    for candidate in (program, restored):
        for operation in (
            lambda: candidate.enclose([(1, 2)]),
            lambda: candidate.solve_certified(1, backend="frankennumpy"),
        ):
            with pytest.raises(ValueError, match="exact source literals"):
                operation()
    exact = s.compile_residual_system(_native_expr(x-s.Rational(5, 4)), x)
    assert exact.solve_certified(1).root_enclosure == ((s.Rational(5, 4), s.Rational(5, 4)),)
    assert s.compile_objective(raw, x).evaluate(2)[0] == 0.75


@pytest.mark.parametrize("arguments", [(), ("symbol",), (-1,), (1 << 65,), (0, 0), (0x7FF0000000000000,)])
def test_malformed_or_nonfinite_native_float_markers_refuse(arguments):
    x = s.Symbol("closed_malformed_float_x")
    entries = [_native_expr(x) if value == "symbol" else _native.py_integer(value)
               for value in arguments]
    source = _native.py_function(_FLOAT_INTERN, *entries)
    with pytest.raises(ValueError, match="floating|finite"):
        s.compile_numeric(x, source)


def test_borrowed_whole_source_limits_precede_float_decoding_and_matrix_cell_getters(monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("closed_borrowed_x")
    huge = _native.py_function(_FLOAT_INTERN, _native.py_integer(1 << 4096))

    def forbidden(*args):
        raise AssertionError("borrowed preflight must refuse before decoding or lifting")

    monkeypatch.setattr(module, "_bits_to_float", forbidden)
    monkeypatch.setattr(s.Matrix, "__getitem__", forbidden)
    for source in (huge, s.Expr(huge), s.Matrix(_native.Matrix(1, 1, [huge]))):
        with pytest.raises(ValueError, match="literal|bit"):
            s.compile_numeric(x, source)


def test_native_subtrees_keep_one_source_counter_and_their_enclosing_depth(monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("closed_shared_count_x")
    source = _native.Add(_native_expr(x), _native.py_integer(1), evaluate=False).as_expr()
    monkeypatch.setattr(module, "_MAX_NODES", 5)
    assert s.compile_numeric(x, source)(2) == 3
    with pytest.raises(ValueError, match="node/depth"):
        s.compile_numeric(x, [source, source])
    monkeypatch.setattr(module, "_MAX_NODES", 65536)
    monkeypatch.setattr(module, "_MAX_DEPTH", 1)
    assert s.compile_numeric(x, source)(2) == 3
    with pytest.raises(ValueError, match="node/depth"):
        s.compile_numeric(x, s.Add(s.Expr(source), 1, evaluate=False))


def test_canonical_and_native_held_domains_are_not_simplified_away():
    x = s.Symbol("closed_domain_x")
    sources = (
        s.Mul(0, s.log(x), evaluate=False),
        s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False),
        s.Pow(s.log(x), -1, evaluate=False),
    )
    for expression in sources:
        source = _native_expr(expression)
        program = s.compile_numeric(x, source)
        with pytest.raises(ValueError):
            program(0)
        system = s.compile_residual_system([source], x)
        with pytest.raises(ValueError):
            system.evaluate(0)


@pytest.mark.parametrize("kind", ["dense", "immutable", "sparse", "retained"])
def test_matrix_layout_uses_private_cells_and_freezes_all_entries_before_conversion(monkeypatch, kind):
    module = importlib.import_module("sympy.utilities.lambdify")
    x = s.Symbol("closed_matrix_snapshot_x")
    raw = [_native_expr(x), _native_expr(x+1)]
    if kind == "sparse":
        matrix = s.SparseMatrix(_native.SparseMatrix(1, 2, {(0, 0): raw[0], (0, 1): raw[1]}))
    elif kind == "retained":
        matrix = s.Matrix([[x, x+1]])
    else:
        cls = s.Matrix if kind == "dense" else s.ImmutableMatrix
        matrix = cls(_native.Matrix(1, 2, raw))

    def forbidden(*args):
        raise AssertionError("compiler must not call the shell matrix element getter")

    with monkeypatch.context() as context:
        context.setattr(type(matrix), "__getitem__", forbidden)
        assert s.compile_numeric(x, matrix)(3) == [[3, 4]]
    if kind != "retained":
        return
    convert, events = module._numeric_source, []

    def mutate_after_capture(value, *args, **kwargs):
        if not events:
            events.append(True)
            matrix[0, 0], matrix[0, 1] = 88, 99
        return convert(value, *args, **kwargs)

    monkeypatch.setattr(module, "_numeric_source", mutate_after_capture)
    program = s.compile_numeric(x, matrix)
    assert events == [True] and program(3) == [[3, 4]]
    assert matrix[0, 0] == 88 and matrix[0, 1] == 99


def test_math_backend_keeps_python_scalar_types_and_does_not_enter_native_source_normalization(monkeypatch):
    module = importlib.import_module("sympy.utilities.lambdify")
    x, y = s.symbols("closed_math_x closed_math_y")

    def forbidden(*args, **kwargs):
        raise AssertionError("modules=math must keep its own scalar arithmetic")

    monkeypatch.setattr(module, "_native_numeric_source", forbidden)
    function = s.lambdify((x, y), (x+y, x*y, s.Rational(1, 2)), modules="math")
    result = function(2, 3)
    assert result == (5, 6, 0.5)
    assert type(result[0]) is type(result[1]) is int and type(result[2]) is float
    with pytest.raises(NotImplementedError, match="matrix numerical outputs"):
        s.lambdify(x, s.Matrix([[x]]), modules="math")
    assert "NumericRootCandidate" in module.__all__


@pytest.mark.parametrize("form", ["native", "generic"])
def test_closed_float_normalization_cannot_erase_equation_provenance_checks(form):
    x = s.Symbol("closed_equation_intern_x")
    raw = _native_expr(s.Float(1.25))
    if form == "native":
        # Explicit Eq(raw_marker, 0) sympifies into a retained Float. To test
        # erased provenance, construct an actual native-only equation instead.
        equations = []
        for side in (raw, _native.Add(raw, _native_expr(x), evaluate=False).as_expr()):
            equation = object.__new__(s.Eq)
            Basic.__dict__["_value"].__set__(
                equation, _native.py_function("Equality", side, _native.py_integer(0)))
            equations.append(equation)
    else:
        erased = s.Expr(raw)
        equations = [s.Eq(side, 0, evaluate=False)
                     for side in (erased, s.Add(erased, x, evaluate=False))]
    for equation in equations:
        with pytest.raises(NotImplementedError, match="retained Python sides"):
            s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})
    genuine = s.Eq(x, s.Float(1.25), evaluate=False)
    assert s.compile_hybrid_residual_system(genuine, x, jacobian=[[1]], callbacks={}).residuals(2) == [0.75]


SOLVER_ROUTES = ("approximate", "budgeted", "python", "frankennumpy", "frankenscipy")


def solve_equation_source(route, system, equation, variable):
    arguments = ([equation], [variable], [1]) if system else (equation, variable, 1)
    if route == "approximate":
        return s.nsolve(*arguments)
    if route == "budgeted":
        return s.nsolve(*arguments, budget=s.EvaluationBudget())
    if route == "python":
        return s.nsolve_certified(*arguments)
    return s.nsolve_certified(*arguments, backend=route)


@pytest.mark.parametrize("route", SOLVER_ROUTES)
@pytest.mark.parametrize("system", [False, True])
def test_numerical_solvers_refuse_native_only_equations_before_lifting_or_getters(monkeypatch, route, system):
    x = s.Symbol("closed_solver_equation_x")
    effects = []

    def forbidden_setter(self, name, value):
        effects.append(("setter", name))
        raise AssertionError("equation admission cannot reconstruct a function class")

    def forbidden_getter(self):
        effects.append(("getter",))
        raise AssertionError("equation admission cannot enter public operand getters")

    opaque_class = FunctionClass("ClosedSolverEquationOpaque", (s.Function,), {
        "__module__": __name__, "__setattr__": forbidden_setter,
    })
    for name in ("args", "lhs", "rhs"):
        monkeypatch.setattr(Relational, name, property(forbidden_getter))
    opaque = _native.py_function("ClosedSolverEquationOpaque", _native_expr(x))
    for side in (_native_expr(x), opaque, _native.py_integer(1 << 4096)):
        equation = object.__new__(s.Eq)
        Basic.__dict__["_value"].__set__(
            equation, _native.py_function("Equality", side, _native.py_integer(2)))
        with pytest.raises(NotImplementedError, match="retained Python sides"):
            solve_equation_source(route, system, equation, x)
    # Function registration is weak; retain the exact hostile class throughout
    # every attempted solve so collection cannot weaken this effect sentinel.
    assert opaque_class.__name__ == "ClosedSolverEquationOpaque"
    assert effects == []


@pytest.mark.parametrize("route", SOLVER_ROUTES)
@pytest.mark.parametrize("system", [False, True])
def test_numerical_solvers_refuse_malformed_equation_pairs_without_conversion_hooks(route, system):
    x = s.Symbol("closed_solver_pair_x")
    effects = []

    class HostileMeta(type):
        @property
        def __mro__(cls):
            effects.append("mro")
            raise AssertionError("equation admission must read actual class ancestry")

        def __eq__(cls, other):
            effects.append("eq")
            raise AssertionError("equation admission must compare class identity")

    class Unknown(metaclass=HostileMeta):
        @property
        def __class__(self):
            effects.append("class")
            raise AssertionError("equation admission cannot inspect a class property")

        def _sympy_(self):
            effects.append("sympify")
            raise AssertionError("malformed retained pairs cannot invoke conversion")

    zero = s.Integer(0)
    for sides in ([x, zero], (), (x,), (x, zero, zero),
                  (_native_expr(x), zero), (Unknown(), zero), (x, 0)):
        equation = s.Eq(x, 2, evaluate=False)
        Relational.__dict__["_relational_sides"].__set__(equation, sides)
        with pytest.raises(ValueError, match="exact captured pair of Basic sides"):
            solve_equation_source(route, system, equation, x)
    assert effects == []


@pytest.mark.parametrize("route", SOLVER_ROUTES)
@pytest.mark.parametrize("system", [False, True])
def test_retained_equations_keep_solver_defaults_without_public_operand_getters(monkeypatch, route, system):
    import inspect

    x = s.Symbol("closed_solver_retained_x")
    equation = s.Eq(2*x, 4, evaluate=False)
    captured = Relational.__dict__["_relational_sides"].__get__(equation, Relational)

    def forbidden(self):
        raise AssertionError("numerical solvers must use the captured operand pair")

    for name in ("args", "lhs", "rhs"):
        monkeypatch.setattr(Relational, name, property(forbidden))
    result = solve_equation_source(route, system, equation, x)
    if route in ("approximate", "budgeted"):
        root = result[0, 0] if system else result
        assert type(root) is s.Float and float(root) == 2
    else:
        assert type(result) is s.RootBoxReceipt
        assert result.root_enclosure == ((2, 2),)
    assert Relational.__dict__["_relational_sides"].__get__(equation, Relational) is captured
    parameters = inspect.signature(s.nsolve).parameters
    assert parameters["tol"].default == 1e-12 and parameters["maxiter"].default == 200
    parameters = inspect.signature(s.nsolve_certified).parameters
    assert parameters["tol"].default is None and parameters["maxiter"].default is None
    assert parameters["backend"].default == "python"
