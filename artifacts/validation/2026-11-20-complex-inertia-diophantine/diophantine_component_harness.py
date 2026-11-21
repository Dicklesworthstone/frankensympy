"""Supplementary checks of the real integer-family helper and publication tail.

Run with the documented math/PyO3 component, for example:

    python -I diophantine_component_harness.py \
        --repo /absolute/path/to/frankensympy \
        --native /absolute/path/to/component/libfsym_python.so \
        --report /absolute/path/to/diophantine-component-results.json

This harness supplies explicit integer coefficient rows. It deliberately does
not call or validate public diophantine(), expand(), polynomial extraction, or
the full production extension. It calls the unchanged production source
admission, _complete_linear_family, native checked Smith solver, original-domain
checker and final source/result snapshots. No native math or verifier is mocked.
Adversarial tests replace only the explicitly untrusted presentation generator
or corrupt candidate/source objects to check the actual publication boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


OMISSIONS = (
    "lib.rs::eval_error", "lib.rs::eval_region", "lib.rs::simplify_expr",
    "lib.rs::expand_expr", "expr.rs::PyExpr.simplify", "expr.rs::PyExpr.expand",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_hashes(repo):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    selected = {name for name in names
                if name.startswith(("crates/", "src/", "python/sympy/"))
                and Path(name).suffix in (".py", ".rs", ".toml")}
    selected.update(("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"))
    return {name: digest(repo / name) for name in sorted(selected)}


def main():
    if not __debug__:
        raise RuntimeError("This harness requires Python assertions to be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    repo, binary, report = (value.resolve() for value in (args.repo, args.native, args.report))
    if not binary.is_file():
        raise FileNotFoundError(binary)
    before = source_hashes(repo)
    evidence = {
        "scope": "documented_math_component: explicit-row Diophantine helper and publication internals",
        "not_validated": ["public diophantine() dispatch", "expand/Poly normalization",
                          "full production extension"],
        "documented_component_omissions": OMISSIONS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
        "python": sys.version,
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "harness_sha256": digest(__file__),
        "requested_cases": args.case,
        "pre_sources": before,
        "results": [],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, indent=2) + "\n")

    spec = importlib.util.spec_from_file_location("fsym_python", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the explicitly selected component binary")
    extension = importlib.util.module_from_spec(spec)
    sys.modules["fsym_python"] = extension
    spec.loader.exec_module(extension)
    sys.path.insert(0, str(repo / "python"))
    s = importlib.import_module("sympy")
    assert Path(s.__file__).resolve() == repo / "python/sympy/__init__.py"
    assert Path(extension.__file__).resolve() == binary
    linear = importlib.import_module("sympy.solvers.diophantine")
    core = importlib.import_module("sympy.core")
    assert core._native is extension
    print("Scope:", evidence["scope"], flush=True)
    print("Native binary:", binary, flush=True)
    print("Public normalization is NOT exercised.", flush=True)

    cases = []
    metrics = {}

    def register(name, function):
        cases.append((name, function))

    def variables(count=3):
        return tuple(s.Symbol("component_axis_%02d" % index, integer=True) for index in range(count))

    def retained(coefficients, constant, axes):
        residual = s.Add(s.Integer(constant), *(coefficient * axis
                         for coefficient, axis in zip(coefficients, axes) if coefficient))
        return s.Add(residual, s.Mul(0, s.Add(*axes), evaluate=False), evaluate=False)

    def publication(source, axes, coefficients, constant, parameter=None):
        # The paired row/source fixtures are explicit. This is intentionally
        # not a replacement for, or a test of, public polynomial extraction.
        admitted, _children, original_axes, work, total, original_snapshot = linear._diophantine_source(source)
        assert set(axes) == original_axes
        family, result_snapshot = linear._complete_linear_family(
            dict(zip(axes, coefficients)), list(axes), constant, original_axes,
            work, parameter, original_snapshot,
        )
        result = linear._check_diophantine_domains(admitted, list(axes), family, work, total)
        linear._linear_source_unchanged(original_snapshot, work)
        linear._linear_source_unchanged(result_snapshot, work)
        return result

    def one(source, axes, coefficients, constant, parameter=None):
        result = publication(source, axes, coefficients, constant, parameter)
        assert type(result) is set and len(result) == 1
        family, = result
        assert len(family) == len(axes)
        return family

    def expect_error(kind, message, function):
        try:
            function()
        except kind as error:
            assert any(fragment in str(error) for fragment in message.split("|")), repr(error)
        else:
            raise AssertionError("Expected %s containing %r" % (kind, message))

    for active in range(3):
        def single_axis(active=active):
            axes = variables()
            coefficients = tuple(3 if index == active else 0 for index in range(3))
            family = one(retained(coefficients, -6, axes), axes, coefficients, -6)
            inactive = [index for index in range(3) if index != active]
            assert family[active] == 2
            n1, n2 = s.symbols("n1 n2", integer=True)
            assert tuple(family[index] for index in inactive) == (n1, n2)
            for first, second in product((-3, 0, 5), repeat=2):
                point = tuple(value.subs({n1: first, n2: second}) for value in family)
                assert point[active] == 2
                assert point[inactive[0]] == first and point[inactive[1]] == second
        register("single_active_coordinate_%d" % active, single_axis)

    signed = (
        (2, 4, -6, (3, 0), (-2, 1)),
        (-1, -2, 3, (3, 0), (-2, 1)),
        (3, 5, -7, (14, -7), (5, -3)),
        (3, 5, 7, (-14, 7), (5, -3)),
        (-2, 4, -6, (-3, 0), (2, 1)),
        (2, -4, -6, (3, 0), (2, 1)),
        (-2, -4, -6, (-3, 0), (-2, 1)),
    )
    for number, (a, b, constant, offsets, directions) in enumerate(signed):
        def signed_family(a=a, b=b, constant=constant, offsets=offsets, directions=directions):
            axes = variables(4)
            coefficients = (a, b, 0, 0)
            source = retained(coefficients, constant, axes)
            t = s.Symbol("t_0", integer=True)
            n1, n2 = s.symbols("n1 n2", integer=True)
            active_values = tuple(offset + direction*t for offset, direction in zip(offsets, directions))
            assert one(source, axes, coefficients, constant) == (*active_values, n1, n2)
            reverse = one(source, tuple(reversed(axes)), tuple(reversed(coefficients)), constant)
            assert reverse == (n2, n1, active_values[1], active_values[0])
            for value in (-5, 0, 7):
                point = tuple(expression.subs(t, value) for expression in active_values)
                assert a*point[0] + b*point[1] + constant == 0
        register("signed_family_and_reversed_coordinates_%d" % number, signed_family)

    def three_active():
        axes = variables(4)
        coefficients = (2, 3, 5, 0)
        family = one(retained(coefficients, -7, axes), axes, coefficients, -7)
        t0, t1 = s.symbols("t_0 t_1", integer=True)
        n = s.Symbol("n1", integer=True)
        assert family == (t0, t0 + 5*t1 + 14, -t0 - 3*t1 - 7, n)
        for target in ((-1, 3, 0), (1, 0, 1), (2, 1, 0), (7, 1, -2)):
            assert 2*target[0] + 3*target[1] + 5*target[2] == 7
            assert (target[1] - target[0] - 14) % 5 == 0
            second = (target[1] - target[0] - 14) // 5
            point = tuple(value.subs({t0: target[0], t1: second, n: 11}) for value in family)
            assert point == (*target, 11)
    register("three_active_primitive_family", three_active)

    for constant in (1, 3, 5, -1):
        def inconsistent(constant=constant):
            axes = variables()
            assert publication(retained((2, 4, 0), -constant, axes), axes, (2, 4, 0), -constant) == set()
        register("checked_divisibility_obstruction_%s" % constant, inconsistent)

    def captured_names():
        axes = (s.Symbol("a", integer=True), s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True))
        family = one(retained((2, 4, 0), -6, axes), axes, (2, 4, 0), -6)
        parameters = set().union(*(value.free_symbols for value in family))
        assert len(parameters) == 2 and not parameters.intersection(axes)
        assert type(family[1]) is s.Dummy and type(family[2]) is s.Dummy
        assert all(value.is_integer is True for value in parameters)
        assert family[0].subs(family[1], -3) == 9
    register("fresh_parameters_prevent_source_capture", captured_names)

    def distinct_dummies():
        x, y = s.Dummy("same", integer=True), s.Dummy("same", integer=True)
        z = s.Symbol("same", integer=True)
        source = retained((2, 4, 0), -6, (x, y, z))
        family = one(source, (z, x, y), (0, 2, 4), -6)
        t, n = s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True)
        assert family == (n, 3 - 2*t, t)
    register("same_printed_dummy_identities_remain_distinct", distinct_dummies)

    def guarded(unknown):
        x, y, z = variables()
        base = z if unknown else (x*x + 1)*(z*z + 1)
        pole = s.Pow(base, -1, evaluate=False)
        residual = retained((0, 1, 0), -1, (x, y, z))
        source = s.Eq(pole, s.Add(pole, residual, evaluate=False), evaluate=False)
        call = lambda: one(source, (x, y, z), (0, 1, 0), -1)
        if unknown:
            expect_error(NotImplementedError, "unresolved parameter conditions", call)
        else:
            assert call() == (s.Symbol("n1", integer=True), 1, s.Symbol("n2", integer=True))
    register("original_always_nonzero_guards", lambda: guarded(False))
    register("original_unknown_guard_refuses_unconditional_family", lambda: guarded(True))

    def finite_index():
        axes = variables()
        def incomplete(_coeff, _axes, _constant, parameters=None):
            t = parameters[0]
            return {(3 - 4*t, 2*t)}
        with patch.object(linear, "_diop_linear", side_effect=incomplete):
            expect_error(ArithmeticError, "omits integer solutions",
                         lambda: publication(retained((2, 4, 0), -6, axes), axes, (2, 4, 0), -6))
    register("residual_zero_finite_index_subfamily_rejected", finite_index)

    for slot_name in ("_args", "_value"):
        def mutate_source(slot_name=slot_name):
            axes = variables()
            source = retained((2, 4, 0), -6, axes)
            original = linear._diop_linear
            owner = s.Add if slot_name == "_args" else s.Basic
            slot = owner.__dict__[slot_name]
            prior = slot.__get__(source, owner)
            def mutate(*values, **kwargs):
                result = original(*values, **kwargs)
                replacement = (s.Integer(1),) if slot_name == "_args" else s.Integer(1)._value
                slot.__set__(source, replacement)
                return result
            try:
                with patch.object(linear, "_diop_linear", side_effect=mutate):
                    expect_error(RuntimeError, "source mutated", lambda: publication(source, axes, (2, 4, 0), -6))
            finally:
                slot.__set__(source, prior)
        register("original_source_mutation_" + slot_name, mutate_source)

    for generated in (False, True):
        def mutate_assumptions(generated=generated):
            axes = variables()
            source = retained((2, 4, 0), -6, axes)
            original = linear._diop_linear
            changed = []
            def mutate(*values, **kwargs):
                result = original(*values, **kwargs)
                atom = values[3][0] if generated else axes[2]
                declared = s.Symbol.__dict__["_assumptions"].__get__(atom, s.Symbol)
                changed.append((declared, dict(declared)))
                declared["integer"] = False
                return result
            try:
                with patch.object(linear, "_diop_linear", side_effect=mutate):
                    expect_error(RuntimeError, "symbol assumptions mutated",
                                 lambda: publication(source, axes, (2, 4, 0), -6))
            finally:
                for declared, prior in changed:
                    declared.clear()
                    declared.update(prior)
        register("generated_parameter_mutation" if generated else "original_symbol_mutation", mutate_assumptions)

    def late_output_mutation():
        axes = variables()
        source = retained((2, 4, 0), -6, axes)
        original = linear._check_diophantine_domains
        slot = s.Basic.__dict__["_value"]
        changed = []
        def mutate(*values, **kwargs):
            result = original(*values, **kwargs)
            family, = result
            changed.append((family[0], slot.__get__(family[0], s.Basic)))
            slot.__set__(family[0], s.Integer(999)._value)
            return result
        try:
            with patch.object(linear, "_check_diophantine_domains", side_effect=mutate):
                expect_error(RuntimeError, "source mutated", lambda: publication(source, axes, (2, 4, 0), -6))
        finally:
            for expression, prior in changed:
                slot.__set__(expression, prior)
    register("post_domain_result_mutation_rejected", late_output_mutation)

    def stale_generator_payload():
        axes = variables()
        source = retained((2, 4, 0), -6, axes)
        original = linear._diop_linear
        retained_candidates = []
        def stale(*values, **kwargs):
            candidate, = original(*values, **kwargs)
            held = s.Add(*candidate[0].args, evaluate=False)
            held._value = s.Integer(999)._value
            retained_candidates.append(held)
            return {(held, *candidate[1:])}
        with patch.object(linear, "_diop_linear", side_effect=stale):
            family = one(source, axes, (2, 4, 0), -6)
        t, n = s.Symbol("t_0", integer=True), s.Symbol("n1", integer=True)
        assert family == (3 - 2*t, t, n)
        assert family[0] is not retained_candidates[0]
        for value in (-3, 0, 7):
            assert family[0].subs(t, value) == 3 - 2*value
    register("canonical_publication_ignores_stale_generator_payload", stale_generator_payload)

    def fake_integer():
        axes = variables()
        candidate = object.__new__(s.Integer)
        candidate._value = s.Rational(2, 3)._value
        with patch.object(linear, "_diop_linear", return_value={(candidate,)}):
            expect_error(ValueError, "literal integer",
                         lambda: publication(retained((3, 0, 0), -6, axes), axes, (3, 0, 0), -6))
    register("integer_class_cannot_disguise_native_rational", fake_integer)

    for cancelled in (False, True):
        def terminal(cancelled=cancelled):
            axes = variables()
            source = retained((2, 4, 0), -6, axes)
            session = core.SupervisedCallbackSession(max_nested_steps=10_000_000 if cancelled else 0)
            if cancelled:
                session.cancel()
            with core.supervised_callback_session(session):
                expect_error((ValueError, RuntimeError), "budget|cancelled|tainted|limit",
                             lambda: publication(source, axes, (2, 4, 0), -6))
        register("terminal_cancelled_request" if cancelled else "terminal_zero_budget_request", terminal)

    def late_cancellation():
        axes = variables()
        source = retained((2, 4, 0), -6, axes)
        original = linear._diop_linear
        with core.supervised_callback_session(max_nested_steps=10_000_000) as session:
            def cancel(*values, **kwargs):
                result = original(*values, **kwargs)
                session.cancel()
                return result
            with patch.object(linear, "_diop_linear", side_effect=cancel):
                expect_error(RuntimeError, "cancelled|tainted", lambda: publication(source, axes, (2, 4, 0), -6))
    register("late_cancellation_cannot_publish_family", late_cancellation)

    def cumulative_budget():
        axes = variables()
        source = retained((2, 4, 0), -6, axes)
        with core.supervised_callback_session(max_nested_steps=10_000_000) as measured:
            one(source, axes, (2, 4, 0), -6)
        cost = measured.consumed_nested_steps
        assert cost > 0
        metrics["explicit_row_pipeline_consumed_nested_steps"] = cost
        with core.supervised_callback_session(max_nested_steps=cost - 1):
            expect_error((ValueError, RuntimeError), "budget|cancelled|tainted|limit",
                         lambda: publication(source, axes, (2, 4, 0), -6))
    register("source_solver_checker_and_publication_share_budget", cumulative_budget)

    selected = [(name, function) for name, function in cases if not args.case or name in args.case]
    if args.case and set(args.case).difference(name for name, _ in selected):
        raise ValueError("Unknown case name")
    for name, function in selected:
        started = time.monotonic()
        try:
            function()
        except Exception as error:
            result = {"name": name, "status": "failed", "exception": type(error).__name__,
                      "message": str(error), "traceback": traceback.format_exc()}
            print("FAIL", name, type(error).__name__, str(error), flush=True)
            print(result["traceback"], flush=True)
        else:
            result = {"name": name, "status": "passed"}
            print("PASS", name, flush=True)
        result["seconds"] = round(time.monotonic() - started, 6)
        evidence["results"].append(result)
        report.write_text(json.dumps(evidence, indent=2) + "\n")

    after = source_hashes(repo)
    evidence.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "post_sources": after,
        "changed_sources": sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name)),
        "post_binary_sha256": digest(binary),
        "metrics": metrics,
        "passed": sum(result["status"] == "passed" for result in evidence["results"]),
        "failed": sum(result["status"] == "failed" for result in evidence["results"]),
        "case_count": len(selected),
    })
    report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({name: evidence[name] for name in ("scope", "case_count", "passed", "failed", "changed_sources")}), flush=True)
    return int(bool(evidence["failed"] or evidence["changed_sources"]
                    or evidence["binary_sha256"] != evidence["post_binary_sha256"]))


if __name__ == "__main__":
    raise SystemExit(main())
