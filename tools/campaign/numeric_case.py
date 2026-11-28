"""One connected numerical run for the exact-source, two-equation campaign case."""

import base64
from fractions import Fraction
import hashlib
import math

CASE_ID = "hero_transcendental_2x2_enclosure"
DOMAIN = ((Fraction(1, 8), 2), (0, 2))
PRECISION = 30
RESIDUAL_LIMITS = {"frankennumpy": Fraction(1, 10**12), "frankenscipy": Fraction(1, 10**8)}


def build_system(sp, x, y):
    return (sp.exp(x / 2) + y - 3, x + sp.sin(y) - sp.Rational(6, 5))


def _bounds(intervals):
    # Preserve the checker's exact endpoints across the JSON boundary.
    return [[str(Fraction(lo.p, lo.q)), str(Fraction(hi.p, hi.q))] for lo, hi in intervals]


def run_numeric_case(sp, x, y):
    sources = build_system(sp, x, y)
    axes = (x, y)
    # Share a finite native work allowance through compilation, replay, both
    # proposals and all original-source checks, including rejected trials.
    budget = sp.EvaluationBudget(max_steps=1 << 32, max_memory_bytes=1 << 32,
                                 max_allocations=1 << 26)
    program_bytes = sp.compile_residual_system(sources, axes, budget=budget).to_bytes(budget=budget)
    # Replay the proof-carrying bytes against the original source and typed axes.
    program = sp.load_residual_system(program_bytes, residuals=sources, args=axes, budget=budget)
    point = [1, 1]
    values, jacobian = program.evaluate(*point, budget=budget)
    box = program.enclose(DOMAIN, precision=PRECISION, budget=budget)
    consumers = {}
    for backend in RESIDUAL_LIMITS:
        candidate = program.solve_candidate(point, backend=backend, budget=budget)
        exact_point = [Fraction.from_float(c) for c in candidate.root]
        checked = program.enclose([(c, c) for c in exact_point], precision=PRECISION, budget=budget)
        consumers[backend] = {
            "backend": candidate.backend,
            "evidence": candidate.evidence,
            "root": list(candidate.root),
            "point_domain": _bounds(checked.domain),
            "residual_bounds": _bounds(checked.residual_values),
        }
    result = {
        "case_id": CASE_ID,
        "status": "completed",
        "num_residuals": program.native_system.residual_count,
        "num_vars": program.native_system.variable_count,
        "program_base64": base64.b64encode(program_bytes).decode("ascii"),
        "program_sha256": hashlib.sha256(program_bytes).hexdigest(),
        "evaluation_point": point,
        "compiled_residuals": values,
        "compiled_jacobian": jacobian,
        "box_enclosure": {
            "precision_digits": box.precision_digits,
            "domain": _bounds(box.domain),
            "residual_bounds": _bounds(box.residual_values),
            "jacobian_bounds": [_bounds(row) for row in box.jacobian],
        },
        "consumers": consumers,
    }
    validate_numeric_case(result)
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_bounds(interval):
    if len(interval) != 2 or not all(isinstance(endpoint, str) for endpoint in interval):
        raise ValueError("expected two exact rational endpoint strings")
    lo, hi = map(Fraction, interval)
    _require(lo <= hi, "reversed enclosure endpoints")
    return lo, hi


def validate_numeric_case(case):
    """Reject detached roots, swapped derivatives and float-only or empty evidence."""
    _require(case["case_id"] == CASE_ID and case["status"] == "completed",
             "missing completed transcendental case")
    _require(case["num_residuals"] == case["num_vars"] == 2, "wrong system shape")
    payload = base64.b64decode(case["program_base64"], validate=True)
    _require(payload and hashlib.sha256(payload).hexdigest() == case["program_sha256"],
             "compiled program digest mismatch")
    _require(case["evaluation_point"] == [1, 1], "wrong evaluation point")
    rows = [case["compiled_residuals"], *case["compiled_jacobian"]]
    expected = [[math.exp(0.5) - 2, 1 + math.sin(1) - 1.2],
                [0.5 * math.exp(0.5), 1], [1, math.cos(1)]]
    for row, reference in zip(rows, expected, strict=True):
        for value, wanted in zip(row, reference, strict=True):
            _require(math.isclose(value, wanted, abs_tol=1e-12, rel_tol=1e-12),
                     "compiled value differs from analytic reference")

    box = case["box_enclosure"]
    _require(box["precision_digits"] == PRECISION and
             tuple(_read_bounds(p) for p in box["domain"]) == DOMAIN,
             "enclosure precision or box mismatch")
    # Interior rational witnesses for extrema of the exact functions on DOMAIN.
    # exp(x/2) ranges from exp(1/16) to e, and [0,2] includes both 0 and pi/2.
    # cos(2) < -1/4. The two cross derivatives are identically one.
    interval_rows = [box["residual_bounds"], *box["jacobian_bounds"]]
    witnesses = [[("-3/2", "3/2"), ("-1", "9/5")],
                 [("3/5", "4/3"), ("1", "1")], [("1", "1"), ("-1/4", "1")]]
    for row, reference in zip(interval_rows, witnesses, strict=True):
        for interval, (low, high) in zip(row, reference, strict=True):
            lo, hi = _read_bounds(interval)
            _require(lo <= Fraction(low) <= Fraction(high) <= hi,
                     "box enclosure excludes a known residual or Jacobian value")

    consumers = case["consumers"]
    _require(consumers.keys() == RESIDUAL_LIMITS.keys(), "missing numerical consumer")
    for backend, limit in RESIDUAL_LIMITS.items():
        candidate = consumers[backend]
        _require(candidate["backend"] == backend and candidate["evidence"] == "approximate",
                 "consumer identity or evidence class mismatch")
        # A small bound at some other point says nothing about this candidate.
        for value, pair, (lo, hi) in zip(candidate["root"], candidate["point_domain"],
                                         DOMAIN, strict=True):
            point = Fraction.from_float(value)
            _require(_read_bounds(pair) == (point, point) and lo <= point <= hi,
                     "certified residual domain differs from the returned iterate")
        x, y = candidate["root"]
        _require(max(abs(math.exp(x / 2) + y - 3), abs(x + math.sin(y) - 1.2))
                 <= float(limit) + 1e-15, "iterate fails the analytic residual diagnostic")
        bounds = candidate["residual_bounds"]
        _require(len(bounds) == 2, "missing consumer residual")
        for interval in bounds:
            lo, hi = _read_bounds(interval)
            _require(-limit <= lo <= hi <= limit, "consumer residual exceeds declared bound")
    for a, b in zip(consumers["frankennumpy"]["root"], consumers["frankenscipy"]["root"],
                    strict=True):
        _require(abs(a - b) <= 1e-6, "consumer iterates disagree")
