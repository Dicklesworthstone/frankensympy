#!/usr/bin/env python3
"""Independent validator for paired benchmark reports (WS22 evidence integrity).

A paired report is admissible evidence only when it is a *complete, internally
consistent, freshly bound* record of one paired run. This validator recomputes
every derived number from the raw samples instead of trusting the harness's own
aggregates, and refuses:

* a report bound to another commit, profile, or host than the caller expects
  (historical-file substitution);
* a stale report (older than the declared window);
* oracle-imports-candidate substitution — a case whose subject and oracle
  samples are the same bytes;
* a failed or noisy A/A control (typed invalid measurement, never a speed win);
* aggregates that do not equal the recomputed values, or divergences/errors
  that are missing from — or padded into — the admitted aggregate.

The validator never certifies a competitive win: a speed claim additionally
requires the registered named-workload gate, which is a separate artifact and is
only checked for presence here.

Usage:
    tools/perf/verify_paired_report.py REPORT.json --expected-subject-head SHA \\
        --expected-profile sympy-1.14.0-cpython [--max-age-days 30] [--now ISO8601]
Exit 0 admissible, 1 refused (typed reason on stdout JSON), 2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "gauntlet.paired_bench.v1"
REQUIRED_KEYS = {
    "schema", "date", "profile_id", "subject_head", "hardware", "toolchains",
    "aa_control", "rounds", "cases", "summary",
}
CASE_STATUSES = {"admitted", "divergence", "error"}
# Only admitted cases carry timings; a failed or divergent case must not be able
# to carry a ratio into the aggregate, so its shape is exact and timing-free.
CASE_KEYS = {
    "case", "status", "subject_ms_median", "oracle_ms_median", "subject_cv_pct",
    "oracle_cv_pct", "subject_dist", "oracle_dist", "subject_samples",
    "oracle_samples", "ratio_oracle_over_subject",
}
DIVERGENCE_KEYS = {"case", "status", "subject_result", "oracle_result"}
ERROR_KEYS = {"case", "status", "subject_error", "oracle_error"}
MEDIAN_TOLERANCE_PCT = 0.5
RATIO_TOLERANCE = 0.005
CV_TOLERANCE_PCT = 0.5
AA_NULL_BAND = 0.25


class Refusal(Exception):
    """Typed refusal: the report is not admissible evidence."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(right))


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise Refusal(code, detail)


def numbers(values: object, expected_len: int, what: str) -> list[float]:
    require(
        isinstance(values, list) and len(values) == expected_len,
        "sample_shape",
        f"{what} must carry exactly {expected_len} raw samples",
    )
    require(
        all(type(value) in (int, float) and value >= 0 for value in values),
        "sample_shape",
        f"{what} must be non-negative numbers",
    )
    return [float(value) for value in values]


def parse_timestamp(raw: object, what: str = "timestamp") -> dt.datetime:
    require(isinstance(raw, str) and raw, "timestamp", f"{what} must be an ISO-8601 string")
    text = raw.replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError as error:
        raise Refusal("timestamp", f"{what} is not ISO-8601: {error}") from None
    require(stamp.tzinfo is not None, "timestamp", f"{what} must carry a timezone")
    return stamp


def validate_aa_control(aa: object, rounds: int) -> dict:
    require(isinstance(aa, dict), "aa_control", "aa_control must be an object")
    required = {
        "case", "arm_a_samples", "arm_b_samples", "arm_a_median_ms",
        "arm_b_median_ms", "ratio_null_baseline", "verified",
    }
    require(set(aa) == required, "aa_control", f"aa_control fields must be exactly {sorted(required)}")
    arm_a = numbers(aa["arm_a_samples"], rounds, "aa_control.arm_a_samples")
    arm_b = numbers(aa["arm_b_samples"], rounds, "aa_control.arm_b_samples")
    median_a, median_b = median(arm_a), median(arm_b)
    require(close(aa["arm_a_median_ms"], median_a, MEDIAN_TOLERANCE_PCT / 100), "aa_control",
            f"arm A median {aa['arm_a_median_ms']} != recomputed {median_a:.4f}")
    require(close(aa["arm_b_median_ms"], median_b, MEDIAN_TOLERANCE_PCT / 100), "aa_control",
            f"arm B median {aa['arm_b_median_ms']} != recomputed {median_b:.4f}")
    require(median_a > 0, "aa_control", "arm A median must be positive")
    baseline = median_b / median_a
    require(close(aa["ratio_null_baseline"], baseline, RATIO_TOLERANCE), "aa_control",
            f"ratio_null_baseline {aa['ratio_null_baseline']} != recomputed {baseline:.6f}")
    # A failed A/A control means the measurement is invalid, not that a win exists.
    require(
        aa["verified"] is True,
        "invalid_measurement",
        "A/A control did not verify; the paired measurement is invalid",
    )
    require(
        abs(baseline - 1.0) <= AA_NULL_BAND,
        "invalid_measurement",
        f"A/A null baseline {baseline:.4f} is outside the declared ±{AA_NULL_BAND:.0%} band",
    )
    return aa


def validate_case(case_id: str, case: object, rounds: int) -> str:
    require(isinstance(case, dict), "case_shape", f"case {case_id} must be an object")
    require(case.get("case") == case_id, "case_shape",
            f"case key {case_id} disagrees with its own case field {case.get('case')!r}")
    status = case.get("status")
    require(status in CASE_STATUSES, "case_shape", f"case {case_id} has unknown status {status!r}")
    if status != "admitted":
        expected = DIVERGENCE_KEYS if status == "divergence" else ERROR_KEYS
        require(set(case) == expected, "case_shape",
                f"{status} case {case_id} fields must be exactly {sorted(expected)}")
        for value in case.values():
            require(isinstance(value, str) and value, "case_shape",
                    f"{status} case {case_id} values must be non-empty strings")
        return status
    require(set(case) == CASE_KEYS, "case_shape",
            f"case {case_id} fields must be exactly {sorted(CASE_KEYS)}")
    subject = numbers(case["subject_samples"], rounds, f"case {case_id} subject_samples")
    oracle = numbers(case["oracle_samples"], rounds, f"case {case_id} oracle_samples")
    require(
        subject != oracle,
        "wrong_engine",
        f"case {case_id} carries identical subject and oracle samples: oracle-imports-candidate",
    )
    subject_median, oracle_median = median(subject), median(oracle)
    require(close(case["subject_ms_median"], subject_median, MEDIAN_TOLERANCE_PCT / 100),
            "aggregate_mismatch",
            f"case {case_id} subject median {case['subject_ms_median']} != {subject_median:.4f}")
    require(close(case["oracle_ms_median"], oracle_median, MEDIAN_TOLERANCE_PCT / 100),
            "aggregate_mismatch",
            f"case {case_id} oracle median {case['oracle_ms_median']} != {oracle_median:.4f}")
    cv_subject = statistics.pstdev(subject) / subject_median * 100 if subject_median else 0.0
    cv_oracle = statistics.pstdev(oracle) / oracle_median * 100 if oracle_median else 0.0
    require(close(case["subject_cv_pct"], cv_subject, CV_TOLERANCE_PCT),
            "aggregate_mismatch", f"case {case_id} subject cv {case['subject_cv_pct']} != {cv_subject:.4f}")
    require(close(case["oracle_cv_pct"], cv_oracle, CV_TOLERANCE_PCT),
            "aggregate_mismatch", f"case {case_id} oracle cv {case['oracle_cv_pct']} != {cv_oracle:.4f}")
    ratio = oracle_median / subject_median if subject_median else 0.0
    require(close(case["ratio_oracle_over_subject"], ratio, RATIO_TOLERANCE),
            "aggregate_mismatch",
            f"case {case_id} ratio {case['ratio_oracle_over_subject']} != {ratio:.6f}")
    for field, samples in (("subject_dist", subject), ("oracle_dist", oracle)):
        dist = case[field]
        require(isinstance(dist, dict) and "median_ms" in dist, "case_shape",
                f"case {case_id} {field} must carry a distribution with median_ms")
        require(close(dist["median_ms"], median(samples), MEDIAN_TOLERANCE_PCT / 100),
                "aggregate_mismatch", f"case {case_id} {field} median disagrees with raw samples")
    return status


def validate_report(report: object, *, expected_head: str | None, expected_profile: str | None,
                    max_age_days: float, now: dt.datetime, require_settings: bool) -> dict:
    require(isinstance(report, dict), "schema", "report must be a JSON object")
    allowed = REQUIRED_KEYS | {"settings"}
    require(set(report) == REQUIRED_KEYS or set(report) == allowed, "schema",
            f"report fields must be exactly {sorted(REQUIRED_KEYS)} plus optional settings")
    require(report["schema"] == SCHEMA, "schema", f"unsupported schema {report['schema']!r}")
    if expected_head is not None:
        require(report["subject_head"] == expected_head, "stale_report",
                f"report is bound to {report['subject_head']}, caller expects {expected_head}")
    if expected_profile is not None:
        require(report["profile_id"] == expected_profile, "stale_report",
                f"report profile {report['profile_id']!r} != expected {expected_profile!r}")
    require(re.fullmatch(r"[0-9a-f]{40}", str(report["subject_head"])) is not None,
            "schema", "subject_head must be a full lowercase commit id")
    for key in ("platform", "machine", "cpu_count"):
        require(key in report["hardware"], "schema", f"hardware is missing {key}")
    for key in ("subject_python", "oracle_python", "rustc"):
        require(isinstance(report["toolchains"], dict) and report["toolchains"].get(key),
                "schema", f"toolchains is missing {key}")
    stamp = parse_timestamp(report["date"], "date")
    age_days = (now - stamp).total_seconds() / 86400
    require(age_days <= max_age_days, "stale_report",
            f"report is {age_days:.2f} days old, limit is {max_age_days}")
    require(age_days >= -1.0, "stale_report", "report date is in the future")

    rounds = report["rounds"]
    require(type(rounds) is int and 2 <= rounds <= 1000, "schema", f"rounds must be 2..1000, got {rounds!r}")
    aa = validate_aa_control(report["aa_control"], rounds)

    cases = report["cases"]
    require(isinstance(cases, dict) and cases, "schema", "cases must be a non-empty object")
    statuses = {case_id: validate_case(case_id, case, rounds) for case_id, case in cases.items()}

    summary = report["summary"]
    require(isinstance(summary, dict), "schema", "summary must be an object")
    admitted = sorted(case_id for case_id, status in statuses.items() if status == "admitted")
    divergences = sorted(case_id for case_id, status in statuses.items() if status == "divergence")
    errors = sorted(case_id for case_id, status in statuses.items() if status == "error")
    require(summary.get("cases_total") == len(cases), "aggregate_mismatch",
            f"cases_total {summary.get('cases_total')} != {len(cases)}")
    require(summary.get("admitted") == len(admitted), "aggregate_mismatch",
            f"admitted {summary.get('admitted')} != recomputed {len(admitted)}")
    require(sorted(summary.get("divergences", [])) == divergences, "aggregate_mismatch",
            f"divergences {summary.get('divergences')} != recomputed {divergences}")
    require(sorted(summary.get("errors", [])) == errors, "aggregate_mismatch",
            f"errors {summary.get('errors')} != recomputed {errors}")
    require(summary.get("aa_control_verified") == aa["verified"], "aggregate_mismatch",
            "summary.aa_control_verified disagrees with the A/A control")
    require(bool(admitted), "no_admitted_cases",
            "no case reached admitted status: nothing can be aggregated")

    low_noise = [
        case_id for case_id in admitted
        if cases[case_id]["subject_cv_pct"] <= 5.0 and cases[case_id]["oracle_cv_pct"] <= 5.0
    ]
    require(summary.get("low_noise_admitted") == len(low_noise), "aggregate_mismatch",
            f"low_noise_admitted {summary.get('low_noise_admitted')} != recomputed {len(low_noise)}")
    product = 1.0
    for case_id in low_noise:
        product *= cases[case_id]["ratio_oracle_over_subject"]
    geomean = product ** (1.0 / len(low_noise)) if low_noise else 0.0
    reported = summary.get("geomean_ratio_oracle_over_subject")
    require(reported is not None and close(float(reported), geomean, 0.01), "aggregate_mismatch",
            f"geomean {reported} != recomputed {geomean:.6f}")

    settings = report.get("settings")
    if require_settings:
        require(isinstance(settings, dict) and settings, "settings_undeclared",
                "report does not declare evidence/cache/thread/budget settings")
    return {
        "schema": SCHEMA,
        "subject_head": report["subject_head"],
        "profile_id": report["profile_id"],
        "rounds": rounds,
        "cases_total": len(cases),
        "admitted": len(admitted),
        "divergences": divergences,
        "errors": errors,
        "low_noise_admitted": len(low_noise),
        "aa_control_verified": aa["verified"],
        "aa_null_baseline": aa["ratio_null_baseline"],
        "settings_declared": isinstance(settings, dict) and bool(settings),
        "competitive_claim": "not_evaluated",
        "note": "admissibility only: a speed claim additionally requires the registered "
                "named-workload gate artifact; failed cases stay outside the aggregate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-subject-head", default=None)
    parser.add_argument("--expected-profile", default=None)
    parser.add_argument("--max-age-days", type=float, default=30.0)
    parser.add_argument("--now", default=None, help="ISO-8601 override for deterministic checks")
    parser.add_argument("--require-settings", action="store_true")
    args = parser.parse_args()
    now = parse_timestamp(args.now, "--now") if args.now else dt.datetime.now(dt.timezone.utc)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"verdict": "usage_error", "detail": str(error)}))
        return 2
    try:
        verdict = validate_report(
            report,
            expected_head=args.expected_subject_head,
            expected_profile=args.expected_profile,
            max_age_days=args.max_age_days,
            now=now,
            require_settings=args.require_settings,
        )
    except Refusal as refusal:
        print(json.dumps({
            "verdict": "refused", "code": refusal.code, "detail": refusal.detail,
            "competitive_claim": "not_supported",
        }, indent=1, sort_keys=True))
        return 1
    verdict["verdict"] = "admissible"
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
