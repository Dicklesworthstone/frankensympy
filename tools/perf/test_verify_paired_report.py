"""Adversarial tests for the paired-benchmark report validator (WS22 evidence).

Each mutant is derived from one synthetic but internally consistent report, so a
refusal is caused by the mutation under test rather than by unrelated damage.
The synthetic report is never presented as a measurement: it exists to exercise
the validator's refusal paths.
"""

from __future__ import annotations

import copy
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import verify_paired_report as validator

HEAD = "0123456789abcdef0123456789abcdef01234567"
PROFILE = "sympy-1.14.0-cpython"


def samples(base: float, spread: float, rounds: int = 5) -> list[float]:
    return [round(base * (1 + spread * (index - 2) / 2), 4) for index in range(rounds)]


def case(case_id: str, status: str, subject_base: float, oracle_base: float, rounds: int = 5) -> dict:
    subject = samples(subject_base, 0.04, rounds)
    oracle = samples(oracle_base, 0.03, rounds)
    subject_median, oracle_median = statistics.median(subject), statistics.median(oracle)
    return {
        "case": case_id,
        "status": status,
        "subject_ms_median": subject_median,
        "oracle_ms_median": oracle_median,
        "subject_cv_pct": statistics.pstdev(subject) / subject_median * 100,
        "oracle_cv_pct": statistics.pstdev(oracle) / oracle_median * 100,
        "subject_dist": {"median_ms": subject_median},
        "oracle_dist": {"median_ms": oracle_median},
        "subject_samples": subject,
        "oracle_samples": oracle,
        "ratio_oracle_over_subject": oracle_median / subject_median,
    }


def report() -> dict:
    rounds = 5
    arm_a = samples(250.0, 0.03, rounds)
    arm_b = samples(252.0, 0.03, rounds)
    cases = {
        "diff_poly": case("diff_poly", "admitted", 2200.0, 340.0, rounds),
        "matrix_det4": case("matrix_det4", "admitted", 250.0, 260.0, rounds),
        "srepr_print": {
            "case": "srepr_print", "status": "divergence",
            "subject_result": "(x**2 - 1)*(x - 1)**(-1)", "oracle_result": "x + 1",
        },
        "integer_arith_rational": {
            "case": "integer_arith_rational", "status": "error",
            "subject_error": "Refused: nesting depth exceeds the bridge bound",
            "oracle_error": "ValueError: digit limit",
        },
    }
    admitted = [cid for cid, entry in cases.items() if entry["status"] == "admitted"]
    low_noise = [cid for cid in admitted if cases[cid]["subject_cv_pct"] <= 5 and cases[cid]["oracle_cv_pct"] <= 5]
    product = 1.0
    for cid in low_noise:
        product *= cases[cid]["ratio_oracle_over_subject"]
    return {
        "schema": validator.SCHEMA,
        "date": "2026-09-11T12:00:00+00:00",
        "profile_id": PROFILE,
        "subject_head": HEAD,
        "hardware": {"platform": "Linux-test", "machine": "x86_64", "cpu_count": 16},
        "toolchains": {"subject_python": "3.14.4", "oracle_python": "3.14.4", "rustc": "1.100.0"},
        "aa_control": {
            "case": "matrix_det4",
            "arm_a_samples": arm_a,
            "arm_b_samples": arm_b,
            "arm_a_median_ms": statistics.median(arm_a),
            "arm_b_median_ms": statistics.median(arm_b),
            "ratio_null_baseline": statistics.median(arm_b) / statistics.median(arm_a),
            "verified": True,
        },
        "rounds": rounds,
        "cases": cases,
        "summary": {
            "cases_total": len(cases),
            "admitted": len(admitted),
            "divergences": sorted(cid for cid, e in cases.items() if e["status"] == "divergence"),
            "errors": sorted(cid for cid, e in cases.items() if e["status"] == "error"),
            "low_noise_admitted": len(low_noise),
            "aa_control_verified": True,
            "geomean_ratio_oracle_over_subject": product ** (1.0 / len(low_noise)),
            "noise_note": "synthetic fixture",
        },
    }


def refuses(mutant: dict, **kwargs) -> str:
    with unittest.TestCase().assertRaises(validator.Refusal) as caught:
        validator.validate_report(
            mutant,
            expected_head=kwargs.get("expected_head", HEAD),
            expected_profile=kwargs.get("expected_profile", PROFILE),
            max_age_days=kwargs.get("max_age_days", 30.0),
            now=kwargs.get("now", validator.parse_timestamp("2026-09-11T13:00:00+00:00")),
            require_settings=kwargs.get("require_settings", False),
        )
    return caught.exception.code


class PairedReportValidatorTests(unittest.TestCase):
    def test_consistent_report_is_admissible_and_never_claims_a_win(self):
        verdict = validator.validate_report(
            report(),
            expected_head=HEAD,
            expected_profile=PROFILE,
            max_age_days=30.0,
            now=validator.parse_timestamp("2026-09-11T13:00:00+00:00"),
            require_settings=False,
        )
        self.assertEqual(verdict["admitted"], 2)
        self.assertEqual(verdict["divergences"], ["srepr_print"])
        self.assertEqual(verdict["errors"], ["integer_arith_rational"])
        self.assertEqual(verdict["competitive_claim"], "not_evaluated")
        self.assertFalse(verdict["settings_declared"])

    def test_historical_file_and_wrong_profile_substitution_are_refused(self):
        mutant = copy.deepcopy(report())
        mutant["subject_head"] = "f" * 40
        self.assertEqual(refuses(mutant), "stale_report")
        mutant = copy.deepcopy(report())
        mutant["profile_id"] = "sympy-1.13.0-cpython"
        self.assertEqual(refuses(mutant), "stale_report")

    def test_stale_and_future_reports_are_refused(self):
        mutant = copy.deepcopy(report())
        mutant["date"] = "2026-01-01T00:00:00+00:00"
        self.assertEqual(refuses(mutant), "stale_report")
        mutant = copy.deepcopy(report())
        mutant["date"] = "2026-09-20T00:00:00+00:00"
        self.assertEqual(refuses(mutant), "stale_report")

    def test_oracle_imports_candidate_is_refused(self):
        mutant = copy.deepcopy(report())
        entry = mutant["cases"]["diff_poly"]
        entry["oracle_samples"] = list(entry["subject_samples"])
        entry["oracle_ms_median"] = entry["subject_ms_median"]
        self.assertEqual(refuses(mutant), "wrong_engine")

    def test_failed_aa_control_is_a_typed_invalid_measurement(self):
        mutant = copy.deepcopy(report())
        mutant["aa_control"]["verified"] = False
        mutant["summary"]["aa_control_verified"] = False
        self.assertEqual(refuses(mutant), "invalid_measurement")
        # A noisy-but-"verified" A/A run is also invalid, not a speed result.
        mutant = copy.deepcopy(report())
        arm_b = [value * 2 for value in mutant["aa_control"]["arm_b_samples"]]
        mutant["aa_control"]["arm_b_samples"] = arm_b
        mutant["aa_control"]["arm_b_median_ms"] = statistics.median(arm_b)
        mutant["aa_control"]["ratio_null_baseline"] = (
            statistics.median(arm_b) / mutant["aa_control"]["arm_a_median_ms"]
        )
        self.assertEqual(refuses(mutant), "invalid_measurement")

    def test_no_admitted_cases_cannot_be_aggregated(self):
        mutant = copy.deepcopy(report())
        for case_id, entry in mutant["cases"].items():
            mutant["cases"][case_id] = {
                "case": case_id, "status": "divergence",
                "subject_result": entry.get("subject_result", "subject"),
                "oracle_result": entry.get("oracle_result", "oracle"),
            }
        mutant["summary"]["admitted"] = 0
        mutant["summary"]["divergences"] = sorted(mutant["cases"])
        mutant["summary"]["errors"] = []
        mutant["summary"]["low_noise_admitted"] = 0
        mutant["summary"]["geomean_ratio_oracle_over_subject"] = 0.0
        self.assertEqual(refuses(mutant), "no_admitted_cases")

    def test_failed_cases_must_stay_outside_the_aggregate(self):
        # Counting the failed case as admitted (a classic laundering move) is refused.
        mutant = copy.deepcopy(report())
        mutant["summary"]["admitted"] = 3
        self.assertEqual(refuses(mutant), "aggregate_mismatch")
        # Dropping a divergence from the summary is refused too.
        mutant = copy.deepcopy(report())
        mutant["summary"]["divergences"] = []
        self.assertEqual(refuses(mutant), "aggregate_mismatch")
        # A late timeout must be retained as an error, not silently removed.
        mutant = copy.deepcopy(report())
        mutant["summary"]["errors"] = []
        self.assertEqual(refuses(mutant), "aggregate_mismatch")

    def test_hand_edited_aggregates_are_refused(self):
        mutations = (
            (lambda m: m["cases"]["diff_poly"].__setitem__("ratio_oracle_over_subject", 0.01),
             "aggregate_mismatch"),
            (lambda m: m["cases"]["diff_poly"].__setitem__("subject_ms_median", 1.0),
             "aggregate_mismatch"),
            (lambda m: m["cases"]["diff_poly"].__setitem__("oracle_cv_pct", 0.1),
             "aggregate_mismatch"),
            (lambda m: m["summary"].__setitem__("geomean_ratio_oracle_over_subject", 0.01),
             "aggregate_mismatch"),
            (lambda m: m["summary"].__setitem__("low_noise_admitted", 99),
             "aggregate_mismatch"),
            (lambda m: m["aa_control"].__setitem__("ratio_null_baseline", 1.25),
             "aa_control"),
        )
        for mutate, expected in mutations:
            mutant = copy.deepcopy(report())
            mutate(mutant)
            with self.subTest(expected=expected):
                self.assertEqual(refuses(mutant), expected)

    def test_schema_drift_and_unknown_fields_fail_closed(self):
        mutant = copy.deepcopy(report())
        mutant["schema"] = "gauntlet.paired_bench.v2"
        self.assertEqual(refuses(mutant), "schema")
        mutant = copy.deepcopy(report())
        mutant["extra"] = True
        self.assertEqual(refuses(mutant), "schema")
        mutant = copy.deepcopy(report())
        del mutant["aa_control"]
        self.assertEqual(refuses(mutant), "schema")
        mutant = copy.deepcopy(report())
        mutant["rounds"] = 1
        self.assertEqual(refuses(mutant), "schema")

    def test_settings_can_be_required_explicitly(self):
        verdict = validator.validate_report(
            {**copy.deepcopy(report()), "settings": {"evidence": "strict", "threads": 1}},
            expected_head=HEAD, expected_profile=PROFILE, max_age_days=30.0,
            now=validator.parse_timestamp("2026-09-11T13:00:00+00:00"), require_settings=True,
        )
        self.assertTrue(verdict["settings_declared"])
        self.assertEqual(refuses(report(), require_settings=True), "settings_undeclared")


class PairedReportCliTests(unittest.TestCase):
    def run_cli(self, report_dict: dict, *extra: str) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report_dict))
            completed = subprocess.run(
                [sys.executable, str(Path(validator.__file__)),
                 str(path), "--expected-subject-head", HEAD, "--expected-profile", PROFILE,
                 "--now", "2026-09-11T13:00:00+00:00", *extra],
                capture_output=True, text=True, timeout=120, check=False,
            )
        return completed.returncode, json.loads(completed.stdout)

    def test_cli_exit_codes_and_typed_refusals(self):
        code, payload = self.run_cli(report())
        self.assertEqual(code, 0)
        self.assertEqual(payload["verdict"], "admissible")
        mutant = copy.deepcopy(report())
        mutant["aa_control"]["verified"] = False
        code, payload = self.run_cli(mutant)
        self.assertEqual(code, 1)
        self.assertEqual(payload["verdict"], "refused")
        self.assertEqual(payload["code"], "invalid_measurement")
        self.assertEqual(payload["competitive_claim"], "not_supported")


if __name__ == "__main__":
    unittest.main()
