"""Adversarial report admission, separate from mathematical/profile parity."""

import copy
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import corpus_gate


class CorpusReportTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"profile_id": "test", "profile_sha256": "profile",
                        "fixtures_sha256": "fixtures", "goldens_sha256": "goldens",
                        "fixture_ids": ["a", "b"]}
        self.report = {
            "schema_version": 1, "input_binding": self.binding,
            "comparator": "construction_only", "paired": 2, "discrepancies": 0,
            "admitted": 2, "admitted_fixture_ids": ["a", "b"],
            "type_matched": 2, "type_matched_fixture_ids": ["a", "b"],
            "by_kind": {}, "broken_candidate": False, "affected_claim": None,
            "named_claims": [], "claims_promoted": False, "ledger_dir": None,
            "ledger_records": 0, "fixture_ids": [], "details": [],
        }

    def test_complete_positive_and_complete_drift(self):
        self.assertEqual(corpus_gate.validate_report(self.report, 0, self.binding), self.report)
        drift = self.drift_report()
        self.assertEqual(corpus_gate.validate_report(drift, 1, self.binding), drift)

    def drift_report(self):
        result = copy.deepcopy(self.report)
        result.update(admitted=1, admitted_fixture_ids=["a"], discrepancies=1,
                      fixture_ids=["b"], by_kind={"surface_identity_drift": 1},
                      details=[{"fixture_id": "b", "kind": "surface_identity_drift",
                                "difference_paths": ["observations.func"],
                                "outcome_classes": None,
                                "value_differences": [{"path": "observations.func",
                                                       "oracle": "Add", "candidate": "Mul"}]}])
        return result

    def test_every_required_field_is_required_and_unknown_fields_refuse(self):
        for key in self.report:
            with self.subTest(missing=key):
                value = copy.deepcopy(self.report)
                del value[key]
                with self.assertRaises(ValueError):
                    corpus_gate.validate_report(value, 0, self.binding)
        value = dict(self.report, unknown=True)
        with self.assertRaises(ValueError):
            corpus_gate.validate_report(value, 0, self.binding)

    def test_crash_cannot_be_a_pass_even_with_complete_json(self):
        for status in (1, 2, -9, 124):
            with self.subTest(status=status), self.assertRaises(ValueError):
                corpus_gate.validate_report(self.report, status, self.binding)
        with self.assertRaises(ValueError):
            corpus_gate.validate_report(self.drift_report(), 0, self.binding)

    def test_coverage_counts_modes_and_binding_mutants(self):
        mutants = [
            ("admitted_fixture_ids", ["a", "a"]), ("admitted_fixture_ids", ["a"]),
            ("admitted_fixture_ids", ["a", "foreign"]), ("paired", 0),
            ("admitted", True), ("discrepancies", 1), ("schema_version", True),
            ("comparator", "exact_surface"), ("claims_promoted", True),
            ("broken_candidate", True), ("type_matched_fixture_ids", []),
            ("details", [{}]), ("by_kind", {"fake": 0}), ("input_binding", {}),
        ]
        for key, value in mutants:
            with self.subTest(key=key), self.assertRaises(ValueError):
                corpus_gate.validate_report(dict(self.report, **{key: value}), 0, self.binding)
        for key in self.binding:
            changed = copy.deepcopy(self.binding)
            changed[key] = "changed"
            with self.subTest(binding=key), self.assertRaises(ValueError):
                corpus_gate.validate_report(self.report, 0, changed)
        empty = dict(self.binding, fixture_ids=[])
        with self.assertRaises(ValueError):
            corpus_gate.validate_report(dict(self.report, input_binding=empty), 0, empty)

    def test_drift_requires_unique_complete_observations(self):
        mutations = [
            ("fixture_id", "foreign"), ("difference_paths", []),
            ("difference_paths", ["x", "x"]), ("value_differences", []),
            ("kind", None),
        ]
        for key, value in mutations:
            report = self.drift_report()
            report["details"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                corpus_gate.validate_report(report, 1, self.binding)

    def test_original_empty_json_crash_returns_error_without_ledger_write(self):
        # Reproduce the actual wrapper bug, not just a helper assertion. Only
        # the crashed subprocess is injected; profile and corpus reads are real.
        for stdout in ("{}", "[]", "null", "{", ""):
            child = subprocess.CompletedProcess([], 1, stdout, "candidate crashed")
            with self.subTest(stdout=stdout), patch.object(sys, "argv", ["corpus_gate.py"]), \
                    patch.object(corpus_gate.subprocess, "run", return_value=child), \
                    patch.object(corpus_gate.Path, "write_text") as write, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(corpus_gate.main(), 2)
                write.assert_not_called()
                self.assertIn("invalid diff result", err.getvalue())
                self.assertIn("candidate crashed", err.getvalue())

    def test_duplicate_json_keys_and_nonfinite_numbers_refuse(self):
        for raw in ('{"admitted": 0, "admitted": 230}', '{"x": NaN}',
                    '{"x": Infinity}', '{"x": {"a": 1, "a": 2}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                corpus_gate.parse_report(raw)

    def test_ledgered_drift_is_development_success_not_certification(self):
        record = {"fixture_id": "b", "difference_paths": ["observations.func"],
                  "status": "open", "profile_id": "test", "comparator": "construction_only"}
        child = subprocess.CompletedProcess([], 1, json.dumps(self.drift_report()), "")
        # A synthetic child/ledger exercises policy, not live mathematical parity.
        for same_profile in (True, False):
            ledger = dict(record, profile_id="test" if same_profile else "foreign")
            with self.subTest(same_profile=same_profile), \
                    patch.object(sys, "argv", ["corpus_gate.py"]), \
                    patch.object(corpus_gate, "load_profile", return_value={
                        "profile_id": "test", "inventory": {"fixtures": ["one"]}}), \
                    patch.object(corpus_gate, "diff_input_binding", return_value=self.binding), \
                    patch.object(corpus_gate.subprocess, "run", return_value=child), \
                    patch.object(corpus_gate.Path, "exists", return_value=True), \
                    patch.object(corpus_gate.Path, "read_text", return_value=json.dumps({"records": [ledger]})), \
                    patch.object(corpus_gate.Path, "mkdir"), \
                    patch.object(corpus_gate.Path, "write_text") as write, \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(corpus_gate.main(), 0 if same_profile else 1)
                # Summary precedes the optional human-readable appended notice.
                report, _ = json.JSONDecoder().raw_decode(output.getvalue())
                self.assertIs(report["certified"], False)
                self.assertEqual(report["scope"], "development_construction_drift")
                self.assertEqual(report["ledgered_open"], int(same_profile))
                self.assertEqual(write.call_count, int(not same_profile))
                if not same_profile:
                    saved = json.loads(write.call_args.args[0])
                    self.assertEqual(len(saved["records"]), 2)
                    self.assertEqual(saved["records"][-1]["profile_id"], "test")


if __name__ == "__main__":
    unittest.main()
