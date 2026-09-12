"""Independent adversarial gate for `corpus_gate.py` (`fra-rc-corpus-gate-emj`).

Reviewer-authored: every case drives `corpus_gate.main()` end to end through the
real profile, real binding computation, and an isolated ledger directory, with
only the child `capture.py` process injected. The point is to check the *wrapper*
contract rather than the report validator in isolation:

* a crashed, empty, truncated, or incomplete child can never reach exit 0;
* no invalid run may write to the ledger (no self-healing on garbage);
* ledgered-open drift is development success, unledgered drift is failure with
  the exact record set appended once;
* a complete matching run is a clean pass with the ledger untouched.

Consumer: the corpus gate closure and any future re-run of that gate.
Deletion condition: fold into `test_corpus_gate.py` once the wrapper contract is
considered stable, keeping these black-box cases.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import corpus_gate
from capture import diff_input_binding, load_profile

LAB_ROOT = Path(__file__).resolve().parent
PROFILE_PATH = LAB_ROOT / "profiles" / "sympy-1.14.0-cpython-r2-corpus.toml"


def fake_child(returncode: int, stdout: str):
    class Completed:
        pass

    completed = Completed()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = ""
    return completed


class CorpusGateWrapperTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile(PROFILE_PATH)
        self.binding = diff_input_binding(self.profile)
        self.fixtures = list(self.binding["fixture_ids"])
        self.assertGreaterEqual(len(self.fixtures), 2, "frozen corpus must be non-trivial")
        self.ledger_dir = Path(tempfile.mkdtemp(prefix="corpus-gate-review-"))
        (self.ledger_dir / self.profile["profile_id"]).mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.ledger_dir / self.profile["profile_id"] / "ledger.json"

    def run_gate(self, child) -> tuple[int, str, str]:
        with (
            patch.object(corpus_gate, "ARTIFACT_ROOT", self.ledger_dir),
            patch.object(corpus_gate.subprocess, "run", return_value=child),
            patch.object(sys, "argv", ["corpus_gate.py", "--profile", str(PROFILE_PATH)]),
        ):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = corpus_gate.main()
        return code, out.getvalue(), err.getvalue()

    def report(self, *, admitted=None, drifted=None, with_details=True) -> dict:
        admitted = list(self.fixtures) if admitted is None else list(admitted)
        drifted = list(drifted or [])
        details = []
        if with_details:
            for fixture_id in drifted:
                details.append(
                    {
                        "fixture_id": fixture_id,
                        "kind": "observation",
                        "difference_paths": ["observations.func"],
                        "outcome_classes": ["observation"],
                        "value_differences": [
                            {"path": "observations.func", "oracle": "oracle", "candidate": "cand"}
                        ],
                    }
                )
        return {
            "schema_version": 1,
            "input_binding": copy.deepcopy(self.binding),
            "comparator": "construction_only",
            "paired": len(self.fixtures),
            "admitted": len(admitted),
            "admitted_fixture_ids": admitted,
            "type_matched": len(admitted),
            "type_matched_fixture_ids": admitted,
            "discrepancies": len(drifted),
            "fixture_ids": drifted,
            "by_kind": {"observation": len(drifted)} if drifted else {},
            "broken_candidate": False,
            "affected_claim": None,
            "named_claims": [],
            "claims_promoted": False,
            "ledger_dir": None,
            "ledger_records": 0,
            "details": details,
        }

    def assert_no_ledger(self):
        self.assertFalse(
            self.ledger_path.exists(), f"invalid run must not write {self.ledger_path}"
        )

    def test_complete_matching_run_passes_and_leaves_the_ledger_alone(self):
        child = fake_child(0, json.dumps(self.report()))
        code, out, _ = self.run_gate(child)
        self.assertEqual(code, 0, out)
        payload = json.loads(out[out.index("{") :])
        self.assertEqual(payload["admitted"], len(self.fixtures))
        self.assertEqual(payload["drift_total"], 0)
        self.assertEqual(payload["unledgered"], 0)
        self.assert_no_ledger()

    def test_crashed_child_with_plausible_json_never_passes(self):
        # The original audit finding: returncode 1 with `{}` used to exit 0.
        for returncode, stdout in (
            (1, "{}"),
            (0, "{}"),
            (1, json.dumps(self.report(admitted=[], drifted=[]))),
            (-9, json.dumps(self.report())),
            (124, json.dumps(self.report())),
        ):
            with self.subTest(returncode=returncode, stdout=stdout[:24]):
                code, _, err = self.run_gate(fake_child(returncode, stdout))
                self.assertNotEqual(code, 0, err)
                self.assert_no_ledger()

    def test_incomplete_or_malformed_reports_never_pass(self):
        base = self.report()
        mutants = {
            "empty-stdout": "".join(["", ""]),
            "truncated-json": json.dumps(base)[:-8],
            "zero-fixtures": json.dumps(self.report(admitted=[], drifted=[])),
            "missing-fixture": json.dumps(
                self.report(admitted=self.fixtures[1:], drifted=self.fixtures[:1], with_details=False)
            ),
            "duplicate-admitted": json.dumps(
                {**base, "admitted_fixture_ids": [self.fixtures[0]] * len(self.fixtures)}
            ),
            "wrong-profile": json.dumps(
                {**base, "input_binding": {**copy.deepcopy(self.binding), "fixture_ids": ["x"]}}
            ),
            "details-missing": json.dumps(
                self.report(
                    admitted=self.fixtures[1:], drifted=self.fixtures[:1], with_details=False
                )
            ),
        }
        for name, raw in mutants.items():
            with self.subTest(case=name):
                code, _, err = self.run_gate(fake_child(1, raw))
                self.assertNotEqual(code, 0, err)
                self.assert_no_ledger()

    def test_child_execution_failure_is_refused_without_ledger_write(self):
        with (
            patch.object(corpus_gate, "ARTIFACT_ROOT", self.ledger_dir),
            patch.object(corpus_gate.subprocess, "run", side_effect=OSError("spawn failed")),
            patch.object(sys, "argv", ["corpus_gate.py", "--profile", str(PROFILE_PATH)]),
        ):
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = corpus_gate.main()
        self.assertNotEqual(code, 0, err.getvalue())
        self.assert_no_ledger()

    def test_ledgered_open_drift_is_development_success(self):
        drifted = self.fixtures[:1]
        admitted = self.fixtures[1:]
        self.ledger_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "records": [
                        {
                            "fixture_id": drifted[0],
                            "difference_paths": ["observations.func"],
                            "status": "open",
                            "profile_id": self.profile["profile_id"],
                            "comparator": "construction_only",
                        }
                    ],
                }
            )
        )
        child = fake_child(1, json.dumps(self.report(admitted=admitted, drifted=drifted)))
        code, out, err = self.run_gate(child)
        self.assertEqual(code, 0, f"{out}\n{err}")
        payload = json.loads(out[out.index("{") :])
        self.assertEqual(payload["drift_total"], 1)
        self.assertEqual(payload["ledgered_open"], 1)
        self.assertEqual(payload["unledgered"], 0)

    def test_unledgered_drift_appends_exactly_once_then_passes(self):
        drifted = self.fixtures[:2]
        admitted = self.fixtures[2:]
        child = fake_child(1, json.dumps(self.report(admitted=admitted, drifted=drifted)))

        code, out, _ = self.run_gate(child)
        self.assertEqual(code, 1, "unledgered drift must fail the gate")
        recorded = json.loads(self.ledger_path.read_text())["records"]
        self.assertEqual(
            {(r["fixture_id"], tuple(sorted(r["difference_paths"]))) for r in recorded},
            {(fixture_id, ("observations.func",)) for fixture_id in drifted},
            "the appended record set must equal the observed drift set",
        )
        self.assertTrue(all(r["status"] == "open" for r in recorded))

        # A re-run with the same complete report is now ledgered-open.
        code, out, _ = self.run_gate(child)
        self.assertEqual(code, 0, out)
        again = json.loads(self.ledger_path.read_text())["records"]
        self.assertEqual(len(again), len(recorded), "records must not be duplicated")

    def test_injected_child_receives_the_declared_profile_and_timeout(self):
        captured = {}

        def record_run(command, **kwargs):
            captured["command"] = command
            captured["timeout"] = kwargs.get("timeout")
            return fake_child(0, json.dumps(self.report()))

        with (
            patch.object(corpus_gate, "ARTIFACT_ROOT", self.ledger_dir),
            patch.object(corpus_gate.subprocess, "run", side_effect=record_run),
            patch.object(sys, "argv", ["corpus_gate.py", "--profile", str(PROFILE_PATH)]),
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = corpus_gate.main()
        self.assertEqual(code, 0)
        self.assertIn("capture.py", " ".join(captured["command"]))
        self.assertIn(str(PROFILE_PATH), captured["command"])
        expected_timeout = 30 + 120 * len(self.profile["inventory"]["fixtures"])
        self.assertEqual(captured["timeout"], expected_timeout)


if __name__ == "__main__":
    unittest.main()
