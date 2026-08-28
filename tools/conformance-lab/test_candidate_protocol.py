"""Candidate process protocol and oracle/candidate isolation gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    CANDIDATE_SIDE,
    ORACLE_SIDE,
    capture_candidate_file,
    isolation_report,
    load_goldens,
    load_profile,
    oracle_python,
)
from comparators import diff_envelopes  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
RUNNER = LAB / "candidate_runner.py"
FIXTURE = LAB / "fixtures" / "seed_core_atoms.json"


def _oracle_python_or_skip() -> str:
    try:
        return oracle_python(None)
    except SystemExit as exc:
        raise unittest.SkipTest(str(exc)) from exc


class CandidateProtocolTests(unittest.TestCase):
    def test_harness_does_not_import_sympy(self) -> None:
        self.assertNotIn("sympy", sys.modules)

    def test_broken_candidate_emits_candidate_side_and_wrong_identity(self) -> None:
        profile = load_profile(PROFILE_PATH)
        envelopes = capture_candidate_file(
            profile,
            FIXTURE,
            sys.executable,
            broken=True,
        )

        self.assertGreaterEqual(len(envelopes), 1)
        for envelope in envelopes:
            self.assertEqual(envelope["side"], CANDIDATE_SIDE)
            self.assertEqual(envelope["outcome_class"], "returned")
            self.assertEqual(envelope["observations"]["type"], "BrokenCandidate")
            self.assertEqual(envelope["observations"]["module"], "broken_candidate")

    def test_construction_only_rejects_broken_candidate_against_oracle_goldens(
        self,
    ) -> None:
        profile = load_profile(PROFILE_PATH)
        goldens = load_goldens(profile)
        oracle = goldens["seed_core_atoms.ndjson"]
        candidate = capture_candidate_file(
            profile, FIXTURE, sys.executable, broken=True
        )

        self.assertEqual(len(oracle), len(candidate))
        rejected = [
            pair[0]["fixture_id"]
            for pair in zip(oracle, candidate)
            if diff_envelopes(pair[0], pair[1], "construction_only")
        ]
        self.assertEqual(rejected, [envelope["fixture_id"] for envelope in oracle])
        self.assertTrue(all(envelope["side"] == ORACLE_SIDE for envelope in oracle))

    def test_candidate_runner_refuses_oracle_sympy_as_the_shell(self) -> None:
        py = _oracle_python_or_skip()
        profile = load_profile(PROFILE_PATH)
        decoy = Path(tempfile.mkdtemp(prefix="fsym-empty-candidate-root-"))
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PYTHON")
        }
        env.update(profile["environment"]["env_overrides"])
        env["LC_ALL"] = profile["environment"]["locale"]
        env["TZ"] = profile["environment"]["timezone"]
        env["FSYM_CANDIDATE_ROOT"] = str(decoy)
        proc = subprocess.run(
            [
                py,
                "-P",
                "-s",
                str(RUNNER),
                str(FIXTURE),
                profile["profile_id"],
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )

        self.assertEqual(proc.returncode, 3, proc.stdout)
        payload = json.loads(proc.stdout.splitlines()[0])
        self.assertEqual(payload["error_class"], "isolation_violation")
        self.assertNotIn("BrokenCandidate", proc.stdout)

    def test_isolation_report_rejects_shared_oracle_import(self) -> None:
        py = _oracle_python_or_skip()
        profile = load_profile(PROFILE_PATH)
        report = isolation_report(profile, py)

        self.assertEqual(report["status"], "passed", report)
        self.assertFalse(report["harness_imported_sympy"])
        self.assertTrue(report["wrong_candidate_root_rejected"])
        self.assertTrue(report["oracle_did_not_import_fsym_python"])
        self.assertFalse(
            Path(report["oracle_sympy_file"])
            .resolve()
            .is_relative_to(LAB.parents[1] / "python" / "sympy")
        )


if __name__ == "__main__":
    unittest.main()
