"""Moving-head drift lane cannot certify and cannot rewrite goldens."""

from __future__ import annotations

import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    cmd_moving_head,
    golden_digest_map,
    load_goldens,
    load_profile,
    moving_head_receipt,
    oracle_python,
    parse_cli,
    validate_envelope,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


class MovingHeadReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile(PROFILE_PATH)
        goldens = load_goldens(self.profile)
        self.golden_envs = [
            envelope for envelopes in goldens.values() for envelope in envelopes
        ]

    def test_matching_head_does_not_certify(self) -> None:
        receipt = moving_head_receipt(
            profile=self.profile,
            golden_envs=self.golden_envs,
            head_envs=copy.deepcopy(self.golden_envs),
            head_python="/nonexistent/moving-head-python",
        )
        self.assertEqual(receipt["lane"], "moving_head")
        self.assertEqual(receipt["kind"], "drift_observation")
        self.assertFalse(receipt["certifies"])
        self.assertFalse(receipt["can_certify"])
        self.assertFalse(receipt["claims_promoted"])
        self.assertFalse(receipt["goldens_written"])
        self.assertTrue(receipt["profile_status_unaffected"])
        self.assertEqual(receipt["discrepancies"], 0)
        self.assertEqual(receipt["drifted_fixture_ids"], [])

    def test_printer_drift_is_detected_and_still_not_certified(self) -> None:
        drifted = copy.deepcopy(self.golden_envs)
        printers = drifted[0]["observations"]["printers"]
        printers["str"] = str(printers["str"]) + "DRIFT"
        receipt = moving_head_receipt(
            profile=self.profile,
            golden_envs=self.golden_envs,
            head_envs=drifted,
            head_python="/nonexistent/moving-head-python",
        )
        self.assertGreater(receipt["discrepancies"], 0)
        self.assertIn(drifted[0]["fixture_id"], receipt["drifted_fixture_ids"])
        self.assertEqual(receipt["details"][0]["kind"], "upstream_surface_drift")
        self.assertFalse(receipt["certifies"])
        self.assertFalse(receipt["can_certify"])

    def test_coverage_gap_is_named(self) -> None:
        receipt = moving_head_receipt(
            profile=self.profile,
            golden_envs=self.golden_envs,
            head_envs=self.golden_envs[1:],
            head_python="/nonexistent/moving-head-python",
        )
        self.assertIn(self.golden_envs[0]["fixture_id"], receipt["drifted_fixture_ids"])
        kinds = {row["kind"] for row in receipt["details"]}
        self.assertIn("coverage_gap", kinds)
        self.assertFalse(receipt["certifies"])

    def test_non_profile_sympy_version_is_allowed_on_the_head_lane_only(self) -> None:
        mutant = copy.deepcopy(self.golden_envs[0])
        mutant["environment"]["sympy_version"] = "9.9.9"
        validate_envelope(mutant, self.profile, pin_upstream_version=False)
        with self.assertRaises(ValueError):
            validate_envelope(mutant, self.profile)


class MovingHeadCliTests(unittest.TestCase):
    def test_certify_flag_is_parsed_and_refused(self) -> None:
        parsed = parse_cli(
            ["moving-head", str(PROFILE_PATH), "--certify"]
        )
        self.assertEqual(parsed["mode"], "moving-head")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_moving_head(profile, "unused-python", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_broken_flag_is_rejected_by_cli(self) -> None:
        parsed = parse_cli(
            ["moving-head", str(PROFILE_PATH), "--broken"]
        )
        self.assertTrue(parsed["broken"])
        self.assertEqual(parsed["mode"], "moving-head")

    def test_cmd_moving_head_does_not_rewrite_goldens(self) -> None:
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        with mock.patch("capture.capture_file", side_effect=AssertionError("must not recapture")):
            status = cmd_moving_head(profile, "unused-python", certify=True)
        self.assertEqual(status, 1)
        self.assertEqual(golden_digest_map(profile), before)

    def test_live_moving_head_observes_without_certifying(self) -> None:
        try:
            py = oracle_python(None)
        except SystemExit:
            self.skipTest("no oracle interpreter")
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cmd_moving_head(profile, py)
        self.assertEqual(status, 0)
        receipt = json.loads(buffer.getvalue())
        self.assertEqual(receipt["lane"], "moving_head")
        self.assertFalse(receipt["certifies"])
        self.assertFalse(receipt["can_certify"])
        self.assertFalse(receipt["claims_promoted"])
        self.assertFalse(receipt["goldens_written"])
        self.assertTrue(receipt["golden_bytes_unchanged"])
        self.assertEqual(golden_digest_map(profile), before)
        # Stale subclass goldens may differ from the current oracle runner.
        # That is drift observation; do not regenerate goldens.


if __name__ == "__main__":
    unittest.main()
