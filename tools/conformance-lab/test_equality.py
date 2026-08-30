"""Equality/hash twin lane cannot certify and cannot rewrite goldens."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    CANDIDATE_SIDE,
    ORACLE_SIDE,
    cmd_equality,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from equality_records import diff_equality_records, make_equality_record  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _twin(side: str, *, equal: bool = True, hashes: bool = True, same: bool = True) -> dict:
    return make_equality_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="core/integer/42",
        side=side,
        construction_outcome="returned",
        equal_to_twin=equal,
        hashes_agree=hashes,
        is_same_object=same,
        probe_error=None,
    )


class EqualityRecordTests(unittest.TestCase):
    def test_matching_twins_are_not_a_certification(self) -> None:
        details = diff_equality_records([_twin(ORACLE_SIDE)], [_twin(CANDIDATE_SIDE)])
        self.assertEqual(details, [])

    def test_twin_inequality_is_detected(self) -> None:
        details = diff_equality_records(
            [_twin(ORACLE_SIDE)],
            [_twin(CANDIDATE_SIDE, equal=False, hashes=False, same=False)],
        )
        self.assertEqual(details[0]["kind"], "equality_identity_drift")
        self.assertTrue(details[0]["oracle"]["equal_to_twin"])
        self.assertFalse(details[0]["candidate"]["equal_to_twin"])

    def test_hash_not_respecting_eq_is_detected(self) -> None:
        details = diff_equality_records(
            [_twin(ORACLE_SIDE, equal=True, hashes=True, same=False)],
            [_twin(CANDIDATE_SIDE, equal=True, hashes=False, same=False)],
        )
        self.assertEqual(details[0]["kind"], "equality_identity_drift")
        self.assertTrue(details[0]["oracle"]["hash_respects_eq"])
        self.assertFalse(details[0]["candidate"]["hash_respects_eq"])

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        oracle = _twin(ORACLE_SIDE)
        candidate = make_equality_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side=CANDIDATE_SIDE,
            construction_outcome="refused",
            equal_to_twin=None,
            hashes_agree=None,
            is_same_object=None,
            probe_error=None,
        )
        details = diff_equality_records([oracle], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class EqualityCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["equality", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "equality")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_equality(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
