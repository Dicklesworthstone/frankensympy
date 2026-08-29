"""Copy/deepcopy lane cannot certify and cannot rewrite goldens."""

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
    cmd_copy_roundtrip,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from copy_records import diff_copy_records, make_copy_record  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _held_mul(side: str, *, deep_args: list[str] | None = None) -> dict:
    args = ["2", "k"]
    return make_copy_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="held/mul_two_k",
        side=side,
        construction_outcome="returned",
        original={"type": "Mul", "module": "sympy.core.mul", "args_repr": args},
        copied={
            "type": "Mul",
            "module": "sympy.core.mul",
            "args_repr": args,
            "is_original": False,
        },
        deepcopied={
            "type": "Mul",
            "module": "sympy.core.mul",
            "args_repr": list(deep_args) if deep_args is not None else args,
            "is_original": False,
        },
    )


class CopyRecordTests(unittest.TestCase):
    def test_matching_copy_is_not_a_certification(self) -> None:
        details = diff_copy_records(
            [_held_mul(ORACLE_SIDE)], [_held_mul(CANDIDATE_SIDE)]
        )
        self.assertEqual(details, [])

    def test_deepcopy_args_collapse_is_detected(self) -> None:
        details = diff_copy_records(
            [_held_mul(ORACLE_SIDE)],
            [_held_mul(CANDIDATE_SIDE, deep_args=["2*k"])],
        )
        self.assertEqual(details[0]["kind"], "copy_identity_drift")
        self.assertTrue(details[0]["oracle"]["deepcopy_preserves_args"])
        self.assertFalse(details[0]["candidate"]["deepcopy_preserves_args"])

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        oracle = _held_mul(ORACLE_SIDE)
        candidate = make_copy_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="held/mul_two_k",
            side=CANDIDATE_SIDE,
            construction_outcome="refused",
            original=None,
            copied=None,
            deepcopied=None,
        )
        details = diff_copy_records([oracle], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class CopyCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["copy-roundtrip", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "copy-roundtrip")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_copy_roundtrip(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
