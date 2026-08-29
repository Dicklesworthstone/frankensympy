"""Reconstruction lane cannot certify and cannot rewrite goldens."""

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
    cmd_reconstruct,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from reconstruction_records import (  # noqa: E402
    diff_reconstruction_records,
    make_reconstruction_record,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _held_mul(side: str, *, recon_args: list[str] | None = None) -> dict:
    args = ["2", "k"]
    return make_reconstruction_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="held/mul_two_k",
        side=side,
        construction_outcome="returned",
        original={"type": "Mul", "module": "sympy.core.mul", "args_repr": args},
        reconstructed={
            "type": "Mul",
            "module": "sympy.core.mul",
            "args_repr": list(recon_args) if recon_args is not None else args,
            "is_original": False,
        },
    )


class ReconstructionRecordTests(unittest.TestCase):
    def test_matching_reconstruction_is_not_a_certification(self) -> None:
        details = diff_reconstruction_records(
            [_held_mul(ORACLE_SIDE)], [_held_mul(CANDIDATE_SIDE)]
        )
        self.assertEqual(details, [])

    def test_reconstruction_args_collapse_is_detected(self) -> None:
        details = diff_reconstruction_records(
            [_held_mul(ORACLE_SIDE)],
            [_held_mul(CANDIDATE_SIDE, recon_args=["2*k"])],
        )
        self.assertEqual(details[0]["kind"], "reconstruction_identity_drift")
        self.assertTrue(details[0]["oracle"]["reconstruction_preserves_args"])
        self.assertFalse(details[0]["candidate"]["reconstruction_preserves_args"])

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        oracle = _held_mul(ORACLE_SIDE)
        candidate = make_reconstruction_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="held/mul_two_k",
            side=CANDIDATE_SIDE,
            construction_outcome="refused",
            original=None,
            reconstructed=None,
        )
        details = diff_reconstruction_records([oracle], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class ReconstructionCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["reconstruct", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "reconstruct")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_reconstruct(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
