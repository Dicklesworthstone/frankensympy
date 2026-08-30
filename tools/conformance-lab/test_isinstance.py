"""isinstance/issubclass module-class lane cannot certify or rewrite goldens."""

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
    cmd_isinstance,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from isinstance_records import diff_isinstance_records, make_isinstance_record  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _ok(side: str, *, module_isinstance: bool = True, type_is: bool = True) -> dict:
    return make_isinstance_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="core/integer/42",
        side=side,
        construction_outcome="returned",
        isinstance_type=True,
        issubclass_type=True,
        module_class_importable=True,
        isinstance_module_class=module_isinstance,
        issubclass_module_class=module_isinstance,
        type_is_module_class=type_is,
        probe_error=None,
    )


class IsinstanceRecordTests(unittest.TestCase):
    def test_matching_module_class_is_not_a_certification(self) -> None:
        details = diff_isinstance_records([_ok(ORACLE_SIDE)], [_ok(CANDIDATE_SIDE)])
        self.assertEqual(details, [])

    def test_module_class_miss_is_detected(self) -> None:
        details = diff_isinstance_records(
            [_ok(ORACLE_SIDE)],
            [_ok(CANDIDATE_SIDE, module_isinstance=False, type_is=False)],
        )
        self.assertEqual(details[0]["kind"], "isinstance_identity_drift")
        self.assertTrue(details[0]["oracle"]["isinstance_module_class"])
        self.assertFalse(details[0]["candidate"]["isinstance_module_class"])

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        candidate = make_isinstance_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side=CANDIDATE_SIDE,
            construction_outcome="refused",
            isinstance_type=None,
            issubclass_type=None,
            module_class_importable=None,
            isinstance_module_class=None,
            issubclass_module_class=None,
            type_is_module_class=None,
            probe_error=None,
        )
        details = diff_isinstance_records([_ok(ORACLE_SIDE)], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class IsinstanceCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["isinstance", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "isinstance")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_isinstance(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
