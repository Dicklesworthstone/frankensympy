"""Warning-class identity lane cannot certify and cannot rewrite goldens."""

from __future__ import annotations

import copy
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    CANDIDATE_SIDE,
    ORACLE_SIDE,
    cmd_warnings,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from warning_records import (  # noqa: E402
    diff_warning_records,
    make_warning_record,
    validate_warning_record,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _record(side: str, *, warnings=None, outcome: str = "returned", fixture: str = "core/integer/42"):
    return make_warning_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id=fixture,
        side=side,
        construction_outcome=outcome,
        warnings=warnings
        if warnings is not None
        else [{"module": "sympy.utilities.exceptions", "name": "SymPyDeprecationWarning"}],
    )


class WarningRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile(PROFILE_PATH)

    def test_matching_classes_are_not_a_certification(self) -> None:
        oracle = [_record(ORACLE_SIDE)]
        candidate = [_record(CANDIDATE_SIDE)]
        self.assertEqual(diff_warning_records(oracle, candidate), [])

    def test_dropped_warning_class_is_detected(self) -> None:
        oracle = [_record(ORACLE_SIDE)]
        candidate = [_record(CANDIDATE_SIDE, warnings=[])]
        details = diff_warning_records(oracle, candidate)
        self.assertEqual(details[0]["kind"], "warning_identity_drift")

    def test_renamed_warning_class_is_detected(self) -> None:
        oracle = [_record(ORACLE_SIDE)]
        candidate = [
            _record(
                CANDIDATE_SIDE,
                warnings=[{"module": "sympy.utilities.exceptions", "name": "WrongWarning"}],
            )
        ]
        details = diff_warning_records(oracle, candidate)
        self.assertEqual(details[0]["kind"], "warning_identity_drift")

    def test_construction_outcome_mismatch_is_not_silent_empty_match(self) -> None:
        oracle = [_record(ORACLE_SIDE, warnings=[], outcome="returned")]
        candidate = [_record(CANDIDATE_SIDE, warnings=[], outcome="refused")]
        details = diff_warning_records(oracle, candidate)
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")

    def test_message_text_is_outside_the_contract(self) -> None:
        record = _record(ORACLE_SIDE)
        validate_warning_record(
            record,
            self.profile,
            expected_side=ORACLE_SIDE,
            expected_id="core/integer/42",
        )
        extra = copy.deepcopy(record)
        extra["message"] = "wording is not identity"
        with self.assertRaises(ValueError):
            validate_warning_record(
                extra,
                self.profile,
                expected_side=ORACLE_SIDE,
                expected_id="core/integer/42",
            )

    def test_duplicate_warning_classes_fail_closed(self) -> None:
        record = _record(
            ORACLE_SIDE,
            warnings=[
                {"module": "builtins", "name": "UserWarning"},
                {"module": "builtins", "name": "UserWarning"},
            ],
        )
        with self.assertRaises(ValueError):
            validate_warning_record(
                record,
                self.profile,
                expected_side=ORACLE_SIDE,
                expected_id="core/integer/42",
            )


class WarningCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["warnings", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "warnings")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_warnings(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
