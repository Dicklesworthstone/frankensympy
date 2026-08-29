"""Cross-process pickle restore cannot certify and cannot rewrite goldens."""

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
    cmd_pickle_roundtrip,
    golden_digest_map,
    load_profile,
    parse_cli,
)
from pickle_records import (  # noqa: E402
    combine_roundtrip,
    diff_pickle_roundtrips,
    make_dump_record,
    make_restore_record,
    validate_dump_record,
    validate_restore_record,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


def _roundtrip(side: str, *, restored_type: str = "Integer", module: str = "sympy.core.numbers"):
    dump = make_dump_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="core/integer/42",
        side=side,
        construction_outcome="returned",
        pickle_sha256="a" * 64,
        pickle_b64="QQ==",
        dump_error=None,
    )
    restore = make_restore_record(
        fixture_id="core/integer/42",
        side=side,
        status="returned",
        restored_type=restored_type,
        module=module,
    )
    return combine_roundtrip(dump, restore)


class PickleRecordsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile(PROFILE_PATH)

    def test_valid_dump_record_validation(self) -> None:
        dump = make_dump_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side="upstream_oracle",
            construction_outcome="returned",
            pickle_sha256="a" * 64,
            pickle_b64="YnVsbGV0",
            dump_error=None,
        )
        validate_dump_record(
            dump, self.profile, expected_side="upstream_oracle", expected_id="core/integer/42"
        )

    def test_dump_record_validation_detects_mismatches(self) -> None:
        dump = make_dump_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side="upstream_oracle",
            construction_outcome="returned",
            pickle_sha256="a" * 64,
            pickle_b64="YnVsbGV0",
            dump_error=None,
        )
        with self.assertRaises(ValueError):
            validate_dump_record(
                dump,
                self.profile,
                expected_side="frankensympy_candidate",
                expected_id="core/integer/42",
            )
        with self.assertRaises(ValueError):
            validate_dump_record(
                dump, self.profile, expected_side="upstream_oracle", expected_id="core/rational/22_7"
            )

    def test_valid_restore_record_returned(self) -> None:
        restore = make_restore_record(
            fixture_id="core/integer/42",
            side="upstream_oracle",
            status="returned",
            restored_type="Integer",
            module="sympy.core.numbers",
        )
        validate_restore_record(
            restore, expected_side="upstream_oracle", expected_id="core/integer/42"
        )

    def test_valid_restore_record_raised(self) -> None:
        restore = make_restore_record(
            fixture_id="core/integer/42",
            side="upstream_oracle",
            status="raised",
            error_class="builtins.ValueError",
            message_head="invalid literal",
        )
        validate_restore_record(
            restore, expected_side="upstream_oracle", expected_id="core/integer/42"
        )

    def test_roundtrip_combination_and_diff(self) -> None:
        details = diff_pickle_roundtrips(
            [_roundtrip(ORACLE_SIDE)], [_roundtrip(CANDIDATE_SIDE)]
        )
        self.assertEqual(details, [])


class PickleRoundtripRecordTests(unittest.TestCase):
    def test_matching_restore_is_not_a_certification(self) -> None:
        details = diff_pickle_roundtrips(
            [_roundtrip(ORACLE_SIDE)], [_roundtrip(CANDIDATE_SIDE)]
        )
        self.assertEqual(details, [])

    def test_restored_type_drift_is_detected(self) -> None:
        details = diff_pickle_roundtrips(
            [_roundtrip(ORACLE_SIDE)],
            [_roundtrip(CANDIDATE_SIDE, restored_type="Rational")],
        )
        self.assertEqual(details[0]["kind"], "pickle_restore_identity_drift")

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        oracle = _roundtrip(ORACLE_SIDE)
        candidate = _roundtrip(CANDIDATE_SIDE)
        candidate["construction_outcome"] = "raised"
        candidate["restore"] = None
        details = diff_pickle_roundtrips([oracle], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class PickleRoundtripCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["pickle-roundtrip", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "pickle-roundtrip")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_pickle_roundtrip(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
