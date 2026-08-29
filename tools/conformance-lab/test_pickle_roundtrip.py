"""Tests for conformance-lab pickle roundtrip records and validation."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import load_profile
from pickle_records import (
    combine_roundtrip,
    diff_pickle_roundtrips,
    make_dump_record,
    make_restore_record,
    validate_dump_record,
    validate_restore_record,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"


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
        dump_o = make_dump_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side="upstream_oracle",
            construction_outcome="returned",
            pickle_sha256="a" * 64,
            pickle_b64="YnVsbGV0",
            dump_error=None,
        )
        restore_o = make_restore_record(
            fixture_id="core/integer/42",
            side="upstream_oracle",
            status="returned",
            restored_type="Integer",
            module="sympy.core.numbers",
        )
        combined_o = combine_roundtrip(dump_o, restore_o)

        dump_c = make_dump_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side="frankensympy_candidate",
            construction_outcome="returned",
            pickle_sha256="b" * 64,
            pickle_b64="YnVsbGV0",
            dump_error=None,
        )
        restore_c = make_restore_record(
            fixture_id="core/integer/42",
            side="frankensympy_candidate",
            status="returned",
            restored_type="Integer",
            module="sympy.core.numbers",
        )
        combined_c = combine_roundtrip(dump_c, restore_c)

        diffs = diff_pickle_roundtrips([combined_o], [combined_c])
        self.assertEqual(diffs, [])


if __name__ == "__main__":
    unittest.main()
