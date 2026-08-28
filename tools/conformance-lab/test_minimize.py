"""Adversarial regression tests for discrepancy pairing and identity."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparators import diff_envelopes, is_valid_discrepancy
from minimize import (
    LEDGER_INDEX_FIELDS,
    MAX_LEDGER_RECORD_BYTES,
    _index_validated_records,
    _validate_ledger_index_entry,
    _validate_ledger_record_size,
    build_records,
    record_identity,
    strict_json_loads,
)

PROFILE = "sympy-1.14.0-cpython"
STAMP = "2026-08-26T00:00:00+00:00"


def envelope(fixture_id: str, *, side: str, observed_type: str = "Integer") -> dict:
    return {
        "schema_version": 1,
        "profile_id": PROFILE,
        "fixture_id": fixture_id,
        "side": side,
        "outcome_class": "returned",
        "observations": {"type": observed_type},
        "environment": {"python": "3.14.4"},
    }


def records(oracle: list[dict], candidate: list[dict]) -> tuple[list[dict], int]:
    return build_records(
        oracle,
        candidate,
        comparator="exact_surface",
        severity="object",
        fallback_profile_id=PROFILE,
        created_at_utc=STAMP,
    )


class DiscrepancyMinimizerTests(unittest.TestCase):
    def test_ledger_index_must_match_record_triage_fields(self) -> None:
        record = records(
            [envelope("fixture/a", side="upstream_oracle")],
            [
                envelope(
                    "fixture/a",
                    side="frankensympy_candidate",
                    observed_type="Wrong",
                )
            ],
        )[0][0]
        index_entry = {field: record[field] for field in LEDGER_INDEX_FIELDS}

        self.assertEqual(
            _validate_ledger_index_entry(index_entry, record, line_number=1),
            record["discrepancy_id"],
        )
        for field, replacement in (
            ("severity", "security"),
            ("status", "closed_verified"),
        ):
            with self.subTest(field=field):
                mutant = copy.deepcopy(index_entry)
                mutant[field] = replacement
                with self.assertRaisesRegex(
                    ValueError, rf"field {field!r} disagrees"
                ):
                    _validate_ledger_index_entry(mutant, record, line_number=1)

    def test_ledger_publication_preflight_rejects_invalid_and_duplicate_records(
        self,
    ) -> None:
        record = records(
            [envelope("fixture/a", side="upstream_oracle")],
            [
                envelope(
                    "fixture/a",
                    side="frankensympy_candidate",
                    observed_type="Wrong",
                )
            ],
        )[0][0]

        with self.assertRaisesRegex(ValueError, "duplicate incoming ledger record"):
            _index_validated_records([record, copy.deepcopy(record)])

        mutant = copy.deepcopy(record)
        mutant["status"] = "silently_accepted"
        with self.assertRaisesRegex(ValueError, "invalid incoming ledger record"):
            _index_validated_records([mutant])

    def test_ledger_record_size_is_admitted_before_read(self) -> None:
        _validate_ledger_record_size(
            MAX_LEDGER_RECORD_BYTES,
            Path("disc-boundary.json"),
        )
        with self.assertRaisesRegex(ValueError, "exceeds the ledger record-size limit"):
            _validate_ledger_record_size(
                MAX_LEDGER_RECORD_BYTES + 1,
                Path("disc-oversized.json"),
            )

    def test_reordered_envelopes_pair_by_fixture_id(self) -> None:
        oracle = [
            envelope("fixture/a", side="upstream_oracle"),
            envelope("fixture/b", side="upstream_oracle"),
        ]
        candidate = [
            envelope("fixture/b", side="frankensympy_candidate"),
            envelope("fixture/a", side="frankensympy_candidate"),
        ]

        found, paired = records(oracle, candidate)

        self.assertEqual(found, [])
        self.assertEqual(paired, 2)

    def test_same_difference_on_two_fixtures_has_distinct_ids(self) -> None:
        oracle = [
            envelope("fixture/a", side="upstream_oracle"),
            envelope("fixture/b", side="upstream_oracle"),
        ]
        candidate = [
            envelope("fixture/a", side="frankensympy_candidate", observed_type="Wrong"),
            envelope("fixture/b", side="frankensympy_candidate", observed_type="Wrong"),
        ]

        found, paired = records(oracle, candidate)

        self.assertEqual(paired, 2)
        self.assertEqual(len(found), 2)
        self.assertNotEqual(found[0]["discrepancy_id"], found[1]["discrepancy_id"])

    def test_missing_and_extra_fixtures_are_independent_records(self) -> None:
        oracle = [envelope("fixture/missing", side="upstream_oracle")]
        candidate = [envelope("fixture/extra", side="frankensympy_candidate")]

        found, paired = records(oracle, candidate)

        self.assertEqual(paired, 0)
        self.assertEqual(
            {record["fixture_id"] for record in found},
            {"fixture/missing", "fixture/extra"},
        )
        self.assertEqual(len({record["discrepancy_id"] for record in found}), 2)

    def test_duplicate_fixture_ids_fail_closed(self) -> None:
        duplicate = envelope("fixture/a", side="upstream_oracle")

        with self.assertRaisesRegex(ValueError, "duplicate oracle fixture_id"):
            records(
                [duplicate, copy.deepcopy(duplicate)],
                [envelope("fixture/a", side="frankensympy_candidate")],
            )

    def test_wrong_side_and_profile_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "required='upstream_oracle'"):
            records(
                [envelope("fixture/a", side="frankensympy_candidate")],
                [envelope("fixture/a", side="frankensympy_candidate")],
            )

        wrong_profile = envelope("fixture/a", side="upstream_oracle")
        wrong_profile["profile_id"] = "wrong-profile"
        with self.assertRaisesRegex(ValueError, "oracle profile does not match"):
            records(
                [wrong_profile],
                [envelope("fixture/a", side="frankensympy_candidate")],
            )

    def test_record_validator_rejects_unknown_difference_fields(self) -> None:
        oracle = [envelope("fixture/a", side="upstream_oracle")]
        candidate = [
            envelope("fixture/a", side="frankensympy_candidate", observed_type="Wrong")
        ]
        found, _ = records(oracle, candidate)
        mutant = copy.deepcopy(found[0])
        mutant["differences"][0]["ignored"] = True

        valid, reason = is_valid_discrepancy(mutant)

        self.assertFalse(valid)
        self.assertIn("difference keys invalid", reason)

    def test_record_validator_recomputes_content_id(self) -> None:
        oracle = [envelope("fixture/a", side="upstream_oracle")]
        candidate = [
            envelope("fixture/a", side="frankensympy_candidate", observed_type="Wrong")
        ]
        found, _ = records(oracle, candidate)
        mutant = copy.deepcopy(found[0])
        mutant["discrepancy_id"] = "disc-" + "0" * 64

        valid, reason = is_valid_discrepancy(mutant)

        self.assertFalse(valid)
        self.assertIn("does not match canonical content", reason)

    def test_exact_comparator_distinguishes_json_boolean_from_integer(self) -> None:
        oracle = envelope("fixture/a", side="upstream_oracle")
        candidate = envelope("fixture/a", side="frankensympy_candidate")
        oracle["observations"] = {"value": True}
        candidate["observations"] = {"value": 1}

        differences = diff_envelopes(oracle, candidate, "exact_surface")

        self.assertEqual(differences[0]["path"], "observations.value")

    def test_construction_comparator_rejects_exception_identity_drift(self) -> None:
        oracle = envelope("fixture/a", side="upstream_oracle")
        candidate = envelope("fixture/a", side="frankensympy_candidate")
        oracle["outcome_class"] = "raised"
        candidate["outcome_class"] = "raised"
        oracle["observations"] = {
            "exception_module": "builtins",
            "exception_type": "ValueError",
            "message_head": "oracle wording",
        }
        candidate["observations"] = {
            "exception_module": "builtins",
            "exception_type": "TypeError",
            "message_head": "candidate wording",
        }

        differences = diff_envelopes(oracle, candidate, "construction_only")

        self.assertEqual(
            [difference["path"] for difference in differences],
            ["observations.exception_type"],
        )

    def test_construction_comparator_requires_declared_observations(self) -> None:
        oracle = envelope("fixture/a", side="upstream_oracle")
        candidate = envelope("fixture/a", side="frankensympy_candidate")

        differences = diff_envelopes(oracle, candidate, "construction_only")

        self.assertEqual(
            [difference["path"] for difference in differences],
            [
                "observations.module",
                "observations.args_repr",
                "observations.func",
            ],
        )

    def test_construction_comparator_requires_exception_identity(self) -> None:
        oracle = envelope("fixture/a", side="upstream_oracle")
        candidate = envelope("fixture/a", side="frankensympy_candidate")
        oracle["outcome_class"] = "raised"
        candidate["outcome_class"] = "raised"
        oracle["observations"] = {"message_head": "same wording"}
        candidate["observations"] = {"message_head": "same wording"}

        differences = diff_envelopes(oracle, candidate, "construction_only")

        self.assertEqual(
            [difference["path"] for difference in differences],
            [
                "observations.exception_module",
                "observations.exception_type",
            ],
        )

    def test_construction_comparator_refuses_unregistered_outcome_policy(self) -> None:
        for outcome_class in ("timeout", "refused"):
            with self.subTest(outcome_class=outcome_class):
                oracle = envelope("fixture/a", side="upstream_oracle")
                candidate = envelope("fixture/a", side="frankensympy_candidate")
                oracle["outcome_class"] = outcome_class
                candidate["outcome_class"] = outcome_class

                with self.assertRaisesRegex(
                    ValueError, "has no registered observation policy"
                ):
                    diff_envelopes(oracle, candidate, "construction_only")

    def test_volatile_metadata_does_not_change_record_identity(self) -> None:
        oracle = [envelope("fixture/a", side="upstream_oracle")]
        candidate = [
            envelope("fixture/a", side="frankensympy_candidate", observed_type="Wrong")
        ]
        found, _ = records(oracle, candidate)
        revised = copy.deepcopy(found[0])
        revised["status"] = "closed_verified"
        revised["created_at_utc"] = "later"

        self.assertEqual(record_identity(found[0]), record_identity(revised))

    def test_non_finite_json_numbers_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            strict_json_loads('{"value": NaN}')

        oracle = [envelope("fixture/a", side="upstream_oracle")]
        candidate = [
            envelope("fixture/a", side="frankensympy_candidate", observed_type="Wrong")
        ]
        found, _ = records(oracle, candidate)
        mutant = copy.deepcopy(found[0])
        mutant["differences"][0]["candidate"] = float("nan")
        valid, reason = is_valid_discrepancy(mutant)
        self.assertFalse(valid)
        self.assertIn("not strict JSON", reason)


if __name__ == "__main__":
    unittest.main()
