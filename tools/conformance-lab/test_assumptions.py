"""Three-valued assumptions lane cannot certify or rewrite goldens."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assumptions_records import (  # noqa: E402
    ASSUMPTION_QUERIES,
    diff_assumptions_records,
    make_assumptions_record,
    validate_assumptions_record,
)
from capture import (  # noqa: E402
    CANDIDATE_SIDE,
    ORACLE_SIDE,
    cmd_assumptions,
    golden_digest_map,
    load_profile,
    parse_cli,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"

INTEGER_QUERIES = {
    "is_positive": "true",
    "is_negative": "false",
    "is_zero": "false",
    "is_real": "true",
    "is_rational": "true",
    "is_integer": "true",
    "is_commutative": "true",
    "is_number": "true",
}


def _integer(side: str, *, queries: dict | None = None) -> dict:
    return make_assumptions_record(
        profile_id="sympy-1.14.0-cpython",
        fixture_id="core/integer/42",
        side=side,
        construction_outcome="returned",
        queries=dict(queries or INTEGER_QUERIES),
    )


class AssumptionsRecordTests(unittest.TestCase):
    def test_registered_query_set_is_complete(self) -> None:
        self.assertEqual(set(INTEGER_QUERIES), set(ASSUMPTION_QUERIES))

    def test_matching_queries_are_not_a_certification(self) -> None:
        details = diff_assumptions_records(
            [_integer(ORACLE_SIDE)], [_integer(CANDIDATE_SIDE)]
        )
        self.assertEqual(details, [])

    def test_true_vs_none_is_detected(self) -> None:
        mutant = dict(INTEGER_QUERIES)
        mutant["is_positive"] = "none"
        details = diff_assumptions_records(
            [_integer(ORACLE_SIDE)], [_integer(CANDIDATE_SIDE, queries=mutant)]
        )
        self.assertEqual(details[0]["kind"], "assumptions_identity_drift")
        self.assertEqual(details[0]["oracle"]["queries"]["is_positive"], "true")
        self.assertEqual(details[0]["candidate"]["queries"]["is_positive"], "none")

    def test_query_errors_are_valid_records(self) -> None:
        profile = load_profile(PROFILE_PATH)
        queries = dict(INTEGER_QUERIES)
        queries["is_positive"] = {
            "error_class": "builtins.AttributeError",
            "message_head": "missing",
        }
        record = _integer(ORACLE_SIDE, queries=queries)
        validate_assumptions_record(
            record,
            profile,
            expected_side=ORACLE_SIDE,
            expected_id="core/integer/42",
        )

    def test_construction_outcome_mismatch_is_not_silent(self) -> None:
        candidate = make_assumptions_record(
            profile_id="sympy-1.14.0-cpython",
            fixture_id="core/integer/42",
            side=CANDIDATE_SIDE,
            construction_outcome="refused",
            queries=None,
        )
        details = diff_assumptions_records([_integer(ORACLE_SIDE)], [candidate])
        self.assertEqual(details[0]["kind"], "construction_outcome_mismatch")


class AssumptionsCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["assumptions", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "assumptions")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_assumptions(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
