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


class PickleAdversarySweepTests(unittest.TestCase):
    """Registered restore-path serialization adversary sweep (fra-toi)."""

    @classmethod
    def setUpClass(cls) -> None:
        import json
        cls.corpus_path = LAB / "fixtures" / "pickle_adversary_corpus.json"
        cls.corpus = json.loads(cls.corpus_path.read_text(encoding="utf-8"))

    def _run_loader(self, fixture_id: str, pickle_b64: str) -> dict:
        import json
        import os
        import subprocess

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "FSYM_CANDIDATE_ROOT": str(LAB.parent.parent / "python"),
        }
        proc = subprocess.run(
            [sys.executable, "-P", "-s", str(LAB / "pickle_loader.py")],
            input=json.dumps(
                {
                    "fixture_id": fixture_id,
                    "side": CANDIDATE_SIDE,
                    "pickle_b64": pickle_b64,
                }
            ),
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        record = json.loads(proc.stdout)
        validate_restore_record(
            record, expected_side=CANDIDATE_SIDE, expected_id=fixture_id
        )
        return record

    def test_registered_adversary_corpus_all_refused_in_fresh_process(self) -> None:
        import base64

        adversaries = self.corpus.get("adversaries", [])
        self.assertGreaterEqual(len(adversaries), 6)
        for entry in adversaries:
            with self.subTest(fixture_id=entry["fixture_id"]):
                record = self._run_loader(entry["fixture_id"], entry["pickle_b64"])
                self.assertEqual(
                    record["status"],
                    "raised",
                    f"adversary {entry['fixture_id']} was NOT refused: {record}",
                )
                self.assertTrue(record.get("error_class"))

        # Oversized payload (> 64 KiB) preflight refusal test
        oversized_b64 = base64.b64encode(b"\x80\x04" + (b"K*" * 40000) + b".").decode("ascii")
        oversized_rec = self._run_loader("adv/pickle/oversized_payload", oversized_b64)
        self.assertEqual(oversized_rec["status"], "raised")
        self.assertIn("ValueError", oversized_rec["error_class"])

    def test_planted_negative_loader_acceptance_or_constructor_execution_fails_sweep(self) -> None:
        # Planted negative: if a weakened loader returned status="returned" on any
        # adversary entry, the sweep grader must flag it immediately.
        def grade_adversary_sweep(records: list[dict]) -> list[str]:
            failures = []
            for rec in records:
                if rec.get("status") != "raised":
                    failures.append(f"{rec.get('fixture_id')}: expected raised, got {rec.get('status')}")
            return failures

        planted_weakened = [
            {
                "schema_version": 1,
                "kind": "pickle_restore",
                "fixture_id": "adv/pickle/hostile_reduce_os_system",
                "side": CANDIDATE_SIDE,
                "protocol": 4,
                "status": "returned",
                "type": "int",
                "module": "builtins",
            }
        ]
        self.assertTrue(
            grade_adversary_sweep(planted_weakened),
            "planted negative (accepted hostile pickle) must fail the sweep grader",
        )


if __name__ == "__main__":
    unittest.main()
