"""Real FrankenSymPy candidate envelopes when the native extension is present."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    capture_candidate_file,
    classify_construction_diff,
    cmd_diff,
    compare,
    compare_construction_only,
    load_goldens,
    load_profile,
    weakened_variants,
)
from extension import find_fsym_python_extension  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
CORE = LAB / "fixtures" / "seed_core_atoms.json"


def _extension_available() -> bool:
    return find_fsym_python_extension() is not None


class RealCandidateTests(unittest.TestCase):
    def test_integer_fixture_returns_native_integer_when_extension_present(self) -> None:
        if not _extension_available():
            self.skipTest("fsym_python cdylib not on CARGO_TARGET_DIR/FSYM_PYTHON_EXT_DIR")
        profile = load_profile(PROFILE_PATH)
        envelopes = capture_candidate_file(profile, CORE, sys.executable, broken=False)
        by_id = {envelope["fixture_id"]: envelope for envelope in envelopes}
        integer = by_id["core/integer/42"]
        self.assertEqual(integer["side"], "frankensympy_candidate")
        self.assertEqual(integer["outcome_class"], "returned")
        self.assertEqual(integer["observations"]["type"], "Integer")
        refused = [
            envelope["fixture_id"]
            for envelope in envelopes
            if envelope["outcome_class"] == "refused"
        ]
        self.assertEqual(refused, [])
        symbol = by_id["core/symbol/x_positive"]
        self.assertEqual(symbol["outcome_class"], "returned")
        self.assertEqual(symbol["observations"]["type"], "Symbol")

    def test_construction_only_admits_integer_identity_without_promoting_claims(self) -> None:
        if not _extension_available():
            self.skipTest("fsym_python cdylib not on CARGO_TARGET_DIR/FSYM_PYTHON_EXT_DIR")
        profile = load_profile(PROFILE_PATH)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cmd_diff(
                profile, sys.executable, broken=False, affected_claim="COMPAT-002"
            )
        self.assertEqual(status, 1)
        summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["claims_promoted"], False)
        self.assertEqual(summary["named_claims"], ["COMPAT-002"])
        self.assertGreaterEqual(summary["type_matched"], summary["admitted"])
        admitted = summary["admitted_fixture_ids"]
        for fixture_id in (
            "core/integer/42",
            "core/rational/22_7",
            "core/symbol/x_positive",
            "core/add/x_plus_y",
            "held/mul_two_k",
            "held/add_noncanonical_order",
            "adversarial/integer/from_float_string",
            "core/add/collapse_duplicates",
            "core/mul/three_x_sq",
        ):
            self.assertIn(fixture_id, admitted)
        by_id = {row["fixture_id"]: row for row in summary["details"]}
        zero = by_id["subclass/ConstitutiveLawZero_zero_collapse"]
        self.assertEqual(zero["kind"], "type_drift")
        self.assertIn("observations.type", zero["difference_paths"])

    def test_real_diff_persists_ledger_without_promoting_claims(self) -> None:
        if not _extension_available():
            self.skipTest("fsym_python cdylib not on CARGO_TARGET_DIR/FSYM_PYTHON_EXT_DIR")
        profile = load_profile(PROFILE_PATH)
        with tempfile.TemporaryDirectory(prefix="fsym-ws01-ledger-") as raw:
            ledger = Path(raw)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = cmd_diff(
                    profile,
                    sys.executable,
                    broken=False,
                    affected_claim="COMPAT-002",
                    ledger_dir=ledger,
                )
            self.assertEqual(status, 1)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["ledger_records"], summary["discrepancies"])
            index = (ledger / "index.ndjson").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(index), summary["discrepancies"])
            first = json.loads((ledger / f"{json.loads(index[0])['discrepancy_id']}.json").read_text())
            self.assertEqual(first["affected_claim"], "COMPAT-002")
            self.assertEqual(first["status"], "open")
            self.assertEqual(summary["claims_promoted"], False)

    def test_two_candidate_subprocesses_match(self) -> None:
        if not _extension_available():
            self.skipTest("fsym_python cdylib not on CARGO_TARGET_DIR/FSYM_PYTHON_EXT_DIR")
        profile = load_profile(PROFILE_PATH)
        first = capture_candidate_file(profile, CORE, sys.executable, broken=False)
        second = capture_candidate_file(profile, CORE, sys.executable, broken=False)
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )


class ConstructionDiffClassifyTests(unittest.TestCase):
    def test_kinds_do_not_collapse_type_into_module_drift(self) -> None:
        self.assertEqual(
            classify_construction_diff(
                ["observations.module", "observations.func"], None
            ),
            "surface_identity_drift",
        )
        self.assertEqual(
            classify_construction_diff(
                ["observations.type", "observations.module"], None
            ),
            "type_drift",
        )
        self.assertEqual(
            classify_construction_diff(
                ["outcome_class"], {"oracle": "returned", "candidate": "raised"}
            ),
            "outcome_mismatch",
        )


class ComparatorFieldContractTests(unittest.TestCase):
    def setUp(self) -> None:
        profile = load_profile(PROFILE_PATH)
        goldens = load_goldens(profile)
        self.golden = goldens[next(iter(goldens))]
        self.variants = weakened_variants(self.golden)

    def test_pickle_digest_is_outside_construction_only(self) -> None:
        mutated = self.variants["pickle-digest-swapped"]
        self.assertTrue(compare(self.golden, mutated))
        self.assertFalse(compare_construction_only(self.golden, mutated))

    def test_held_form_args_are_inside_construction_only(self) -> None:
        mutated = self.variants["held-form-args-collapsed"]
        self.assertTrue(compare_construction_only(self.golden, mutated))

    def test_mro_class_drift_is_exact_surface_only(self) -> None:
        mutated = self.variants["mro-class-swapped"]
        self.assertTrue(compare(self.golden, mutated))
        self.assertFalse(compare_construction_only(self.golden, mutated))


if __name__ == "__main__":
    unittest.main()
