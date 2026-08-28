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
    load_profile,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
CORE = LAB / "fixtures" / "seed_core_atoms.json"


def _extension_available() -> bool:
    import os

    names = ("fsym_python.so", "libfsym_python.so")
    dirs = []
    explicit = os.environ.get("FSYM_PYTHON_EXT_DIR")
    if explicit:
        dirs.append(Path(explicit))
    cargo = os.environ.get("CARGO_TARGET_DIR")
    if cargo:
        dirs.append(Path(cargo) / "debug")
        dirs.append(Path(cargo) / "release")
    for directory in dirs:
        for name in names:
            if (directory / name).is_file():
                return True
    return False


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
        self.assertEqual(symbol["outcome_class"], "raised")
        self.assertEqual(symbol["observations"]["exception_type"], "NotImplementedError")

    def test_construction_only_reports_module_drift_not_a_false_admit(self) -> None:
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
        self.assertEqual(summary["admitted"], 0)
        self.assertEqual(summary["claims_promoted"], False)
        self.assertEqual(summary["named_claims"], ["COMPAT-002"])
        by_id = {row["fixture_id"]: row for row in summary["details"]}
        integer = by_id["core/integer/42"]
        self.assertNotIn("observations.type", integer["difference_paths"])
        self.assertIn("observations.module", integer["difference_paths"])
        self.assertIn("observations.func", integer["difference_paths"])
        symbol = by_id["core/symbol/x_positive"]
        self.assertEqual(
            symbol["outcome_classes"],
            {"oracle": "returned", "candidate": "raised"},
        )
        self.assertEqual(integer["kind"], "surface_identity_drift")
        self.assertEqual(symbol["kind"], "outcome_mismatch")
        self.assertGreaterEqual(summary["by_kind"].get("surface_identity_drift", 0), 1)

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


if __name__ == "__main__":
    unittest.main()
