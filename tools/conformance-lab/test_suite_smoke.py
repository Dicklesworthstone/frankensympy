"""Oracle upstream-suite smoke receipts: digest-pinned, no port-status claims."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    cmd_suite_smoke,
    load_profile,
    oracle_python,
    parse_cli,
    validate_suite_receipt,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
SMOKE_PATH = "utilities/tests/test_source.py"
COMPAT_PATH = "core/tests/test_compatibility.py"


def _oracle_python_or_skip() -> str:
    try:
        return oracle_python(None)
    except SystemExit as exc:
        raise unittest.SkipTest(str(exc)) from exc


class SuiteSmokeTests(unittest.TestCase):
    @staticmethod
    def _inventory_entry() -> dict:
        return {
            "path": SMOKE_PATH,
            "sha256": "a" * 64,
            "bytes": 289,
        }

    @classmethod
    def _valid_receipt(cls) -> dict:
        entry = cls._inventory_entry()
        return {
            "schema_version": 1,
            "kind": "oracle_suite_receipt",
            "profile_id": "sympy-1.14.0-cpython",
            "test_path": SMOKE_PATH,
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "runner": "sympy.testing.runtests.test",
            "pytest_installed": False,
            "legacy_return_true": True,
            "counts": {"passed": 2, "failed": 0, "skipped": 0},
            "status_note": (
                "oracle execution receipt only; no FrankenSymPy port status is claimed"
            ),
        }

    def _run_mocked_receipt(self, receipt: object) -> int:
        profile = load_profile(PROFILE_PATH)
        entry = self._inventory_entry()
        inventory = {"upstream_test_tree": {"files": [entry]}}
        proc = mock.Mock(
            returncode=0,
            stdout=json.dumps(receipt, sort_keys=True) + "\n",
            stderr="",
        )
        with (
            mock.patch("capture.load_inventory_artifact", return_value=inventory),
            mock.patch("capture.subprocess.run", return_value=proc),
            mock.patch("sys.stderr", new=io.StringIO()),
            redirect_stdout(io.StringIO()),
        ):
            return cmd_suite_smoke(profile, "unused-python", SMOKE_PATH)

    def test_cli_requires_test_path(self) -> None:
        parsed = parse_cli(["suite-smoke", str(PROFILE_PATH)])
        self.assertEqual(parsed["mode"], "suite-smoke")
        self.assertIsNone(parsed["test_path"])

    def test_unknown_inventory_path_fails_closed(self) -> None:
        profile = load_profile(PROFILE_PATH)
        py = _oracle_python_or_skip()
        with mock.patch("sys.stderr", new=io.StringIO()):
            status = cmd_suite_smoke(profile, py, "not/a/real/test_file.py")
        self.assertEqual(status, 1)

    def test_unsafe_path_fails_closed(self) -> None:
        profile = load_profile(PROFILE_PATH)
        py = _oracle_python_or_skip()
        with mock.patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(cmd_suite_smoke(profile, py, "../secret.py"), 1)
            self.assertEqual(cmd_suite_smoke(profile, py, "/etc/passwd"), 1)

    def test_digest_mismatch_fails_closed(self) -> None:
        profile = load_profile(PROFILE_PATH)
        py = _oracle_python_or_skip()
        with mock.patch("capture.load_inventory_artifact") as loader:
            loader.return_value = {
                "upstream_test_tree": {
                    "files": [{"path": SMOKE_PATH, "sha256": "0" * 64, "bytes": 1}]
                }
            }
            with mock.patch("sys.stderr", new=io.StringIO()):
                status = cmd_suite_smoke(profile, py, SMOKE_PATH)
        self.assertEqual(status, 1)

    def test_suite_receipt_is_bound_to_the_exact_request(self) -> None:
        profile = load_profile(PROFILE_PATH)
        entry = self._inventory_entry()
        self.assertTrue(
            validate_suite_receipt(self._valid_receipt(), profile, SMOKE_PATH, entry)
        )

        forged_values = {
            "schema_version": 2,
            "profile_id": "moving-head",
            "test_path": COMPAT_PATH,
            "bytes": entry["bytes"] + 1,
            "sha256": "b" * 64,
            "runner": "unregistered.runner",
            "pytest_installed": True,
            "status_note": "certified compatible",
        }
        for field, forged in forged_values.items():
            with self.subTest(field=field):
                receipt = self._valid_receipt()
                receipt[field] = forged
                with self.assertRaises((TypeError, ValueError)):
                    validate_suite_receipt(receipt, profile, SMOKE_PATH, entry)

    def test_suite_receipt_rejects_malformed_and_zero_run_shapes(self) -> None:
        profile = load_profile(PROFILE_PATH)
        entry = self._inventory_entry()

        malformed = []
        receipt = self._valid_receipt()
        del receipt["counts"]
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["unexpected"] = "field"
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["port_status"] = "ported"
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["legacy_return_true"] = 1
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["counts"] = {"passed": True, "failed": 0, "skipped": 0}
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["counts"] = {"passed": 0, "failed": 0, "skipped": 0}
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["counts"] = {"passed": 1, "failed": 1, "skipped": 0}
        malformed.append(receipt)
        receipt = self._valid_receipt()
        receipt["counts"] = {"passed": -1, "failed": 0, "skipped": 0}
        malformed.append(receipt)

        for index, candidate in enumerate(malformed):
            with self.subTest(index=index):
                with self.assertRaises((TypeError, ValueError)):
                    validate_suite_receipt(candidate, profile, SMOKE_PATH, entry)

    def test_cmd_suite_smoke_rejects_a_forged_final_receipt(self) -> None:
        receipt = self._valid_receipt()
        receipt["test_path"] = COMPAT_PATH
        self.assertEqual(self._run_mocked_receipt(receipt), 1)

    def test_live_oracle_source_tests_emit_receipt_without_port_status(self) -> None:
        profile = load_profile(PROFILE_PATH)
        py = _oracle_python_or_skip()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cmd_suite_smoke(profile, py, SMOKE_PATH)
        self.assertEqual(status, 0, buffer.getvalue())
        receipt = json.loads(buffer.getvalue())
        self.assertEqual(receipt["kind"], "oracle_suite_receipt")
        self.assertEqual(receipt["test_path"], SMOKE_PATH)
        self.assertNotIn("port_status", receipt)
        self.assertIn("no FrankenSymPy port status", receipt["status_note"])
        self.assertGreaterEqual(receipt["counts"]["passed"], 2)
        self.assertTrue(receipt["legacy_return_true"])
        self.assertFalse(receipt["pytest_installed"])

    def test_live_oracle_compatibility_file_is_a_second_inventoried_receipt(self) -> None:
        profile = load_profile(PROFILE_PATH)
        py = _oracle_python_or_skip()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cmd_suite_smoke(profile, py, COMPAT_PATH)
        self.assertEqual(status, 0, buffer.getvalue())
        receipt = json.loads(buffer.getvalue())
        self.assertEqual(receipt["test_path"], COMPAT_PATH)
        self.assertNotEqual(receipt["test_path"], SMOKE_PATH)
        self.assertNotIn("port_status", receipt)
        self.assertGreaterEqual(receipt["counts"]["passed"], 1)


if __name__ == "__main__":
    unittest.main()
