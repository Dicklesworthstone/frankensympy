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
