"""Environment/build receipt cannot certify and cannot rewrite goldens."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import cmd_environment, golden_digest_map, load_profile, parse_cli  # noqa: E402
from environment_records import (  # noqa: E402
    extension_identity,
    make_environment_receipt,
    oracle_pin_mismatches,
    sha256_file,
    validate_extension_identity,
)

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
MISSING_EXTENSION = {
    "present": False,
    "path": None,
    "size": None,
    "sha256": None,
}


class EnvironmentRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile(PROFILE_PATH)
        self.oracle = {
            "sympy_version": self.profile["upstream"]["version"],
            "python": "3.14.4",
            "implementation": "CPython",
            "platform": "Linux",
            "sympy_path": "/oracle/sympy/__init__.py",
            "env": dict(self.profile["environment"]["env_overrides"]),
        }

    def test_matching_oracle_pin_does_not_certify(self) -> None:
        receipt = make_environment_receipt(
            profile=self.profile,
            oracle_environment=self.oracle,
            candidate_environment=None,
            extension=MISSING_EXTENSION,
        )
        self.assertEqual(receipt["kind"], "environment_build_receipt")
        self.assertFalse(receipt["certifies"])
        self.assertFalse(receipt["can_certify"])
        self.assertFalse(receipt["claims_promoted"])
        self.assertFalse(receipt["goldens_written"])
        self.assertEqual(receipt["mismatches"], [])
        self.assertFalse(receipt["extension"]["present"])

    def test_sympy_version_pin_drift_is_named(self) -> None:
        mutant = dict(self.oracle)
        mutant["sympy_version"] = "0.0.0"
        mismatches = oracle_pin_mismatches(self.profile, mutant)
        self.assertEqual(mismatches[0]["field"], "sympy_version")
        receipt = make_environment_receipt(
            profile=self.profile,
            oracle_environment=mutant,
            candidate_environment=None,
            extension=MISSING_EXTENSION,
        )
        self.assertFalse(receipt["certifies"])
        self.assertTrue(receipt["mismatches"])

    def test_missing_extension_cannot_carry_a_digest(self) -> None:
        with self.assertRaises(ValueError):
            validate_extension_identity(
                {
                    "present": False,
                    "path": None,
                    "size": 1,
                    "sha256": "a" * 64,
                }
            )

    def test_present_extension_identity_from_a_tiny_file(self) -> None:
        with tempfile.NamedTemporaryFile(prefix="fsym-ext-", suffix=".so", delete=False) as fh:
            fh.write(b"cdylib-bytes")
            path = Path(fh.name)
        try:
            record = extension_identity(path)
            validate_extension_identity(record)
            self.assertTrue(record["present"])
            self.assertEqual(record["size"], len(b"cdylib-bytes"))
            self.assertEqual(record["sha256"], sha256_file(path))
        finally:
            path.unlink()


class EnvironmentCliTests(unittest.TestCase):
    def test_certify_is_refused_without_touching_goldens(self) -> None:
        parsed = parse_cli(["environment", str(PROFILE_PATH), "--certify"])
        self.assertEqual(parsed["mode"], "environment")
        self.assertTrue(parsed["certify"])
        profile = load_profile(PROFILE_PATH)
        before = golden_digest_map(profile)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cmd_environment(profile, "unused", "unused", certify=True)
        self.assertEqual(status, 1)
        self.assertIn("cannot certify", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(golden_digest_map(profile), before)


if __name__ == "__main__":
    unittest.main()
