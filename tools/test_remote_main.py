"""Exercise the read-only verifier against real isolated local Git repositories."""

import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from verify_remote_main import verify


class RemoteMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Retained deliberately: repository policy prohibits deleting test artifacts.
        cls.root = Path(tempfile.mkdtemp(prefix="fsym-remote-main-"))
        cls.remote = cls.root / "remote"
        cls.remote.mkdir()
        cls.run_git(cls.remote, "init", "-q", "-b", "main")
        (cls.remote / "source.txt").write_text("first revision\n")
        cls.run_git(cls.remote, "add", "source.txt")
        cls.run_git(cls.remote, "-c", "user.name=GateTest", "-c",
                    "user.email=gate-test@example.invalid", "commit", "-qm", "first")
        cls.sha = cls.run_git(cls.remote, "rev-parse", "HEAD").strip()
        cls.run_git(cls.remote, "branch", "feature-preserve")
        cls.run_git(cls.remote, "tag", "preserve-tag")
        cls.client = cls.root / "client"
        cls.client.mkdir()
        cls.run_git(cls.client, "init", "-q")
        cls.run_git(cls.client, "remote", "add", "origin", str(cls.remote))

    @staticmethod
    def run_git(root, *args):
        return subprocess.run(["git", *args], cwd=root, check=True,
                              capture_output=True, text=True, timeout=30).stdout

    def test_matching_main_preserves_every_remote_ref_and_file(self):
        before = self.run_git(self.remote, "show-ref")
        status = self.run_git(self.remote, "status", "--porcelain")
        self.assertEqual(verify(self.client, self.sha), self.sha)
        self.assertEqual(self.run_git(self.remote, "show-ref"), before)
        self.assertEqual(self.run_git(self.remote, "status", "--porcelain"), status)

    def test_stale_revision_and_invalid_ids_refuse_without_writes(self):
        before = self.run_git(self.remote, "show-ref")
        for expected in ("0" * 40, "main", "--upload-pack=evil", self.sha[:7]):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                verify(self.client, expected)
        self.assertEqual(self.run_git(self.remote, "show-ref"), before)

    def test_missing_remote_refuses(self):
        root = self.root / "no-remote"
        root.mkdir()
        self.run_git(root, "init", "-q")
        with self.assertRaises(ValueError):
            verify(root, self.sha)

    def test_remote_without_main_refuses(self):
        root = self.root / "no-main"
        root.mkdir()
        self.run_git(root, "init", "-q")
        self.run_git(root, "remote", "add", "origin", str(root))
        with self.assertRaises(ValueError):
            verify(root, self.sha)

    def test_workflow_has_read_only_permissions_and_main_trigger_guard(self):
        workflow = (Path(__file__).resolve().parents[1]
                    / ".github/workflows/branch-topology.yml").read_text()
        self.assertIn("contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("if: github.ref == 'refs/heads/main'", workflow)
        self.assertIn("branches: [main]", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("tools/verify_remote_main.py", workflow)
        for forbidden in ("contents: write", "git push", "--force", "--delete",
                          "pull_request_target", "refs/heads/master"):
            self.assertNotIn(forbidden, workflow)

    def test_ci_python_version_matches_immutable_conformance_profile(self):
        root = Path(__file__).resolve().parents[1]
        profile = tomllib.loads(
            (root / "tools/conformance-lab/profiles/sympy-1.14.0-cpython.toml").read_text()
        )
        workflow = (root / ".github/workflows/ci.yml").read_text()
        version = profile["environment"]["python_version"]
        self.assertIn(f'python-version: "{version}"', workflow)


if __name__ == "__main__":
    unittest.main()
