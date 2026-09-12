"""Independent static review of branch-policy automation (`fra-rc-workflow-kzu`).

Reviewer-authored, separate from `tools/test_remote_main.py` (which pins the
runtime behaviour of the read-only verifier). This file checks the *automation
surface* instead: no workflow may carry a remote-mutating command or a write
token, every workflow must declare read-only permissions, and the branch
topology lane must stay main-guarded and credential-free.

Assertions are deliberately about the *absence* of mutation commands rather than
about successful execution of cleanup: a workflow that never mutates cannot
delete a branch no matter how it is triggered (feature branch, tag, manual
dispatch, fork, denied token, or absent remote).

Consumer: the `fra-rc-workflow-kzu` closure and any future workflow addition.
Deletion condition: merge into `tools/test_remote_main.py` if the repository
keeps a single branch-policy workflow.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"

# Label -> pattern. Each is a mutating verb a workflow must never contain.
MUTATION_PATTERNS = {
    "git push": r"git\s+push",
    "force flag": r"--force",
    "delete flag": r"--delete",
    "branch delete": r"git\s+branch\s+-[dD]\b",
    "write permission": r"contents:\s*write",
    "privileged trigger": r"pull_request_target",
    "hard reset": r"git\s+reset\s+--hard",
    "recursive delete": r"rm\s+-rf",
    "api delete": r"-X\s*DELETE",
    "admin merge": r"gh\s+pr\s+merge[^\n]*--admin",
}


class WorkflowMutationGuardTests(unittest.TestCase):
    def workflows(self) -> list[Path]:
        files = sorted(WORKFLOW_DIR.glob("*.yml"))
        self.assertTrue(files, f"no workflows found under {WORKFLOW_DIR}")
        return files

    def test_no_workflow_contains_a_remote_mutating_command(self):
        for path in self.workflows():
            text = path.read_text(encoding="utf-8")
            for label, pattern in MUTATION_PATTERNS.items():
                with self.subTest(workflow=path.name, pattern=label):
                    self.assertIsNone(
                        re.search(pattern, text),
                        f"{path.name} contains a mutating construct: {label}",
                    )

    def test_every_workflow_declares_read_only_token_permissions(self):
        for path in self.workflows():
            text = path.read_text(encoding="utf-8")
            with self.subTest(workflow=path.name):
                self.assertIn("permissions:", text)
                self.assertIn("contents: read", text)

    def test_branch_topology_stays_main_guarded_and_credential_free(self):
        text = (WORKFLOW_DIR / "branch-topology.yml").read_text(encoding="utf-8")
        # A tag push or a fork cannot satisfy the main-ref guard, and a denied
        # token cannot mutate because no step asks the token to.
        self.assertIn("branches: [main]", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("if: github.ref == 'refs/heads/main'", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("tools/verify_remote_main.py", text)
        self.assertNotIn("refs/heads/master", text)

    def test_read_only_verifier_can_only_observe_refs(self):
        source = (ROOT / "tools" / "verify_remote_main.py").read_text(encoding="utf-8")
        for pattern in (r"git\s+push", r"--delete", r"--force", r"rm\s+-rf", r"update-ref"):
            self.assertIsNone(
                re.search(pattern, source), f"verify_remote_main.py matches {pattern}"
            )
        self.assertIn("ls-remote", source, "the verifier must observe refs, not change them")
        self.assertIn("OpenAI File Downloader, XaiImageApiFetch/1.0", source)


if __name__ == "__main__":
    unittest.main()
