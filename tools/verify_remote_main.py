"""Read-only remote-main observation, not a release or implementation gate."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def verify(root: Path, expected: str) -> str:
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected) is None:
        raise ValueError("expected commit must be a full lowercase Git object ID")
    result = subprocess.run(
        ["git", "-c", f"http.userAgent={USER_AGENT}", "ls-remote",
         "--exit-code", "--refs", "origin", "refs/heads/main"],
        cwd=root, capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"remote main could not be read (git exit {result.returncode})")
    rows = [row.split() for row in result.stdout.splitlines() if row.strip()]
    if rows != [[expected, "refs/heads/main"]]:
        raise ValueError("remote main is missing, ambiguous, or differs from expected commit")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()
    try:
        observed = verify(Path(__file__).resolve().parents[1], args.expected)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"REFUSE: {error}")
        return 1
    print(f"OBSERVED refs/heads/main={observed}; no remote writes performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
