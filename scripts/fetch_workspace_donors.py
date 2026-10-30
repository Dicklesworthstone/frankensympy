#!/usr/bin/env python3
"""Materialize the existing workspace's exact registry-pinned path dependencies.

New checkouts are created beside this repository, matching Cargo.toml. Existing
directories are inspected and accepted only when clean and already at the pin;
the script never resets, cleans, deletes, or updates an existing checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DONORS = {
    "asupersync": "asupersync",
    "franken_numpy": "franken_numpy-pin",
    "frankenscipy": "frankenscipy-pin",
}
USER_AGENT = "frankensympy-agent (+https://github.com/Dicklesworthstone/frankensympy)"


def git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(directory), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def main() -> None:
    with (ROOT / "registries/donor_sources.toml").open("rb") as stream:
        registry = tomllib.load(stream)
    sources = {}
    for source in registry["source"]:
        if source["id"] in sources:
            raise SystemExit(f"duplicate donor registry ID: {source['id']}")
        sources[source["id"]] = source

    # Validate the complete requested set before creating any directories.
    requested = []
    for identifier, directory in DONORS.items():
        source = sources[identifier]
        repository, revision = source["repository"], source["commit"]
        if repository != f"Dicklesworthstone/{identifier}":
            raise SystemExit(f"unexpected repository for donor {identifier}")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit(f"donor {identifier} has no immutable full commit pin")
        requested.append((ROOT.parent / directory, repository, revision))

    for target, repository, revision in requested:
        if target.is_symlink():
            raise SystemExit(f"refusing a symlink donor path: {target}")
        if target.exists():
            if not target.is_dir() or not (target / ".git").is_dir():
                raise SystemExit(f"refusing an existing non-checkout donor path: {target}")
            if git(target, "rev-parse", "HEAD") != revision:
                raise SystemExit(f"existing donor is not at its registry pin: {target}")
            if git(target, "status", "--porcelain=v1", "--untracked-files=normal"):
                raise SystemExit(f"existing donor has local changes: {target}")
        else:
            target.mkdir()
            git(target, "-c", "init.defaultBranch=main", "init", ".")
            git(
                target, "fetch", "--depth=1", "--no-tags",
                f"https://github.com/{repository}.git", revision,
            )
            git(target, "checkout", "--detach", "FETCH_HEAD")
            if git(target, "rev-parse", "HEAD") != revision:
                raise SystemExit(f"fetched donor does not match its registry pin: {target}")
        print(f"{target.name}: {repository}@{revision}", flush=True)


if __name__ == "__main__":
    main()
