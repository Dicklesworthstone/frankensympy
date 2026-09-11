"""FrankenSymPy preview package (distribution name: ``frankensympy``).

This package owns exactly one import name: ``frankensympy``. It never ships,
creates, or imports the top-level ``sympy`` package, so it can be installed in
an environment that already contains upstream SymPy without overlapping
ownership of that namespace.

It is a *preview* channel, not a drop-in replacement: the preview distribution
deliberately does not claim to satisfy ``Requires-Dist: sympy``. The
resolver-transparent replacement channel is a separate, separately certified
artifact (see ``registries/packaging_profiles.toml``).

A missing or mismatched native extension is a hard failure, never a silent
fallback to another engine.
"""

from __future__ import annotations

__all__ = ["__version__", "engine_version", "native"]
__version__ = "0.1.0"

try:
    from . import fsym_python as native
except ImportError as exc:  # pragma: no cover - exercised by the packaging battery
    raise ImportError(
        "frankensympy requires its bundled fsym_python extension; the installed "
        f"wheel is incomplete or built for a different interpreter: {exc}"
    ) from None


def engine_version() -> str:
    """Version string reported by the bundled native engine."""
    return str(native.version())
