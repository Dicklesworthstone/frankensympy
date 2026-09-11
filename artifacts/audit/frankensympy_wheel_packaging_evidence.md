# Preview wheel packaging evidence (`fra-rc-package-mcj`)

Implementation evidence for the coexistable `frankensympy` preview wheel. The
companion independent gate `fra-rc-package-gate-gjd` requires a reviewer other
than the implementation author; this file records what was built and observed so
that review can be adversarial rather than trust-based.

## What exists now

- `python/frankensympy/__init__.py` — the preview package. It owns exactly one
  import name (`frankensympy`), imports its bundled `fsym_python` extension as a
  submodule, and raises a named `ImportError` when the extension is missing or
  built for another interpreter. It never imports or ships `sympy`; the
  resolver-transparent replacement channel stays a separate profile.
- `scripts/build_frankensympy_wheel.py` — stdlib-only deterministic wheel
  assembler. It refuses (exit 2) on a missing/misnamed/wrong-ABI extension, a
  version disagreement across `pyproject.toml`/`Cargo.toml`/package, any member
  that would own the `sympy` namespace, and a LICENSE that is not the repository
  MIT rider. It stages inside the output directory, replaces atomically, and
  deletes only its own staging directory.
- `scripts/check_frankensympy_wheel.py` — the adversarial battery below.
- `scripts/build_python_extension.sh` — now installs the same extension bytes
  both as `python/fsym_python.so` (development shell) and as
  `python/frankensympy/fsym_python<EXT_SUFFIX>` (the wheel layout).
- `pyproject.toml` — license corrected from `Apache-2.0` to
  `MIT License with the repository rider`, `python-packages = ["frankensympy"]`
  so no build backend can package the `sympy` tree from this manifest.

## Reproduce

```bash
scripts/build_python_extension.sh                      # needs a cargo binding; see below
python scripts/build_frankensympy_wheel.py --dist dist
python scripts/check_frankensympy_wheel.py --wheel dist/frankensympy-*.whl \
    --wheelhouse <dir with sympy-1.14.0 and mpmath wheels>
```

Built wheel at the recorded revision: `frankensympy-0.1.0-cp314-cp314-linux_x86_64.whl`,
sha256 `0be44c8b35a7e9420b6953957f11ae4afba7fe61f5d252730b01f2bb5d703a4e`,
6 members (package `__init__`, ABI-tagged extension, METADATA, WHEEL,
`licenses/LICENSE`, RECORD).

## Battery output (exit 0, all six cases)

```
[1] wheel audit
    ok  no member owns the sympy namespace (6 members)
    ok  package layout is flat under frankensympy/
    ok  every member is in RECORD (missing: [])
    ok  RECORD hashes and sizes match (6 members, bad: [])
    ok  METADATA names the frankensympy distribution
    ok  METADATA license is the MIT rider
    ok  METADATA contains no Apache attribution
    ok  METADATA declares Requires-Python
    ok  wheel carries the LICENSE file (frankensympy-0.1.0.dist-info/licenses/LICENSE)
    ok  bundled LICENSE is the repository MIT rider
    ok  extension present as frankensympy/fsym_python.cpython-314-x86_64-linux-gnu.so
    ok  extension is not empty
    ok  wheel tag is interpreter specific (cp314)
[2] coexistence: offline install next to pinned upstream SymPy
    ok  offline install exit 0 (upstream=True)
    ok  import smoke exit 0 from neutral cwd
    ok  preview version 0.1.0
    ok  engine version 0.1.0
    ok  upstream sympy is 1.14.0
    ok  preview and upstream live in different directories
    ok  native engine resolves inside the preview package (.../site-packages/frankensympy/fsym_python.cpython-314-x86_64-linux-gnu.so)
[3] isolation: same wheel with no upstream installed
    ok  offline install exit 0 (upstream=False)
    ok  import without upstream exit 0
    ok  importing the preview never imports sympy
    ok  wheel sources contain no sympy import (found: [])
[4] missing extension refuses, no fallback
    ok  import of an incomplete wheel fails
    ok  refusal names the missing bundled extension
[5] wrong module name refuses, no fallback
    ok  import with a misnamed extension fails
    ok  refusal is explicit
[6] uninstall leaves upstream intact
    ok  uninstall exit 0
    ok  upstream import after uninstall exit 0
    ok  upstream version unchanged
    ok  no preview leftovers in site-packages (found: [])
```

Case 2 and 3 install with `pip install --no-index --find-links <wheelhouse>`
(no network at install time) into throwaway virtual environments; case 2 installs
pinned upstream `sympy==1.14.0` alongside the preview and imports both from a
neutral working directory with `PYTHONPATH` removed.

Refusal text observed for the incomplete wheel:

```
ImportError: frankensympy requires its bundled fsym_python extension; the
installed wheel is incomplete or built for a different interpreter: cannot
import name 'fsym_python' from partially initialized module 'frankensympy' ...
```

## Known limits (not claimed)

- The wheel contains a **debug-profile** extension (435 MB `.so`, 87 MB wheel).
  A release/stripped packaging profile is an open packaging decision; no
  performance or size claim is made here.
- The wheel is interpreter-specific (`cp314-cp314`), matching the current pyo3
  build. No abi3 or free-threaded variant is claimed.
- `satisfies_requires_dist_sympy` is deliberately false: this is the preview
  channel. The replacement channel needs its own distribution identity and
  release matrix.
- The environment needs a cargo binding to build the extension: the pinned
  toolchain `nightly-2026-08-20` ships without its cargo component on this
  workstation, so the extension build exports the pinned `rustc` explicitly.
