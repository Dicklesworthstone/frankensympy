# fra-rc-package-mcj — wheel build + install validation (TurquoiseHorizon 2026-09-19)

## Fix landed in pyproject.toml
- module-name: fsym_python -> frankensympy.fsym_python (maturin mixed layout:
  the cdylib nests inside the frankensympy package, matching the
  `from . import fsym_python` import in __init__.py)
- [tool.maturin] manifest-path = "crates/fsym-python/Cargo.toml" (workspace
  member build; the root manifest has no matching cdylib and maturin fell
  back to bin mode without it)

## Frozen build command
    uv build --wheel -o dist/
(isolated build env; maturin pep517 backend; builds the pinned fsym-python
cdylib with pyo3/extension-module and packages python/frankensympy)

## Artifact
dist/frankensympy-0.1.0-cp314-cp314-linux_x86_64.whl (retained in this
directory)

## Install validation (fresh venv /tmp/wheelcheck, python 3.14)
- uv pip install <wheel> -> 1 package installed
- find_spec('frankensympy') -> True (site-packages origin, not the source tree)
- import frankensympy -> OK (bundled fsym_python extension loads)
- importlib.metadata.version('frankensympy') -> 0.1.0
- License metadata -> "MIT License with the repository rider" (matches the
  repository rider; the Apache-2.0 mismatch from the bead background is fixed)
- Summary/Description accurate (preview channel, not a drop-in claim)

## Coexistability validation
- wheel top-level entries: frankensympy/ + dist-info ONLY; no top-level sympy/
- upstream sympy==1.14.0 installed alongside in the same venv
- both import simultaneously; upstream smoke (sympy.integrate(2*t, t) -> t**2)
  unaffected

## Acceptance mapping (bead)
- find_spec('frankensympy') after install: MET
- accurate metadata (license rider, name, version): MET
- bridge manifest selected (module-name frankensympy.fsym_python -> bundled
  extension): MET
- wheel validation (not local-copy smoke): MET (fresh venv install)
