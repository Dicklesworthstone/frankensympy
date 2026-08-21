# Packaging and resolver-transparent drop-in deployment

**Status:** normative architecture contract  
**Scope:** Python distribution identity, import ownership, wheels, dependency resolution, installation channels, upgrade/uninstall behavior, coexistence, profiles, and certification

## 1. Three separate claims

### Import compatibility

`import sympy` and registered submodule imports resolve to FrankenSymPy’s Python surface.

### Behavioral compatibility

The imported surface matches a named immutable compatibility profile over declared observations and corpus.

### Distribution compatibility

Python package resolvers treat the installed artifact as satisfying dependencies that require the distribution `sympy` under the certified version policy.

None implies the others.

## 2. Distribution name

A wheel named only `frankensympy` that installs a `sympy/` package does not generally satisfy `Requires-Dist: sympy`. Resolver-transparent deployment therefore requires a controlled replacement artifact or resolver/source override whose distribution identity is `sympy` for the target environment.

The exact distribution/version policy is a product and ecosystem decision. It is not solved by `Provides-Dist`, import hooks, or writing overlapping files.

## 3. Exclusive path ownership

Exactly one installed distribution owns the `sympy/` import tree in a certified environment. Co-installing upstream SymPy and another wheel that writes the same paths is forbidden because uninstall/upgrade order can corrupt either distribution.

Upstream differential oracles run in isolated environments or processes.

## 4. Supported deployment channels

Potential certified channels:

- private or custom package index serving a replacement `sympy` wheel;
- direct URL/path replacement in a lockfile;
- environment-manager source override;
- vendored application environment;
- container or hermetic bundle;
- future coordinated upstream distribution arrangement.

Each channel has its own resolver matrix. Documentation must not call a channel “drop-in” until dependency installation, upgrade, uninstall, and rollback are tested.

## 5. Version semantics

A replacement version must satisfy resolver constraints without misleading users about upstream identity. The profile records:

- upstream compatibility target;
- FrankenSymPy implementation version;
- distribution version mapping;
- local/build metadata policy;
- pre-release behavior;
- upper/lower-bound handling;
- yanked/revoked release behavior.

Exact pins such as `sympy==X` are separately tested from ranges such as `sympy>=X,<Y`.

## 6. Wheel contents

The wheel manifest declares:

- all owned Python modules/packages;
- native extension modules and ABI tags;
- metadata files;
- type information;
- console scripts;
- licenses and provenance;
- optional assets;
- no overlapping unowned paths;
- reproducible build inputs.

Native core crates remain separately embeddable from the Python wheel.

## 7. Platform and ABI matrix

Certification records:

- CPython versions;
- classic/free-threaded ABI;
- macOS Apple Silicon;
- Linux x86-64 baseline and optimized classes;
- Windows if/when admitted;
- manylinux/musllinux policy;
- wheel repair/audit tooling;
- source-build path;
- subinterpreter support status.

An unsupported tag fails installation clearly rather than loading a mismatched binary.

## 8. Resolver matrix

Tests include:

- clean install by name;
- downstream package requiring `sympy`;
- exact and ranged requirements;
- extras;
- lockfile generation and sync;
- upgrade upstream→FrankenSymPy;
- downgrade/rollback;
- uninstall and reinstall;
- cached wheel/index metadata;
- editable/development installs;
- conflicting user requests;
- offline/hermetic install;
- multiple environment managers.

## 9. Import matrix

After installation, test:

- top-level and deep imports;
- namespace/package metadata;
- `importlib.metadata` distribution lookup;
- resources and data files;
- pickling module/class references;
- console entry points;
- plugin discovery;
- reload and subinterpreter import;
- no accidental import of upstream files from another path.

## 10. Delegated compatibility profile

A development profile may import upstream SymPy under a private isolated name or helper process for unsupported operations. Such a profile is not sovereign, not portable, and not proof of independent implementation. The packaging manifest and runtime receipt disclose delegation.

## 11. Security

- no `.pth` or `sitecustomize` hijack as the primary replacement mechanism;
- no runtime download of native binaries;
- signed wheel and release manifest;
- hash-pinned dependencies;
- bounded, non-executable mathematical artifact formats;
- pickle compatibility documented as unsafe for untrusted input;
- rollback and revocation channel.

## 12. Evidence states

- `import_only`;
- `resolver_verified`;
- `behavior_profile_verified`;
- `wheel_matrix_verified`;
- `upgrade_rollback_verified`;
- `release_signed`;
- `ecosystem_certified` for a named downstream corpus.

“Drop-in replacement” requires the gates selected by the release profile, not merely `import_only`.
