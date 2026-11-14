"""Record exact inputs and omissions for the scoped current-source PyO3 build."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
SOURCE = Path('/workspace/scratch/aac5c0ea4661/frankensympy')
DONORS = [SOURCE.parent / name for name in ('asupersync', 'franken_numpy-pin', 'frankenscipy-pin')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(package):
    return package['name'], package['version'], package.get('source')


composition = json.loads((HERE / 'bridge-composition.json').read_text())
for name, expected in composition['original_bridge_files'].items():
    if sha(SOURCE / 'crates/fsym-python/src' / name) != expected:
        raise SystemExit(f'Original bridge changed after composition: {name}')
for name, expected in composition['component_bridge_files'].items():
    if sha(HERE / 'src' / name) != expected:
        raise SystemExit(f'Compiled bridge differs from recorded composition: {name}')

metadata = json.loads((HERE / 'metadata.json').read_text())
original = tomllib.loads((SOURCE / 'Cargo.lock').read_text())
component = tomllib.loads((HERE / 'Cargo.lock').read_text())
expected_registry = {identity(p): p for p in original['package']}
registry = []
for package in component['package']:
    if not package.get('source'):
        continue
    expected = expected_registry.get(identity(package))
    if expected is None or package.get('checksum') != expected.get('checksum'):
        raise SystemExit(f'Registry identity differs from production lock: {package["name"]}')
    registry.append({field: package[field] for field in ('name', 'version', 'source', 'checksum')})

source_files = set((SOURCE / 'crates/fsym-python/src').rglob('*.rs'))
source_files.update((SOURCE / 'python').rglob('*.py'))
source_files.update(SOURCE / name for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'registries/dependencies.toml', 'registries/donor_sources.toml'))
packages = []
for package in metadata['packages']:
    if package.get('source'):
        continue
    directory = Path(package['manifest_path']).parent.resolve()
    if directory in (HERE, HERE / 'runtime'):
        continue
    if not (directory.is_relative_to(SOURCE / 'crates') or any(directory.is_relative_to(donor) for donor in DONORS)):
        raise SystemExit(f'Unexpected source package: {directory}')
    packages.append({'name': package['name'], 'manifest_path': str(directory / 'Cargo.toml')})
    source_files.update(path for path in directory.rglob('*') if path.is_file() and 'target' not in path.parts and '.git' not in path.parts)
for donor in DONORS:
    source_files.update(path for path in (donor / 'Cargo.toml', donor / 'Cargo.lock', donor / 'rust-toolchain.toml') if path.exists())

harness = set((HERE / 'src').rglob('*'))
harness = {path for path in harness if path.is_file()}
harness.update(HERE / name for name in ('Cargo.toml', 'Cargo.lock', 'metadata.json', 'runtime/Cargo.toml', 'runtime/lib.rs', 'bridge-composition.json', 'record_scope.py'))
harness.add(HERE.parent / 'prepare_python_component.py')
record = {
    'scope': composition['scope'],
    'not_covered': composition['not_covered'],
    'omitted_members': composition['omitted_members'],
    'source_function_replacements': [],
    'repository_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
    'donor_heads': {str(donor): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=donor, text=True).strip() for donor in DONORS},
    'original_path_packages': sorted(packages, key=lambda package: package['name']),
    'registry_identities_match_original_lock': True,
    'registry_packages': registry,
    'resolved_features': {node['id']: node['features'] for node in metadata['resolve']['nodes']},
    'source_sha256': {str(path): sha(path) for path in sorted(source_files)},
    'harness_sha256': {str(path): sha(path) for path in sorted(harness)},
}
destination = HERE / sys.argv[1]
destination.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'record': str(destination), 'registry_packages': len(registry), 'original_path_packages': len(packages), 'source_files': len(source_files), 'harness_files': len(harness), 'omitted_helpers': len(composition['omitted_members'])}))
