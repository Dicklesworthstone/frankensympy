"""Run selected tests against the actual production extension, with source bookends."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/workspace/scratch/7fe32b905529/frankensympy')
BINARY = Path(os.environ.get('FSYM_TEST_BINARY', str(ROOT / 'target/debug/libfsym_python.so')))
SCOPE = os.environ.get('FSYM_TEST_SCOPE', 'production')
assert SCOPE in ('production', 'documented_math_component')
PREFIX = Path(sys.argv[1])
TARGETS = sys.argv[2:]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def sources():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = {path for path in tracked
             if path.startswith(('crates/', 'src/', 'python/sympy/'))
             and Path(path).suffix in ('.rs', '.py', '.toml')}
    paths.update(('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'))
    paths.update(target.split('::')[0] for target in TARGETS if target.endswith('.py') or '::' in target)
    return {path: digest(ROOT / path) for path in sorted(paths)}


os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
before = sources()
evidence = {
    'started_at': datetime.now(timezone.utc).isoformat(),
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
    'python': sys.version,
    'binary': str(BINARY),
    'scope': SCOPE,
    'binary_sha256': digest(BINARY),
    'targets': TARGETS,
    'pre_sources': before,
}
PREFIX.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
spec = importlib.util.spec_from_file_location('fsym_python', BINARY)
extension = importlib.util.module_from_spec(spec)
sys.modules['fsym_python'] = extension
spec.loader.exec_module(extension)
sys.path.insert(0, str(ROOT / 'python'))
import sympy
assert Path(sympy.__file__).resolve() == ROOT / 'python/sympy/__init__.py'
assert hasattr(sympy.Matrix([[1]]), '_native_storage')
print('Extension scope:', SCOPE, flush=True)
print('Extension binary:', extension.__file__, flush=True)
print('Candidate Python:', sympy.__file__, flush=True)
import pytest
result = pytest.main(['-q', *TARGETS])
after = sources()
evidence.update({
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'pytest_exit_code': int(result),
    'post_sources': after,
    'changed_sources': sorted(path for path in set(before) | set(after)
                              if before.get(path) != after.get(path)),
    'post_binary_sha256': digest(BINARY),
})
PREFIX.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
print('Source changes during tests:', evidence['changed_sources'], flush=True)
sys.exit(result)
