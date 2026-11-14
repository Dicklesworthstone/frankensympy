"""Validate the exact frozen mathematical PyO3 component with the CI pytest pin.

Run after sourcing validation/env.sh and setting the recorded serialized flags.
No production source, Cargo manifest, or test is changed by this script.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/scratch/aac5c0ea4661')
HERE = ROOT / 'validation/python-component'
SOURCE = ROOT / 'frankensympy'
PYTHON = '/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3'
RUNNER = ROOT / 'validation/python-runner-8.4.2'
commands = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(name):
    subprocess.run([PYTHON, 'record_scope.py', name], cwd=HERE, check=True)
    return json.loads((HERE / name).read_text())


def run(name, argv, env=None):
    print('COMMAND ' + json.dumps(argv), flush=True)
    started = time.monotonic()
    with (HERE / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(argv, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = proc.wait()
    result = {'name': name, 'command': argv, 'cwd': str(HERE), 'returncode': code,
              'elapsed_seconds': time.monotonic() - started,
              'log': str(HERE / (name + '.log'))}
    commands.append(result)
    return result


def write_result(name, before, after, extra):
    result = {
        'source_hashes_unchanged': before['source_sha256'] == after['source_sha256'],
        'harness_hashes_unchanged': before['harness_sha256'] == after['harness_sha256'],
        'source_hash_count': len(before['source_sha256']),
        'harness_hash_count': len(before['harness_sha256']),
        'repository_head_before': before['repository_head'],
        'repository_head_after': after['repository_head'],
        'source_snapshot_includes_reviewed_bridge_corrections': True,
        'rustflags': os.environ['RUSTFLAGS'],
        'pyo3_python': os.environ['PYO3_PYTHON'],
        'cargo_target_dir': os.environ['CARGO_TARGET_DIR'],
        'scope': before['scope'],
        'omitted_members': before['omitted_members'],
        'source_function_replacements': [],
        'runner_script_sha256': digest(__file__),
        'max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        **extra,
    }
    (HERE / name).write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT ' + json.dumps(result), flush=True)
    if not result['source_hashes_unchanged'] or not result['harness_hashes_unchanged']:
        raise SystemExit('Frozen validation input changed during execution')
    return result


before_build = record('python-final-pre-build.json')
build = run('python-final-build', ['cargo', 'build', '--offline', '--locked',
                                 '--lib', '--features', 'extension-module'])
after_build = record('python-final-post-build.json')
build_result = write_result('python-final-build-result.json', before_build, after_build,
                           {'command': build, 'returncode': build['returncode']})
if build['returncode']:
    raise SystemExit(build['returncode'])

binary = Path(os.environ['CARGO_TARGET_DIR']) / 'debug/libfsym_python.so'
extension = HERE / 'extension/fsym_python.so'
extension.parent.mkdir(exist_ok=True)
shutil.copyfile(binary, extension)
binary_sha = digest(binary)
assert digest(extension) == binary_sha
pythonpath = ':'.join([str(extension.parent), str(SOURCE / 'python'), str(RUNNER / 'site-packages')])
env = dict(os.environ, PYTHONPATH=pythonpath, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
smoke_code = (
    'import json,pathlib,sympy,fsym_python,pytest; '
    f'assert pathlib.Path(sympy.__file__).resolve()==pathlib.Path({str(SOURCE / "python/sympy/__init__.py")!r}).resolve(); '
    f'assert pathlib.Path(fsym_python.__file__).resolve()==pathlib.Path({str(extension)!r}).resolve(); '
    'assert pytest.__version__=="8.4.2"; '
    'assert hasattr(fsym_python,"solve_real_rational_constraints_native") and hasattr(fsym_python,"real_constraint_source_preflight_native"); '
    'print(json.dumps({"sympy":sympy.__file__,"native":fsym_python.__file__,"pytest_version":pytest.__version__,"new_bridge_registered":True}))'
)
smoke = run('python-pinned-runner-smoke', [PYTHON, '-c', smoke_code], env)
if smoke['returncode']:
    raise SystemExit(smoke['returncode'])

before_tests = record('python-pinned-runner-pre-tests.json')
tests = run('python-pinned-runner-tests', [PYTHON, '-m', 'pytest', '-q',
            str(SOURCE / 'python/tests/test_real_constraint_systems.py'),
            str(SOURCE / 'python/tests/test_real_rational_inequalities.py')], env)
after_tests = record('python-pinned-runner-post-tests.json')
write_result('python-pinned-runner-suite-result.json', before_tests, after_tests,
             {'commands': list(commands), 'returncode': tests['returncode'],
              'loaded_binary': str(extension), 'binary_sha256': binary_sha,
              'pythonpath': pythonpath, 'pytest_version': '8.4.2',
              'pytest_wheel_manifest': str(RUNNER / 'wheel-manifest.json'),
              'pytest_wheel_manifest_sha256': digest(RUNNER / 'wheel-manifest.json')})
if tests['returncode']:
    raise SystemExit(tests['returncode'])

before_clippy = record('python-final-pre-clippy.json')
clippy = run('python-final-bridge-clippy', ['cargo', 'clippy', '--offline', '--locked',
             '-p', 'fsym-python-math-source-component', '--no-deps', '--lib',
             '--features', 'extension-module', '--', '-D', 'warnings'])
after_clippy = record('python-final-post-clippy.json')
write_result('python-final-bridge-clippy-result.json', before_clippy, after_clippy,
             {'command': clippy, 'returncode': clippy['returncode']})
raise SystemExit(clippy['returncode'])
