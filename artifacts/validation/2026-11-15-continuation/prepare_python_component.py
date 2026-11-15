"""Prepare the existing, explicitly bounded mathematical PyO3 composition.

Only scratch copies of the two runtime-dependent bridge files are shortened.
All retained functions, mathematical packages, and shell sources stay exact.
The six omitted helpers are the same ones recorded in the existing Nov 14
math-component scope; no replacement implementation or upstream engine is added.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/workspace/scratch/4a3a39075b27/frankensympy')
OUTPUT = Path('/workspace/scratch/4a3a39075b27/python-component')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise SystemExit(f'Expected one reviewed bridge item: {old!r}')
    return source.replace(old, new, 1)


def omit_block(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise SystemExit('Reviewed omission anchors differ from current source')
    first = source.index(begin)
    last = source.index(end, first)
    return source[:first] + source[last:], source[first:last]


def main():
    if OUTPUT.exists():
        raise SystemExit('Refusing to overwrite an existing Python component')
    reference = json.loads((ROOT / 'artifacts/audit/2026-11-14-math-component-scope.json').read_text())
    original = ROOT / 'crates/fsym-python/src'
    before = {str(p.relative_to(original)): digest(p.read_bytes()) for p in original.rglob('*') if p.is_file()}
    OUTPUT.mkdir()
    shutil.copytree(original, OUTPUT / 'src')
    (OUTPUT / 'runtime').mkdir()
    for key, destination in (('manifest', 'Cargo.toml'), ('runtime_manifest', 'runtime/Cargo.toml'), ('runtime_wrapper', 'runtime/lib.rs')):
        text = reference[key].replace('/tmp/frankensympy-work', str(ROOT))
        (OUTPUT / destination).write_text(text)
    shutil.copyfile(ROOT / 'Cargo.lock', OUTPUT / 'Cargo.lock')

    lib = (original / 'lib.rs').read_text()
    lib, omitted_lib = omit_block(lib, '/// Maps a metered evaluation refusal onto the Python surface.', '/// Taylor polynomial of `src` around `var = at` through degree `order`.')
    lib = replace_once(lib, 'use fsym_runtime::{Budget, BudgetLimits, FsymCx, RuntimeBudget};', 'use fsym_runtime::{Budget, BudgetLimits};')
    lib = replace_once(lib, 'use fsym_simplify::{SimplifyError, expand_with, simplify_with};\n', '')
    for function in ('simplify_expr', 'expand_expr'):
        lib = replace_once(lib, f'    m.add_function(wrap_pyfunction!({function}, m)?)?;\n', '')
    (OUTPUT / 'src/lib.rs').write_text(lib)

    expr = (original / 'expr.rs').read_text()
    expr, omitted_expr = omit_block(expr, '    /// Simplify expression under budgeted region.', '    /// Evaluate expression to floating point number.')
    expr = replace_once(expr, 'use fsym_runtime::{Budget, BudgetLimits, FsymCx, RuntimeBudget};\n', '')
    expr = replace_once(expr, 'use fsym_simplify::{expand_with, simplify_with};\n', '')
    (OUTPUT / 'src/expr.rs').write_text(expr)

    after = {str(p.relative_to(OUTPUT / 'src')): digest(p.read_bytes()) for p in (OUTPUT / 'src').rglob('*') if p.is_file()}
    changed = sorted(name for name in before if before[name] != after[name])
    if changed != ['expr.rs', 'lib.rs'] or before.keys() != after.keys():
        raise SystemExit('Unexpected bridge copy differences')
    record = {
        'source_root': str(ROOT),
        'scope': 'Supplementary mathematical/PyO3 component only; production async runtime and six helpers absent; no replacement functions or upstream production engine',
        'reference_recipe': 'artifacts/audit/2026-11-14-math-component-scope.json',
        'original_bridge_files': before,
        'component_bridge_files': after,
        'omitted_members': reference['omitted_members'],
        'omitted_block_sha256': {'lib.rs': digest(omitted_lib.encode()), 'expr.rs': digest(omitted_expr.encode())},
        'allowed_non_function_changes': ['Remove imports used only by omitted helpers', 'Remove the two omitted free-function PyO3 registrations', 'Alias runtime only to the original fsym_budget implementation'],
        'source_function_replacements': [],
        'not_covered': ['Full production Cargo dependency graph', 'Asynchronous runtime and cancellation', 'Simplify/expand runtime helpers', 'Compatibility profile certification'],
    }
    (OUTPUT / 'bridge-composition.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'component': str(OUTPUT), 'retained_complete_bridge_files': len(before)-2, 'shortened_files': changed, 'omitted_helpers': len(record['omitted_members'])}))


if __name__ == '__main__':
    main()
