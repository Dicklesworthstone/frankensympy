import json, subprocess, sys, os, tempfile
sys.path.insert(0, 'python/tests')
from test_latex_corpus import REPO

DRIVER = '''
import json, sys
import sympy
x, y, z = sympy.symbols("x y z")
out = {}
def rec(name, fn):
    try:
        out[name] = sympy.srepr(fn())
    except Exception as exc:
        out[name] = f"ERR: {type(exc).__name__}: {str(exc)[:60]}"

exprs = {
    "trig1": sympy.sin(x)**2 + sympy.cos(x)**2,
    "trig2": sympy.sin(2*x),
    "trig3": sympy.tan(x)*sympy.cos(x),
    "rad1": sympy.sqrt(8),
    "rad2": sympy.sqrt(x**2),
    "pow1": x**2*x**3,
    "pow2": (x**2)**3,
    "pow3": sympy.exp(x)*sympy.exp(y),
    "rat1": (x**2 - 1)/(x + 1),
    "rat2": 1/x + 1/y,
    "rat3": (x**2 + 3*x + 2)/(x + 2),
    "mul1": x*(x + 1) + x**2,
    "nest": sympy.sin(sympy.cos(x))**2 + sympy.sin(sympy.cos(x))**2,
}
for nm, e in exprs.items():
    rec(f"simplify_{nm}", lambda e=e: sympy.simplify(e))
    rec(f"expand_{nm}", lambda e=e: sympy.expand(e))
    rec(f"factor_{nm}", lambda e=e: sympy.factor(e))
    rec(f"trigsimp_{nm}", lambda e=e: sympy.trigsimp(e))
    rec(f"radsimp_{nm}", lambda e=e: sympy.radsimp(e))
    rec(f"powsimp_{nm}", lambda e=e: sympy.powsimp(e))
    rec(f"ratsimp_{nm}", lambda e=e: sympy.ratsimp(e))
    rec(f"separatevars_{nm}", lambda e=e: sympy.separatevars(e))
    rec(f"together_{nm}", lambda e=e: sympy.together(e))
    rec(f"cancel_{nm}", lambda e=e: sympy.cancel(e))
    rec(f"apart_{nm}", lambda e=e: sympy.apart(e))
    rec(f"collect_xy_{nm}", lambda e=e: sympy.collect(e, x))
print(json.dumps(out))
'''

with tempfile.NamedTemporaryFile("w", suffix="_sd.py", delete=False) as h:
    h.write(DRIVER); drv = h.name

def run(py, env_extra):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([py, drv], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=900)
    if r.returncode != 0:
        print("fail:", r.stderr[-200:]); return None
    return json.loads(r.stdout.strip().splitlines()[-1])

o = run("/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python", {})
s = run(str(REPO / ".venv-conformance" / "bin" / "python"), {"PYTHONPATH": str(REPO / "python")})
if o and s:
    diffs = {k: (s.get(k), o[k]) for k in o if s.get(k) != o[k]}
    print(f"probes={len(o)} divergences={len(diffs)}")
    for k, v in list(diffs.items())[:12]:
        print("---", k, "shell=", repr(v[0])[:80], "oracle=", repr(v[1])[:80])
