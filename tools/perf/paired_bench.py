"""Paired live-incumbent benchmark harness (WS22, bead fra-fra-ws22-paired-incumbent-adoption-lfx).

Adopted from the frankensympy__gauntlet_workspace round-2 harness (SilverWillow, 2026-09-04).
Contract (repo AGENTS.md section 16 + gauntlet keep-gate rules):
- semantic admission FIRST: a case is timed only if subject and pinned-oracle outputs
  agree exactly (T1); divergent/failed cases are reported in the outcome mix, never scored;
- controlled paired run: alternating subject/oracle subprocesses, per-case isolation
  (a crashing case cannot take down the sweep), identical loop text for both engines;
- cv<=5% on BOTH sides required for a keep-eligible case; raw samples retained;
- competitive claims require a release-perf build (debug runs are labeled as such).
"""
#!/usr/bin/env python3
"""Paired perf harness: FrankenSymPy shell (subject) vs upstream SymPy 1.14.0 (oracle).

Design per repo AGENTS.md §16 + gauntlet keep-gate rules:
- Controlled paired run: identical harness code, alternating subject/oracle subprocesses,
  same host, same minute window, same corpus, identical loop cost per engine.
- Semantic admission FIRST: each case runs once per engine and its printed result is
  compared (T1 string equality). Only admitted cases enter the ratio; divergences and
  errors are reported in the outcome mix and never timed into the score.
- cv% reported per case; cv>5 => noise, ineligible for any keep claim.
- Raw paired data retained in JSON.

Usage:
  paired_bench.py worker --engine subject|oracle --cases cases.json --rounds N
  paired_bench.py run    --out artifacts/perf/baseline_paired_<date>.json
"""
import json
import subprocess
import sys
import time
import statistics
import os

REPO = os.environ.get("FSYM_REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUBJECT_ENV = {**os.environ, "PYTHONPATH": os.path.join(REPO, "python")}
ORACLE_ENV = dict(os.environ)  # venv python carries sympy 1.14.0
ORACLE_PY = os.environ.get("FSYM_ORACLE_PY", os.path.join(REPO, ".venv-conformance/bin/python3"))
SUBJECT_PY = os.environ.get("FSYM_SUBJECT_PY", sys.executable)

# (name, code) — code must define run() -> str result. Identical text for both engines.
CASES = [
    ("symbol_construct", """
def run():
    from sympy import Symbol
    for _ in range(50000):
        s = Symbol('x')
    return str(s)
"""),
    ("integer_arith_rational", """
def run():
    from sympy import Integer, Rational
    a = Integer(2)
    for i in range(1, 4000):
        a = a * Integer(i) + Rational(1, i)
    return str(a)
"""),
    ("poly_build_deg12", """
def run():
    from sympy import Symbol, Pow, Add
    x = Symbol('x')
    def run_once():
        e = x
        for k in range(2, 13):
            e = e + x**k
        return e
    for _ in range(3000):
        e = run_once()
    return str(e)
"""),
    ("expand_sq20", """
def run():
    from sympy import Symbol, expand
    x = Symbol('x')
    e = (x + 1)**20
    for _ in range(200):
        e2 = expand(e)
    return str(e2)
"""),
    ("diff_poly", """
def run():
    from sympy import Symbol, diff, Rational
    x = Symbol('x')
    e = x**7 + 3*x**4 - 5*x + Rational(1, 2)
    for _ in range(5000):
        d = diff(e, x)
    return str(d)
"""),
    ("simplify_rational_cancel", """
def run():
    from sympy import Symbol, simplify
    x = Symbol('x')
    e = (x**2 - 1)/(x - 1)
    for _ in range(2000):
        s = simplify(e)
    return str(s)
"""),
    ("solve_linear", """
def run():
    from sympy import Symbol, solve
    x = Symbol('x')
    for _ in range(3000):
        r = solve(3*x + 6, x)
    return str(r)
"""),
    ("factorint_mersenne", """
def run():
    from sympy import factorint
    for _ in range(300):
        f = factorint(2**61 - 1)
    return str(f)
"""),
    ("isprime_mersenne31", """
def run():
    from sympy import isprime
    for _ in range(5000):
        v = isprime(2**31 - 1)
    return str(v)
"""),
    ("trig_build", """
def run():
    from sympy import Symbol, sin, cos, exp
    x = Symbol('x')
    for _ in range(3000):
        e = sin(x) + cos(x) + exp(x)
    return str(e)
"""),
    ("srepr_print", """
def run():
    from sympy import Symbol, srepr
    x = Symbol('x')
    e = x**3 + 2*x
    for _ in range(3000):
        s = srepr(e)
    return s
"""),
    ("hash_roundtrip", """
def run():
    from sympy import Symbol
    import builtins
    x = Symbol('x')
    e = x**3 + 2*x
    for _ in range(20000):
        h = builtins.hash(e)
    return str(h % 10**9)
"""),
    ("matrix_det4", """
def run():
    from sympy import Matrix
    m = Matrix([[1, 2, 0, 1], [0, 1, 3, 0], [2, 1, 1, 1], [1, 0, 2, 2]])
    for _ in range(300):
        d = m.det()
    return str(d)
"""),
    ("str_print", """
def run():
    from sympy import Symbol
    x = Symbol('x')
    e = x**3 + 2*x
    for _ in range(300):
        s = str(e)
    return s
"""),
    ("evalf_pi", """
def run():
    from sympy import N, pi
    for _ in range(200):
        v = N(pi, 30)
    return str(v)
"""),
]


def case_worker(case_name):
    """Run ONE case (in this process); print a JSON line. Isolated so a native
    crash (segfault) cannot take down the whole sweep."""
    code = dict(CASES)[case_name]
    ns = {}
    exec(code, ns)
    t0 = time.perf_counter_ns()
    try:
        out = ns["run"]()
        err = None
    except Exception as exc:  # noqa: BLE001 — harness records all failures
        out, err = None, f"{type(exc).__name__}: {exc}"
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    print(json.dumps({"case": case_name, "result": out, "error": err,
                      "ms": round(dt_ms, 4)}))


def run_case_isolated(engine, case_name):
    """Spawn one grandchild process for a single case; return its JSON dict."""
    py = SUBJECT_PY if engine == "subject" else ORACLE_PY
    env = SUBJECT_ENV if engine == "subject" else ORACLE_ENV
    proc = subprocess.run(
        [py, os.path.abspath(__file__), "case", "--engine", engine, "--case", case_name],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"case": case_name, "result": None,
                "error": f"process-failure exit={proc.returncode}",
                "ms": None}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def worker(engine, rounds):
    """Admission pass, then `rounds` timed rounds — every case isolated."""
    results = {"engine": engine, "admission": {}, "rounds": []}
    for name, _ in CASES:
        r = run_case_isolated(engine, name)
        results["admission"][name] = {"result": r["result"], "error": r["error"],
                                      "untimed_ms": r["ms"]}
    for _ in range(rounds):
        round_data = {}
        for name, _ in CASES:
            if results["admission"][name]["error"] is not None:
                continue
            r = run_case_isolated(engine, name)
            if r["error"] is None:
                round_data[name] = r["ms"]
        results["rounds"].append(round_data)
    print(json.dumps(results))


def get_hardware_info():
    import platform
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
    }


def get_toolchain_info():
    rustc = "unknown"
    try:
        proc = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        if proc.returncode == 0:
            rustc = proc.stdout.strip()
    except Exception:
        pass
    oracle_py_ver = "unknown"
    try:
        proc = subprocess.run([ORACLE_PY, "-c", "import sys; print(sys.version.split()[0])"], capture_output=True, text=True)
        if proc.returncode == 0:
            oracle_py_ver = proc.stdout.strip()
    except Exception:
        pass
    return {
        "subject_python": sys.version.split()[0],
        "oracle_python": oracle_py_ver,
        "rustc": rustc,
    }


def compute_distribution(samples):
    if not samples:
        return {}
    xs = sorted(samples)
    n = len(xs)
    med = statistics.median(xs)
    mean = statistics.mean(xs)
    std_dev = statistics.pstdev(xs) if n > 1 else 0.0
    cv = round(100.0 * std_dev / mean, 2) if mean > 0 else 0.0
    def pctl(p):
        idx = int(round((p / 100.0) * (n - 1)))
        return xs[min(max(idx, 0), n - 1)]
    return {
        "median_ms": round(med, 4),
        "mean_ms": round(mean, 4),
        "std_dev_ms": round(std_dev, 4),
        "cv_pct": cv,
        "p90_ms": round(pctl(90), 4),
        "p95_ms": round(pctl(95), 4),
        "p99_ms": round(pctl(99), 4),
    }


def run_aa_control(rounds=5):
    """Run reference engine against itself to measure null-hypothesis baseline noise floor."""
    case = "matrix_det4"
    arm_a, arm_b = [], []
    for _ in range(rounds):
        ra = run_case_isolated("oracle", case)
        rb = run_case_isolated("oracle", case)
        if ra["error"] is None and rb["error"] is None and ra["ms"] and rb["ms"]:
            arm_a.append(ra["ms"])
            arm_b.append(rb["ms"])
    if not arm_a or not arm_b:
        return {"case": case, "verified": false, "error": "AA run failed"}
    med_a, med_b = statistics.median(arm_a), statistics.median(arm_b)
    ratio = round(med_b / med_a, 3) if med_a > 0 else None
    return {
        "case": case,
        "arm_a_samples": arm_a,
        "arm_b_samples": arm_b,
        "arm_a_median_ms": round(med_a, 4),
        "arm_b_median_ms": round(med_b, 4),
        "ratio_null_baseline": ratio,
        "verified": ratio is not None and 0.8 <= ratio <= 1.25,
    }


def run_all(out_path, rounds=7):
    """Alternate engine subprocesses; compute paired stats on T1-admitted cases."""
    pairs = {}
    admitted = {}
    for r in range(rounds):
        for engine in ("subject", "oracle"):
            py = SUBJECT_PY if engine == "subject" else ORACLE_PY
            env = SUBJECT_ENV if engine == "subject" else ORACLE_ENV
            proc = subprocess.run(
                [py, os.path.abspath(__file__), "worker", "--engine", engine,
                 "--cases", "inline", "--rounds", "1"],
                capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
            if proc.returncode != 0:
                raise RuntimeError(f"{engine} worker failed r{r}: {proc.stderr[-2000:]}")
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            if r == 0:
                admitted[engine] = data["admission"]
            for case, ms in data["rounds"][0].items():
                pairs.setdefault(case, {"subject": [], "oracle": []})[engine].append(ms)

    # A/A control execution
    aa_ctrl = run_aa_control(rounds=min(rounds, 5))

    report = {
        "schema": "gauntlet.paired_bench.v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile_id": "sympy-1.14.0-cpython",
        "subject_head": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "hardware": get_hardware_info(),
        "toolchains": get_toolchain_info(),
        "aa_control": aa_ctrl,
        "rounds": rounds,
        "cases": {},
    }
    divergences, errors = [], []
    for name, _ in CASES:
        s_adm, o_adm = admitted["subject"][name], admitted["oracle"][name]
        entry = {"case": name}
        if s_adm["error"] or o_adm["error"]:
            entry["status"] = "error"
            entry["subject_error"] = s_adm["error"]
            entry["oracle_error"] = o_adm["error"]
            errors.append(name)
        elif s_adm["result"] != o_adm["result"]:
            entry["status"] = "divergence"
            entry["subject_result"] = s_adm["result"]
            entry["oracle_result"] = o_adm["result"]
            divergences.append(name)
        else:
            s, o = pairs[name]["subject"], pairs[name]["oracle"]
            s_dist = compute_distribution(s)
            o_dist = compute_distribution(o)
            s_med = s_dist["median_ms"]
            o_med = o_dist["median_ms"]
            entry.update({
                "status": "admitted",
                "subject_ms_median": s_med,
                "oracle_ms_median": o_med,
                "subject_cv_pct": s_dist["cv_pct"],
                "oracle_cv_pct": o_dist["cv_pct"],
                "subject_dist": s_dist,
                "oracle_dist": o_dist,
                "subject_samples": s,
                "oracle_samples": o,
                "ratio_oracle_over_subject": round(o_med / s_med, 3) if s_med else None,
            })
        report["cases"][name] = entry

    admitted_cases = [e for e in report["cases"].values() if e["status"] == "admitted"]
    ratios = [e["ratio_oracle_over_subject"] for e in admitted_cases
              if e["ratio_oracle_over_subject"] and e["subject_cv_pct"] <= 5 and e["oracle_cv_pct"] <= 5]
    import math
    report["summary"] = {
        "cases_total": len(CASES),
        "admitted": len(admitted_cases),
        "divergences": divergences,
        "errors": errors,
        "low_noise_admitted": len(ratios),
        "aa_control_verified": aa_ctrl.get("verified", False),
        "geomean_ratio_oracle_over_subject": round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3) if ratios else None,
        "noise_note": "ratio>1 means subject faster; ratio<1 means oracle faster; cv>5 cases excluded from geomean but retained above",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "case":
        case_worker(sys.argv[sys.argv.index("--case") + 1])
    elif sys.argv[1] == "worker":
        eng = sys.argv[sys.argv.index("--engine") + 1]
        worker(eng, 1)
    elif sys.argv[1] == "run":
        out = sys.argv[sys.argv.index("--out") + 1]
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 7
        run_all(out, rounds)
