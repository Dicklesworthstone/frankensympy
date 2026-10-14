#!/usr/bin/env python3
"""Paired live-incumbent benchmark harness (WS22, beads fra-fra-ws22-paired-incumbent-adoption-lfx & fra-rc-perf-yub).

Contract (repo AGENTS.md section 16 + gauntlet keep-gate rules):
- Engine attestation FIRST: verify subject loads FrankenSymPy (`python/sympy` + `fsym_python`)
  and oracle loads isolated upstream SymPy 1.14.0 (`fsym_python` absent from `sys.modules`).
- Semantic + structural admission FIRST: a case is timed only if subject and pinned-oracle
  outputs agree on both structural type/class/args envelope AND mathematical representation.
  Divergent or failed cases are retained in the outcome mix and never scored into speed ratios.
- Controlled paired run: alternating subject/oracle subprocesses, per-case isolation
  (a crashing or timing-out case cannot take down the sweep), identical loop text and
  evidence/cache/durability/thread/budget policies for both engines.
- cv<=5% on BOTH sides required for a low-noise case; raw samples retained.
- No competitive win is claimed unless the separate registered named-workload gate passes.

Usage:
  paired_bench.py case   --engine subject|oracle --case <case_name>
  paired_bench.py worker --engine subject|oracle --cases inline --rounds N
  paired_bench.py run    --out artifacts/perf/paired_incumbent_hero.json [--rounds 5]
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

REPO = os.environ.get(
    "FSYM_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
SUBJECT_ENV = {
    **os.environ,
    "PYTHONPATH": os.path.join(REPO, "python"),
    "PYTHONHASHSEED": "0",
}
ORACLE_ENV = {
    k: v for k, v in os.environ.items() if k != "PYTHONPATH"
}
ORACLE_ENV["PYTHONHASHSEED"] = "0"

DEFAULT_ORACLE_PY = (
    "/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python"
    if os.path.exists("/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python")
    else os.path.join(REPO, ".venv-conformance/bin/python3")
)
ORACLE_PY = os.environ.get("FSYM_ORACLE_PY", DEFAULT_ORACLE_PY)
SUBJECT_PY = os.environ.get("FSYM_SUBJECT_PY", sys.executable)

# (name, code) — code must define run() -> symbolic/Python value. Identical text for both engines.
CASES = [
    ("symbol_construct", """
def run():
    from sympy import Symbol
    for _ in range(5000):
        s = Symbol('x')
    return s
"""),
    ("integer_arith_rational", """
def run():
    from sympy import Integer, Rational
    a = Integer(2)
    for i in range(1, 400):
        a = a * Integer(i) + Rational(1, i)
    return a
"""),
    ("poly_build_deg12", """
def run():
    from sympy import Symbol
    x = Symbol('x')
    def run_once():
        e = x
        for k in range(2, 13):
            e = e + x**k
        return e
    for _ in range(300):
        e = run_once()
    return e
"""),
    ("expand_sq20", """
def run():
    from sympy import Symbol, expand
    x = Symbol('x')
    e = (x + 1)**20
    for _ in range(40):
        e2 = expand(e)
    return e2
"""),
    ("diff_poly", """
def run():
    from sympy import Symbol, diff, Rational
    x = Symbol('x')
    e = x**7 + 3*x**4 - 5*x + Rational(1, 2)
    for _ in range(500):
        d = diff(e, x)
    return d
"""),
    ("simplify_rational_cancel", """
def run():
    from sympy import Symbol, simplify
    x = Symbol('x')
    e = (x**2 - 1)/(x - 1)
    for _ in range(200):
        s = simplify(e)
    return s
"""),
    ("solve_linear", """
def run():
    from sympy import Symbol, solve
    x = Symbol('x')
    for _ in range(300):
        r = solve(3*x + 6, x)
    return r
"""),
    ("factorint_mersenne", """
def run():
    from sympy import factorint
    for _ in range(60):
        f = factorint(2**61 - 1)
    return f
"""),
    ("isprime_mersenne31", """
def run():
    from sympy import isprime
    for _ in range(500):
        v = isprime(2**31 - 1)
    return v
"""),
    ("trig_build", """
def run():
    from sympy import Symbol, sin, cos, exp
    x = Symbol('x')
    for _ in range(300):
        e = sin(x) + cos(x) + exp(x)
    return e
"""),
    ("srepr_print", """
def run():
    from sympy import Symbol, srepr
    x = Symbol('x')
    e = x**3 + 2*x
    for _ in range(300):
        s = srepr(e)
    return s
"""),
    ("hash_roundtrip", """
def run():
    from sympy import Symbol
    import builtins
    x = Symbol('x')
    e = x**3 + 2*x
    for _ in range(2000):
        h = builtins.hash(e)
    return h % 10**9
"""),
    ("matrix_det4", """
def run():
    from sympy import Matrix
    m = Matrix([[1, 2, 0, 1], [0, 1, 3, 0], [2, 1, 1, 1], [1, 0, 2, 2]])
    for _ in range(100):
        d = m.det()
    return d
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
    for _ in range(50):
        v = N(pi, 30)
    return v
"""),
    ("hero_constitutive_jacobian", """
def run():
    from sympy import Add, Function, ImmutableDenseMatrix, S, Symbol, cos, exp, sin
    x = Symbol('x')
    y = Symbol('y')
    a = Symbol('a', positive=True)
    b = Symbol('b', nonzero=True)
    k = Symbol('k', positive=True)

    class ConstitutiveLaw(Function):
        nargs = 2

        @classmethod
        def eval(cls, u, stiffness):
            if getattr(u, 'is_zero', False):
                return S.Zero
            return None

        def _eval_derivative(self, s):
            u, stiffness = self.args
            du = u.diff(s)
            dk = stiffness.diff(s)
            return du * stiffness * cos(u) + dk * sin(u)

    held = Add(a * x**2, -a * x**2, b * x * y, evaluate=False)
    res = ImmutableDenseMatrix([
        held + (x - y) * (x**2 + x * y + y**2) + sin(y) + ConstitutiveLaw(x, k),
        (x + y) * (x**2 - 2 * x * y + y**2) + exp(x) + ConstitutiveLaw(y, k),
    ])
    for _ in range(20):
        jac = res.jacobian([x, y])
    return jac
"""),
    ("hero_factor_univariate_zz", """
def run():
    from sympy import Symbol, expand, factor
    x = Symbol('x')
    p = expand((x**2 + 1) * (x**2 + 2) * (x - 3))
    for _ in range(40):
        f = factor(p)
    return f
"""),
    ("hero_certified_evalf_transcendental", """
def run():
    from sympy import Rational, cos, exp, sin
    expr = sin(Rational(1, 3)) + cos(Rational(1, 5)) + exp(Rational(1, 7))
    for _ in range(25):
        val = expr.evalf(30)
    return val
"""),
]


def compute_structural_envelope(val: Any) -> str:
    """Compute a canonical structural + mathematical envelope for admission comparison.

    Two values with identical `str(val)` but different Python class identity or
    argument structure produce distinct envelopes and fail semantic admission.
    """
    cls = type(val)
    type_tag = cls.__name__
    if isinstance(val, (list, tuple)):
        items = ", ".join(compute_structural_envelope(x) for x in val)
        return f"{type_tag}[{items}]"
    if isinstance(val, dict):
        items = ", ".join(
            f"{compute_structural_envelope(k)}:{compute_structural_envelope(v)}"
            for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
        )
        return f"dict{{{items}}}"
    if hasattr(val, "shape") and hasattr(val, "__getitem__") and type_tag.endswith("Matrix"):
        rows, cols = val.shape
        entries = ", ".join(
            compute_structural_envelope(val[r, c])
            for r in range(rows)
            for c in range(cols)
        )
        return f"{type_tag}({rows}x{cols})[{entries}]"
    args = getattr(val, "args", None)
    if args is not None and isinstance(args, tuple):
        names = [type(a).__name__ for a in args]
        if type_tag in ("Add", "Mul"):
            names.sort()
        arg_types = ",".join(names)
        return f"{type_tag}({arg_types}):{val}"
    return f"{type_tag}:{val}"



def attest_engine_worker(engine: str) -> None:
    """Inspect imported sympy and fsym_python inside the isolated engine process."""
    import sympy

    sympy_file = os.path.abspath(getattr(sympy, "__file__", ""))
    sympy_version = getattr(sympy, "__version__", "unknown")
    fsym_loaded = "fsym_python" in sys.modules
    fsym_file = None
    if not fsym_loaded and engine == "subject":
        try:
            import fsym_python  # type: ignore[import-not-found]

            fsym_loaded = True
            fsym_file = os.path.abspath(getattr(fsym_python, "__file__", ""))
        except ImportError:
            fsym_loaded = False
    elif fsym_loaded:
        fsym_file = os.path.abspath(getattr(sys.modules["fsym_python"], "__file__", ""))

    repo_python = os.path.abspath(os.path.join(REPO, "python"))
    if engine == "subject":
        if not sympy_file.startswith(repo_python) or not fsym_loaded:
            raise RuntimeError(
                f"subject engine attestation failed: sympy_file={sympy_file}, fsym_loaded={fsym_loaded}"
            )
    elif engine == "oracle":
        if sympy_file.startswith(repo_python) or fsym_loaded or sympy_version != "1.14.0":
            raise RuntimeError(
                f"oracle engine attestation failed: sympy_file={sympy_file}, "
                f"fsym_loaded={fsym_loaded}, version={sympy_version}"
            )
    print(
        json.dumps(
            {
                "engine": engine,
                "sympy_file": sympy_file,
                "sympy_version": sympy_version,
                "fsym_python_loaded": fsym_loaded,
                "fsym_python_file": fsym_file,
                "python_executable": sys.executable,
            }
        )
    )


def attest_engine_isolated(engine: str) -> dict[str, Any]:
    py = SUBJECT_PY if engine == "subject" else ORACLE_PY
    env = SUBJECT_ENV if engine == "subject" else ORACLE_ENV
    proc = subprocess.run(
        [py, os.path.abspath(__file__), "attest", "--engine", engine],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"{engine} engine attestation failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def case_worker(case_name: str) -> None:
    """Run ONE case (in this process); print a JSON line."""
    code = dict(CASES)[case_name]
    ns: dict[str, Any] = {}
    exec(code, ns)
    t0 = time.perf_counter_ns()
    try:
        raw_out = ns["run"]()
        out = compute_structural_envelope(raw_out)
        err = None
    except Exception as exc:  # noqa: BLE001 — harness records all failures
        out, err = None, f"{type(exc).__name__}: {exc}"
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    print(
        json.dumps(
            {
                "case": case_name,
                "result": out,
                "error": err,
                "ms": round(dt_ms, 4),
            }
        )
    )


def run_case_isolated(engine: str, case_name: str, timeout_s: int = 120) -> dict[str, Any]:
    """Spawn one grandchild process for a single case; return its JSON dict."""
    py = SUBJECT_PY if engine == "subject" else ORACLE_PY
    env = SUBJECT_ENV if engine == "subject" else ORACLE_ENV
    try:
        proc = subprocess.run(
            [py, os.path.abspath(__file__), "case", "--engine", engine, "--case", case_name],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "case": case_name,
            "result": None,
            "error": f"TimeoutExpired: case exceeded {timeout_s}s budget",
            "ms": None,
        }
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "case": case_name,
            "result": None,
            "error": f"process-failure exit={proc.returncode}",
            "ms": None,
        }
    return json.loads(proc.stdout.strip().splitlines()[-1])


def worker(engine: str, rounds: int) -> None:
    """Admission pass, then `rounds` timed rounds — every case isolated."""
    results: dict[str, Any] = {"engine": engine, "admission": {}, "rounds": []}
    for name, _ in CASES:
        r = run_case_isolated(engine, name)
        results["admission"][name] = {
            "result": r["result"],
            "error": r["error"],
            "untimed_ms": r["ms"],
        }
    for _ in range(rounds):
        round_data = {}
        for name, _ in CASES:
            if results["admission"][name]["error"] is not None:
                continue
            r = run_case_isolated(engine, name)
            if r["error"] is None and r["ms"] is not None:
                round_data[name] = r["ms"]
            else:
                results["admission"][name]["error"] = r["error"] or "late_round_failure"
        results["rounds"].append(round_data)
    print(json.dumps(results))


def get_hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
    }


def get_toolchain_info() -> dict[str, str]:
    rustc = "unknown"
    for candidate in (
        "/home/ubuntu/.rustup/toolchains/nightly-2026-08-20-x86_64-unknown-linux-gnu/bin/rustc",
        "rustc",
    ):
        try:
            proc = subprocess.run([candidate, "--version"], capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                rustc = proc.stdout.strip()
                break
        except Exception:
            pass
    oracle_py_ver = "unknown"
    try:
        proc = subprocess.run(
            [ORACLE_PY, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            oracle_py_ver = proc.stdout.strip()
    except Exception:
        pass
    return {
        "subject_python": sys.version.split()[0],
        "oracle_python": oracle_py_ver,
        "rustc": rustc,
    }


def compute_distribution(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    xs = sorted(samples)
    n = len(xs)
    med = float(statistics.median(xs))
    mean = float(statistics.mean(xs))
    std_dev = float(statistics.pstdev(xs)) if n > 1 else 0.0
    cv = round(100.0 * std_dev / med, 2) if med > 0 else 0.0

    def pctl(p: float) -> float:
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


def run_aa_control(rounds: int = 5) -> dict[str, Any]:
    """Run reference engine against itself to measure null-hypothesis baseline noise floor."""
    case = "matrix_det4"
    arm_a: list[float] = []
    arm_b: list[float] = []
    for _ in range(rounds):
        ra = run_case_isolated("oracle", case)
        rb = run_case_isolated("oracle", case)
        if ra["error"] is None and rb["error"] is None and ra["ms"] and rb["ms"]:
            arm_a.append(float(ra["ms"]))
            arm_b.append(float(rb["ms"]))
    if len(arm_a) != rounds or len(arm_b) != rounds:
        padded_a = (arm_a + [0.0] * rounds)[:rounds]
        padded_b = (arm_b + [0.0] * rounds)[:rounds]
        return {
            "case": case,
            "arm_a_samples": padded_a,
            "arm_b_samples": padded_b,
            "arm_a_median_ms": 0.0,
            "arm_b_median_ms": 0.0,
            "ratio_null_baseline": 0.0,
            "verified": False,
        }
    med_a = float(statistics.median(arm_a))
    med_b = float(statistics.median(arm_b))
    ratio = round(med_b / med_a, 3) if med_a > 0 else 0.0
    return {
        "case": case,
        "arm_a_samples": arm_a,
        "arm_b_samples": arm_b,
        "arm_a_median_ms": round(med_a, 4),
        "arm_b_median_ms": round(med_b, 4),
        "ratio_null_baseline": ratio,
        "verified": bool(med_a > 0 and 0.8 <= ratio <= 1.25),
    }


def run_all(out_path: str, rounds: int = 3) -> dict[str, Any]:
    """Verify engine attestation, alternate engine subprocesses, and compute paired stats."""
    subj_attestation = attest_engine_isolated("subject")
    orac_attestation = attest_engine_isolated("oracle")

    pairs: dict[str, dict[str, list[float]]] = {}
    admitted: dict[str, dict[str, Any]] = {}
    for r in range(rounds):
        for engine in ("subject", "oracle"):
            py = SUBJECT_PY if engine == "subject" else ORACLE_PY
            env = SUBJECT_ENV if engine == "subject" else ORACLE_ENV
            proc = subprocess.run(
                [
                    py,
                    os.path.abspath(__file__),
                    "worker",
                    "--engine",
                    engine,
                    "--cases",
                    "inline",
                    "--rounds",
                    "1",
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO,
                timeout=600,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"{engine} worker failed r{r}: {proc.stderr[-2000:]}")
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            if r == 0:
                admitted[engine] = data["admission"]
            else:
                for cname, adm_info in data["admission"].items():
                    if adm_info.get("error") is not None:
                        admitted[engine][cname]["error"] = adm_info["error"]
            for case, ms in data["rounds"][0].items():
                pairs.setdefault(case, {"subject": [], "oracle": []})[engine].append(float(ms))

    # A/A control execution with exact same round count
    aa_ctrl = run_aa_control(rounds=rounds)

    shared_policy = {
        "durability_mode": "in_memory_none",
        "evidence_policy": "parity_gated_t1_structural_and_mathematical",
        "cache_policy": "fresh_per_case_isolated_process",
        "thread_count": 1,
        "budget_policy": "120s_per_case_wall_clock",
    }

    report: dict[str, Any] = {
        "schema": "gauntlet.paired_bench.v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile_id": "sympy-1.14.0-cpython",
        "subject_head": subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "hardware": get_hardware_info(),
        "toolchains": get_toolchain_info(),
        "aa_control": aa_ctrl,
        "rounds": rounds,
        "cases": {},
        "settings": {
            "structural_admission": True,
            "subject_settings": dict(shared_policy),
            "oracle_settings": dict(shared_policy),
            "engine_attestation": {
                "subject": subj_attestation,
                "oracle": orac_attestation,
            },
            "competitive_claim": "not_evaluated",
        },
    }
    divergences: list[str] = []
    errors: list[str] = []
    for name, _ in CASES:
        s_adm, o_adm = admitted["subject"][name], admitted["oracle"][name]
        s_samples = pairs.get(name, {}).get("subject", [])
        o_samples = pairs.get(name, {}).get("oracle", [])
        entry: dict[str, Any] = {"case": name}
        if s_adm["error"] or o_adm["error"] or len(s_samples) != rounds or len(o_samples) != rounds:
            entry["status"] = "error"
            entry["subject_error"] = str(s_adm["error"] or "ok")
            entry["oracle_error"] = str(o_adm["error"] or "ok")
            errors.append(name)
        elif s_adm["result"] != o_adm["result"]:
            entry["status"] = "divergence"
            entry["subject_result"] = str(s_adm["result"])
            entry["oracle_result"] = str(o_adm["result"])
            divergences.append(name)
        else:
            s_dist = compute_distribution(s_samples)
            o_dist = compute_distribution(o_samples)
            s_med = s_dist["median_ms"]
            o_med = o_dist["median_ms"]
            entry.update(
                {
                    "status": "admitted",
                    "subject_ms_median": s_med,
                    "oracle_ms_median": o_med,
                    "subject_cv_pct": s_dist["cv_pct"],
                    "oracle_cv_pct": o_dist["cv_pct"],
                    "subject_dist": s_dist,
                    "oracle_dist": o_dist,
                    "subject_samples": s_samples,
                    "oracle_samples": o_samples,
                    "ratio_oracle_over_subject": round(o_med / s_med, 3) if s_med else None,
                }
            )
        report["cases"][name] = entry

    admitted_cases = [e for e in report["cases"].values() if e["status"] == "admitted"]
    ratios = [
        e["ratio_oracle_over_subject"]
        for e in admitted_cases
        if e["ratio_oracle_over_subject"]
        and e["subject_cv_pct"] <= 5.0
        and e["oracle_cv_pct"] <= 5.0
    ]
    report["summary"] = {
        "cases_total": len(CASES),
        "admitted": len(admitted_cases),
        "divergences": sorted(divergences),
        "errors": sorted(errors),
        "low_noise_admitted": len(ratios),
        "aa_control_verified": aa_ctrl.get("verified", False),
        "geomean_ratio_oracle_over_subject": (
            round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 3) if ratios else None
        ),
        "noise_note": (
            "ratio>1 means subject faster; ratio<1 means oracle faster; cv>5 cases "
            "excluded from geomean but retained above; competitive_claim=not_evaluated"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    if sys.argv[1] == "attest":
        eng = sys.argv[sys.argv.index("--engine") + 1]
        attest_engine_worker(eng)
    elif sys.argv[1] == "case":
        case_worker(sys.argv[sys.argv.index("--case") + 1])
    elif sys.argv[1] == "worker":
        eng = sys.argv[sys.argv.index("--engine") + 1]
        worker(eng, 1)
    elif sys.argv[1] == "run":
        out = sys.argv[sys.argv.index("--out") + 1]
        rounds_arg = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3
        run_all(out, rounds_arg)
