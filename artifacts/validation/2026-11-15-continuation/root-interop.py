"""Exercise actual CLI/Python root-certificate transport without rebuilding.

Run with the existing testenv Python and -I -B. The report retains the actual
NDJSON exchanges, exact proof bytes, source/binary hashes and bounded scope.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    repo = options.repo.resolve()
    evidence = repo / "artifacts/validation/2026-11-15-continuation"
    inventory_records = {
        name: {"path": str(evidence / name), "sha256": digest(evidence / name)}
        for name in ("root-session-after.json", "public-postlint-after.json")
    }
    named_sources = [
        "src/bin/frankensympy.rs", "src/session.rs", "src/certification.rs",
        "src/certification_session.rs", "crates/fsym-core/Cargo.toml",
        "crates/fsym-core/src/root_certificate.rs",
        "crates/fsym-core/src/root_certificate/artifact.rs",
        "crates/fsym-assumptions/src/lib.rs", "crates/fsym-python/src/numeric.rs",
        "crates/fsym-python/src/numeric/root_artifact.rs",
        "crates/fsym-python/src/numeric_budget.rs", "crates/fsym-python/src/binding.rs",
        "python/sympy/__init__.py", "python/sympy/core/__init__.py",
        "python/sympy/utilities/_certified_numeric.py",
        "python/sympy/utilities/_root_certificate.py",
        "python/sympy/utilities/source_certification.py",
        "python/sympy/utilities/lambdify.py",
    ]
    sources = {str(repo / path) for path in named_sources}
    sources.add(str(Path(__file__).resolve()))
    before_sources = {path: digest(path) for path in sorted(sources)}
    binaries = {
        "cli": str(options.cli.resolve()),
        "extension": str(options.extension.resolve()),
    }
    before_binaries = {name: digest(path) for name, path in binaries.items()}
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
        "scope": (
            "Two actual cross-language root-proof replays using the existing original-source "
            "Session component CLI and existing mathematical PyO3 component. No build, "
            "source substitution or production/test edit. This is not full-runtime or "
            "compatibility-profile validation. The CLI predates the separately validated "
            "unrelated Product enum-layout change. Existing exhaustive build inventories "
            "are referenced by path/hash; only the named integration sources are rehashed here."
        ),
        "statement": {"residuals": ["x"], "variables": ["x"], "domain": [[-1, 1]],
                      "center": [0], "preconditioner": [[1]], "precision_digits": 24},
        "source_inventories": inventory_records,
        "source_sha256": before_sources,
        "binaries": binaries,
        "binary_sha256": before_binaries,
        "cli_exchanges": [],
        "success": False,
    }

    def envelope(kind, **payload):
        return {"id": kind, "type": kind, "payload": {"schema_version": 1, **payload}}

    register = envelope("RegisterResidualSystem", system="system:interop",
                        residuals=["x"], variables=["x"])
    cli_arguments = [binaries["cli"], "--compute-steps", "2000000",
                     "--memory-bytes", "1073741824", "--allocations", "1000000",
                     "--verifier-steps", "100000000"]

    def run_cli(label, requests):
        data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in requests)
        with subprocess.Popen(cli_arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as process:
            try:
                stdout, stderr = process.communicate(data, timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise AssertionError(f"CLI timeout: {stdout!r}; {stderr!r}")
            exchange = {"label": label, "pid": process.pid, "command": cli_arguments,
                        "stdin": data, "stdout": stdout, "stderr": stderr,
                        "returncode": process.returncode}
            report["cli_exchanges"].append(exchange)
            assert process.returncode == 0, exchange
        responses = [json.loads(line) for line in stdout.splitlines()]
        assert len(responses) == len(requests), exchange
        for request, response in zip(requests, responses, strict=True):
            assert response["id"] == request["id"], response
            assert response["status"] == "success", response
        return responses

    def exact_native_claim(result):
        assert result["domain"] == [{"lower": "-1", "upper": "1"}], result
        assert result["root_enclosure"] == [{"lower": "0", "upper": "0"}], result
        assert result["contraction_bound"] == "0", result
        assert result["evidence"] == "certificate_verified", result
        assert result["claim"] == "unique_nonsingular_real_root_in_box", result

    try:
        witness = {"domain": [{"lower": "-1", "upper": "1"}], "center": ["0"],
                   "preconditioner": [["1"]], "precision_digits": 24}
        sender = run_cli("native_producer", [
            register,
            envelope("CertifyRootBox", system="system:interop", certificate="certificate:native",
                     witness=witness),
            envelope("ExportRootCertificate", certificate="certificate:native"),
        ])
        exact_native_claim(sender[1]["result"])
        native_bytes = sender[2]["result"]["artifact"].encode("utf-8")

        sys.dont_write_bytecode = True
        assert "sympy" not in sys.modules
        spec = importlib.util.spec_from_file_location("fsym_python", binaries["extension"])
        assert spec is not None and spec.loader is not None
        native = importlib.util.module_from_spec(spec)
        sys.modules["fsym_python"] = native
        spec.loader.exec_module(native)
        sys.path.insert(0, str(repo / "python"))
        import sympy as s
        from sympy.core import _native, _symbol_binding
        assert _native is native
        assert Path(s.__file__).resolve() == repo / "python/sympy/__init__.py"
        assert Path(_native.__file__).resolve() == options.extension.resolve()
        x = s.Symbol("x")
        assert _symbol_binding(x).plain

        def budget():
            return s.EvaluationBudget(max_steps=100_000_000, max_memory_bytes=1 << 30,
                                      max_allocations=1_000_000)

        def exact_python_claim(receipt):
            assert type(receipt) is s.RootBoxReceipt
            assert receipt.residuals == (x,)
            assert len(receipt.variables) == 1 and receipt.variables[0].plain
            assert receipt.variables[0].name == "x"
            assert receipt.domain == ((s.Integer(-1), s.Integer(1)),)
            assert receipt.center == (s.Integer(0),)
            assert receipt.preconditioner == ((s.Integer(1),),)
            assert receipt.precision_digits == 24
            assert receipt.root_enclosure == ((s.Integer(0), s.Integer(0)),)
            assert receipt.contraction_bound == 0

        replay_budget = budget()
        replayed = s.RootBoxReceipt.from_bytes(native_bytes, residuals=x, args=x,
                                              budget=replay_budget)
        exact_python_claim(replayed)
        assert replayed.to_bytes(budget=replay_budget) == native_bytes
        report["cli_to_python"] = {"independent_replay_accepted": True,
                                   "exact_root_enclosure": [["0", "0"]],
                                   "contraction_bound": "0", "canonical_bytes_preserved": True}

        produce_budget = budget()
        produced = s.certify_root_box(x, x, [(-1, 1)], center=[0], preconditioner=[[1]],
                                      precision=24, budget=produce_budget)
        exact_python_claim(produced)
        python_bytes = produced.to_bytes(budget=produce_budget)
        receiver = run_cli("native_receiver", [
            register,
            envelope("ReplayRootCertificate", system="system:interop", certificate="certificate:python",
                     artifact=python_bytes.decode("utf-8")),
            envelope("ExportRootCertificate", certificate="certificate:python"),
        ])
        exact_native_claim(receiver[1]["result"])
        assert receiver[2]["result"]["artifact"].encode("utf-8") == python_bytes
        assert python_bytes == native_bytes
        assert sender[1]["result"]["context_digest"] == receiver[1]["result"]["context_digest"]
        report["python_to_cli"] = {"independent_replay_accepted": True,
                                   "exact_root_enclosure": [["0", "0"]],
                                   "contraction_bound": "0", "canonical_bytes_preserved": True}
        report["artifact"] = {"utf8": native_bytes.decode("utf-8"), "bytes": len(native_bytes),
                              "sha256": hashlib.sha256(native_bytes).hexdigest(),
                              "both_independent_producers_byte_identical": True}
        report["loaded_python"] = {"executable": sys.executable, "shell": s.__file__,
                                   "extension": _native.__file__}
        report["success"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["source_hashes_unchanged"] = all(digest(path) == value for path, value in before_sources.items())
        report["binary_hashes_unchanged"] = all(digest(binaries[name]) == value for name, value in before_binaries.items())
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        options.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["source_hashes_unchanged"] and report["binary_hashes_unchanged"], report
    print(json.dumps({"success": True, "directions": ["CLI -> Python", "Python -> CLI"],
                      "root_enclosure": [["0", "0"]], "contraction_bound": "0",
                      "artifact_bytes": len(native_bytes),
                      "artifact_sha256": report["artifact"]["sha256"],
                      "byte_identical_independent_producers": True,
                      "source_hashes_unchanged": len(before_sources),
                      "binary_sha256": before_binaries, "report": str(options.output)}, indent=2))


if __name__ == "__main__":
    main()
