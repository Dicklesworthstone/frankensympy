# FrankenSymPy

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT%20with%20rider-blue.svg" alt="License: MIT with OpenAI/Anthropic Rider">
  <img src="https://img.shields.io/badge/rust-2024%20edition%20%C2%B7%20nightly-orange.svg" alt="Rust 2024 nightly">
  <img src="https://img.shields.io/badge/unsafe-%23!%5Bforbid(unsafe__code)%5D-brightgreen.svg" alt="No unsafe code">
  <img src="https://img.shields.io/badge/async-asupersync%20(no%20tokio)-purple.svg" alt="asupersync, no tokio">
  <img src="https://img.shields.io/badge/conformance-differential%20oracle-success.svg" alt="Differential Oracle Conformance">
</p>

> **FrankenSymPy is a clean-room, memory-safe Rust reimplementation of SymPy
> (Python's library for symbolic mathematics and computer algebra systems).**
> Built from first principles in safe Rust, FrankenSymPy brings high-performance
> symbolic manipulation, exact arithmetic, algebraic simplification, calculus,
> equation solving, matrices, and differential conformance testing to Rust and Python workloads.

---

## TL;DR

### The Problem

SymPy is the gold standard for symbolic mathematics in Python, but Python's runtime characteristics impose structural limitations:

- **Performance & Memory Footprint:** Expression trees in Python carry significant object allocation overhead, pointer chasing, and reference-counting pressure.
- **CPython GIL:** Multithreaded parallel evaluation of algebraic expressions and symbolic matrices is bottlenecked by the Global Interpreter Lock.
- **Embedded & Edge Constraints:** Embedding Python and SymPy into standalone systems, microservices, WebAssembly, or safety-critical Rust runtimes is heavy and complex.
- **Execution Resource Bounds:** Preventing pathological recursion or infinite expansion loops in computer algebra requires strict, deterministic resource budgets and structured timeouts.

### The Solution

FrankenSymPy rebuilds the symbolic algebra surface in idiomatic, safe Rust with three core guarantees:

1. **Memory- and Thread-Safety by Construction:** `#![forbid(unsafe_code)]` workspace-wide across all numeric and symbolic crates.
2. **Structured Async & Deterministic Bounding:** Powered by [asupersync](/dp/asupersync) for deterministic evaluation budgets, cancel-safe pipelines, and virtual time execution.
3. **Differential Conformance Against SymPy:** A continuous differential oracle harness capturing reference behavior from upstream SymPy and diffing results to prevent behavioral drift.

---

## Why FrankenSymPy?

| | SymPy (Python) | Typical CAS / Symbolic Rust Crates | **FrankenSymPy** |
|---|---|---|---|
| **Memory safety** | Dynamic / C-extensions | Safe Rust | **`#![forbid(unsafe_code)]`** |
| **Async runtime** | N/A (Python sync) | None / tokio | **asupersync** (structured concurrency, no tokio) |
| **Surface area** | Comprehensive CAS | Focused / partial | **Full symbolic algebra suite** |
| **Differential conformance** | Self-checking | None | **Live Python SymPy Oracle validation** |
| **Exact arithmetic** | `mpmath` / Python `int` | `num-bigint` / `num-rational` | **Arbitrary-precision rational / algebraic core** |
| **Execution bounds** | Recursion limit | Manual | **Deterministic compute & step budgets** |
| **Multi-agent ready** | Standard repo | Standard repo | **Beads (`br`) + MCP Agent Mail workflow** |

---

## Workspace Architecture

FrankenSymPy is structured as a modular Cargo workspace:

```
frankensympy/
├── Cargo.toml                         # Workspace definition
├── crates/
│   ├── fsym-core/                     # Core AST (Expr, Symbol, Integer, Rational, Add, Mul, Pow)
│   ├── fsym-polys/                    # Polynomial rings, GCD, factorization, Gröbner bases
│   ├── fsym-simplify/                 # Algebraic simplification, rewrites, expansion, canonicalization
│   ├── fsym-calculus/                 # Differentiation, integration, limits, series expansions
│   ├── fsym-solvers/                  # Algebraic solvers, polynomial/linear systems, ODEs
│   ├── fsym-matrices/                 # Symbolic matrices, determinants, eigenvalues, decompositions
│   ├── fsym-functions/                # Elementary & special functions (trig, gamma, zeta, special)
│   ├── fsym-logic/                    # Boolean algebra, truth tables, CNF/DNF, SAT solving
│   ├── fsym-ntheory/                  # Primality tests, factorization, totient, modular arithmetic
│   ├── fsym-sets/                     # Symbolic sets, intervals, unions, finite sets
│   ├── fsym-geometry/                 # 2D/3D geometric primitives, intersections, distances
│   ├── fsym-tensor/                   # Symbolic tensors, index notation, contractions
│   ├── fsym-assumptions/              # Assumptions system, predicate deduction, refinement
│   ├── fsym-printing/                 # LaTeX emission, code generation (Rust, C, Python), pretty-printing
│   ├── fsym-runtime/                  # Evaluation budgets, timeouts, audit logs, asupersync integration
│   ├── fsym-conformance/              # Differential testing harness against Python SymPy
│   └── fsym-python/                   # PyO3 bindings for drop-in Python integration
├── docs/                              # Architecture, schemas, and design docs
└── .beads/                            # Dependency-aware issue tracking database
```

---

## Toolchain & Quality Gates

FrankenSymPy enforces strict CI gates (G1–G8):

- **G1 (fmt + clippy):** Zero clippy warnings under `-D warnings` and strict formatting.
- **G2 (unit + property tests):** Unit test coverage across all domain modules.
- **G3 (differential conformance):** Diffing outputs against live `sympy` reference oracle.
- **G4 (adversarial & security):** Fuzzing and adversarial boundary checking.
- **G5 (E2E scenarios):** Multi-step symbolic workflow scenarios.
- **G6 (performance baselines):** Tracking evaluation throughput and allocation budgets.
- **G7 (schema validation):** Machine-readable contract validation.
- **G8 (durability):** RaptorQ sidecar verification and artifact proofs.

---

## Multi-Agent Development

This repository is built collaboratively with autonomous coding agents using:

- **[beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`)**: Dependency-aware issue tracking.
- **MCP Agent Mail**: Async coordination and file reservation leases between agents.
- **[asupersync](/dp/asupersync)**: Structured async runtime across all async workflows.

---

## License

This project is licensed under the [MIT License (with OpenAI/Anthropic Rider)](LICENSE).
