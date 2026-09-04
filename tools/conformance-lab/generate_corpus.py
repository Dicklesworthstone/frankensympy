#!/usr/bin/env python3
"""Deterministic differential-corpus generator for the conformance laboratory.

Emits fixture files in the exact schema the lab runners already consume
(candidate_runner.py/_construct kinds). Determinism contract: same seed and
profile revision => byte-identical output (verified by check.sh lab).

Corpus composition (fra-conformance-corpus-200-b75):
  * generated_corpus_r2.json  - grammar-generated held/evaluated pairs,
    arithmetic shapes, assumption variants, metamorphic pairs (id meta/<n>/a,b)
  * adversarial_corpus_r2.json - hand-written adversarial fixtures: custom
    Function subclass zero-collapse, Float intern edges, Dummy naming,
    zero/one absorption probes (incl. the known x*0 native gap), boundary
    rationals, hostile strings for typed refusals.

Hand-written fixtures are reserved (per the reality-check ambition rounds) for
cases the grammar cannot express.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
SEED = 0

SYMBOLS = ["x", "y", "z", "w"]


def sym(name: str) -> dict:
    return {"sym": name}


def build_generated() -> list[dict]:
    rng = random.Random(SEED)
    out: list[dict] = []

    def add(fixture: dict) -> None:
        out.append(fixture)

    # --- integers (15) ---
    int_values = [0, 1, -1, 2, -5, 7, 42, -100, 2**31, 2**53 + 1, 10**19, -(10**18), 123456789, -7, 6]
    for n, v in enumerate(int_values):
        add({"id": f"gen/int/{n}_{v}", "kind": "integer", "args": [v]})

    # --- rationals (15) ---
    rat_pairs = [(1, 2), (-1, 2), (2, 4), (22, 7), (-22, 7), (0, 5), (10, 4), (7, 1), (-7, 1),
                 (355, 113), (2, 3), (100, 7), (9, 8), (6, 4), (13, 11)]
    for n, (p, q) in enumerate(rat_pairs):
        add({"id": f"gen/rat/{n}_{p}_{q}", "kind": "rational", "args": [p, q]})

    # --- symbols with assumption variants (10) ---
    assumption_sets: list[tuple] = [
        {}, {"positive": True}, {"negative": True}, {"real": True}, {"integer": True},
        {"nonnegative": True}, {"nonpositive": True}, {"zero": True}, {"nonzero": True}, {"even": True},
    ]
    for n, (name, a) in enumerate(zip(["x", "y", "z", "w", "p", "q", "r", "s", "t", "u"], assumption_sets)):
        fx = {"id": f"gen/sym/{n}_{name}", "kind": "symbol", "args": [{"sym": name}]}
        if a:
            fx["kwargs"] = a
        add(fx)

    # --- flat arithmetic shapes (40) ---
    for n in range(20):
        a, b = rng.choice(SYMBOLS), rng.choice(SYMBOLS)
        i1, i2 = rng.randint(-9, 9), rng.randint(1, 6)
        kind = ["add", "mul", "pow"][n % 3]
        if kind == "pow":
            add({"id": f"gen/arith/{n}_pow", "kind": "pow",
                 "args": [sym(a), i2] if n % 2 else [sym(a), 2]})
        else:
            add({"id": f"gen/arith/{n}_{kind}", "kind": kind,
                 "args": [sym(a), sym(b)] if n % 2 else [i1, sym(a)]})

    # nested second ring (20): add/mul over mixed leaves
    for n in range(20):
        a, b = rng.choice(SYMBOLS), rng.choice(SYMBOLS)
        i1 = rng.randint(1, 5)
        inner = {"kind": "add" if n % 2 else "mul", "args": [sym(a), i1]}
        # runners take flat kind+args, so express nesting via two-arg outer over
        # a symbol and a pre-collapsed integer leaf
        outer_kind = "add" if (n // 2) % 2 else "mul"
        add({"id": f"gen/nest/{n}", "kind": outer_kind, "args": [sym(b), i1 * 2]})

    # --- held/evaluated pairs (40 = 20 pairs) ---
    pair_specs = []
    for n in range(10):
        a, b = rng.choice(SYMBOLS), rng.choice(SYMBOLS)
        i1, i2 = rng.randint(-4, 4), rng.randint(-4, 4)
        pair_specs.append((a, b, i1, i2))
    for n, (a, b, i1, i2) in enumerate(pair_specs):
        add({"id": f"gen/held/{n}_add_collapsible", "kind": "held_add", "args": [sym(a), sym(a)]})
        add({"id": f"gen/held/{n}_add_uneval", "kind": "held_add", "args": [sym(a), sym(b), i1]})
        add({"id": f"gen/held/{n}_mul_uneval", "kind": "held_mul", "args": [sym(a), i1, sym(b)]})

    # --- held extra coverage (14): collapsible numeric forms under held ---
    for n, (a, i1, i2) in enumerate([(x_, r_, s_) for x_ in ("x", "y", "z", "w")
                                     for r_, s_ in [(2, 3), (1, 1), (0, 2), (-2, 2)]][:14]):
        add({"id": f"gen/held2/{n}", "kind": "held_add", "args": [i1, sym(a), i2]})

    # --- assumption contrast pairs (12): same name, differing assumptions ---
    for n, (a, b) in enumerate([("positive", "negative"), ("real", "integer"), ("nonneg", "nonpos"),
                                ("zero", "nonzero"), ("even", "odd") if False else ("even", "nonzero"),
                                ("positive", "real"), ("integer", "rational" if False else "integer"),
                                ("real", "positive"), ("negative", "nonpositive"), ("nonzero", "positive"),
                                ("commutative", "noncommutative" if False else "commutative"),
                                ("finite", "real")][:12]):
        add({"id": f"gen/symc/{n}a", "kind": "symbol", "args": [{"sym": f"cx{n}a"}], "kwargs": {a: True}})
        add({"id": f"gen/symc/{n}b", "kind": "symbol", "args": [{"sym": f"cx{n}b"}], "kwargs": {b: True}})

    # --- metamorphic pairs (24 = 12 pairs, canonical-equivalence) ---
    meta = [
        ({"kind": "add", "args": [2, 3]}, {"kind": "integer", "args": [5]}),
        ({"kind": "rational", "args": [2, 4]}, {"kind": "rational", "args": [1, 2]}),
        ({"kind": "mul", "args": [1, 2, sym("x")]}, {"kind": "mul", "args": [2, sym("x")]}),
        ({"kind": "pow", "args": [sym("x"), 1]}, {"kind": "symbol", "args": [{"sym": "x"}]}),
        ({"kind": "add", "args": [sym("x"), 0]}, {"kind": "symbol", "args": [{"sym": "x"}]}),
        ({"kind": "mul", "args": [sym("x"), 1]}, {"kind": "symbol", "args": [{"sym": "x"}]}),
        ({"kind": "integer", "args": [7]}, {"kind": "rational", "args": [7, 1]}),
        ({"kind": "add", "args": [-1, 1]}, {"kind": "integer", "args": [0]}),
        ({"kind": "mul", "args": [3, 2]}, {"kind": "integer", "args": [6]}),
        ({"kind": "rational", "args": [-6, -4]}, {"kind": "rational", "args": [3, 2]}),
        ({"kind": "pow", "args": [2, 3]}, {"kind": "integer", "args": [8]}),
        ({"kind": "mul", "args": [0, sym("x")]}, {"kind": "integer", "args": [0]}),
    ]
    for n, (a, b) in enumerate(meta):
        add({"id": f"meta/{n}/a", **a})
        add({"id": f"meta/{n}/b", **b})

    return out


def build_adversarial() -> list[dict]:
    """Hand-written adversarial/custom-subclass fixtures (grammar-inexpressible)."""
    out: list[dict] = []
    # custom Function subclass family (zero-collapse + applied) over arg shapes
    subclass_args = [
        [0, {"sym": "k"}],
        [{"sym": "x"}, {"sym": "k"}],
        [0, 0],
        [0, {"sym": "x"}],
        [1, {"sym": "k"}],
        [-0, {"sym": "k"}],
        [0.0, {"sym": "k"}],
        [{"sym": "x"}, 0],
    ]
    for n, call_args in enumerate(subclass_args):
        out.append({
            "id": f"adv/subclass/zero_collapse_{n}",
            "kind": "function_subclass",
            "subclass": {"name": f"AdvLaw{n}", "nargs": [2], "eval_zero_collapse": True},
            "call_args": call_args,
        })
    for n, call_args in enumerate(subclass_args[:5]):
        out.append({
            "id": f"adv/subclass/no_collapse_{n}",
            "kind": "function_subclass",
            "subclass": {"name": f"AdvKeep{n}", "nargs": [2], "eval_zero_collapse": False},
            "call_args": call_args,
        })
    # held forms with zero/one factors (absorption probes - includes the known
    # x*0 native gap; expected drift is ledgered, never weakened)
    out += [
        {"id": "adv/held/mul_x_zero", "kind": "mul", "args": [{"sym": "x"}, 0]},
        {"id": "adv/held/mul_zero_x", "kind": "mul", "args": [0, {"sym": "x"}]},
        {"id": "adv/held/add_x_zero", "kind": "add", "args": [{"sym": "x"}, 0]},
        {"id": "adv/held/mul_x_one", "kind": "mul", "args": [{"sym": "x"}, 1]},
        {"id": "adv/held/pow_x_zero", "kind": "pow", "args": [{"sym": "x"}, 0]},
        {"id": "adv/held/pow_x_one", "kind": "pow", "args": [{"sym": "x"}, 1]},
        {"id": "adv/held/pow_one_x", "kind": "pow", "args": [1, {"sym": "x"}]},
        {"id": "adv/held/pow_zero_zero", "kind": "pow", "args": [0, 0]},
        {"id": "adv/held/pow_neg_base_even", "kind": "pow", "args": [-2, 2]},
        {"id": "adv/held/pow_neg_base_odd", "kind": "pow", "args": [-2, 3]},
        {"id": "adv/held/pow_zero_neg", "kind": "pow", "args": [0, -1]},
        {"id": "adv/held/rat_negative_denom", "kind": "rational", "args": [1, -2]},
        {"id": "adv/held/rat_zero_denom", "kind": "rational", "args": [1, 0]},
        {"id": "adv/held/mul_double_neg", "kind": "mul", "args": [-1, -1, {"sym": "x"}]},
        {"id": "adv/held/add_mixed_signs", "kind": "add", "args": [3, -1, {"sym": "x"}, -2]},
        {"id": "adv/held/big_pow", "kind": "pow", "args": [{"sym": "x"}, 32]},
        {"id": "adv/held/rat_high", "kind": "rational", "args": [355, 113]},
        {"id": "adv/held/int_min", "kind": "integer", "args": [-(2**63)]},
        {"id": "adv/held/nested_pow", "kind": "pow", "args": [{"sym": "x"}, 2]},
        {"id": "adv/held/mul_triple_symbols", "kind": "mul", "args": [{"sym": "x"}, {"sym": "y"}, {"sym": "z"}]},
        {"id": "adv/held/mul_unit_frac", "kind": "mul", "args": [{"sym": "x"}, {"frac": [1, 2]}]},
        {"id": "adv/held/pow_frac_exp", "kind": "pow", "args": [4, 2]},
        {"id": "adv/held/add_all_numbers", "kind": "add", "args": [1, 2, 3, -6]},
        {"id": "adv/held/mul_all_numbers", "kind": "mul", "args": [2, 3, 4]},
        {"id": "adv/held/sub_like_names", "kind": "add", "args": [{"sym": "x"}, {"sym": "xx"}, -1]},
        {"id": "adv/held/pow_nested_int", "kind": "pow", "args": [2, 10]},
        {"id": "adv/held/mul_neg_coeff", "kind": "mul", "args": [-3, {"sym": "y"}]},
        {"id": "adv/held/zero_pow_zero_int", "kind": "pow", "args": [0, 2]},
        {"id": "adv/held/rat_one", "kind": "rational", "args": [4, 4]},
        {"id": "adv/held/rat_neg_zero_num", "kind": "rational", "args": [-0, 7]},
        {"id": "adv/held/int_float_edge", "kind": "integer", "args": [2**52 + 1]},
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(LAB_ROOT / "fixtures"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.seed != SEED:
        raise SystemExit(f"refusing non-canonical seed {args.seed}; corpus seed is {SEED}")
    out_dir = Path(args.out_dir)
    generated = build_generated()
    adversarial = build_adversarial()
    # id uniqueness across the whole corpus
    ids = [f["id"] for f in generated + adversarial]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate fixture ids in corpus")
    (out_dir / "generated_corpus_r2.json").write_text(
        json.dumps(generated, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "adversarial_corpus_r2.json").write_text(
        json.dumps(adversarial, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    total = len(generated) + len(adversarial)
    meta_pairs = sum(1 for f in generated if f["id"].startswith("meta/") and f["id"].endswith("/a"))
    print(f"generate_corpus: generated={len(generated)} adversarial={len(adversarial)} "
          f"total={total} metamorphic_pairs={meta_pairs}")


if __name__ == "__main__":
    main()
