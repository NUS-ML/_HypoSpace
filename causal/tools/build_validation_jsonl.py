#!/usr/bin/env python3
"""
Build a small validation JSONL (messages format) by randomly sampling observation sets
from node 2..4 datasets and generating prompts+answers in the same format as training.

Defaults are small-scale and deterministic with --seed.

Examples:
  # Build a tiny validation set (~8 per node, prior size 0 only, i.e., Prior: None)
  python -m causal.tools.build_validation_jsonl \
    --node-range 2-4 \
    --per-node 8 \
    --output data/val_nodes2-4.jsonl

  # Use all-subsets prior mode but cap to size <=1 to keep small
  python -m causal.tools.build_validation_jsonl \
    --node-range 2-4 \
    --per-node 6 \
    --prior-mode all-subsets \
    --limit-prior-size 1 \
    --output data/val_nodes2-4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..generate_causal_dataset import CausalDatasetGenerator
from .build_lora_jsonl import (
    _compose_user_prompt,
    _edges_to_graph_line,
    _build_prior_block,
    _infer_max_edges,
)


ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]


def _parse_range(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _iter_collection_obs_sets(coll: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    meta = coll.get("metadata", {})
    if "datasets" in coll:
        for ds in coll["datasets"]:
            yield meta, ds
    elif "datasets_by_n_observations" in coll:
        for _k, lst in coll["datasets_by_n_observations"].items():
            for ds in lst:
                yield meta, ds


def _load_or_generate_collection(n: int, seed: Optional[int], n_samples: Optional[int]) -> Dict[str, Any]:
    # Prefer reuse existing files
    base = Path(f"causal/datasets/node0{n}")
    candidates = [base / f"n{n}_all_observations.json"]
    if base.exists():
        candidates += sorted(base.glob("*.json"))
    for p in candidates:
        if p.exists():
            try:
                with p.open("r") as f:
                    return json.load(f)
            except Exception:
                continue
    # Fallback: generate a small collection in-memory
    nodes = ALPHABET[:n]
    return CausalDatasetGenerator.generate_complete_dataset_collection(
        nodes=nodes,
        n_observations=n,
        fixed=True,
        max_edges=None,
        seed=seed,
        n_samples=n_samples,
    )


def _sample_obs_sets(coll: Dict[str, Any], k: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    all_sets = [ds for _m, ds in _iter_collection_obs_sets(coll)]
    if not all_sets:
        return []
    rng = random.Random(seed)
    if k >= len(all_sets):
        return list(all_sets)
    return rng.sample(all_sets, k)


def main():
    ap = argparse.ArgumentParser(description="Build a small validation JSONL from node2-4 datasets")
    ap.add_argument("--node-range", default="2-4", help="Range/list of node counts, e.g., '2-4' or '2,3,4'")
    ap.add_argument("--per-node", type=int, default=8, help="Number of observation sets to sample per node count")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--n-samples", type=int, default=200, help="Generation sampling cap for speed when no files exist")
    ap.add_argument("--prior-mode", default="size-only", choices=["size-only", "all-subsets"],
                    help="size-only: prior sizes 0..limit-prior-size; all-subsets: enumerate all subsets (optionally capped)")
    ap.add_argument("--limit-prior-size", type=int, default=0, help="Cap prior size (applies to either mode). 0 means only None")
    ap.add_argument("--output", required=True, help="Output JSONL path for validation set")

    args = ap.parse_args()

    random.seed(args.seed)

    node_counts = _parse_range(args.node_range)

    examples: List[Dict[str, Any]] = []

    for n in node_counts:
        coll = _load_or_generate_collection(n=n, seed=args.seed, n_samples=args.n_samples)
        picked = _sample_obs_sets(coll, args.per_node, seed=args.seed + n)

        for obs_set in picked:
            nodes = obs_set.get("nodes") or coll.get("metadata", {}).get("nodes") or ALPHABET[:n]
            observations = obs_set.get("observations", [])
            gts = obs_set.get("ground_truth_graphs", [])
            if not gts or not observations:
                continue

            max_edges = _infer_max_edges(coll.get("metadata", {}), obs_set)
            constraint_info = f"\nConstraint: The graph should have at most {max_edges} edges." if isinstance(max_edges, int) else ""

            # For each target GT, build small number of prior variants
            for idx, target in enumerate(gts):
                target_edges = target.get("edges", [])

                others = [g for j, g in enumerate(gts) if j != idx]
                def edges_key(g: Dict[str, Any]) -> str:
                    return _edges_to_graph_line(g.get("edges", []))
                others_sorted = sorted(others, key=edges_key)

                m = len(others_sorted)
                max_size = min(args.limit_prior_size, m)

                if args.prior_mode == "size-only":
                    sizes = range(0, max_size + 1)
                    for r in sizes:
                        priors_edges = [g.get("edges", []) for g in others_sorted[:r]]
                        prior_block = _build_prior_block(priors_edges)
                        user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block)
                        assistant_content = _edges_to_graph_line(target_edges)
                        examples.append({
                            "messages": [
                                {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ]
                        })
                else:  # all-subsets
                    # Enumerate all subsets up to size cap
                    for r in range(0, max_size + 1):
                        for subset in combinations(others_sorted, r):
                            priors_edges = [g.get("edges", []) for g in subset]
                            prior_block = _build_prior_block(priors_edges)
                            user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block)
                            assistant_content = _edges_to_graph_line(target_edges)
                            examples.append({
                                "messages": [
                                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": assistant_content},
                                ]
                            })

    # Write output (small, so single file)
    out_path = Path(args.output)
    if "aliyun" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem + "_aliyun" + out_path.suffix)
    if "val" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem.replace("_aliyun", "_val_aliyun"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as w:
        for ex in examples:
            w.write(json.dumps(ex, ensure_ascii=False))
            w.write("\n")

    print(f"Validation JSONL written: {out_path}  (examples: {len(examples)})")


if __name__ == "__main__":
    main()
