#!/usr/bin/env python3
"""
Auto-generate causal datasets for node counts (e.g., 2-4) and convert to LoRA JSONL with dedup.

This script will:
1) For each n in the requested node range, generate a dataset collection in-memory using
   CausalDatasetGenerator.generate_complete_dataset_collection (no changes to existing code).
   - If you prefer to reuse existing JSON datasets, you can pass --reuse-files to load from disk instead.
2) Convert each observation set into a LoRA SFT entry using the same prompt template as run_causal_benchmark.
3) Aggregate across nodes and seeds, and de-duplicate by hashing (user+assistant) content.

Example:
  python causal/tools/auto_build_lora_dataset.py \
    --node-range 2-4 \
    --seeds 11,12,13 \
    --mode all-gts --include-priors 1 \
    --n-samples 200 \
    --output data/lora_nodes2-4.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from itertools import combinations

from ..generate_causal_dataset import CausalDatasetGenerator

# Reuse conversion helpers from build_lora_jsonl to keep prompts consistent
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


def _dedup_key(user_content: str, assistant_content: str) -> str:
    h = hashlib.sha256()
    h.update(user_content.strip().encode("utf-8"))
    h.update(b"\n\n")
    h.update(assistant_content.strip().encode("utf-8"))
    return h.hexdigest()


def _build_examples_from_collection(
    coll: Dict[str, Any],
    mode: str,
    include_priors: int,
    prior_mode: str,
    seed: Optional[int],
    dedup_seen: set,
    limit_per_set: Optional[int] = None,
    max_sets: Optional[int] = None,
    max_targets_per_set: Optional[int] = None,
    limit_prior_size: Optional[int] = None,
    writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    examples: List[Dict[str, Any]] = []

    # Optionally subsample observation sets deterministically with seed
    all_sets = list(_iter_collection_obs_sets(coll))
    if max_sets is not None and len(all_sets) > max_sets:
        rng = random.Random(seed)
        picked = rng.sample(all_sets, max_sets)
    else:
        picked = all_sets

    for meta, obs_set in picked:
        nodes = obs_set.get("nodes") or meta.get("nodes") or []
        observations = obs_set.get("observations", [])
        gts = obs_set.get("ground_truth_graphs", [])

        if not nodes or not observations:
            continue

        max_edges = _infer_max_edges(meta, obs_set)
        constraint_info = f"\nConstraint: The graph should have at most {max_edges} edges." if isinstance(max_edges, int) else ""

        # Choose target GT(s)
        selected_indices: List[int] = []
        if not gts:
            targets: List[Optional[List[List[str]]]] = [None]
        else:
            if mode == "all-gts":
                selected_indices = list(range(len(gts)))
            elif mode == "first-gt":
                selected_indices = [0]
            elif mode == "random-gt":
                selected_indices = [random.randrange(len(gts))]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if limit_per_set is not None:
                selected_indices = selected_indices[: max(0, limit_per_set)]
            if max_targets_per_set is not None:
                selected_indices = selected_indices[: max(0, max_targets_per_set)]

        if not gts:
            # No priors available
            user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block="")
            assistant_content = "Graph: "
            key = _dedup_key(user_content, assistant_content)
            if key not in dedup_seen:
                item = {
                    "messages": [
                        {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
                dedup_seen.add(key)
                if writer:
                    writer(item)
                else:
                    examples.append(item)
        else:
            for idx in selected_indices:
                target = gts[idx]
                target_edges = target.get("edges", [])

                # Build a canonical ordering of other GTs (exclude the current target)
                others = [g for j, g in enumerate(gts) if j != idx]
                def edges_key(g: Dict[str, Any]) -> str:
                    # Canonical key by stringified edge list
                    return _edges_to_graph_line(g.get("edges", []))
                others_sorted = sorted(others, key=edges_key)

                # Determine cap of prior size if provided
                prior_cap = None
                if limit_prior_size is not None:
                    prior_cap = max(0, min(limit_prior_size, len(others_sorted)))
                if prior_mode == "size-only":
                    # Prior sizes 0..len(others), using first-N in canonical order
                    max_size = prior_cap if prior_cap is not None else len(others_sorted)
                    for prior_size in range(0, max_size + 1):
                        priors_edges = [g.get("edges", []) for g in others_sorted[:prior_size]]
                        prior_block = _build_prior_block(priors_edges)

                        user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block=prior_block)
                        assistant_content = _edges_to_graph_line(target_edges)

                        key = _dedup_key(user_content, assistant_content)
                        if key not in dedup_seen:
                            item = {
                                "messages": [
                                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": assistant_content},
                                ]
                            }
                            dedup_seen.add(key)
                            if writer:
                                writer(item)
                            else:
                                examples.append(item)
                elif prior_mode == "all-subsets":
                    m = len(others_sorted)
                    if prior_cap is not None:
                        m = min(m, prior_cap)
                    # Enumerate all subsets (including empty) in canonical order per subset size
                    for r in range(0, m + 1):
                        for subset in combinations(others_sorted, r):
                            priors_edges = [g.get("edges", []) for g in subset]
                            prior_block = _build_prior_block(priors_edges)

                            user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block=prior_block)
                            assistant_content = _edges_to_graph_line(target_edges)

                            key = _dedup_key(user_content, assistant_content)
                            if key not in dedup_seen:
                                item = {
                                    "messages": [
                                        {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                                        {"role": "user", "content": user_content},
                                        {"role": "assistant", "content": assistant_content},
                                    ]
                                }
                                dedup_seen.add(key)
                                if writer:
                                    writer(item)
                                else:
                                    examples.append(item)
                else:
                    raise ValueError(f"Unknown prior_mode: {prior_mode}")

    return examples if not writer else []


def _generate_collection_for_n(
    n: int,
    seed: Optional[int],
    fixed: bool,
    n_samples: Optional[int],
    max_edges: Optional[int],
) -> Dict[str, Any]:
    nodes = ALPHABET[:n]
    coll = CausalDatasetGenerator.generate_complete_dataset_collection(
        nodes=nodes,
        n_observations=n,
        fixed=fixed,
        max_edges=max_edges,
        seed=seed,
        n_samples=n_samples,
    )
    return coll


def main():
    ap = argparse.ArgumentParser(description="Auto-generate node[range] datasets and build LoRA JSONL with dedup")
    ap.add_argument("--node-range", default="2-4", help="Range or list of node counts, e.g., '2-4' or '2,3,4'")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g., '1,2,3'")
    ap.add_argument("--n-samples", type=int, default=None, help="Sample up to N observation combos per n_obs (for speed)")
    ap.add_argument("--large-n-samples", type=int, default=None, help="Override --n-samples for n >= large-n-threshold")
    ap.add_argument("--fixed", action="store_true", help="Use exactly n_observations=n (default: generate 1..n)")
    ap.add_argument("--max-edges", type=int, default=None, help="Optional max_edges constraint")
    ap.add_argument("--mode", default="all-gts", choices=["all-gts", "first-gt", "random-gt"],
                    help="Which GT(s) to output as assistant answers")
    ap.add_argument("--include-priors", type=int, default=0, help="(Deprecated) Ignored when --prior-mode is used; size-only/all-subsets enumerate priors deterministically")
    ap.add_argument("--prior-mode", default="size-only", choices=["size-only", "all-subsets"],
                    help="size-only: prior sizes 0..(k-1) using first-N canonical GTs; all-subsets: enumerate all subsets of other GTs as priors")
    ap.add_argument("--limit-per-set", type=int, default=None, help="Limit examples per observation set")
    # Scaling controls for larger node sizes
    ap.add_argument("--large-n-threshold", type=int, default=4, help="For n >= threshold, apply scaling controls")
    ap.add_argument("--large-max-sets", type=int, default=300, help="Max observation sets processed per large-n node count")
    ap.add_argument("--large-max-targets", type=int, default=3, help="Max target GTs per observation set for large-n")
    ap.add_argument("--large-prior-mode", default=None, choices=["size-only", "all-subsets", None],
                    help="Override prior mode for large-n; default None means use --prior-mode")
    ap.add_argument("--large-limit-prior-size", type=int, default=1, help="Cap prior size for large-n (0 means None only)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--reuse-files", action="store_true", help="Reuse on-disk datasets if found (fallback to generate)")

    # Per-n overrides and caps (focused on n=4 and n=5)
    ap.add_argument("--n4-n-samples", type=int, default=None, help="Override --n-samples for n=4")
    ap.add_argument("--n5-n-samples", type=int, default=None, help="Override --n-samples for n=5")
    ap.add_argument("--n4-max-sets", type=int, default=None, help="Max observation sets for n=4 (overrides large/small policy)")
    ap.add_argument("--n5-max-sets", type=int, default=None, help="Max observation sets for n=5 (overrides large policy)")
    ap.add_argument("--n4-max-targets", type=int, default=None, help="Max target GTs per set for n=4")
    ap.add_argument("--n5-max-targets", type=int, default=None, help="Max target GTs per set for n=5")
    ap.add_argument("--n4-limit-prior-size", type=int, default=None, help="Cap prior size for n=4 (size-only mode)")
    ap.add_argument("--n5-limit-prior-size", type=int, default=None, help="Cap prior size for n=5 (size-only mode)")
    ap.add_argument("--n4-cap-examples", type=int, default=None, help="Stop after writing this many examples for n=4")
    ap.add_argument("--n5-cap-examples", type=int, default=None, help="Stop after writing this many examples for n=5")

    args = ap.parse_args()

    node_counts = _parse_range(args.node_range)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    dedup_seen = set()

    # Prepare streaming writer with 200MB splitting up-front
    out_path = Path(args.output)
    if "aliyun" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem + "_aliyun" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    MAX_BYTES = 200 * 1024 * 1024
    def next_part_path(base: Path, part_idx: int) -> Path:
        stem, suffix = base.stem, base.suffix
        return base.with_name(f"{stem}_part{part_idx:03d}{suffix}")

    part_idx = 1
    cur_path = next_part_path(out_path, part_idx)
    cur_f = cur_path.open("wb")
    cur_bytes = 0
    total_written = 0

    # Per-n counters and caps
    per_n_written: Dict[int, int] = {}
    per_n_caps: Dict[int, Optional[int]] = {4: args.n4_cap_examples, 5: args.n5_cap_examples}
    per_n_stop: Dict[int, bool] = {}
    active_n: Optional[int] = None

    def write_item(item: Dict[str, Any]):
        nonlocal cur_f, cur_path, part_idx, cur_bytes, total_written, active_n
        # Enforce per-n caps if configured
        if active_n is not None:
            cap = per_n_caps.get(active_n)
            cur = per_n_written.get(active_n, 0)
            if cap is not None and cur >= cap:
                per_n_stop[active_n] = True
                return
        line = json.dumps(item, ensure_ascii=False)
        b = line.encode("utf-8")
        nb = len(b) + 1
        if cur_bytes > 0 and cur_bytes + nb > MAX_BYTES:
            cur_f.close()
            print(f"Rotated file at ~{cur_bytes/1024/1024:.1f} MB: {cur_path}")
            part_idx += 1
            cur_path = next_part_path(out_path, part_idx)
            cur_f = cur_path.open("wb")
            cur_bytes = 0
        cur_f.write(b)
        cur_f.write(b"\n")
        cur_bytes += nb
        total_written += 1
        if active_n is not None:
            per_n_written[active_n] = per_n_written.get(active_n, 0) + 1
            cap = per_n_caps.get(active_n)
            if cap is not None and per_n_written[active_n] >= cap:
                per_n_stop[active_n] = True

    for n in node_counts:
        per_n_written.setdefault(n, 0)
        for seed in seeds:
            collection: Optional[Dict[str, Any]] = None

            # Optionally reuse existing dataset files if present
            if args.reuse_files:
                candidate_paths = []
                # canonical name pattern used elsewhere
                base = Path(f"causal/datasets/node0{n}")
                candidate_paths.append(base / f"n{n}_all_observations.json")
                # also allow any json under the node directory
                if base.exists():
                    candidate_paths.extend(sorted(base.glob("*.json")))

                for p in candidate_paths:
                    if p.exists():
                        try:
                            with p.open("r") as f:
                                collection = json.load(f)
                            break
                        except Exception:
                            continue

            if collection is None:
                eff_n_samples = args.n_samples
                if n >= args.large_n_threshold and args.large_n_samples is not None:
                    eff_n_samples = args.large_n_samples
                # Per-n overrides for sampling
                if n == 4 and args.n4_n_samples is not None:
                    eff_n_samples = args.n4_n_samples
                if n == 5 and args.n5_n_samples is not None:
                    eff_n_samples = args.n5_n_samples

                collection = _generate_collection_for_n(
                    n=n,
                    seed=seed,
                    fixed=args.fixed,
                    n_samples=eff_n_samples,
                    max_edges=args.max_edges,
                )

            # Decide scaling policy based on node count
            if n >= args.large_n_threshold:
                eff_prior_mode = args.large_prior_mode or args.prior_mode
                eff_limit_prior_size = args.large_limit_prior_size
                eff_max_sets = args.large_max_sets
                eff_max_targets = args.large_max_targets
            else:
                eff_prior_mode = args.prior_mode
                eff_limit_prior_size = None  # full coverage on small n
                eff_max_sets = None
                eff_max_targets = None

            # Per-n overrides regardless of threshold
            if n == 4:
                if args.n4_max_sets is not None:
                    eff_max_sets = args.n4_max_sets
                if args.n4_max_targets is not None:
                    eff_max_targets = args.n4_max_targets
                if args.n4_limit_prior_size is not None:
                    eff_limit_prior_size = args.n4_limit_prior_size
            if n == 5:
                if args.n5_max_sets is not None:
                    eff_max_sets = args.n5_max_sets
                if args.n5_max_targets is not None:
                    eff_max_targets = args.n5_max_targets
                if args.n5_limit_prior_size is not None:
                    eff_limit_prior_size = args.n5_limit_prior_size

            # Convert collection to LoRA examples with cross-seed dedup, streaming to disk
            active_n = n
            _ = _build_examples_from_collection(
                coll=collection,
                mode=args.mode,
                include_priors=args.include_priors,
                prior_mode=eff_prior_mode,
                seed=seed,
                dedup_seen=dedup_seen,
                limit_per_set=args.limit_per_set,
                max_sets=eff_max_sets,
                max_targets_per_set=eff_max_targets,
                limit_prior_size=eff_limit_prior_size,
                writer=write_item,
            )
            print(f"[n={n}, seed={seed}] streamed; per-n {per_n_written.get(n,0)}, running total {total_written}")
            # Stop early if this n hit cap
            if per_n_stop.get(n):
                print(f"[n={n}] Reached cap ({per_n_caps.get(n)}), skipping remaining seeds.")
                break
    # Close stream and summarize
    try:
        cur_f.close()
    except Exception:
        pass
    print(f"Wrote {total_written} deduplicated examples across {part_idx} file(s), last part size ~{cur_bytes/1024/1024:.1f} MB. First part: {next_part_path(out_path, 1)}")


if __name__ == "__main__":
    main()
