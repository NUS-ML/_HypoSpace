#!/usr/bin/env python3
"""
Build LoRA fine-tuning JSONL for the causal DAG task.

Given one or more dataset JSON files produced by the generators in this repo,
emit a JSONL file with entries in the form:

{"conversations": [
  {"role": "user", "content": "...filled prompt..."},
  {"role": "assistant", "content": "Graph: ..."}
]}

Key features:
- Supports datasets structured as {"datasets": [...]} or {"datasets_by_n_observations": {k: [...]}}.
- Fills placeholders: nodes_str, constraint_info, obs_block, prior_block.
- Selects target graph from ground_truth_graphs by mode:
  * all-gts: one training line per GT (maximizes coverage)
  * first-gt: first GT only
  * random-gt: random GT (seeded)
- Optional prior predictions block (sampled from other GTs of the same observation set).
- Optional de-duplication.

Example:
  python causal/tools/build_lora_jsonl.py \
    --inputs causal/datasets/node03/n3_all_observations.json \
    --output data/lora_causal_n3.jsonl \
    --mode all-gts --include-priors 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Any, Optional
import os


def _iter_observation_sets(dataset_json: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Yield (meta, obs_set) pairs from a dataset JSON object.

    Supports two shapes:
    - {"metadata": {...}, "datasets": [ ... ]}
    - {"metadata": {...}, "datasets_by_n_observations": {n: [ ... ]}}
    """
    meta = dataset_json.get("metadata", {})
    if "datasets" in dataset_json:
        for ds in dataset_json["datasets"]:
            yield meta, ds
    elif "datasets_by_n_observations" in dataset_json:
        for _n, lst in dataset_json["datasets_by_n_observations"].items():
            for ds in lst:
                yield meta, ds
    else:
        # Fallback: try treat the file itself as a single sample list
        if isinstance(dataset_json, list):
            for ds in dataset_json:
                yield meta, ds


def _edges_to_graph_line(edges: List[List[str]]) -> str:
    """Convert [[u,v], ...] into canonical "Graph: A->B, C->D" or "Graph: No edges"."""
    if not edges:
        return "Graph: No edges"
    # Keep order as-is to augment natural variation; optionally sort for determinism
    parts = [f"{u}->{v}" for u, v in edges]
    return "Graph: " + ", ".join(parts)


def _build_prior_block(priors: List[List[List[str]]]) -> str:
    if not priors:
        return "None"
    lines = []
    for e in priors:
        lines.append(_edges_to_graph_line(e))
    return "\n".join(lines)


def _infer_max_edges(meta: Dict[str, Any], obs_set: Dict[str, Any]) -> Optional[int]:
    me = obs_set.get("max_edges")
    if isinstance(me, int):
        return me
    me = meta.get("max_edges")
    if isinstance(me, int):
        return me
    # Infer from GTs if present
    gts = obs_set.get("ground_truth_graphs") or []
    if gts:
        return max(len(g.get("edges", [])) for g in gts)
    return None


def _compose_user_prompt(nodes: List[str], constraint_info: str, observations: List[Dict[str, Any]], prior_block: str) -> str:
    nodes_str = ", ".join(nodes)
    obs_block = "\n".join(o.get("string", "") for o in observations)
    # Always include Prior predictions section; use 'None' when empty
    prior_display = prior_block.strip() if (prior_block and prior_block.strip()) else "None"
    prior_section = f"\n        Prior predictions (do not repeat if avoidable):\n        {prior_display}"
    content = f"""
        You are given observations from perturbation experiments on a causal system.
        
        Semantics:
        - When a node is perturbed, the perturbed node is 0.
        - A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
        - All other nodes are 0.
        Nodes: {nodes_str}{constraint_info}
        Observations:
        {obs_block}{prior_section}
        Task:
        Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.
        
        Diversity rule:
        - A "diverse" graph is any valid graph whose edge set is NOT identical to any prior prediction.
        - Generate diverse graphs when possible to explore the solution space.
        
        Formatting rules:
        1) Use only the listed nodes. No self-loops. No cycles.
        2) Respond with exactly one line:
        - If there are edges: Graph: A->B, B->C
        - If there are no edges: Graph: No edges"""
    return content.strip()


def build_examples_from_file(
    path: Path,
    mode: str = "all-gts",
    include_priors: int = 0,
    seed: Optional[int] = None,
    limit_per_set: Optional[int] = None,
    dedup: bool = False,
) -> List[Dict[str, Any]]:
    """Create conversation examples from a single dataset JSON file."""
    if seed is not None:
        random.seed(seed)

    with path.open("r") as f:
        data = json.load(f)

    examples: List[Dict[str, Any]] = []
    seen_keys = set()

    for meta, obs_set in _iter_observation_sets(data):
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
            # No GTs; create an example with empty assistant (optional)
            if mode == "skip-no-gt":
                continue
            target_edges_list = [None]
        else:
            if mode == "all-gts":
                selected_indices = list(range(len(gts)))
            elif mode == "first-gt":
                selected_indices = [0]
            elif mode == "random-gt":
                selected_indices = [random.randrange(len(gts))]
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Limit per set if requested
        if limit_per_set is not None and gts:
            selected_indices = selected_indices[: max(0, limit_per_set)]

        # Build examples
        if not gts:
            # assistant empty (rare; prefer datasets with GTs)
            prior_block = "None"
            user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block)
            assistant_content = "Graph: "
            key = (user_content, assistant_content)
            if (not dedup) or (key not in seen_keys):
                examples.append({
                    "conversations": [
                        {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                })
                seen_keys.add(key)
        else:
            for idx in selected_indices:
                target = gts[idx]
                target_edges = target.get("edges", [])

                # Build priors from other GTs (optional)
                priors_edges: List[List[List[str]]] = []
                if include_priors > 0 and len(gts) > 1:
                    # candidates: all other GTs
                    cand = [g["edges"] for j, g in enumerate(gts) if j != idx]
                    random.shuffle(cand)
                    priors_edges = cand[: include_priors]

                prior_block = _build_prior_block(priors_edges)
                user_content = _compose_user_prompt(nodes, constraint_info, observations, prior_block)
                assistant_content = _edges_to_graph_line(target_edges)

                key = (user_content, assistant_content)
                if (not dedup) or (key not in seen_keys):
                    examples.append({
                        "messages": [
                            {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ]
                    })
                    seen_keys.add(key)

    return examples


def main():
    ap = argparse.ArgumentParser(description="Build LoRA JSONL for causal DAG task")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Input dataset JSON files or directories (will scan *.json)")
    ap.add_argument("--output", required=True, help="Output JSONL file path")
    ap.add_argument("--mode", default="all-gts", choices=["all-gts", "first-gt", "random-gt"],
                    help="Which ground truth(s) to use as assistant outputs")
    ap.add_argument("--include-priors", type=int, default=0,
                    help="How many prior predictions to include (sampled from other GTs of the same set)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--limit-per-set", type=int, default=None, help="Limit examples per observation set")
    ap.add_argument("--dedup", action="store_true", help="De-duplicate identical (prompt, answer) pairs")
    ap.add_argument("--recursive", action="store_true", help="Scan directories recursively for *.json")

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Collect input files
    input_files: List[Path] = []
    for p in args.inputs:
        path = Path(p)
        if path.is_dir():
            pattern = "**/*.json" if args.recursive else "*.json"
            input_files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            input_files.append(path)
        else:
            # Allow simple globbing
            input_files.extend(sorted(Path().glob(p)))

    if not input_files:
        raise SystemExit("No input JSON files found")

    all_examples: List[Dict[str, Any]] = []
    for fp in input_files:
        try:
            ex = build_examples_from_file(
                fp,
                mode=args.mode,
                include_priors=args.include_priors,
                seed=args.seed,
                limit_per_set=args.limit_per_set,
                dedup=args.dedup,
            )
            all_examples.extend(ex)
            print(f"[OK] {fp} -> {len(ex)} examples")
        except Exception as e:
            print(f"[WARN] Skipped {fp}: {e}")

    # Ensure output filename includes 'aliyun' marker
    out_path = Path(args.output)
    if "aliyun" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem + "_aliyun" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with 200MB splitting
    MAX_BYTES = 200 * 1024 * 1024

    def next_part_path(base: Path, part_idx: int) -> Path:
        stem, suffix = base.stem, base.suffix
        return base.with_name(f"{stem}_part{part_idx:03d}{suffix}")

    part_idx = 1
    cur_path = next_part_path(out_path, part_idx)
    cur_f = cur_path.open("wb")
    cur_bytes = 0
    written = 0

    try:
        for ex in all_examples:
            line = json.dumps(ex, ensure_ascii=False)
            b = line.encode("utf-8")
            nb = len(b) + 1  # include newline
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
            written += 1
    finally:
        try:
            cur_f.close()
        except Exception:
            pass

    print(f"Wrote {written} examples across {part_idx} file(s), last part size ~{cur_bytes/1024/1024:.1f} MB. First part: {next_part_path(out_path, 1)}")


if __name__ == "__main__":
    main()
