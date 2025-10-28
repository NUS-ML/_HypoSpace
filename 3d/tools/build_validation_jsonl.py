#!/usr/bin/env python3
"""
Build a small validation JSONL by randomly sampling observation sets from 3d datasets.

Defaults are deterministic with --seed. The script will prefer existing JSON files passed as inputs.

Example:
  python 3d/tools/build_validation_jsonl.py --inputs 3d/datasets/3d_complete.json --per-file 8 --output data/val_3d.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iter_observation_sets(coll: Dict[str, Any]):
    meta = coll.get("metadata", {})
    if "observation_sets" in coll:
        for ds in coll["observation_sets"]:
            yield meta, ds
    elif isinstance(coll, list):
        for ds in coll:
            yield meta, ds


def _sample_obs_sets(data: Dict[str, Any], k: int, seed: Optional[int]):
    all_sets = [ds for _m, ds in _iter_observation_sets(data)]
    if not all_sets:
        return []
    rng = random.Random(seed)
    if k >= len(all_sets):
        return list(all_sets)
    return rng.sample(all_sets, k)


def main():
    ap = argparse.ArgumentParser(description="Build 3D validation JSONL by sampling observation sets")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSON files or directories")
    ap.add_argument("--per-file", type=int, default=8, help="Samples per input file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid-size", type=int, default=None, help="If supplied, force grid size when formatting prompts")

    args = ap.parse_args()
    random.seed(args.seed)

    input_files: List[Path] = []
    for p in args.inputs:
        path = Path(p)
        if path.is_dir():
            input_files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            input_files.append(path)

    if not input_files:
        raise SystemExit("No input files found")

    examples: List[Dict[str, Any]] = []

    for fp in input_files:
        try:
            with fp.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        sampled = _sample_obs_sets(data, args.per_file, seed=args.seed)
        for obs in sampled:
            observation = obs.get('observation')
            if not observation:
                continue
            grid_size = args.grid_size or data.get('metadata', {}).get('grid_size', 3)
            # reuse prompt template similar to training
            rows = []
            for r in range(grid_size):
                row = observation[r * grid_size:(r + 1) * grid_size]
                rows.append(' '.join(list(row)))
            top_view = '\n'.join(rows)
            prompt = (
                f"You are given the TOP view of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.\n"
                f"Top view:\n{top_view}\n"
                "Rules: Each '1' indicates a column that may contain one or more unit blocks stacked vertically. "
                "Return a valid 3D structure as layers from bottom (Layer 1) upward, using 0/1 in a grid per layer.\n"
            )
            # choose a canonical assistant answer (first GT if exists)
            gts = obs.get('ground_truth_structures', [])
            if not gts:
                assistant = "Structure:"
            else:
                layers = list(reversed(gts[0].get('layers', [])))
                # format assistant
                parts = []
                for i, layer in enumerate(layers, start=1):
                    rlines = []
                    for rr in range(grid_size):
                        row = layer[rr * grid_size:(rr + 1) * grid_size]
                        rlines.append(' '.join(list(row)))
                    parts.append(f"Layer {i}:\n" + '\n'.join(rlines))
                assistant = "Structure:\n" + "\n\n".join(parts)

            examples.append({
                "messages": [
                    {"role": "system", "content": "You are an expert at 3D reconstruction from top views."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant},
                ]
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as w:
        for ex in examples:
            w.write(json.dumps(ex, ensure_ascii=False))
            w.write('\n')

    print(f"Validation JSONL written: {out_path}  (examples: {len(examples)})")


if __name__ == '__main__':
    main()
