#!/usr/bin/env python3
"""
Convert 3D dataset JSON -> instruction-style JSONL
Writes either a single-canonical completion per observation (first ground truth) and optionally all compatible completions.

Usage:
  python dataset_to_jsonl_3d.py input.json output_single.jsonl [--output-all output_all.jsonl]

The script writes JSONL lines of the form {"prompt":..., "completion":...}
"""
import json
import sys
import os
from pathlib import Path


def format_top_view(obs_str, grid_size=3):
    # obs_str like '100000000' length grid_size^2
    rows = []
    for r in range(grid_size):
        row = obs_str[r*grid_size:(r+1)*grid_size]
        rows.append(' '.join(list(row)))
    return '\n'.join(rows)


def format_structure(layers, grid_size=3):
    parts = []
    for i, layer in enumerate(layers, start=1):
        rows = []
        for r in range(grid_size):
            row = layer[r*grid_size:(r+1)*grid_size]
            rows.append(' '.join(list(row)))
        parts.append(f"Layer {i}:\n" + '\n'.join(rows))
    return 'Structure:\n' + '\n\n'.join(parts)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python dataset_to_jsonl_3d.py input.json output_single.jsonl [--output-all output_all.jsonl]")
        sys.exit(2)

    input_path = Path(sys.argv[1])
    out_single = Path(sys.argv[2])
    out_all = None
    if '--output-all' in sys.argv:
        idx = sys.argv.index('--output-all')
        if idx + 1 < len(sys.argv):
            out_all = Path(sys.argv[idx+1])

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(2)

    out_single.parent.mkdir(parents=True, exist_ok=True)
    if out_all:
        out_all.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    grid_size = metadata.get('grid_size', 3)

    obs_sets = data.get('observation_sets', [])

    count_single = 0
    count_all = 0

    with out_single.open('w', encoding='utf-8') as sf:
        if out_all:
            with out_all.open('w', encoding='utf-8') as af:
                for obs in obs_sets:
                    observation = obs.get('observation')
                    if observation is None:
                        continue

                    top_view = format_top_view(observation, grid_size=grid_size)
                    prompt = (
                        f"You are given the TOP view of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.\n"
                        f"Top view:\n{top_view}\n"
                        "Rules: Each '1' indicates a column that may contain one or more unit blocks stacked vertically. Return a valid 3D structure as layers from bottom (Layer 1) upward, using 0/1 in a grid per layer.\n"
                    )

                    gts = obs.get('ground_truth_structures', [])
                    if not gts:
                        continue

                    first = gts[0]
                    # dataset layers are stored top->bottom; present them bottom->up so Layer 1 is bottom
                    layers = first.get('layers', [])
                    completion = format_structure(list(reversed(layers)), grid_size=grid_size) + '\n'
                    obj = {"prompt": prompt, "completion": completion}
                    sf.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    count_single += 1

                    for gt in gts:
                        layers2 = gt.get('layers', [])
                        comp2 = format_structure(list(reversed(layers2)), grid_size=grid_size) + '\n'
                        # note: reversed so bottom is Layer 1
                        obj2 = {"prompt": prompt, "completion": comp2}
                        af.write(json.dumps(obj2, ensure_ascii=False) + '\n')
                        count_all += 1
        else:
            for obs in obs_sets:
                observation = obs.get('observation')
                if observation is None:
                    continue

                top_view = format_top_view(observation, grid_size=grid_size)
                prompt = (
                    f"You are given the TOP view of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.\n"
                    f"Top view:\n{top_view}\n"
                    "Rules: Each '1' indicates a column that may contain one or more unit blocks stacked vertically. Return a valid 3D structure as layers from top (Layer 1) to bottom, using 0/1 in a grid per layer.\n"
                )

                gts = obs.get('ground_truth_structures', [])
                if not gts:
                    continue

                first = gts[0]
                layers = first.get('layers', [])
                completion = format_structure(list(reversed(layers)), grid_size=grid_size) + '\n'
                obj = {"prompt": prompt, "completion": completion}
                sf.write(json.dumps(obj, ensure_ascii=False) + '\n')
                count_single += 1

    print(f"Wrote {count_single} examples to {out_single}")
    if out_all:
        print(f"Wrote {count_all} examples to {out_all}")
