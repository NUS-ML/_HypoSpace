#!/usr/bin/env python3
"""
Build LoRA / SFT JSONL for the 3D reconstruction task.

Input: one or more 3D dataset JSON files produced by `3d/generate_3d_dataset_complete.py`.
Output: JSONL lines in messages/conversations format, e.g.: 
  {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}

Modes:
  - all-gts (one line per ground-truth structure)
  - first-gt
  - random-gt

Options: include priors (other compatible structures from same observation set), dedup, limit per set, seed.

Example:
  python 3d/tools/build_lora_jsonl.py --inputs 3d/datasets/3d_complete.json --output data/lora_3d.jsonl --mode all-gts
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_observation_sets(dataset_json: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield (meta, obs_set) pairs from a 3d dataset JSON object."""
    meta = dataset_json.get("metadata", {})
    if "observation_sets" in dataset_json:
        for ds in dataset_json["observation_sets"]:
            yield meta, ds
    else:
        # fallback: if file is list
        if isinstance(dataset_json, list):
            for ds in dataset_json:
                yield meta, ds


def _parse_single_observation(data: Any, grid_size: int) -> List[List[int]]:
    """Convert observation data into a matrix of ints."""
    if isinstance(data, str):
        length = len(data)
        inferred = int(length ** 0.5)
        size = grid_size or inferred
        matrix = []
        for r in range(size):
            row = [int(data[r * size + c]) for c in range(size)]
            matrix.append(row)
        return matrix
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            return [[int(cell) for cell in row] for row in data]
        return [[int(cell) for cell in row] for row in data]
    if isinstance(data, dict):
        return _parse_single_observation(data.get("observation", []), grid_size)
    return []


def _format_matrix(matrix: List[List[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in matrix)


def _gather_observation_matrices(observations: Any, grid_size: int) -> List[List[List[int]]]:
    if observations is None:
        return []
    matrices: List[List[List[int]]] = []
    if isinstance(observations, list) and observations and isinstance(observations[0], dict):
        for obs in observations:
            matrix = _parse_single_observation(obs.get("observation", []), grid_size)
            if matrix:
                matrices.append(matrix)
    else:
        matrix = _parse_single_observation(observations, grid_size)
        if matrix:
            matrices.append(matrix)
    return matrices


def _format_structure_for_assistant(layers: List[str], grid_size: int = 3) -> str:
    # layers here are expected bottom->top order (Layer 1 bottom)
    parts = []
    for i, layer in enumerate(layers, start=1):
        rows = []
        for r in range(grid_size):
            row = layer[r * grid_size:(r + 1) * grid_size]
            rows.append(" ".join(list(row)))
        parts.append(f"Layer {i}:\n" + "\n".join(rows))
    return "Structure:\n" + "\n\n".join(parts)


def _format_prior_structure(layers_bottom_up: List[str], grid_size: int) -> str:
    lines: List[str] = []
    for idx, layer in enumerate(layers_bottom_up, start=1):
        lines.append(f"Layer {idx}:")
        for r in range(grid_size):
            row = layer[r * grid_size:(r + 1) * grid_size]
            lines.append(" ".join(list(row)))
    return "\n".join(lines)


def _compose_user_prompt(
    grid_size: int,
    max_height: int,
    observation_views: List[str],
    prior_structures: List[str],
) -> str:
    prompt = f"""You are given observations of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.
        Each observation shows a view of the structure from a specific angle.
        The maximum height of the structure is {max_height} layers.

        Observations (Top View - shows 1 if ANY layer has a block at that position):
        """
    for view in observation_views:
        prompt += "\nTop view:\n"
        prompt += view
        prompt += "\n"

    if prior_structures:
        prompt += "\n\nPrior 3D structure generated (do not repeat if avoidable):\n"
        for idx, struct in enumerate(prior_structures, 1):
            prompt += f"\nAttempt {idx}:\n{struct}\n"

    prompt += f"""
        
        Task: Infer the complete 3D structure that could produce these observations.

        Structure specifications:
        - Grid size: {grid_size}x{grid_size}
        - Maximum height: {max_height} layers
        - The structure consists of layers stacked from bottom to top, where each layer is a {grid_size}x{grid_size} grid with 0 (empty) or 1 (block).
        
        Important constraints:
        1. Layer 1 is the BOTTOM layer (ground level) - it should contain blocks, not be all zeros
        2. Blocks must be supported from below (a block at height h requires a block at height h-1 in the same position)
        3. Do not add unnecessary empty layers at the bottom or top
        4. Layers are numbered from bottom (Layer 1) to top (Layer N)
        5. Do not exceed the maximum height of {max_height} layers
        """

    if prior_structures:
        prompt += f"""
        6. Generate a DIFFERENT valid structure from the {len(prior_structures)} prior attempts shown above
        7. Two structures are considered equivalent if they have the same blocks in the same positions across all layers
        """

    prompt += f"""

        Provide your answer as a 3D structure with each layer specified.

        Output format:
        Structure:
        Layer 1:  (bottom layer - should contain at least one block)
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        Layer 2:
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        (continue for all layers needed, up to maximum {max_height} layers)
        
        Example of CORRECT output format for a 3x3 grid:
        Structure:
        Layer 1:
        1 0 1
        0 0 0
        1 0 1
        Layer 2:
        1 0 0
        0 0 0
        0 0 1
        
        Note: This represents a 3x3x2 structure where:
        - Layer 1 (bottom) has blocks at corners: (0,0), (0,2), (2,0), (2,2)
        - Layer 2 (top) has blocks at (0,0) and (2,2), both supported by blocks in Layer 1 below
        - Every '1' in Layer 2 has a '1' directly below it in Layer 1 (physical support requirement)
        - Uses spaces between digits (not commas or other separators)
        
        INCORRECT formats to avoid:
        1. Don't start with empty bottom layer: Layer 1 should contain blocks
        2. Don't use commas: "1,0,1" - use spaces instead: "1 0 1"
        3. Don't add blocks without support below them (physics violation)
        4. Don't exceed {max_height} layers in height
    """

    return prompt


def build_examples_from_file(
    path: Path,
    mode: str = "all-gts",
    include_priors: int = 0,
    seed: Optional[int] = None,
    limit_per_set: Optional[int] = None,
    dedup: bool = False,
    grid_size_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[Dict[str, Any]] = []
    seen = set()

    for meta, obs in _iter_observation_sets(data):
        grid_size = grid_size_override or meta.get("grid_size") or 3
        observation_data = obs.get("observation", obs.get("observations"))
        if observation_data is None:
            continue
        observation_matrices = _gather_observation_matrices(observation_data, grid_size)
        if not observation_matrices:
            continue
        observation_views = [_format_matrix(mat) for mat in observation_matrices]
        max_height = obs.get("max_height", meta.get("max_height", 3))
        gts = obs.get("ground_truth_structures", [])
        if not gts:
            # create a prompt with empty assistant if requested
            if mode == "skip-no-gt":
                continue
            user = _compose_user_prompt(grid_size, max_height, observation_views, [])
            assistant = "Structure:"
            item = {"messages": [
                {"role": "system", "content": "You are an expert at 3D reconstruction from top views."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]}
            key = (user.strip(), assistant.strip())
            if (not dedup) or (key not in seen):
                examples.append(item)
                seen.add(key)
            continue

        indices: List[int] = []
        if mode == "all-gts":
            indices = list(range(len(gts)))
        elif mode == "first-gt":
            indices = [0]
        elif mode == "random-gt":
            indices = [random.randrange(len(gts))]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if limit_per_set is not None:
            indices = indices[:limit_per_set]

        for idx in indices:
            gt = gts[idx]
            # gt['layers'] in dataset are top->bottom; we want bottom->up for Layer 1
            layers_top_to_bottom = gt.get("layers", [])
            layers_bottom_up = list(reversed(layers_top_to_bottom))

            # build priors: sample other GTs' layers (bottom-up) up to include_priors
            priors = []
            if include_priors > 0 and len(gts) > 1:
                other = [list(reversed(g.get("layers", []))) for j, g in enumerate(gts) if j != idx]
                random.shuffle(other)
                priors = other[:include_priors]

            prior_texts = [_format_prior_structure(p, grid_size=grid_size) for p in priors]
            height_for_prompt = max_height
            if layers_bottom_up:
                height_for_prompt = max(height_for_prompt, len(layers_bottom_up))
            user = _compose_user_prompt(grid_size, height_for_prompt, observation_views, prior_texts)
            assistant = _format_structure_for_assistant(layers_bottom_up, grid_size=grid_size)

            item = {"messages": [
                {"role": "system", "content": "You are an expert at 3D reconstruction from top views."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]}

            key = (user.strip(), assistant.strip())
            if (not dedup) or (key not in seen):
                examples.append(item)
                seen.add(key)

    return examples


def main():
    ap = argparse.ArgumentParser(description="Build LoRA JSONL for 3D reconstruction task")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input dataset JSON files or directories")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--mode", default="all-gts", choices=["all-gts", "first-gt", "random-gt"]) 
    ap.add_argument("--include-priors", type=int, default=0, help="How many prior structures to include (sampled from other GTs)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit-per-set", type=int, default=None)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--grid-size", type=int, default=None, help="Override grid size when formatting views")

    args = ap.parse_args()

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
                grid_size_override=args.grid_size,
            )
            all_examples.extend(ex)
            print(f"[OK] {fp} -> {len(ex)} examples")
        except Exception as e:
            print(f"[WARN] Skipped {fp}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as w:
        for ex in all_examples:
            w.write(json.dumps(ex, ensure_ascii=False))
            w.write("\n")

    print(f"Wrote {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
