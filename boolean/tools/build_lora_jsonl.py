#!/usr/bin/env python3
"""Build LoRA / SFT JSONL for the Boolean function reconstruction task."""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_OPERATOR_DESCRIPTIONS = {
    "AND": "AND (conjunction)",
    "OR": "OR (disjunction)",
    "NOT": "NOT (negation)",
    "XOR": "XOR (exclusive or)",
    "NOR": "NOR (NOT (x OR y))",
    "NAND": "NAND (NOT (x AND y))",
    "IMPLIES": "IMPLIES (logical implication)",
    "EQUIV": "EQUIV (logical equivalence)",
}


def _iter_observation_sets(dataset_json: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield (meta, obs_set) pairs from Boolean dataset JSON."""
    meta = dataset_json.get("metadata", {})
    if "datasets" in dataset_json:
        for ds in dataset_json["datasets"]:
            yield meta, ds
    elif "datasets_by_n_observations" in dataset_json:
        for _n, sets in dataset_json["datasets_by_n_observations"].items():
            for ds in sets:
                yield meta, ds
    else:
        if isinstance(dataset_json, list):
            for ds in dataset_json:
                yield meta, ds


def _format_observations(observations: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for obs in observations:
        if isinstance(obs, dict):
            if "string" in obs:
                lines.append(obs["string"])
                continue
            inputs = obs.get("inputs", {})
            inputs_str = ", ".join(f"{k}={v}" for k, v in sorted(inputs.items()))
            lines.append(f"({inputs_str}) -> {obs.get('output', 0)}")
        else:
            lines.append(str(obs))
    return "\n".join(lines)


def _format_prior_block(priors: List[str]) -> str:
    if not priors:
        return "None"
    return "\n".join(f"Expression: {expr}" for expr in priors)


def _build_operator_string(operators: Iterable[str]) -> str:
    desc: List[str] = []
    for op in operators:
        label = _OPERATOR_DESCRIPTIONS.get(op)
        if label:
            desc.append(label)
        else:
            desc.append(op)
    return ", ".join(desc)


def _build_rules_text(mech_opts: Dict[str, Any]) -> str:
    comm = mech_opts.get("apply_commutativity", True)
    idem = mech_opts.get("apply_idempotence_and_or", True)
    flat = mech_opts.get("flatten_associativity", True)

    rules: List[str] = []
    if comm:
        rules.append("- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x).")
    if idem:
        rules.append("- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x).")
    if flat:
        rules.append("- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses (e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y).")
    else:
        rules.append("- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT (e.g., (x AND y) AND x ≠ x AND (y AND x)).")
    rules.append("- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT (e.g., (x AND y) OR z ≠ x AND (y OR z)).")

    return "\n            ".join(rules)


def _compose_user_prompt(
    variables: List[str],
    operators: List[str],
    max_depth: int,
    observations_block: str,
    prior_block: str,
    mech_opts: Dict[str, Any],
) -> str:
    operators_str = _build_operator_string(operators)
    rules_text = _build_rules_text(mech_opts)
    prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {observations_block}

            Prior expressions generated (avoid repeating any expression 
            that is equivalent under the rules below):
            {prior_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(variables)}
            2. Use ONLY these operators: {', '.join(operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.

            Output format: 
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators. 
            - Use lowercase for variables: {', '.join(variables)}
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression
            
            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)
            
            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
    return textwrap.dedent(prompt).strip()


def _choose_indices(mode: str, count: int) -> List[int]:
    if count == 0:
        return []
    if mode == "all-gts":
        return list(range(count))
    if mode == "first-gt":
        return [0]
    if mode == "random-gt":
        return [random.randrange(count)]
    raise ValueError(f"Unknown mode: {mode}")


def build_examples_from_file(
    path: Path,
    mode: str = "all-gts",
    include_priors: int = 0,
    seed: Optional[int] = None,
    limit_per_set: Optional[int] = None,
    dedup: bool = False,
) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for meta, obs_set in _iter_observation_sets(data):
        observations = obs_set.get("observations", [])
        gts = obs_set.get("ground_truth_expressions", [])
        if not observations:
            continue

        variables = obs_set.get("variables") or meta.get("variables") or []
        operators = obs_set.get("operators") or meta.get("operators") or []
        max_depth = obs_set.get("max_depth") or meta.get("max_depth") or 2
        mech_opts = meta.get("mechanistic_opts", {})

        obs_block = _format_observations(observations)

        indices = _choose_indices(mode, len(gts)) if gts else []
        if limit_per_set is not None and indices:
            indices = indices[:limit_per_set]

        if not gts:
            prior_block = "None"
            user = _compose_user_prompt(variables, operators, max_depth, obs_block, prior_block, mech_opts)
            assistant = "Expression: "
            key = (user, assistant)
            if (not dedup) or (key not in seen):
                examples.append({
                    "messages": [
                        {"role": "system", "content": "You are an expert in Boolean logic."},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                })
                seen.add(key)
            continue

        for idx in indices:
            target = gts[idx]
            formula = target.get("formula")
            if not formula:
                continue

            priors: List[str] = []
            if include_priors > 0 and len(gts) > 1:
                candidates = [g.get("formula") for j, g in enumerate(gts) if j != idx and g.get("formula")]
                random.shuffle(candidates)
                priors = candidates[:include_priors]

            prior_block = _format_prior_block(priors)
            user = _compose_user_prompt(variables, operators, max_depth, obs_block, prior_block, mech_opts)
            assistant = f"Expression: {formula}"

            key = (user, assistant)
            if (not dedup) or (key not in seen):
                examples.append({
                    "messages": [
                        {"role": "system", "content": "You are an expert in Boolean logic."},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                })
                seen.add(key)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LoRA JSONL for Boolean reconstruction task")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset JSON files or directories")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--mode", default="all-gts", choices=["all-gts", "first-gt", "random-gt"], help="Which ground truths to include")
    parser.add_argument("--include-priors", type=int, default=0, help="How many prior expressions to include")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--limit-per-set", type=int, default=None, help="Limit examples per observation set")
    parser.add_argument("--dedup", action="store_true", help="De-duplicate identical examples")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for JSON files")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_files: List[Path] = []
    for entry in args.inputs:
        p = Path(entry)
        if p.is_dir():
            pattern = "**/*.json" if args.recursive else "*.json"
            input_files.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            input_files.append(p)
        else:
            input_files.extend(sorted(Path().glob(entry)))

    if not input_files:
        raise SystemExit("No input JSON files found")

    all_examples: List[Dict[str, Any]] = []
    for fp in input_files:
        try:
            examples = build_examples_from_file(
                fp,
                mode=args.mode,
                include_priors=args.include_priors,
                seed=args.seed,
                limit_per_set=args.limit_per_set,
                dedup=args.dedup,
            )
            all_examples.extend(examples)
            print(f"[OK] {fp} -> {len(examples)} examples")
        except Exception as exc:
            print(f"[WARN] Skipped {fp}: {exc}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for example in all_examples:
            fh.write(json.dumps(example, ensure_ascii=False))
            fh.write("\n")

    print(f"Wrote {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
