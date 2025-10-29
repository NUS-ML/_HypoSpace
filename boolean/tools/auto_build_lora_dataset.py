#!/usr/bin/env python3
"""Stream-build Boolean LoRA dataset with optional multi-seed deduplication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .build_lora_jsonl import build_examples_from_file


def _dedup_key(user_content: str, assistant_content: str) -> str:
    digest = hashlib.sha256()
    digest.update(user_content.strip().encode("utf-8"))
    digest.update(b"\n\n")
    digest.update(assistant_content.strip().encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-build Boolean LoRA dataset with streaming and dedup")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset JSON files or directories")
    parser.add_argument("--output", required=True, help="Output JSONL base path (will split if large)")
    parser.add_argument("--seeds", default="42", help="Comma-separated list of seeds")
    parser.add_argument("--mode", default="all-gts", choices=["all-gts", "first-gt", "random-gt"], help="Which ground truths to include")
    parser.add_argument("--include-priors", type=int, default=0, help="How many prior expressions to include")
    parser.add_argument("--limit-per-set", type=int, default=None, help="Limit examples per observation set")
    parser.add_argument("--dedup", action="store_true", help="Enable deduplication across seeds/files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for JSON files")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    input_files: List[Path] = []
    for entry in args.inputs:
        path = Path(entry)
        if path.is_dir():
            pattern = "**/*.json" if args.recursive else "*.json"
            input_files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            input_files.append(path)
        else:
            input_files.extend(sorted(Path().glob(entry)))

    if not input_files:
        raise SystemExit("No input JSON files found")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = 200 * 1024 * 1024

    def _next_part(base: Path, idx: int) -> Path:
        stem, suffix = base.stem, base.suffix
        return base.with_name(f"{stem}_part{idx:03d}{suffix}")

    part_idx = 1
    current_path = _next_part(out_path, part_idx)
    current_file = current_path.open("wb")
    current_bytes = 0
    total_examples = 0

    dedup_seen = set()

    try:
        for seed in seeds:
            for fp in input_files:
                try:
                    examples = build_examples_from_file(
                        fp,
                        mode=args.mode,
                        include_priors=args.include_priors,
                        seed=seed,
                        limit_per_set=args.limit_per_set,
                        dedup=args.dedup,
                    )
                except Exception as exc:
                    print(f"[WARN] Skipped {fp}: {exc}")
                    continue

                for ex in examples:
                    messages = ex.get("messages") or ex.get("conversations") or []
                    user = ""
                    assistant = ""
                    for message in messages:
                        role = message.get("role")
                        if role == "user":
                            user = message.get("content", "")
                        elif role == "assistant":
                            assistant = message.get("content", "")
                    key = _dedup_key(user, assistant)
                    if args.dedup and key in dedup_seen:
                        continue
                    dedup_seen.add(key)

                    line = json.dumps(ex, ensure_ascii=False)
                    payload = line.encode("utf-8") + b"\n"
                    if current_bytes > 0 and current_bytes + len(payload) > max_bytes:
                        current_file.close()
                        part_idx += 1
                        current_path = _next_part(out_path, part_idx)
                        current_file = current_path.open("wb")
                        current_bytes = 0
                    current_file.write(payload)
                    current_bytes += len(payload)
                    total_examples += 1
            print(f"[seed={seed}] streamed, total_examples={total_examples}")
    finally:
        try:
            current_file.close()
        except Exception:
            pass

    print(f"Wrote {total_examples} examples across {part_idx} file(s). First part: {_next_part(out_path, 1)}")


if __name__ == "__main__":
    main()
