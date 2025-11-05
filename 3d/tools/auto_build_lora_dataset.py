#!/usr/bin/env python3
"""
Auto-build LoRA dataset for 3D by scanning inputs (or generating if you add a generator)
and streaming deduplicated examples to disk. This reuses `build_lora_jsonl.build_examples_from_file`.

Example:
  python 3d/tools/auto_build_lora_dataset.py --inputs 3d/datasets/3d_complete.json --output data/lora_3d.jsonl --seeds 42,43
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from itertools import combinations
import os

from build_lora_jsonl import build_examples_from_file


def _dedup_key(user_content: str, assistant_content: str) -> str:
    h = hashlib.sha256()
    h.update(user_content.strip().encode('utf-8'))
    h.update(b"\n\n")
    h.update(assistant_content.strip().encode('utf-8'))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Auto-build LoRA dataset for 3D with dedup and streaming")
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seeds', default='42', help='Comma-separated seeds')
    ap.add_argument('--mode', default='all-gts', choices=['all-gts','first-gt','random-gt'])
    ap.add_argument('--include-priors', type=int, default=0)
    ap.add_argument('--limit-per-set', type=int, default=None)
    ap.add_argument('--dedup', action='store_true')
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--grid-size', type=int, default=None)

    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    input_files: List[Path] = []
    for p in args.inputs:
        path = Path(p)
        if path.is_dir():
            pattern = '**/*.json' if args.recursive else '*.json'
            input_files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            input_files.append(path)
        else:
            input_files.extend(sorted(Path().glob(p)))

    if not input_files:
        raise SystemExit('No input files found')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    MAX_BYTES = 200 * 1024 * 1024
    def next_part_path(base: Path, part_idx: int) -> Path:
        stem, suffix = base.stem, base.suffix
        return base.with_name(f"{stem}_part{part_idx:03d}{suffix}")

    part_idx = 1
    cur_path = next_part_path(out_path, part_idx)
    cur_f = cur_path.open('wb')
    cur_bytes = 0
    total_written = 0

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
                        grid_size_override=args.grid_size,
                    )
                except Exception as e:
                    print(f"[WARN] Skipped {fp}: {e}")
                    continue

                for ex in examples:
                    # compute dedup key
                    # extract user+assistant
                    msgs = ex.get('messages') or ex.get('conversations') or []
                    user = ''
                    assistant = ''
                    for m in msgs:
                        if m.get('role') == 'user':
                            user = m.get('content','')
                        elif m.get('role') == 'assistant':
                            assistant = m.get('content','')
                    key = _dedup_key(user, assistant)
                    if args.dedup and key in dedup_seen:
                        continue
                    dedup_seen.add(key)

                    line = json.dumps(ex, ensure_ascii=False)
                    b = line.encode('utf-8')
                    nb = len(b) + 1
                    if cur_bytes > 0 and cur_bytes + nb > MAX_BYTES:
                        cur_f.close()
                        part_idx += 1
                        cur_path = next_part_path(out_path, part_idx)
                        cur_f = cur_path.open('wb')
                        cur_bytes = 0
                    cur_f.write(b)
                    cur_f.write(b"\n")
                    cur_bytes += nb
                    total_written += 1
            print(f"[seed={seed}] streamed, total_written={total_written}")
    finally:
        try:
            cur_f.close()
        except Exception:
            pass

    print(f"Wrote {total_written} examples across {part_idx} file(s). First part: {next_part_path(out_path,1)}")


if __name__ == '__main__':
    main()
