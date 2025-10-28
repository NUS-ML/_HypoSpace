#!/usr/bin/env python3
"""
Rename top-level key `messages` to `conversations` for each JSON line in a JSONL file.
- Preserves key order (Python 3.7+ dicts are ordered).
- Streams line-by-line to handle large files.
- Defaults to writing to a new output file to avoid destructive edits.

Usage examples:
  python -m causal.tools.replace_messages_with_conversations \
      --input data/sample_lora_nodes2-5_aliyun_part001.jsonl \
      --output data/sample_lora_nodes2-5_aliyun_part001_conversations.jsonl

  # In-place (creates optional backup if provided)
  python -m causal.tools.replace_messages_with_conversations \
      --input data/file.jsonl --in-place --backup .bak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple


def rename_key_preserve_order(d: Dict[str, Any], old_key: str, new_key: str, overwrite: bool) -> Tuple[Dict[str, Any], bool]:
    """Return a new dict where old_key is renamed to new_key, preserving order.
    Returns (new_dict, changed_flag).
    If overwrite is False and new_key exists, leaves dict unchanged for that key.
    If overwrite is True and both exist, keeps the value from old_key at the old position
    and drops the original new_key occurrence.
    """
    if not isinstance(d, dict):
        return d, False

    if old_key not in d:
        return d, False

    new_d: Dict[str, Any] = {}
    changed = False

    for k, v in d.items():
        if k == old_key:
            # We are renaming here
            if (new_key in d) and not overwrite:
                # Do not rename if target exists and no overwrite requested
                new_d[k] = v
                # changed remains False for this key
            else:
                new_d[new_key] = v
                changed = True
        elif overwrite and (k == new_key) and (old_key in d):
            # We'll skip this occurrence because we will insert new_key at old_key position
            # using the value from old_key (done above when k==old_key)
            continue
        else:
            new_d[k] = v

    return new_d, changed


def process_file(inp: str, out: str, in_place: bool, backup_ext: str | None, overwrite: bool) -> None:
    total = 0
    changed = 0

    # Prepare output path
    if in_place:
        if not out:
            out = inp + ".tmp_rewrite"
    else:
        if not out:
            root, ext = os.path.splitext(inp)
            out = f"{root}_conversations{ext or '.jsonl'}"

    # Stream read/write
    with open(inp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Write original line if it's not valid JSON
                fout.write(line)
                continue

            if isinstance(obj, dict):
                new_obj, did_change = rename_key_preserve_order(
                    obj, "messages", "conversations", overwrite=overwrite
                )
                if did_change:
                    changed += 1
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            else:
                # Not a dict at top level; write as-is
                fout.write(line)

    # If in-place, atomically replace original, optionally create backup
    if in_place:
        if backup_ext:
            backup_path = inp + backup_ext
            # Remove existing backup if any to avoid errors
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            except Exception:
                pass
            os.replace(inp, backup_path)
        else:
            # No backup requested; just remove original
            os.remove(inp)
        os.replace(out, inp)
        final_path = inp
    else:
        final_path = out

    print(
        json.dumps(
            {
                "input": inp,
                "output": final_path,
                "lines_total": total,
                "lines_changed": changed,
            },
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rename top-level 'messages' to 'conversations' in JSONL.")
    p.add_argument("--input", required=True, help="Input JSONL file path")
    p.add_argument("--output", default="", help="Output JSONL file path (default: <input>_conversations.jsonl)")
    p.add_argument("--in-place", action="store_true", help="Rewrite input file in place (creates temp and swap)")
    p.add_argument("--backup", default="", help="When using --in-place, optional backup extension, e.g. .bak")
    p.add_argument("--overwrite", action="store_true", help="If target 'conversations' already exists, overwrite it with 'messages'")

    args = p.parse_args(argv)

    inp = args.input
    out = args.output
    in_place = args.in_place
    backup_ext = args.backup or None

    if not os.path.isfile(inp):
        print(f"Input file not found: {inp}", file=sys.stderr)
        return 2

    if in_place and out:
        print("--output is ignored when --in-place is used", file=sys.stderr)

    try:
        process_file(inp, out, in_place, backup_ext, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
