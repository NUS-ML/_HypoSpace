#!/usr/bin/env python3
"""
Rename top-level key `messages` to `conversations` for each JSON line in a JSONL file.
Copied/adapted from causal/tools implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple


def rename_key_preserve_order(d: Dict[str, Any], old_key: str, new_key: str, overwrite: bool) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(d, dict):
        return d, False
    if old_key not in d:
        return d, False
    new_d: Dict[str, Any] = {}
    changed = False
    for k, v in d.items():
        if k == old_key:
            if (new_key in d) and not overwrite:
                new_d[k] = v
            else:
                new_d[new_key] = v
                changed = True
        elif overwrite and (k == new_key) and (old_key in d):
            continue
        else:
            new_d[k] = v
    return new_d, changed


def process_file(inp: str, out: str, in_place: bool, backup_ext: str | None, overwrite: bool) -> None:
    total = 0
    changed = 0

    if in_place:
        if not out:
            out = inp + ".tmp_rewrite"
    else:
        if not out:
            root, ext = os.path.splitext(inp)
            out = f"{root}_conversations{ext or '.jsonl'}"

    with open(inp, 'r', encoding='utf-8') as fin, open(out, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue
            if isinstance(obj, dict):
                new_obj, did_change = rename_key_preserve_order(obj, 'messages', 'conversations', overwrite=overwrite)
                if did_change:
                    changed += 1
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            else:
                fout.write(line)

    if in_place:
        if backup_ext:
            backup_path = inp + backup_ext
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            except Exception:
                pass
            os.replace(inp, backup_path)
        else:
            os.remove(inp)
        os.replace(out, inp)
        final_path = inp
    else:
        final_path = out

    print(json.dumps({"input": inp, "output": final_path, "lines_total": total, "lines_changed": changed}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rename top-level 'messages' to 'conversations' in JSONL.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--backup", default="")
    p.add_argument("--overwrite", action="store_true")
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


if __name__ == '__main__':
    raise SystemExit(main())
