#!/usr/bin/env python3
"""
Convert instruction-style JSONL (prompt/completion) into ChatML-style JSONL
Each output line will be a JSON object: {"messages": [ {role, content}, ... ] }

Usage:
  python convert_to_chatml.py --input /path/finetune_all.jsonl --output /path/chatml_all.jsonl \
    --system "You are a reasoning model that infers 3D structures from 2D views." 

The script accepts lines with keys: 'prompt' and 'completion' (fallback to 'input'/'target').
It will place the prompt in the user role and the completion in the assistant role.
"""
import argparse
import json
from pathlib import Path
import sys


def clean_prompt(text: str) -> str:
    # Remove common separator markers and trim
    if text is None:
        return ""
    # Remove trailing '###' separators used in prompts
    text = text.replace('\r\n', '\n')
    parts = text.split('\n')
    # drop trailing lines that are only hashes or separators
    while parts and parts[-1].strip() in ['', '###', '--', '---']:
        parts.pop()
    # join and strip
    return '\n'.join(parts).strip()


def clean_completion(text: str) -> str:
    if text is None:
        return ""
    # normalize newlines and strip leading/trailing whitespace but keep internal newlines
    text = text.replace('\r\n', '\n')
    return text.strip() + '\n'


def convert_line(obj: dict, system_prompt: str) -> dict:
    # Accept different field names
    prompt = obj.get('prompt') or obj.get('input') or obj.get('instruction') or ''
    completion = obj.get('completion') or obj.get('target') or obj.get('output') or ''

    user_content = clean_prompt(prompt)
    assistant_content = clean_completion(completion)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Convert finetune JSONL to ChatML-style JSONL")
    parser.add_argument('--input', '-i', required=True, help='Input JSONL (prompt/completion)')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL with ChatML messages')
    parser.add_argument('--system', '-s', default='You are a reasoning model that infers 3D structures from 2D views.', help='System message content')
    parser.add_argument('--preview', action='store_true', help='Print a preview (first converted object) to stdout')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count_in = 0
    count_out = 0
    preview_obj = None

    with input_path.open('r', encoding='utf-8') as inf, output_path.open('w', encoding='utf-8') as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                # skip malformed lines but report
                print(f"Skipping malformed JSON line: {e}", file=sys.stderr)
                continue
            count_in += 1
            out_obj = convert_line(obj, args.system)
            outf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            count_out += 1
            if preview_obj is None:
                preview_obj = out_obj

    print(f"Converted {count_in} -> {count_out} lines. Output: {output_path}")
    if args.preview and preview_obj is not None:
        print(json.dumps(preview_obj, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
