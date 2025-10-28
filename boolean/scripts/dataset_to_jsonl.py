import json, sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: dataset_to_jsonl.py <input_json> <output_jsonl>")
    sys.exit(1)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

data = json.load(open(input_path, 'r', encoding='utf-8'))

# Normalize to a list of items (observation sets)
items = []
if 'datasets' in data:
    items = data['datasets']
elif 'datasets_by_n_observations' in data:
    for n, ds in data['datasets_by_n_observations'].items():
        items.extend(ds)
elif 'observation_sets' in data:
    items = data['observation_sets']
elif 'datasets' in data.get('metadata', {}):
    items = data.get('datasets', [])
else:
    items = data.get('datasets', []) or data.get('observation_sets', []) or []

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as out:
    for item in items:
        # Build prompt lines
        obs_lines = []
        if 'observations' in item:
            for obs in item['observations']:
                if isinstance(obs, dict) and 'string' in obs:
                    obs_lines.append(obs['string'])
                elif isinstance(obs, dict) and 'inputs' in obs and 'output' in obs:
                    inputs = obs['inputs']
                    # If inputs is dict
                    if isinstance(inputs, dict):
                        s = ", ".join(f"{k}={v}" for k,v in inputs.items())
                    else:
                        # if inputs is list like [0,1] map to x,y
                        vars_names = ['x','y','z']
                        s = ", ".join(f"{vars_names[i]}={val}" for i,val in enumerate(inputs))
                    obs_lines.append(f"({s}) -> {obs['output']}")
                else:
                    obs_lines.append(str(obs))
        prompt = "Observations:\n" + "\n".join(obs_lines) + "\n\nProvide a Boolean expression over variables x and y that matches the observations.\n"

        # completion selection
        completion = ""
        if 'ground_truth_expressions' in item and item['ground_truth_expressions']:
            gt = item['ground_truth_expressions'][0]
            completion = gt.get('canonical_form') or gt.get('formula') or str(gt)
        elif 'ground_truth_structures' in item and item['ground_truth_structures']:
            completion = json.dumps(item['ground_truth_structures'][0], ensure_ascii=False)
        else:
            completion = ""

        # normalise
        if not completion.startswith(" "):
            completion = " " + completion
        if not completion.endswith("\n"):
            completion = completion + "\n"

        out.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

print(f"Wrote JSONL to {output_path}")
