Run benchmark

```
cd  causal
uv run python run_causal_benchmark.py \
  --dataset "datasets/node03/n3_all_observations.json" \
  --config "config/config_ollama.yaml" \
  --n-samples 30 \
  --query-multiplier 1.0 \
  --seed 33550336
```

Generate LoRa dataset
```
uv run python -m causal.tools.auto_build_lora_dataset \
  --node-range 2-4 \
  --seeds 11,12,13 \
  --mode all-gts \
  --prior-mode size-only \
  --large-n-threshold 4 \
  --output data/sample_lora_nodes2-4.jsonl
```
may come with an import error, just add a 'causal.modules'