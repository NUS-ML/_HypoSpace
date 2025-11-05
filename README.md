# NUS-EEC4300 

This is for the final project of Machine Learning, instructed by Dianbo Liu.

Completed by Haoyan Yang, Chijin Yu, Hanwen Zhang, Jifeng Zhu

Great appreciate for my teammates.

## Report 
Our report is [here](Association_for_Computational_Linguistics__ACL__conference.pdf). Please refer it for full experimental result and methods.

## QuickStart
Our aim is to fine-tune the qwen2.5:0.5b, so we need to generate dataset for 3 tasks and then use lora.

### Generate Dataset
Take 3d as example(the same as other two tasks)
1. You need to install the virtual env 
```zsh
$ uv sync
```
2. Run the original script from HypoSpace
```zsh
$ cd 3d
$ uv run python generate_3d_dataset_complete.py \
  --grid-size 3 \
  --max-height 3 \
  --max-blocks 1 \
  --fixed \
  --seed 33550336 \
  --output "datasets/3d_complete.json"
```
3. Run the script to change to loRa
```zsh
$ uv run tools/auto_build_lora_dataset.py --inputs ./datasets/3d_complete.json --output ./datasets/lora_3d.jsonl --seeds 42,43 --mode all-gts --include-priors 0 --limit-per-set 100 --dedup 
```
4. An other part of dataset distill from GPT-5. Didn't give out here.

### Do LoRa 
1. move train.jsonl into lora/data, and do lora
```zsh
$ cd lora
$ mlx_lm.lora --model <base-model-path> --train --data ./data/causal<or other task> --report-to swanlab
```
2. combine model with lora module
```zsh
$ mlx_lm.fuse --model <base-model-path>  --adapter-path adapters --save-path <your-model-name>
```
3. use ollama to deploy the llm
```zsh
$ ollama create <your-model-name> -f <Modelfile path>
```
4. run the model in ollama
```zsh
$ ollama run <your-model-name>
```