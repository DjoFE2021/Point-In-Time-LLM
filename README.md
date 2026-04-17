# Description

## Datasets

Pre-Training:

```bash
cd data && python get_train_set.py --all --output-dir <OUTPUT_DIR>
python data/get_validation_set.py <OUTPUT_DIR>
```

SFT:

```bash
bash post_training/store_tokens.sh
```

## Training

Pre-Training:

```bash
torchrun --nproc_per_node=$NGPUS train_gpt.py \
    --model <1B|4B> \
    --data-dir <TRAIN_SHARDS_DIR> \
    --val-dir <VAL_SHARDS_DIR>
```

SFT:

```bash
bash post_training/sft_all.sh <CHECKPOINT_1.pt> [CHECKPOINT_2.pt] ...
```

## General Language Understanding

PIT-4B

```python3
python3 eval/eval_multigpu.py --model 4B --checkpoints <CHECKPOINT_1.pt> [CHECKPOINT_2.pt] ...
```

HF Models
```python3
python3 eval/eval_hf_models.py --models <model_name> [model_name_2]
```

## Ifeval

PIT-4B:

```bash
./post_training/sft_all.sh
```
Benchmarks:

```bash
 torchrun --nproc_per_node=1 eval/ifeval_test.py \
    --candidate manelalab/chrono-gpt-instruct-v1-20241231 &&
torchrun --nproc_per_node=1 eval/ifeval_test.py \
    --candidate Qwen/Qwen1.5-1.8B-Chat
```