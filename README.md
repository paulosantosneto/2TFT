# 2T-FT: Two-Token Fine-tuning Improves Zero-Shot Performance With Minimal Training

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

See the LICENSE file for details.

## Code Details

Main entrypoint: `main.py`

---

# Features

## CoT Decoding

The CoT-decoding code is a modified version of the Xueqing Wu code. For more information on the original code, please visit: https://github.com/shirley-wu/cot_decoding

For each question, the model generates **k reasoning paths**.  
Each branch is processed to:

- extract answer with task-specific regex
- optionally extract answer with BERT QA
- compute a score based on token probability margins
- store reasoning + extracted answers + scores

---

## Answer Aggregation

Final prediction is selected using one of:

- **sum** — sum scores grouped by identical answer

---

## Dataset Generation

The pipeline can export multiple JSONL datasets from the same run.

Per extractor (regex / bert):

- greedy candidate
- best-by-score candidate
- aggregated candidate

All exported datasets are also packed into:

```
all_datasets.zip
```

---

## LoRA Finetuning (Masked Loss)

Training supports LoRA adapters with selective token loss masking.

Supported mask modes:

- full (FFT)
- two_token (2TFT)
- three_token
- first_token
- eos_only
- random_two
- answer_span

Loss is computed only where `loss_mask == 1`.

---

# Installation

Requires Python 3.10+ recommended.

Install core dependencies:

```bash
pip install -r requirements.txt
```

---

# Model Aliases

| Alias   | Model |
|---------|--------|
| mistral | mistralai/Mistral-7B-v0.1 |
| qwen2   | Qwen/Qwen2.5-1.5B |
| phi2    | microsoft/phi-2 |

---

# Supported Datasets

- gsm / gsm8k
- svamp
- multiarith

---

# CLI Overview

Core parameters:

```bash
--mode prompt | dataset_generation | dataset_evaluation | finetuning
--model MODEL_NAME_OR_ALIAS
--k NUM_BRANCHES
--aggregate max | sum | self_consistency
--prompt_style qa | instruct | plain | train
--max_new_tokens N
--no_qa
--stop_strings STR1 STR2 ...
```

Dataset parameters:

```bash
--dataset_name NAME
--dataset_config CONFIG
--dataset_split SPLIT
--dataset_path FILE.jsonl
--limit N
```

Adapter parameters:

```bash
--adapter_dir PATH
--merge_lora
```

Training parameters:

```bash
--base_model MODEL
--train_jsonl FILE
--out_dir DIR
--mask_mode MODE
--epochs N
--lr LR
--batch_size N
--grad_accum N
```

---

# Mode: prompt

Runs CoT decoding for a single question.

## Example — self-consistency aggregation

```bash
python main.py \
  --mode prompt \
  --model mistral \
  --k 20 \
  --aggregate self_consistency \
  --prompt_style qa \
  --max_new_tokens 256 \
  --question "I have 3 apples, my dad has 2 more than me. Total?"
```

---

## Example — regex-only extraction

```bash
python main.py \
  --mode prompt \
  --model mistral \
  --no_qa \
  --k 10 \
  --aggregate sum \
  --question "John had 10 and used 3. Remaining?"
```

---

## Example — using LoRA adapter

```bash
python main.py \
  --mode prompt \
  --model mistralai/Mistral-7B-v0.1 \
  --adapter_dir finetuned/2tft_two_token \
  --merge_lora \
  --k 10 \
  --aggregate sum \
  --question "..."
```

---

## Output Files

Stored in:

```
runs/<RUN_ID>/
```

Files:

```
cot_decoding.jsonl
final.jsonl
resume_aggregation.jsonl
```

---

# Mode: dataset_generation

Runs decoding over a dataset and exports training-ready JSONLs.

## Example — GSM8K subset

```bash
python main.py \
  --mode dataset_generation \
  --dataset_name gsm8k \
  --dataset_config main \
  --dataset_split train \
  --limit 200 \
  --task gsm \
  --model mistral \
  --k 10 \
  --aggregate sum
```

---

## Example — SVAMP

```bash
python main.py \
  --mode dataset_generation \
  --dataset_name svamp \
  --dataset_split train \
  --task svamp \
  --model qwen2 \
  --k 15 \
  --aggregate self_consistency
```

---

## Exported Dataset Files

```
generated_dataset.jsonl

dataset_greedy_regex.jsonl
dataset_cot_max_regex.jsonl
dataset_cot_agg_regex.jsonl

dataset_greedy_bert.jsonl
dataset_cot_max_bert.jsonl
dataset_cot_agg_bert.jsonl

all_datasets.zip
```

Training input:

```
dataset_cot_max_regex.jsonl
```

---

# Mode: dataset_evaluation

Scores prediction JSONLs against a dataset.

## Example

```bash
python main.py \
  --mode dataset_evaluation \
  --task gsm \
  --dataset_path runs/RUN/generated_dataset.jsonl \
  --pred_jsonls runs/RUN/dataset_cot_agg_regex.jsonl \
  --pred_answer_field answer
```

---

# Mode: finetuning (LoRA)

Uses masked-token loss trainer.

Training JSONL format:

```json
{
  "question": "...",
  "reasoning": "model reasoning and final answer"
}
```

---

# Mask Modes (Loss Targeting)

| Mode | Description |
|--------|----------------|
full | train on all answer tokens |
two_token | first answer token + EOS |
three_token | first + second + EOS |
first_token | only first answer token |
eos_only | only EOS token |
random_two | two random answer tokens |
answer_span | regex-detected numeric span |

---

## Example — FFT (full mask)

```bash
python main.py train \
  --base_model mistralai/Mistral-7B-v0.1 \
  --train_jsonl runs/RUN/dataset_cot_max_regex.jsonl \
  --out_dir runs/TRAIN/fft \
  --mask_mode full \
  --prompt_style plain \
  --epochs 3 \
  --lr 5e-5 \
  --batch_size 1 \
  --grad_accum 4
```

---

## Example — 2TFT (two-token mask)

```bash
python main.py train \
  --base_model mistralai/Mistral-7B-v0.1 \
  --train_jsonl runs/RUN/dataset_cot_max_regex.jsonl \
  --out_dir runs/TRAIN/2tft \
  --mask_mode two_token \
  --prompt_style plain \
  --epochs 3 \
  --lr 5e-5 \
  --batch_size 1 \
  --grad_accum 4
```



# Notes

- BERT QA extractor loads automatically unless `--no_qa` is used
- Failed answer extraction is marked as `[invalid]`
- Default stop strings assume Q/A prompt format
- Confidence score uses token probability margin over answer span

