from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from constants import model_family, LORA_TARGET_MODULES_BY_FAMILY
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
import random
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model

class MaskStrategy(Protocol):
    name: str
    def build_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        answer_len: int,
        tokenizer: Any,
    ) -> torch.Tensor: ...


@dataclass
class FirstTokenMask:
    name: str = "first_token"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        first_answer_pred_pos = prompt_len - 1

        if 0 <= first_answer_pred_pos < mask.shape[0]:
            mask[first_answer_pred_pos] = 1.0

        return mask

@dataclass
class ThreeTokenMask:
    name: str = "three_token"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        first_pos = prompt_len - 1       
        second_pos = prompt_len        
        eos_pos = prompt_len + answer_len - 1 

        for pos in (first_pos, second_pos, eos_pos):
            if 0 <= pos < mask.shape[0]:
                mask[pos] = 1.0

        return mask

@dataclass
class RandomTwoAnswerTokenMask:
    name: str = "random_two"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        start = prompt_len - 1
        end = prompt_len + answer_len - 1

        valid_positions = list(range(start, end + 1))
        if not valid_positions:
            return mask

        k = min(2, len(valid_positions))
        chosen = random.sample(valid_positions, k=k)

        for pos in chosen:
            if 0 <= pos < mask.shape[0]:
                mask[pos] = 1.0

        return mask

@dataclass
class FullAnswerMask:
    name: str = "full"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        seq_len = input_ids.shape[0]

        start = max(prompt_len - 1, 0)
        end = min(prompt_len + answer_len, seq_len)
        
        if start < end:
            mask[start:end] = 1.0
        return mask

@dataclass
class EosOnlyMask:
    name: str = "eos_only"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        eos_pred_pos = prompt_len + answer_len - 1

        if 0 <= eos_pred_pos < mask.shape[0]:
            mask[eos_pred_pos] = 1.0

        return mask

@dataclass
class TwoTokenMask:
    name: str = "two_token"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        first_answer_pred_pos = prompt_len - 1

        eos_pred_pos = prompt_len + answer_len - 1

        if 0 <= first_answer_pred_pos < mask.shape[0]:
            mask[first_answer_pred_pos] = 1.0
        if 0 <= eos_pred_pos < mask.shape[0]:
            mask[eos_pred_pos] = 1.0

        return mask

@dataclass
class RegexAnswerSpanTokenMask:
    name: str = "regex_answer_span"

    def build_mask(self, input_ids, attention_mask, prompt_len, answer_len, tokenizer):
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        ans_start = max(prompt_len, 0)
        ans_end = min(prompt_len + answer_len, int(input_ids.shape[0]))
        if ans_start >= ans_end:
            return mask

        ans_ids = input_ids[ans_start:ans_end].tolist()
        ans_toks = tokenizer.convert_ids_to_tokens(ans_ids)

        def is_num_tok(t: str) -> bool:
            t2 = t.replace("▁", "").replace("Ġ", "")
            t2 = t2.replace(",", "").replace(".", "")
            t2 = t2.lstrip("-")
            return t2.isdigit() and len(t2) > 0

        i = len(ans_toks) - 1
        while i >= 0 and not is_num_tok(ans_toks[i]):
            i -= 1
        if i < 0:
            return mask 

        j = i
        while j >= 0 and is_num_tok(ans_toks[j]):
            j -= 1
        j += 1

        for k in range(j, i + 1):
            abs_pos = ans_start + k
            pred_pos = abs_pos - 1
            if 0 <= pred_pos < mask.shape[0]:
                mask[pred_pos] = 1.0

        return mask

def get_mask_strategy(name: str) -> MaskStrategy:
    name = (name or "").lower()
    if name in ("full", "fft", "all"):
        return FullAnswerMask()
    if name in ("two_token", "2tft", "two", "2t"):
        return TwoTokenMask()
    if name in ("first_token", "first", "1t", "one_token"):
        return FirstTokenMask()
    if name in ("eos_only", "eos", "only_eos"):
        return EosOnlyMask()
    if name in ("three_token", "3tft", "three"):
        return ThreeTokenMask()
    if name in ("random_two", "rand2"):
        return RandomTwoAnswerTokenMask()
    if name in ("regex_answer_span", "answer_span", "span"):
        return RegexAnswerSpanTokenMask()
    raise ValueError(f"mask_mode invalid: {name}")

class JsonlTextDataset(TorchDataset):
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        prompt_lens: List[int],
        answer_lens: List[int],
        tokenizer,
        mask_strategy: MaskStrategy,
    ):
        self.encodings = encodings
        self.prompt_lens = prompt_lens
        self.answer_lens = answer_lens
        self.tokenizer = tokenizer
        self.mask_strategy = mask_strategy

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}

        labels = item["input_ids"].clone()
        labels[:-1] = labels[1:].clone()
        labels[-1] = self.tokenizer.eos_token_id
        item["labels"] = labels

        prompt_len = int(self.prompt_lens[idx])
        answer_len = int(self.answer_lens[idx])
        item["loss_mask"] = self.mask_strategy.build_mask(
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
            prompt_len=prompt_len,
            answer_len=answer_len,
            tokenizer=self.tokenizer,
        )

        return item

def debug_loss_mask(ds, tokenizer, idx=0, show_context=8):
    ex = ds[idx]
    input_ids = ex["input_ids"]
    labels = ex["labels"]
    mask = ex["loss_mask"]

    active = (mask > 0).nonzero(as_tuple=False).view(-1).tolist()

    print(f"[debug] idx={idx}")
    print(f"active_positions = {active}  (count={len(active)})")

    for pos in active:
        in_tok = tokenizer.decode([int(input_ids[pos])], skip_special_tokens=False)

        lab_tok = tokenizer.decode([int(labels[pos])], skip_special_tokens=False)

        print(f"\n  pos={pos}")
        print(f"    input_id={int(input_ids[pos])} token={in_tok!r}")
        print(f"    label_id={int(labels[pos])} token={lab_tok!r}")

        lo = max(0, pos - show_context)
        hi = min(len(input_ids), pos + show_context + 1)
        ctx_ids = input_ids[lo:hi].tolist()
        ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        print(f"    context[{lo}:{hi}]: {ctx_text!r}")

def format_example(question: str, reasoning: str, prompt_style: str) -> Tuple[str, str]:
    if prompt_style == "plain":
        prompt = question.rstrip() + " "
        answer = reasoning.lstrip()
        return prompt, answer
    if prompt_style == "train":
        return question, reasoning
    if prompt_style == "qa":
        prompt = f"Q: {question}\nA:"
        answer = reasoning
        return prompt, answer
    if prompt_style == "instruct":
        prompt = f"### Instruction:\nSolve step by step.\n\n### Input:\n{question}\n\n### Response:\n"
        answer = reasoning
        return prompt, answer
    raise ValueError("prompt_style invalid")


def prepare_dataset(
    jsonl_path: str,
    tokenizer,
    prompt_style: str,
    max_length: int,
    question_field: str,
    reasoning_field: str,
) -> Tuple[TorchDataset, int]:
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    prompts: List[str] = []
    answers: List[str] = []
    prompt_lens: List[int] = []
    answer_lens: List[int] = []

    for ex in ds:
        q = str(ex.get(question_field, "")).strip()
        r = str(ex.get(reasoning_field, "")).strip()
        prompt_text, answer_text = format_example(q, r, prompt_style)

        prompts.append(prompt_text)
        answers.append(answer_text)

        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        a_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        prompt_lens.append(len(p_ids))
        answer_lens.append(len(a_ids))

    full_texts = [p + a for p, a in zip(prompts, answers)]

    enc = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False, 
    )

    return enc, prompt_lens, answer_lens, len(ds)

class MaskedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")        # [B, T]
        loss_mask = inputs.pop("loss_mask")  # [B, T] float/bool

        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        vocab = logits.shape[-1]

        active = loss_mask.reshape(-1).to(dtype=torch.bool)

        if active.sum().item() == 0:
            loss = logits.sum() * 0.0 
            return (loss, outputs) if return_outputs else loss

        active_logits = logits.reshape(-1, vocab)[active]  # [N, V]
        active_labels = labels.reshape(-1)[active]         # [N]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss


def load_llm_with_lora(
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device_map: str,
    dtype: str,
):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    torch_dtype = (
        torch.bfloat16 if dtype == "bf16"
        else torch.float16 if dtype == "fp16"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    family = model_family(base_model)
    target_modules = LORA_TARGET_MODULES_BY_FAMILY[family]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_rslora=True,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok

def add_args(subparsers):

    tr = subparsers.add_parser("train", help="Train FFT (full) or 2TFT (two_token) with LoRA on a jsonl dataset")
    tr.add_argument("--base_model", required=True, help="e.g.: mistralai/Mistral-7B-v0.1")
    tr.add_argument("--train_jsonl", required=True, help="jsonl with question+reasoning (e.g.: dataset_cot_max_regex.jsonl)")
    tr.add_argument("--out_dir", required=True)

    tr.add_argument("--prompt_style", choices=["qa","instruct"], default="qa")
    tr.add_argument("--question_field", default="question")
    tr.add_argument("--reasoning_field", default="reasoning")

    tr.add_argument("--mask_mode", choices=["full","two_token"], default="two_token",
                    help="full=FFT (all answer tokens); two_token=2TFT (first answer token + EOS-prediction token)")

    tr.add_argument("--max_length", type=int, default=512)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--batch_size", type=int, default=1)
    tr.add_argument("--grad_accum", type=int, default=4)
    tr.add_argument("--warmup_ratio", type=float, default=0.1)

    tr.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    tr.add_argument("--device_map", default="auto")

    # LoRA params
    tr.add_argument("--lora_r", type=int, default=64)
    tr.add_argument("--lora_alpha", type=int, default=32)
    tr.add_argument("--lora_dropout", type=float, default=0.05)

    tr.add_argument("--logging_steps", type=int, default=10)
    tr.add_argument("--save_strategy", choices=["steps","epoch"], default="epoch")
    tr.add_argument("--save_total_limit", type=int, default=2)

    tr.add_argument("--neftune_noise_alpha", type=float, default=0.0,
                    help="0 disables it. Example: 5.0 for small datasets.")


def run_training(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_strategy = get_mask_strategy(args.mask_mode)

    model, tokenizer = load_llm_with_lora(
        base_model=args.base_model,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        device_map=str(args.device_map),
        dtype=str(args.dtype),
    )

    enc, prompt_lens, answer_lens, nrows = prepare_dataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        prompt_style=args.prompt_style,
        max_length=int(args.max_length),
        question_field=args.question_field,
        reasoning_field=args.reasoning_field,
    )

    train_ds = JsonlTextDataset(
        encodings=enc,
        prompt_lens=prompt_lens,
        answer_lens=answer_lens,
        tokenizer=tokenizer,
        mask_strategy=mask_strategy,
    )

    def collate(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "loss_mask": torch.stack([x["loss_mask"] for x in batch]),
        }

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        num_train_epochs=int(args.epochs),
        warmup_ratio=float(args.warmup_ratio),
        lr_scheduler_type="cosine",
        logging_steps=int(args.logging_steps),
        save_strategy=str(args.save_strategy),
        save_total_limit=int(args.save_total_limit),
        remove_unused_columns=False,
        fp16=(args.dtype == "fp16"),
        bf16=(args.dtype == "bf16"),
        optim="paged_adamw_32bit",
        neftune_noise_alpha=float(args.neftune_noise_alpha) if float(args.neftune_noise_alpha) > 0 else None,
        report_to=[],
    )

    trainer = MaskedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collate,
    )

    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "train_jsonl": args.train_jsonl,
            "prompt_style": args.prompt_style,
            "question_field": args.question_field,
            "reasoning_field": args.reasoning_field,
            "mask_mode": args.mask_mode,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "n_train_rows": nrows,
        }, f, ensure_ascii=False, indent=2)

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"[train] done. saved to: {out_dir}")
