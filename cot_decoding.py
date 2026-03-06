# Adapted from code by Xueqing Wu
# Source: https://github.com/shirley-wu/cot_decoding
#
# Original code copyright (c) 2026 Xueqing Wu
# Licensed under the MIT License.
#
# Modifications copyright (c) 2026 Paulo Santos Neto
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from constants import DEFAULT_STOP_STRINGS
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, StopStringCriteria


class StopOnSequence(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequence: str):
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence
        self.stop_tokens = tokenizer(stop_sequence, add_special_tokens=False).input_ids

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < len(self.stop_tokens):
            return False
        return input_ids[0][-len(self.stop_tokens):].tolist() == self.stop_tokens


class SimpleBatch:
    def __init__(self, tokenized_dict):
        self.input_ids = tokenized_dict["input_ids"]
        self.attention_mask = tokenized_dict["attention_mask"]


@dataclass
class DecodingArguments:
    encode_format: str = "instruct"
    max_new_tokens: int = 256
    decoding: str = "cot"
    cot_n_branches: int = 10
    cot_aggregate: str = "sum"


def encode_function(example: Dict[str, Any], tokenizer, task) -> Dict[str, torch.Tensor]:
    prompt = task.encode_prompt(example)
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _decode_with_offsets(tokenizer, generation_ids: torch.Tensor) -> Tuple[str, List[int]]:
    tokens = tokenizer.convert_ids_to_tokens(generation_ids.tolist())
    text = ""
    offsets: List[int] = []
    for i, tok in enumerate(tokens):
        if tokenizer.eos_token is not None and tok == tokenizer.eos_token:
            break
        text = tokenizer.convert_tokens_to_string(tokens[: i + 1])
        offsets.append(len(text))
    offsets += [-1 for _ in range(len(tokens) - len(offsets))]
    return text, offsets


def _match_answer_span(answer_span: Tuple[int, int], offsets: List[int]) -> List[int]:
    answer_s, answer_e = answer_span
    inds: List[int] = []
    for i, offset in enumerate(offsets):
        if offset < 0:
            continue
        if answer_s < offset:
            inds.append(i)
            if answer_e <= offset:
                break
    return inds


def _get_cot_score(probs: torch.Tensor) -> float:
    # probs: [T, V]
    top2 = probs.topk(k=2, dim=-1, sorted=True).values  # [T, 2]
    score = (top2[:, 0] - top2[:, 1]).mean()
    return float(score.detach().cpu())

def strip_stop_strings(text: str, stop_strings: list[str]) -> str:
    changed = True
    while changed:
        changed = False
        for s in stop_strings:
            if text.endswith(s):
                text = text[: -len(s)]
                changed = True
    return text.rstrip()


def cot_decoding_solve(
    model,
    tokenizer,
    task,
    questions_text: List[str],
    batch: SimpleBatch,
    args: DecodingArguments,
    qa_pipeline=None,
    stop_strings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:

    bsz = batch.input_ids.shape[0]
    n_branches = int(args.cot_n_branches)

    stopping_criteria = StoppingCriteriaList([
        StopStringCriteria(tokenizer, DEFAULT_STOP_STRINGS)
    ])

    input_ids = model.generate(
        input_ids=batch.input_ids.to(model.device),
        attention_mask=batch.attention_mask.to(model.device),
        do_sample=False,
        num_beams=n_branches,
        num_return_sequences=n_branches,
        max_new_tokens=1,
        min_new_tokens=1,
        early_stopping=True,
        stopping_criteria=stopping_criteria if len(stopping_criteria) else None,
    )

    attention_mask = batch.attention_mask.to(model.device).repeat_interleave(n_branches, 0)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((len(attention_mask), 1), device=attention_mask.device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max(int(args.max_new_tokens) - 1, 1),
        output_logits=True,
        return_dict_in_generate=True,
        early_stopping=True,
        stopping_criteria=stopping_criteria if len(stopping_criteria) else None,
    )

    # sequences: [bsz*n_branches, prompt+gen]
    gen_ids = outputs["sequences"][:, input_ids.shape[1] - 1 :].reshape(bsz, n_branches, -1)

    # stack -> [gen_len-1, bsz*n_branches, vocab] => transpose for [bsz*n_branches, gen_len-1, vocab]
    gen_probs = torch.stack(outputs["logits"], dim=0).softmax(-1).transpose(0, 1)
    n_vocab = gen_probs.shape[-1]

    uniform = torch.full((bsz * n_branches, 1, n_vocab), 1 / n_vocab, dtype=gen_probs.dtype, device=gen_probs.device)
    gen_probs = torch.cat([uniform, gen_probs], dim=1).reshape(bsz, n_branches, -1, n_vocab)

    ret: List[Dict[str, Any]] = []

    for i in range(bsz):
        candidates: List[Dict[str, Any]] = []
        q_text = questions_text[i] if i < len(questions_text) else ""

        for j in range(n_branches):
            STOP_STRINGS = DEFAULT_STOP_STRINGS

            text, offsets = _decode_with_offsets(tokenizer, gen_ids[i, j])

            cut = None
            for s in STOP_STRINGS:
                idx = text.find(s)
                if idx != -1:
                    cut = idx if cut is None else min(cut, idx)

            if cut is not None:
                text = text[:cut].rstrip()
                new_offsets = []
                for off in offsets:
                    if off is None:
                        continue
                    if isinstance(off, (tuple, list)) and len(off) >= 1:
                        if off[0] < cut:
                            new_offsets.append(off)
                    elif isinstance(off, int):
                        if off <= cut:
                            new_offsets.append(off)
                    elif isinstance(off, dict) and "start" in off:
                        if int(off["start"]) <= cut:
                            new_offsets.append(off)

                offsets = new_offsets

            # ----------- REGEX EXTRACTION  -----------
            regex_answer, regex_span = task.extract_model_answer(text)
            regex_score = -100.0
            if regex_span is not None:
                answer_tokens = _match_answer_span(regex_span, offsets)
                if answer_tokens:
                    answer_probs = gen_probs[i, j][torch.as_tensor(answer_tokens, device=gen_probs.device)]
                    regex_score = _get_cot_score(answer_probs)

            # ----------- BERT QA EXTRACTION -----------
            bert_answer = "[invalid]"
            bert_score = -100.0

            if qa_pipeline is not None and text.strip():
                try:
                    bert_out = qa_pipeline(question=q_text, context=text)
                    bert_answer = bert_out.get("answer", "[invalid]")

                    start = int(bert_out.get("start", -1))
                    end = int(bert_out.get("end", -1))
                    if start >= 0 and end > start:
                        bert_span = (start, end)
                        bert_tokens = _match_answer_span(bert_span, offsets)
                        if bert_tokens:
                            bert_probs = gen_probs[i, j][torch.as_tensor(bert_tokens, device=gen_probs.device)]
                            bert_score = _get_cot_score(bert_probs)
                except Exception:
                    bert_answer = "[invalid]"
                    bert_score = -100.0

            candidates.append(
                {
                    "text": text,
                    "regex_answer": regex_answer,
                    "regex_score": float(regex_score),
                    "bert_answer": bert_answer,
                    "bert_score": float(bert_score),
                }
            )

        ret.append({"question_idx": i, "candidates": candidates})

    return ret
