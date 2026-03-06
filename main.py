from __future__ import annotations
from transformers import set_seed
import zipfile
import collections
import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from datasets import load_dataset
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from evaluation import GSMTask, SVAMPTask, MultiArithTask
from evaluation import load_task_dataset
from cot_decoding import ( 
    DecodingArguments,
    SimpleBatch,
    encode_function,
    cot_decoding_solve,
)
from training import run_training
from peft import PeftModel
from constants import DEFAULT_STOP_STRINGS, DEFAULT_K, _MODEL_ALIASES

def load_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    limit: Optional[int] = None,
    start: int = 0,
):
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config)[split]
    else:
        ds = load_dataset(dataset_name)[split]

    n = len(ds)

    if start < 0:
        start = 0
    if start >= n:
        return []

    if limit is not None:
        end = min(start + limit, n)
    else:
        end = n

    ds = ds.select(range(start, end))

    return [dict(ex) for ex in ds]

# -------------------- runs/ helpers -------------------- #
def make_run_dir(base: str = "runs") -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def resolve_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


def build_task(task_name: str, prompt_style: str):
    task_name = (task_name or "gsm").lower()
    if task_name in ("gsm", "gsm8k"):
        return GSMTask(encode_format=prompt_style)
    if task_name in ("svamp",):
        return SVAMPTask()
    if task_name in ("multiarith", "multi_arith"):
        return MultiArithTask()
    return GSMTask(encode_format=prompt_style)

def load_llm(model_name: str, device: Optional[str] = None, adapter_dir: Optional[str] = None, merge_lora: bool = False):
    model_name = resolve_model_name(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge_lora:
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, device

def load_qa_pipeline(device: str):
    dev = 0 if device == "cuda" else -1
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=dev)

def _set_attr(obj, name: str, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass

def _aggregate_final_answer(
    results: List[Dict[str, Any]],
    aggregate: str
) -> Tuple[str, Dict[str, Any]]:

    if not results or "candidates" not in results[0]:
        return "[no-result]", {}

    cands = results[0]["candidates"]

    def get_answer(c: Dict[str, Any]) -> str:
        ans = c.get("regex_answer")
        if not ans or ans == "[invalid]":
            ans = c.get("bert_answer")
        if not ans:
            ans = "[invalid]"
        return str(ans)

    def get_score(c: Dict[str, Any]) -> float:
        rs = float(c.get("regex_score", -100.0))
        bs = float(c.get("bert_score", -100.0))
        return max(rs, bs)

    if aggregate == "self_consistency":

        buckets: Dict[str, List[Dict[str, Any]]] = {}

        for c in cands:
            ans = get_answer(c)
            buckets.setdefault(ans, []).append(c)

        answers_json = []
        votes_dict = {}

        for ans, items in buckets.items():
            votes = len(items)
            votes_dict[ans] = votes

            scores = [get_score(x) for x in items]

            answers_json.append({
                "answer": ans,
                "votes": votes,
                "scores": scores,
                "score_max": max(scores) if scores else -1e9,
                "score_mean": (sum(scores)/votes) if votes > 0 else -1e9,
                "candidates": items, 
            })

        answers_json.sort(
            key=lambda x: (x["votes"], x["score_max"], x["score_mean"]),
            reverse=True
        )

        best_answer = answers_json[0]["answer"]

        meta = {
            "method": "self_consistency",
            "best_answer": best_answer,
            "votes": votes_dict,
            "answers": answers_json,
        }

        return best_answer, meta

    def score_sum(c):
        return float(c.get("regex_score", -100.0)) + float(c.get("bert_score", -100.0))

    if aggregate == "sum":
        best_c = max(cands, key=score_sum)
        ans = get_answer(best_c)
        return ans, {
            "method": "sum",
            "picked_answer": ans,
            "picked_candidate": best_c
        }

    best_c = max(cands, key=get_score)
    ans = get_answer(best_c)
    return ans, {
        "method": "max",
        "picked_answer": ans,
        "picked_candidate": best_c
    }


def _extract_question(task_name: str, ex: Dict[str, Any]) -> str:
    t = (task_name or "").lower()
    if t == "svamp":
        body = ex.get("Body", "")
        q = ex.get("Question", "")
        if body:
            return f"{body}\n{q}".strip()
        return str(q).strip()
    return str(ex.get("question", "")).strip()

def normalize_example_for_task(task_name: str, ex: Dict[str, Any]) -> Dict[str, Any]:
    t = (task_name or "").lower()

    if t == "svamp":
        qc = ex.get("question_concat")
        if qc:
            ex = dict(ex) 
            ex["question"] = qc

    return ex

def _extract_gt(task, task_name: str, ex: Dict[str, Any]) -> str:
    try:
        return str(task.extract_gt_answer(ex)).strip()
    except Exception:
        t = (task_name or "").lower()
        if t in ("multiarith", "multi_arith"):
            return str(ex.get("final_ans", "[invalid]")).strip()
        if t == "svamp":
            return str(ex.get("Answer", "[invalid]")).strip()
        if t in ("gsm", "gsm8k"):
            return str(ex.get("answer", "[invalid]")).strip()
        return "[invalid]"


def _normalize_pred_for_eval(task, pred: str) -> str:
    try:
        ans, _ = task.extract_model_answer(str(pred))
        return str(ans).strip()
    except Exception:
        return str(pred).strip()


def run_cot_for_one(
    model,
    tokenizer,
    task,
    question_text: str,
    decoding_args,
    qa_pipeline,
    stop_strings: List[str],
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    example = {"question": question_text}
    tokenized = encode_function(example, tokenizer, task)
    batch = SimpleBatch(tokenized)

    results = cot_decoding_solve(
        model=model,
        tokenizer=tokenizer,
        task=task,
        questions_text=[question_text],
        batch=batch,
        args=decoding_args,
        qa_pipeline=qa_pipeline,
        stop_strings=stop_strings,
    )
    final_answer, final_meta = _aggregate_final_answer(results, decoding_args.cot_aggregate)
    return results, final_answer, final_meta


# -------------------- modes -------------------- #
def run_prompt_mode(args) -> int:
    run_dir = make_run_dir("runs")
    print(f"[run] saving outputs to: {run_dir}")

    task = build_task(args.task, args.prompt_style)
    model, tokenizer, device = load_llm(
        args.model,
        args.device,
        adapter_dir=getattr(args, "adapter_dir", None),
        merge_lora=getattr(args, "merge_lora", False),
    )

    qa_pipeline = None
    if not args.no_qa:
        qa_pipeline = load_qa_pipeline(device)

    decoding_args = DecodingArguments()
    _set_attr(decoding_args, "encode_format", args.prompt_style)
    _set_attr(decoding_args, "max_new_tokens", int(args.max_new_tokens))
    _set_attr(decoding_args, "cot_n_branches", int(args.k))
    _set_attr(decoding_args, "cot_aggregate", str(args.aggregate))

    stop_strings = DEFAULT_STOP_STRINGS

    results, final_answer, final_meta = run_cot_for_one(
        model=model,
        tokenizer=tokenizer,
        task=task,
        question_text=args.question,
        decoding_args=decoding_args,
        qa_pipeline=qa_pipeline,
        stop_strings=stop_strings,
    )

    out_path = run_dir / "cot_decoding.jsonl"
    for q in results:
        for cand in q.get("candidates", []):
            append_jsonl(
                out_path,
                {
                    "run_id": run_dir.name,
                    "mode": "prompt",
                    "technique": "cot_decoding",
                    "model": resolve_model_name(args.model),
                    "k": int(args.k),
                    "prompt_style": args.prompt_style,
                    "max_new_tokens": int(args.max_new_tokens),
                    "aggregate": args.aggregate,
                    "stop_strings": stop_strings,
                    "question": args.question,
                    "text": cand.get("text", ""),
                    "regex_answer": cand.get("regex_answer", "[invalid]"),
                    "regex_score": cand.get("regex_score", -100.0),
                    "bert_answer": cand.get("bert_answer", "[invalid]"),
                    "bert_score": cand.get("bert_score", -100.0),
                },
            )

    answers_resume = [
        {
            "answer": a["answer"],
            "votes": a["votes"],
            "score_max": a["score_max"],
            "score_mean": a["score_mean"],
        }
        for a in final_meta.get("answers", [])
    ]

    append_jsonl(
        run_dir / "final.jsonl",
        {
            "run_id": run_dir.name,
            "mode": "prompt",
            "technique": "cot_decoding",
            "model": resolve_model_name(args.model),
            "k": int(args.k),
            "prompt_style": args.prompt_style,
            "max_new_tokens": int(args.max_new_tokens),
            "aggregate": args.aggregate,
            "question": args.question,
            "final_answer": final_answer,
            "meta": final_meta,
        },
    )

    append_jsonl(
        run_dir / "resume_aggregation.jsonl",
        {
            "run_id": run_dir.name,
            "question": args.question,
            "aggregate": args.aggregate,
            "best_answer": final_answer,
            "votes": final_meta.get("votes", {}),
            "answers_resume": answers_resume
        },
    )

    print(f"[run] saved: {out_path}")
    print(f"[run] saved: {run_dir / 'final.jsonl'}")
    print(f"\nFinal answer ({args.aggregate}): {final_answer}\n")
    return 0

def export_strategies_to_jsonl(results, questions_text, out_dir=".", aggregation="self_consistency"):
    os.makedirs(out_dir, exist_ok=True)
    all_generated_files = []

    def save_jsonl(filename, data):
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"File saved: {path} ({len(data)} records)")
        all_generated_files.append(path)

    def export_mode(mode):
        data_greedy = []
        data_cot_max = []
        data_cot_agg = []

        ans_key = f"{mode}_answer"
        score_key = f"{mode}_score"

        for idx, item in enumerate(results):
            question = questions_text[idx] if idx < len(questions_text) else ""
            candidates = item.get("candidates", [])
            if not candidates:
                common_entry = {"question": question, "answer": "[invalid]", "reasoning": ""}
                data_greedy.append(common_entry)
                data_cot_max.append(common_entry)
                data_cot_agg.append(common_entry)
                continue

            valid_candidates = [
                c for c in candidates
                if c.get(ans_key) not in (None, "", "[invalid]")
            ]

            if not valid_candidates:
                fb = candidates[0]
                common_entry = {
                    "question": question,
                    "answer": fb.get(ans_key, "[invalid]"),
                    "reasoning": fb.get("text", ""),
                    "score": fb.get(score_key, -100.0),
                }
                data_greedy.append({k: common_entry[k] for k in ("question","answer","reasoning")})
                data_cot_max.append(common_entry)
                data_cot_agg.append(common_entry)
                continue

            greedy_cand = candidates[0]
            data_greedy.append({
                "question": question,
                "answer": greedy_cand.get(ans_key, "[invalid]"),
                "reasoning": greedy_cand.get("text", ""),
            })

            best_max = max(valid_candidates, key=lambda x: x.get(score_key, -100.0))
            data_cot_max.append({
                "question": question,
                "answer": best_max.get(ans_key, "[invalid]"),
                "reasoning": best_max.get("text", ""),
                "score": best_max.get(score_key, -100.0),
            })

            if aggregation == "self_consistency":
                counts = collections.Counter([c.get(ans_key) for c in valid_candidates])
                best_ans = counts.most_common(1)[0][0]

                reps = [c for c in valid_candidates if c.get(ans_key) == best_ans]
                rep = max(reps, key=lambda x: x.get(score_key, -100.0))

                data_cot_agg.append({
                    "question": question,
                    "answer": best_ans,
                    "reasoning": rep.get("text", ""),
                    "votes": int(counts[best_ans]),
                    "votes_all": dict(counts), 
                })

            elif aggregation == "sum":
                score_sum = {}
                rep_by_ans = {}

                for cand in valid_candidates:
                    ans = cand.get(ans_key)
                    sc = float(cand.get(score_key, -100.0))
                    score_sum[ans] = score_sum.get(ans, 0.0) + sc

                    if ans not in rep_by_ans or sc > float(rep_by_ans[ans].get(score_key, -100.0)):
                        rep_by_ans[ans] = cand

                best_ans = max(score_sum, key=score_sum.get)
                rep = rep_by_ans[best_ans]

                data_cot_agg.append({
                    "question": question,
                    "answer": best_ans,
                    "reasoning": rep.get("text", ""),
                    "aggregated_score": float(score_sum[best_ans]),
                    "scores_all": score_sum,
                })

            else:
                raise ValueError("aggregation must be 'self_consistency' or 'sum'")

        save_jsonl(f"dataset_greedy_{mode}.jsonl", data_greedy)
        save_jsonl(f"dataset_cot_max_{mode}.jsonl", data_cot_max)
        save_jsonl(f"dataset_cot_agg_{mode}.jsonl", data_cot_agg)

    export_mode("regex")
    export_mode("bert")


    final_zip = os.path.join(out_dir, "all_datasets.zip")
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for file in all_generated_files:
            z.write(file, arcname=os.path.basename(file))

    print(f"\nZIP FINAL criado: {final_zip}")
    return final_zip, all_generated_files


def run_dataset_generation(args) -> int:
    run_dir = make_run_dir("runs")
    print(f"[run] saving outputs to: {run_dir}")

    task = build_task(args.task, args.prompt_style)
    model, tokenizer, device = load_llm(
        args.model,
        args.device,
        adapter_dir=getattr(args, "adapter_dir", None),
        merge_lora=getattr(args, "merge_lora", False),
    )

    qa_pipeline = None
    if not args.no_qa:
        qa_pipeline = load_qa_pipeline(device)

    decoding_args = DecodingArguments()
    _set_attr(decoding_args, "encode_format", args.prompt_style)
    _set_attr(decoding_args, "max_new_tokens", int(args.max_new_tokens))
    _set_attr(decoding_args, "cot_n_branches", int(args.k))
    _set_attr(decoding_args, "cot_aggregate", str(args.aggregate))

    stop_strings = DEFAULT_STOP_STRINGS

    if not args.dataset_name:
        raise ValueError("--dataset_name is required in dataset_generation")

    examples = load_hf_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        limit=args.limit,
    )
    if not examples:
        raise ValueError("No examples loaded. Check --dataset_name/--dataset_split.")


    logs_path = run_dir / "cot_decoding.jsonl"
    final_path = run_dir / "final.jsonl"
    gen_path = run_dir / (args.output_name or "generated_dataset.jsonl")

    print(f"[data] loaded {len(examples)} examples from HF: {args.dataset_name} split={args.dataset_split}")


    all_results = [] 
    all_questions_text = []

    for idx, ex in enumerate(examples):
        ex = normalize_example_for_task(args.task, ex)

        q_text = _extract_question(args.task, ex)
        if not q_text:
            continue

        results, final_answer, final_meta = run_cot_for_one(
            model=model,
            tokenizer=tokenizer,
            task=task,
            question_text=q_text,
            decoding_args=decoding_args,
            qa_pipeline=qa_pipeline,
            stop_strings=stop_strings,
        )

        if results:
            all_results.append(results[0])
            all_questions_text.append(q_text)

        for q in results:
            for cand in q.get("candidates", []):
                append_jsonl(
                    logs_path,
                    {
                        "run_id": run_dir.name,
                        "mode": "dataset_generation",
                        "example_idx": idx,
                        "technique": "cot_decoding",
                        "model": resolve_model_name(args.model),
                        "k": int(args.k),
                        "prompt_style": args.prompt_style,
                        "max_new_tokens": int(args.max_new_tokens),
                        "aggregate": args.aggregate,
                        "question": q_text,
                        "text": cand.get("text", ""),
                        "regex_answer": cand.get("regex_answer", "[invalid]"),
                        "regex_score": cand.get("regex_score", -100.0),
                        "bert_answer": cand.get("bert_answer", "[invalid]"),
                        "bert_score": cand.get("bert_score", -100.0),
                    },
                )

        gt = _extract_gt(task, args.task, ex) if args.keep_gt else None

        append_jsonl(
            final_path,
            {
                "run_id": run_dir.name,
                "mode": "dataset_generation",
                "example_idx": idx,
                "model": resolve_model_name(args.model),
                "k": int(args.k),
                "prompt_style": args.prompt_style,
                "max_new_tokens": int(args.max_new_tokens),
                "aggregate": args.aggregate,
                "question": q_text,
                "final_answer": final_answer,
                "gt": gt,
                "meta": final_meta,
            },
        )

        out_ex: Dict[str, Any] = {
            "question": q_text,
            "model_answer": final_answer,
        }
        if args.keep_gt:
            out_ex["gt"] = gt
        if args.keep_source_fields:
            out_ex["source"] = ex

        append_jsonl(gen_path, out_ex)

        if args.log_every and (idx + 1) % int(args.log_every) == 0:
            print(f"[data] processed {idx+1}/{len(examples)}")


    if all_results:
        agg_mode = "self_consistency" if args.aggregate == "self_consistency" else "sum" if args.aggregate == "sum" else "self_consistency"

        export_strategies_to_jsonl(
            results=all_results,
            questions_text=all_questions_text,
            out_dir=str(run_dir),
            aggregation=agg_mode
        )
        print(f"[run] saved 6 strategy jsonls in: {run_dir}")

    print(f"[run] saved logs: {logs_path}")
    print(f"[run] saved finals: {final_path}")
    print(f"[run] saved generated dataset: {gen_path}")
    return 0


def run_dataset_evaluation(args) -> int:
    run_dir = make_run_dir("runs")
    print(f"[run] saving outputs to: {run_dir}")

    task = build_task(args.task, args.prompt_style)

    original = load_hf_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        limit=args.limit,
    )
    if not original:
        raise ValueError("Original dataset is empty. Check --dataset_name/--dataset_split/--limit.")

    gt_norm = []
    questions_norm = []
    for ex in original:
        q_text = _extract_question(args.task, ex)
        gt = _extract_gt(task, args.task, ex)
        gt_n = _normalize_pred_for_eval(task, gt)
        gt_norm.append(gt_n)
        questions_norm.append(q_text)

    all_metrics = {
        "task": args.task,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "limit": args.limit,
        "pred_answer_field": args.pred_answer_field,
        "files": []
    }

    for pred_path in args.pred_jsonls:
        rows = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            print(f"[warn] {pred_path}: empty file, skipping.")
            continue

        n = min(len(rows), len(gt_norm))
        if args.limit is not None:
            n = min(n, int(args.limit))

        correct = 0
        total = 0

        per_item_path = run_dir / (Path(pred_path).stem + "__eval.jsonl")

        for i in range(n):
            r = rows[i]
            pred_raw = r.get(args.pred_answer_field)

            if pred_raw is None:
                pred_raw = r.get("model_answer") or r.get("final_answer") or r.get("answer") or "[invalid]"

            pred_n = _normalize_pred_for_eval(task, str(pred_raw))
            ok = int(pred_n == gt_norm[i])
            correct += ok
            total += 1

            append_jsonl(
                per_item_path,
                {
                    "idx": i,
                    "question": questions_norm[i],
                    "gt_norm": gt_norm[i],
                    "pred_raw": pred_raw,
                    "pred_norm": pred_n,
                    "correct": ok,
                },
            )

        acc = correct / max(total, 1)

        file_metrics = {
            "pred_jsonl": pred_path,
            "n_compared": total,
            "correct": correct,
            "accuracy": acc,
            "per_item_eval": str(per_item_path),
        }

        all_metrics["files"].append(file_metrics)

        print(f"[eval] {pred_path} | acc={acc:.4f} ({correct}/{total})")

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"[run] saved metrics: {metrics_path}")
    return 0

# -------------------- CLI -------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project runner (CoT decoding, datasets, finetuning).")

    p.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "dataset_generation", "dataset_evaluation", "finetuning"],
        help="Execution mode.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name on HuggingFace Hub (e.g.: ChilleD/MultiArith, gsm8k)",
    )

    p.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="HF dataset config (e.g.: main for gsm8k).",
    )

    p.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split (train|test|validation).",
    )
    p.add_argument("--model", type=str, default="mistral-7b-instruct")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--prompt_style", choices=["qa","instruct","plain","train"], default="qa")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--aggregate", type=str, default="sum", choices=["max", "sum", "self_consistency"])
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--task", type=str, default="gsm")

    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--no_qa", action="store_true")

    p.add_argument("--stop_strings", type=str, nargs="*", default=None)

    # Dataset args
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log_every", type=int, default=0)

    # generation extras
    p.add_argument("--output_name", type=str, default="generated_dataset.jsonl")
    p.add_argument("--keep_gt", action="store_true")
    p.add_argument("--keep_source_fields", action="store_true")

    p.add_argument(
        "--pred_jsonls",
        type=str,
        nargs="*",
        default=None,
        help="List of generated JSONL paths (e.g.: dataset_greedy_regex.jsonl dataset_cot_max_regex.jsonl ...).",
    )

    p.add_argument(
        "--pred_answer_field",
        type=str,
        default="answer",
        help="Field in the JSONL that contains the model answer (default: answer). Common alternative: model_answer",
    )

    p.add_argument("--adapter_dir", type=str, default=None,
               help="Directory of the trained LoRA/PEFT adapter (e.g.: finetuned/2tft_two_token).")

    p.add_argument("--merge_lora", action="store_true",
                help="If set, merges the LoRA on load (merge_and_unload).")


    # ---- TRAINING ARGS ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_model", type=str)
    p.add_argument("--train_jsonl", type=str)
    p.add_argument("--out_dir", type=str)

    p.add_argument(
        "--mask_mode",
        choices=["full","two_token","three_token","first_token","eos_only", "random_two", "answer_span"],
        default="two_token"
    )

    p.add_argument("--question_field", default="question")
    p.add_argument("--reasoning_field", default="reasoning")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)

    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--max_length", default=512)
    p.add_argument("--warmup_ratio", default=0.1)
    p.add_argument("--logging_steps", default=5)

    p.add_argument("--save_strategy", choices=["steps","epoch"], default="epoch")
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--neftune_noise_alpha", type=float, default=0.0,
                    help="0 disables it. Example: 5.0 for small datasets.")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "prompt":
        if not args.question:
            parser.error("--question is required in prompt mode.")
        return run_prompt_mode(args)

    if args.mode == "dataset_generation":
        return run_dataset_generation(args)

    if args.mode == "dataset_evaluation":
        if not args.pred_jsonls:
            parser.error("--pred_jsonls is required in dataset_evaluation mode.")
        if not args.dataset_name:
            parser.error("--dataset_name is required in dataset_evaluation mode.")
        return run_dataset_evaluation(args)

    if args.mode == "finetuning":
        return run_training(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

