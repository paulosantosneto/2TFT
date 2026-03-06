from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re
from abc import ABC, abstractmethod
from datasets import load_dataset

class MathTaskBase(ABC):

    @abstractmethod
    def encode_prompt(self, example):
        pass

    @abstractmethod
    def extract_gt_answer(self, example):
        pass

    @abstractmethod
    def extract_model_answer(self, text):
        pass

    @abstractmethod
    def is_correct(self, gt_example, model_answer):
        pass

import re


class GSMTask(MathTaskBase):
    # partially adapted from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
    GT_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")

    def __init__(self, encode_format='qa'):
        self.encode_format = encode_format

    def encode_prompt(self, example):
        return f"Q: {example['question']}\nA:"

    def extract_gt_answer(self, example):
        match = self.GT_ANS_RE.search(example["answer"])
        if match:
            return match.group(1).replace(",", "")
        return self.INVALID_ANS

    def extract_model_answer(self, text):
        matches = list(re.finditer(self.MODEL_ANS_RE, text))
        if matches:
            match = matches[-1]
            return match.group(), (match.start(), match.end())
        return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        gt = self.extract_gt_answer(gt_example)
        return gt == model_answer

class MultiArithTask:

    INVALID_ANS = "[invalid]"

    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")

    def encode_prompt(self, example):
        return f"Q: {example['question']}\nA:"

    def extract_gt_answer(self, example):
        return str(example["final_ans"])

    def extract_model_answer(self, text):
        matches = list(re.finditer(self.MODEL_ANS_RE, text))
        if matches:
            match = matches[-1]
            return match.group(), (match.start(), match.end())

        return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        gt = self.extract_gt_answer(gt_example)
        return gt == model_answer


class SVAMPTask(MathTaskBase):

    INVALID_ANS = "[invalid]"
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")

    def encode_prompt(self, example):
        q = example["question"]
        return f"Q: {q}\nA:"

    def extract_gt_answer(self, example):
        return str(example["Answer"])

    def extract_model_answer(self, text):
        matches = list(re.finditer(self.MODEL_ANS_RE, text))
        if matches:
            match = matches[-1]
            return match.group(), (match.start(), match.end())
        return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        return self.extract_gt_answer(gt_example) == model_answer


def _read_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_dataset_file(
    path: Union[str, Path],
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)

    if path.suffix.lower() == ".json":
        data = _read_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if split is not None:
                if split not in data:
                    raise KeyError(
                        f"Split '{split}' does not exist in the file. Available splits: {list(data.keys())}"
                    )
                if not isinstance(data[split], list):
                    raise TypeError(f"Split '{split}' is not a list.")
                return data[split]

            for key in ("test", "validation", "val", "dev", "train"):
                if key in data and isinstance(data[key], list):
                    return data[key]

            for v in data.values():
                if isinstance(v, list):
                    return v
            raise TypeError("JSON dict does not contain any split in list format.")
        raise TypeError("Invalid JSON format (expected list or dict).")

    raise ValueError(f"Unsupported extension: {path.suffix} (use .jsonl or .json)")


def _to_str_number(x: Any) -> str:
    if x is None:
        return "[invalid]"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)
    return str(x).strip()


def normalize_examples_for_task(
    task_name: str,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    t = (task_name or "").lower()
    out: List[Dict[str, Any]] = []

    if t in ("gsm", "gsm8k"):
        for ex in examples:
            if "question" in ex and "answer" in ex:
                out.append(ex)
                continue

            q = ex.get("question") or ex.get("input") or ex.get("prompt")
            a = ex.get("answer") or ex.get("output") or ex.get("target")
            if q is None or a is None:
                continue
            out.append({"question": q, "answer": a})
        return out

    if t in ("svamp",):
        for ex in examples:
            if all(k in ex for k in ("Body", "Question", "Answer")):
                out.append(ex)
                continue

            body = ex.get("Body") or ex.get("body") or ""
            ques = ex.get("Question") or ex.get("question") or ex.get("query")
            ans = ex.get("Answer") or ex.get("answer") or ex.get("target")

            if ques is None or ans is None:
                continue
            out.append({"Body": body, "Question": ques, "Answer": ans})
        return out

    if t in ("multiarith", "multi_arith"):
        for ex in examples:
            if "question" in ex and "final_ans" in ex:
                ex2 = dict(ex)
                ex2["final_ans"] = _to_str_number(ex2["final_ans"])
                out.append(ex2)
                continue

            q = ex.get("question") or ex.get("query") or ex.get("input")
            fa = ex.get("final_ans")
            if fa is None:
                fa = ex.get("answer") or ex.get("target") or ex.get("label")

            if q is None or fa is None:
                continue

            out.append({"question": q, "final_ans": _to_str_number(fa)})
        return out

    return examples


def load_task_dataset(
    task_name: str,
    path: Union[str, Path],
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:

    raw = load_dataset_file(path, split=split)
    norm = normalize_examples_for_task(task_name, raw)
    if limit is not None:
        norm = norm[: int(limit)]
    return norm
