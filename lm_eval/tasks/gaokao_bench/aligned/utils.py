"""GaokaoBench aligned benchmark for lm-eval-harness.

Implements the RUCAIBox/gaokao-bench dataset with:
- Generation-based MCQ scoring (not log-likelihood)
- GPT-OSS channel stripping
- Support for both single-answer and multi-answer question formats
- Chinese + English answer extraction patterns

Dataset: https://huggingface.co/datasets/RUCAIBox/gaokao-bench
"""

import re
from typing import Dict, List

import datasets

from lm_eval.tasks._gptoss_utils import extract_final_channel


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SINGLE_ANSWER_PROMPT_ZH = (
    "以下是一道单选题，请直接给出正确答案的选项字母。\n"
    "在回答最后一行，请按如下格式给出答案：\"答案：X\"（X为A/B/C/D中的一个）。\n\n"
    "{question}"
)

_SINGLE_ANSWER_PROMPT_EN = (
    "Answer the following multiple choice question. "
    "The last line of your response should be of the following format: "
    "'ANSWER: X' (without quotes) where X is one of A,B,C,D.\n\n"
    "{question}"
)

_MULTI_ANSWER_PROMPT_ZH = (
    "以下是一道包含多个小题的选择题，请依次给出每小题的正确答案。\n"
    "在回答最后，请按如下格式给出所有答案：\"答案：X, Y, Z\"（每个字母对应一个小题）。\n\n"
    "{question}"
)

_MULTI_ANSWER_PROMPT_EN = (
    "Answer the following passage with multiple sub-questions. "
    "At the end of your response, list all answers in order: "
    "'ANSWER: X, Y, Z' where each letter corresponds to one sub-question.\n\n"
    "{question}"
)


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------

# Configs where all questions have exactly 1 answer
_SINGLE_ANSWER_CONFIGS = {
    "2010-2022_Biology_MCQs",
    "2010-2022_Chemistry_MCQs",
    "2010-2022_History_MCQs",
    "2010-2022_Math_I_MCQs",
    "2010-2022_Math_II_MCQs",
    "2010-2022_Physics_MCQs",
    "2010-2022_Political_Science_MCQs",
    "2010-2013_English_MCQs",
}

# Configs with English-language questions
_ENGLISH_CONFIGS = {
    "2010-2013_English_MCQs",
    "2010-2022_English_Fill_in_Blanks",
    "2010-2022_English_Reading_Comp",
    "2012-2022_English_Cloze_Test",
}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Preprocess GaokaoBench questions for generation-based evaluation."""

    def _transform(doc):
        answer_list = doc["answer"]
        question = doc["question"].strip()
        is_single = len(answer_list) == 1

        # Determine language for prompt
        # Use config name from dataset if available, fallback to content heuristic
        is_english = any(
            keyword in question[:100].lower()
            for keyword in ["read", "choose", "following", "answer"]
        ) and not any(
            ch in question[:50] for ch in "的是在了"
        )

        if is_single:
            template = _SINGLE_ANSWER_PROMPT_EN if is_english else _SINGLE_ANSWER_PROMPT_ZH
        else:
            template = _MULTI_ANSWER_PROMPT_EN if is_english else _MULTI_ANSWER_PROMPT_ZH

        return {
            "prompt": template.format(question=question),
            "answer_list": ",".join(answer_list),  # "D" or "B,C"
            "num_answers": len(answer_list),
            "question": question,
            "year": doc.get("year", ""),
            "category": doc.get("category", ""),
            "score": doc.get("score", 0),
        }

    return dataset.map(_transform)


def doc_to_text(doc: dict) -> str:
    """Return the formatted prompt."""
    return doc["prompt"]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_choice_single(response: str) -> str | None:
    """Extract a single MCQ answer (A/B/C/D) from response.

    Same cascading logic as ceval aligned — handles Chinese and English formats.
    """
    # Pattern 1: 答案：X / 答案: X / 答案是X (Chinese)
    m = re.search(r"答案\s*[：:是]\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 2: ANSWER: X (English)
    m = re.search(r"(?i)ANSWER\s*:\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 3: The answer is (X)
    m = re.search(r"[Tt]he answer is\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 4: \boxed{X}
    m = re.search(r"\\boxed\{?\(?([A-D])\)?\}?", response)
    if m:
        return m.group(1).upper()

    # Pattern 5: Last (A)/(B)/(C)/(D) in parentheses
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return matches[-1].upper()

    # Pattern 6: Last standalone letter on its own line
    matches_line = list(
        re.finditer(r"(?:^|\n)\s*([A-D])\s*\.?\s*$", response, re.MULTILINE)
    )
    if matches_line:
        return matches_line[-1].group(1).upper()

    return None


def extract_choices_multi(response: str, expected_count: int) -> list[str | None]:
    """Extract multiple MCQ answers from a response with numbered sub-questions.

    Strategies (in priority order):
    1. "答案：A, B, C" or "ANSWER: A, B, C" — comma-separated list
    2. Per-sub-question patterns: "1. A", "第1题：A", etc.
    3. Fall back to finding all standalone letters in order
    """
    answers = [None] * expected_count

    # Strategy 1: Comma/space-separated answer list
    # Match: 答案：A, B, C  or  答案：A B C  or  ANSWER: A, B, C
    m = re.search(
        r"(?:答案\s*[：:是]|[Aa][Nn][Ss][Ww][Ee][Rr]\s*:)\s*([A-D](?:\s*[,，、\s]\s*[A-D])*)",
        response,
    )
    if m:
        letters = re.findall(r"[A-D]", m.group(1))
        if len(letters) == expected_count:
            return [l.upper() for l in letters]

    # Strategy 2: Numbered sub-question answers
    # Match patterns like: "1. A", "1、A", "1：A", "第1题 A", "(1) A"
    numbered = re.findall(
        r"(?:(?:第?\s*(\d+)\s*[.、：:题)）]\s*)|(?:\((\d+)\)\s*))([A-D])",
        response,
    )
    if numbered:
        for match in numbered:
            idx = int(match[0] or match[1]) - 1
            if 0 <= idx < expected_count:
                answers[idx] = match[2].upper()
        if all(a is not None for a in answers):
            return answers

    # Strategy 3: Find all standalone A-D letters in order
    all_letters = re.findall(r"(?:^|\s)\(?([A-D])\)?(?:\s|$|[.。，,])", response, re.MULTILINE)
    if len(all_letters) >= expected_count:
        # Take the last N letters (model's final answers)
        return [l.upper() for l in all_letters[-expected_count:]]

    return answers


# ---------------------------------------------------------------------------
# process_results — lm-eval entry point
# ---------------------------------------------------------------------------

def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    """Extract answer(s) from generation and compute accuracy."""
    response = extract_final_channel(results[0])
    answer_list = doc["answer_list"].split(",")
    num_answers = doc["num_answers"]

    if num_answers == 1:
        pred = extract_choice_single(response)
        correct = pred is not None and pred == answer_list[0]
        return {"exact_match": 1.0 if correct else 0.0}
    else:
        preds = extract_choices_multi(response, num_answers)
        correct_count = sum(
            1 for pred, gold in zip(preds, answer_list)
            if pred is not None and pred == gold
        )
        return {"exact_match": correct_count / num_answers}
