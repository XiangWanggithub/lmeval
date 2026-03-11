"""LiveCodeBench v6 aligned benchmark for lm-eval-harness.

Evalscope-aligned implementation:
- Same dataset: livecodebench/code_generation_lite (release_latest)
- Same prompt template: question + format instructions
- Same code extraction: last fenced code block
- Same execution engine: stdio-based + call-based with timeout
- Same scoring: pass@1 (binary — all test cases must pass)

Plus GPT-OSS channel stripping for offline lm-eval mode.
"""

import base64
import json
import multiprocessing
import pickle  # Required by upstream LiveCodeBench dataset format (compressed test cases)
import re
import zlib
from typing import Dict, List

import datasets
import numpy as np

from lm_eval.tasks._gptoss_utils import extract_final_channel


# ---------------------------------------------------------------------------
# Prompt constants (from evalscope)
# ---------------------------------------------------------------------------

_FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)
_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows."
)

_TIMEOUT = 6  # seconds per test case (same as evalscope default)


# ---------------------------------------------------------------------------
# Dataset preprocessing (process_docs)
# ---------------------------------------------------------------------------

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Decompress test cases and build evaluation samples."""

    def _transform(doc):
        # Build format prompt
        if doc["starter_code"]:
            format_prompt = f"### Format: {_FORMATTING_WITH_STARTER_CODE}\n"
            format_prompt += f"```python\n{doc['starter_code']}\n```\n\n"
        else:
            format_prompt = f"### Format: {_FORMATTING_WITHOUT_STARTER_CODE}\n"
            format_prompt += "```python\n# YOUR CODE HERE\n```\n\n"

        # Load public test cases
        public_test_cases = json.loads(doc["public_test_cases"])

        # Load private test cases (may be compressed via pickle+zlib+base64
        # by the upstream LiveCodeBench dataset — this is their standard format)
        private_raw = doc["private_test_cases"]
        try:
            private_test_cases = json.loads(private_raw)
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(  # noqa: S301 — upstream LiveCodeBench dataset format
                    zlib.decompress(base64.b64decode(private_raw.encode("utf-8")))
                )
            )

        # Load metadata for fn_name
        metadata = json.loads(doc["metadata"])

        # Build evaluation sample (same format as evalscope)
        all_tests = public_test_cases + private_test_cases
        evaluation_sample = json.dumps({
            "inputs": [t["input"] for t in all_tests],
            "outputs": [t["output"] for t in all_tests],
            "fn_name": metadata.get("func_name", None),
        })

        return {
            "format_prompt": format_prompt,
            "evaluation_sample": evaluation_sample,
            # Keep original fields
            "question_content": doc["question_content"],
            "question_title": doc.get("question_title", ""),
            "contest_date": doc.get("contest_date", ""),
            "platform": doc.get("platform", ""),
            "difficulty": doc.get("difficulty", ""),
            "starter_code": doc["starter_code"],
        }

    return dataset.map(_transform)


# ---------------------------------------------------------------------------
# Prompt formatting (doc_to_text)
# ---------------------------------------------------------------------------

def doc_to_text(doc: dict) -> str:
    """Build the prompt — same template as evalscope."""
    return (
        f"### Question:\n{doc['question_content']}\n\n"
        f"{doc['format_prompt']}"
        f"### Answer: (use the provided format with backticks)\n\n"
    )


# ---------------------------------------------------------------------------
# Code extraction (from evalscope extract_utils.py)
# ---------------------------------------------------------------------------

def extract_code(model_output: str) -> str:
    """Extract code from the last fenced code block in model output.

    Same logic as evalscope: find backtick-delimited blocks, take the last one.
    Returns empty string if no fenced block found.
    """
    lines = model_output.split("\n")
    index_lines = [i for i, line in enumerate(lines) if "```" in line]
    if len(index_lines) < 2:
        return ""
    return "\n".join(lines[index_lines[-2] + 1 : index_lines[-1]])


# ---------------------------------------------------------------------------
# Code execution (from evalscope evaluate_utils.py)
# ---------------------------------------------------------------------------

def _subprocess_runner(sample, generation, result_list, metadata_list, timeout):
    """Target function for multiprocessing — must be top-level for pickling."""
    from lm_eval.tasks.livecodebench.aligned.testing_util import run_test

    res, metadata = run_test(sample, test=generation, debug=False, timeout=timeout)
    result_list.append(res)
    metadata_list.append(metadata)


def check_correctness(sample: dict, generation: str, timeout: int = _TIMEOUT) -> bool:
    """Execute generated code against test cases. Returns True if all pass."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_subprocess_runner,
        args=(sample, generation, result, metadata_list, timeout),
    )
    p.start()
    # Global timeout: per-test timeout * num_tests + buffer
    in_outs = json.loads(sample["input_output"])
    global_timeout = (timeout + 1) * len(in_outs["inputs"])
    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
        p.join(timeout=5)

    if not result:
        # Global timeout — all tests considered failed
        return False

    # Check if all test results are positive (True)
    return all(r is True or (isinstance(r, (int, float)) and r > 0) for r in result[0])


# ---------------------------------------------------------------------------
# process_results — lm-eval entry point
# ---------------------------------------------------------------------------

def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    """Extract code, execute against test cases, return pass@1."""
    response = extract_final_channel(results[0])
    code = extract_code(response)

    if not code.strip():
        return {"pass_at_1": 0.0}

    # Build sample in the format expected by testing_util
    sample = {"input_output": doc["evaluation_sample"]}

    try:
        passed = check_correctness(sample, code, timeout=_TIMEOUT)
    except Exception:
        passed = False

    return {"pass_at_1": 1.0 if passed else 0.0}
