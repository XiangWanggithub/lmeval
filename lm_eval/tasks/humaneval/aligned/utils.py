"""Evalscope-aligned HumanEval-Plus scoring with robust code extraction.

Key difference from stock lm-eval HumanEval:
- Instruction-style prompt (not code completion)
- 3-stage code extraction (fenced -> incomplete fenced -> heuristic)
- Always prepends doc["prompt"] to extracted code for test assembly
  (Python allows redefining functions, so prompt + complete_function works)
- Channel-aware: strips GPT-OSS analysis channel before code extraction

Uses process_results (not filter_list + metric_list) so that repeats_mode=average
works correctly. lm_eval's repeats_mode=average path bypasses filters and calls
process_results directly with raw model output — so all extraction logic must live
inside process_results.
"""

import re

import evaluate as hf_evaluate

from lm_eval.tasks._gptoss_utils import extract_final_channel


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


# ---------------------------------------------------------------------------
# 3-stage code extraction (aligned with evalscope)
# ---------------------------------------------------------------------------

_FENCED_PATTERN = re.compile(
    r"```([^\n]*)\n(.*?)\n\s*```", re.DOTALL | re.MULTILINE
)
_INCOMPLETE_FENCED_PATTERN = re.compile(
    r"```([^\n]*)\n(.*)", re.DOTALL | re.MULTILINE
)
_PYTHON_ALIASES = {"python", "Python", "py", "Python3", "python3", "PY"}


def extract_code(completion: str) -> str:
    """Evalscope-aligned 3-stage code extraction for Python."""
    code = ""

    # Stage 1: complete fenced code blocks
    matches = _FENCED_PATTERN.findall(completion)
    if matches:
        for lang, content in matches:
            if lang.strip() in _PYTHON_ALIASES:
                code = content
                break
        if not code:
            code = matches[0][1]
    else:
        # Stage 2: incomplete fenced code blocks
        matches = _INCOMPLETE_FENCED_PATTERN.findall(completion)
        if matches:
            for lang, content in matches:
                if lang.strip() in _PYTHON_ALIASES:
                    code = content
                    break
            if not code:
                code = matches[0][1]
        else:
            # Stage 3: bare code — use as-is
            # No fenced blocks found. After channel stripping, the output
            # is clean Python code. The old heuristic regex failed on blank
            # lines within function bodies. Just use the full text.
            code = completion

    code = code.replace("\r", "")

    # Remove if __name__ == "__main__"
    if 'if __name__ ==' in code:
        code = code[: code.index('if __name__ ==')].rstrip()

    # Remove "# Example usage"
    if "# Example usage" in code:
        code = code[: code.index("# Example usage")].rstrip()

    return code


# ---------------------------------------------------------------------------
# process_results — lm-eval entry point (called once per repeat)
# ---------------------------------------------------------------------------

def process_results(doc: dict, results: list) -> dict:
    """Evaluate one candidate generation for pass@1.

    lm_eval calls this once per (doc, repeat) with the raw model output in
    results[0]. With repeats_mode=average and repeats=N, this is called N times
    per doc; lm_eval then averages the N binary 0/1 results to get the pass rate
    (equivalent to pass@1 estimated with N samples).

    Handles GPT-OSS channel-tagged output transparently.
    """
    response = extract_final_channel(results[0])
    extracted = extract_code(response)
    # Prepend the function signature/docstring so the definition is complete.
    # Python allows redefining functions — the model's definition overrides the
    # partial one from the prompt.
    code = doc["prompt"] + extracted

    if not extracted.strip():
        return {"pass_at_1": 0.0}

    test_case = doc["test"] + "\ncheck(" + doc["entry_point"] + ")"

    try:
        res, _ = compute_.compute(
            references=[test_case],
            predictions=[[code]],
            k=[1],
        )
        return {"pass_at_1": res["pass@1"]}
    except Exception:
        return {"pass_at_1": 0.0}
