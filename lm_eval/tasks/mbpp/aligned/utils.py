"""Evalscope-aligned MBPP-Plus scoring with robust code extraction.

Implements evalscope's 3-stage code extraction pipeline:
1. Complete fenced code blocks (```python ... ```)
2. Incomplete fenced code blocks (``` without closing)
3. Heuristic extraction (detect def/class patterns)

Plus post-processing: strip [DONE], stop at \nassert/\n\"\"\", remove __main__.
Channel-aware: strips GPT-OSS analysis channel before code extraction.

Uses process_results (not filter_list + metric_list) so that repeats_mode=average
works correctly. lm_eval's repeats_mode=average path bypasses filters and calls
process_results directly with raw model output — so all extraction logic must live
inside process_results.
"""

import re

import evaluate as hf_evaluate

from lm_eval.tasks._gptoss_utils import extract_final_channel


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
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
_STOP_WORDS = ["\nassert", '\n"""']


def extract_code(completion: str) -> str:
    """Evalscope-aligned 3-stage code extraction for Python."""
    # Strip [DONE] marker if present
    if "[DONE]" in completion:
        completion = completion[: completion.index("[DONE]")]

    code = ""

    # Stage 1: complete fenced code blocks
    matches = _FENCED_PATTERN.findall(completion)
    if matches:
        # Prefer python-tagged blocks
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

    # Post-processing: stop words
    for sw in _STOP_WORDS:
        idx = code.find(sw)
        if idx != -1:
            code = code[:idx]

    # Post-processing: remove if __name__ == "__main__"
    if 'if __name__ ==' in code:
        code = code[: code.index('if __name__ ==')].rstrip()

    # Post-processing: remove "# Example usage"
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
    code = extract_code(response)

    if not code.strip():
        return {"pass_at_1": 0.0}

    # MBPP test reference: concatenated test_list assertions
    test_case = "\n".join(doc["test_list"])

    try:
        res, _ = pass_at_k.compute(
            references=[test_case],
            predictions=[[code]],
            k=[1],
        )
        return {"pass_at_1": res["pass@1"]}
    except Exception:
        return {"pass_at_1": 0.0}
