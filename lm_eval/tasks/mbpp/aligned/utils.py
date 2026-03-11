"""Evalscope-aligned MBPP-Plus scoring with robust code extraction.

Implements evalscope's 3-stage code extraction pipeline:
1. Complete fenced code blocks (```python ... ```)
2. Incomplete fenced code blocks (``` without closing)
3. Heuristic extraction (detect def/class patterns)

Plus post-processing: strip [DONE], stop at \\nassert/\\n\"\"\", remove __main__.
"""

import re
from typing import Union

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


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
_HEURISTIC_PATTERN = re.compile(
    r"(?:^(?:import|from|#)[^\n]+\n)*"
    r"^(?:def|class) [^\n]+\n"
    r"(?:\s+[^\n]+\n)+",
    re.MULTILINE,
)
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
            # Stage 3: heuristic — detect def/class patterns
            heuristic_matches = _HEURISTIC_PATTERN.findall(completion)
            if heuristic_matches:
                code = heuristic_matches[0]
            else:
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


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Filter function for lm-eval: extract code from each response."""
    return [[extract_code(r) for r in resp] for resp in resps]
