"""Evalscope-aligned HumanEval-Plus scoring with robust code extraction.

Key difference from stock lm-eval HumanEval:
- Instruction-style prompt (not code completion)
- 3-stage code extraction (fenced -> incomplete fenced -> heuristic)
- Always prepends doc["prompt"] to extracted code for test assembly
  (Python allows redefining functions, so prompt + complete_function works)
- Channel-aware: strips GPT-OSS analysis channel before code extraction
"""

import re
from typing import Union

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(
    references: list[str], predictions: list[list[str]], k: list[int] = None
):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


# ---------------------------------------------------------------------------
# GPT-OSS channel stripping (same as IFEval aligned)
# ---------------------------------------------------------------------------

_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.DOTALL,
)
_TRAILING_SPECIAL_RE = re.compile(
    r"<\|(end|start|channel|message|return|im_end|endoftext|eot_id)\|>.*$",
    re.DOTALL,
)


def extract_final_channel(response: str) -> str:
    """Extract 'final' channel content from GPT-OSS multi-channel output.

    Returns the original response unchanged for non-GPT-OSS models.
    """
    m = _FINAL_CHANNEL_RE.search(response)
    if m:
        content = m.group(1)
        content = _TRAILING_SPECIAL_RE.sub("", content)
        return content.strip()
    if "<|channel|>analysis" in response:
        return ""
    return response


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


def build_predictions(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Extract code and prepend doc prompt for test assembly.

    Evalscope always prepends prompt (function signature) to completion.
    Python allows redefining functions, so prompt + complete_function works:
    the model's definition overrides the incomplete one from prompt.
    """
    return [[doc["prompt"] + extract_code(extract_final_channel(r)) for r in resp] for resp, doc in zip(resps, docs)]
