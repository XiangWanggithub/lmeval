import re
from typing import Dict, List


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evalscope-aligned MMLU-Redux scoring with cascading letter extraction."""
    response = results[0]
    pred = extract_choice(response)
    target = ["A", "B", "C", "D"][doc["answer"]]
    correct = pred is not None and pred == target
    return {"exact_match": int(correct)}


def extract_choice(response: str) -> str | None:
    """Extract multiple-choice answer A/B/C/D with cascading patterns.

    Priority:
    1. "ANSWER: X" format (evalscope style)
    2. "The answer is (X)" / "the answer is X"
    3. "\\boxed{X}"
    4. Last standalone A/B/C/D in parentheses or after colon
    5. Last single A/B/C/D letter on its own line
    """
    # Pattern 1: ANSWER: X (case-insensitive)
    m = re.search(r"(?i)ANSWER\s*:\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 2: The answer is (X)
    m = re.search(r"[Tt]he answer is\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 3: \boxed{X}
    m = re.search(r"\\boxed\{?\(?([A-D])\)?\}?", response)
    if m:
        return m.group(1).upper()

    # Pattern 4: Last (A)/(B)/(C)/(D) in parentheses
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return matches[-1].upper()

    # Pattern 5: Last standalone letter on its own line or after colon
    m = re.search(r"(?:^|\n)\s*([A-D])\s*\.?\s*$", response, re.MULTILINE)
    matches_line = list(re.finditer(r"(?:^|\n)\s*([A-D])\s*\.?\s*$", response, re.MULTILINE))
    if matches_line:
        return matches_line[-1].group(1).upper()

    return None
