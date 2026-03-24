"""
Standalone test for the proposed repeats_mode='average' evaluator change.
No conda env, no GPU, no lm_eval imports needed — just plain Python.

Run with:  python test_repeats_average.py
"""
from collections import defaultdict
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# The exact new logic extracted from the evaluator loop (our proposed change)
# ---------------------------------------------------------------------------

def compute_doc_metrics(task, requests, filter_key):
    """
    Drop-in replacement for the current evaluator block at lines 624-627.
    Supports both take_first (existing) and average (new) modes.
    """
    if getattr(task.config, "repeats_mode", "take_first") == "average" \
            and task.config.repeats > 1:
        k = len(requests[0].resps)
        per_repeat = [
            task.process_results(None, [req.resps[i] for req in requests])
            for i in range(k)
        ]
        return {m: sum(r[m] for r in per_repeat) / k for m in per_repeat[0]}
    else:
        return task.process_results(
            None, [req.filtered_resps[filter_key] for req in requests]
        )


# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------

def make_instance(doc_id, resps, filtered_resp):
    """resps = list of k raw responses; filtered_resp = take_first result."""
    inst = SimpleNamespace()
    inst.doc_id = doc_id
    inst.idx = doc_id
    inst.resps = resps                          # list of k strings
    inst.filtered_resps = {"none": filtered_resp}
    return inst


def make_task(repeats, repeats_mode, score_fn):
    """score_fn(doc, [resp]) → dict of metrics."""
    cfg = SimpleNamespace(repeats=repeats, repeats_mode=repeats_mode)
    task = SimpleNamespace(config=cfg)
    task.process_results = score_fn
    return task


# ---------------------------------------------------------------------------
# A realistic scorer: 1.0 if response contains the correct answer, else 0.0
# ---------------------------------------------------------------------------

CORRECT_ANSWER = "B"

def gpqa_score(doc, resps):
    resp = resps[0]  # for generate_until, single response per request
    return {"exact_match": 1.0 if CORRECT_ANSWER in resp else 0.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_take_first_unchanged():
    """Existing behaviour: only the first repeat is scored."""
    task = make_task(repeats=3, repeats_mode="take_first", score_fn=gpqa_score)
    # Doc 0: first repeat correct, others wrong
    inst = make_instance(0, ["The answer is B", "wrong", "also wrong"], "The answer is B")
    metrics = compute_doc_metrics(task, [inst], "none")
    assert metrics["exact_match"] == 1.0, "take_first: should use first (correct) repeat"

    # Doc 1: first repeat wrong, second correct
    inst2 = make_instance(1, ["wrong", "The answer is B", "wrong"], "wrong")
    metrics2 = compute_doc_metrics(task, [inst2], "none")
    assert metrics2["exact_match"] == 0.0, "take_first: should use first (wrong) repeat, ignore second"
    print("PASS  test_take_first_unchanged")


def test_average_mode_basic():
    """Average mode: score each repeat, average the scores."""
    task = make_task(repeats=4, repeats_mode="average", score_fn=gpqa_score)
    # 2 of 4 repeats are correct → average = 0.5
    inst = make_instance(0,
        ["The answer is B", "wrong", "The answer is B", "wrong"],
        "The answer is B"  # filtered_resp (unused in average mode)
    )
    metrics = compute_doc_metrics(task, [inst], "none")
    assert metrics["exact_match"] == 0.5, f"expected 0.5, got {metrics['exact_match']}"
    print("PASS  test_average_mode_basic")


def test_average_mode_all_correct():
    task = make_task(repeats=3, repeats_mode="average", score_fn=gpqa_score)
    inst = make_instance(0, ["B is correct", "Answer: B", "B"], "B is correct")
    metrics = compute_doc_metrics(task, [inst], "none")
    assert metrics["exact_match"] == 1.0
    print("PASS  test_average_mode_all_correct")


def test_average_mode_none_correct():
    task = make_task(repeats=3, repeats_mode="average", score_fn=gpqa_score)
    inst = make_instance(0, ["wrong", "also wrong", "nope"], "wrong")
    metrics = compute_doc_metrics(task, [inst], "none")
    assert metrics["exact_match"] == 0.0
    print("PASS  test_average_mode_none_correct")


def test_average_mode_single_repeat_same_as_take_first():
    """With repeats=1, average mode should behave identically to take_first."""
    task_avg = make_task(repeats=1, repeats_mode="average", score_fn=gpqa_score)
    task_tf  = make_task(repeats=1, repeats_mode="take_first", score_fn=gpqa_score)
    inst = make_instance(0, ["The answer is B"], "The answer is B")
    assert compute_doc_metrics(task_avg, [inst], "none") == \
           compute_doc_metrics(task_tf,  [inst], "none")
    print("PASS  test_average_mode_single_repeat_same_as_take_first")


def test_multiple_docs_aggregation():
    """Simulate the full raw_metrics accumulation loop for 3 docs."""
    task = make_task(repeats=4, repeats_mode="average", score_fn=gpqa_score)
    docs = [
        make_instance(0, ["B", "wrong", "B", "wrong"], "B"),       # avg = 0.5
        make_instance(1, ["B", "B", "B", "B"], "B"),               # avg = 1.0
        make_instance(2, ["wrong", "wrong", "wrong", "wrong"], "wrong"),  # avg = 0.0
    ]
    raw_metrics = defaultdict(list)
    for inst in docs:
        m = compute_doc_metrics(task, [inst], "none")
        for metric, value in m.items():
            raw_metrics[metric].append(value)

    scores = raw_metrics["exact_match"]
    task_accuracy = sum(scores) / len(scores)
    assert scores == [0.5, 1.0, 0.0], f"per-doc scores wrong: {scores}"
    assert abs(task_accuracy - 0.5) < 1e-9, f"task accuracy wrong: {task_accuracy}"
    print(f"PASS  test_multiple_docs_aggregation  (task acc = {task_accuracy:.2f})")


def test_process_results_called_k_times():
    """Verify process_results is called exactly k times per doc in average mode."""
    call_count = {"n": 0}
    def counting_scorer(doc, resps):
        call_count["n"] += 1
        return {"exact_match": 1.0}

    task = make_task(repeats=7, repeats_mode="average", score_fn=counting_scorer)
    inst = make_instance(0, ["r"] * 7, "r")
    compute_doc_metrics(task, [inst], "none")
    assert call_count["n"] == 7, f"expected 7 calls, got {call_count['n']}"
    print("PASS  test_process_results_called_k_times")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_take_first_unchanged()
    test_average_mode_basic()
    test_average_mode_all_correct()
    test_average_mode_none_correct()
    test_average_mode_single_repeat_same_as_take_first()
    test_multiple_docs_aggregation()
    test_process_results_called_k_times()
    print("\nAll tests passed.")
