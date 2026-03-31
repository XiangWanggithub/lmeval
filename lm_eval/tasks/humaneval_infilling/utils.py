import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references, predictions, k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    # lm_eval calls per-doc: references=str, predictions=list[str]
    # code_eval expects: references=list[str], predictions=list[list[str]]
    if isinstance(references, str):
        references = [references]
    if predictions and isinstance(predictions[0], str):
        predictions = [predictions]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0][f"pass@{k[0]}"]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + r + doc["suffix"] for r in resp]
        for resp, doc in zip(resps, docs)
    ]
