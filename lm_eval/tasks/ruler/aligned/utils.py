"""Ruler aligned utils — adds GPT-OSS channel stripping to Ruler scoring."""

import logging
import re
from typing import Union

from lm_eval.tasks._gptoss_utils import extract_final_channel
from lm_eval.tasks.ruler.common_utils import (
    DEFAULT_SEQ_LENGTHS,
    postprocess_pred,
    string_match_all,
    string_match_part,
    aggregate_metrics,
)

eval_logger = logging.getLogger(__name__)


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Process results with GPT-OSS channel stripping."""
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    # Strip GPT-OSS channel format before scoring
    stripped = [extract_final_channel(r) for r in results]
    pred = postprocess_pred(stripped)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    """Process results (partial match) with GPT-OSS channel stripping."""
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    # Strip GPT-OSS channel format before scoring
    stripped = [extract_final_channel(r) for r in results]
    pred = postprocess_pred(stripped)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics
