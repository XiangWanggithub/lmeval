"""Ruler aligned utils — adds GPT-OSS channel stripping to Ruler scoring.

Also re-exports dataset functions from parent ruler/ so YAML !function
references resolve from this directory.
"""

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

# Re-export dataset functions, filtering out unhashable kwargs (e.g. chat_template_args)
# that break @cache in get_tokenizer.
from lm_eval.tasks.ruler import niah_utils as _niah
from lm_eval.tasks.ruler import vt_utils as _vt
from lm_eval.tasks.ruler import cwe_utils as _cwe
from lm_eval.tasks.ruler import fwe_utils as _fwe
from lm_eval.tasks.ruler import qa_utils as _qa


def _clean_kwargs(kwargs):
    """Remove unhashable values that break @cache in get_tokenizer."""
    return {k: v for k, v in kwargs.items() if not isinstance(v, (dict, list, set))}


def niah_single_1(**kwargs):
    return _niah.niah_single_1(**_clean_kwargs(kwargs))

def niah_single_2(**kwargs):
    return _niah.niah_single_2(**_clean_kwargs(kwargs))

def niah_single_3(**kwargs):
    return _niah.niah_single_3(**_clean_kwargs(kwargs))

def niah_multikey_1(**kwargs):
    return _niah.niah_multikey_1(**_clean_kwargs(kwargs))

def niah_multikey_2(**kwargs):
    return _niah.niah_multikey_2(**_clean_kwargs(kwargs))

def niah_multikey_3(**kwargs):
    return _niah.niah_multikey_3(**_clean_kwargs(kwargs))

def niah_multiquery(**kwargs):
    return _niah.niah_multiquery(**_clean_kwargs(kwargs))

def niah_multivalue(**kwargs):
    return _niah.niah_multivalue(**_clean_kwargs(kwargs))

def get_vt_dataset(**kwargs):
    return _vt.get_vt_dataset(**_clean_kwargs(kwargs))

def get_cw_dataset(**kwargs):
    return _cwe.get_cw_dataset(**_clean_kwargs(kwargs))

def fwe_download(**kwargs):
    return _fwe.fwe_download(**_clean_kwargs(kwargs))

def get_squad(**kwargs):
    return _qa.get_squad(**_clean_kwargs(kwargs))

def get_hotpotqa(**kwargs):
    return _qa.get_hotpotqa(**_clean_kwargs(kwargs))

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
