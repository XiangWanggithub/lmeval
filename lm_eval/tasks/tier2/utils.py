"""Tier 2 fast feedback set — filters datasets to sensitive samples only.

Reads sample IDs from sensitive_samples.json (tier2_set) and filters
each task's dataset to only include degraded + stable_correct_sampled samples.
"""

import json
import logging
import os
from functools import cache, partial

eval_logger = logging.getLogger(__name__)

# Path to sensitive_samples.json — configurable via env var
SENSITIVE_SAMPLES_PATH = os.environ.get(
    "TIER2_SAMPLES_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                 "eval_results", "sensitivity_filtering", "fm_gptq",
                 "sensitive_samples.json")
)


@cache
def _load_tier2_ids():
    """Load tier2 sample IDs from sensitive_samples.json."""
    # Try the configured path first, then fallback to AutoQuant root
    paths_to_try = [
        SENSITIVE_SAMPLES_PATH,
        "/home/w00857628/AutoQuant/eval_results/sensitivity_filtering/fm_gptq/sensitive_samples.json",
    ]
    for path in paths_to_try:
        path = os.path.realpath(path)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            tier2 = data.get("tier2_set", {})
            # Merge degraded + stable_correct_sampled per task
            result = {}
            for category in ["degraded", "stable_correct_sampled"]:
                for task, ids in tier2.get(category, {}).items():
                    result.setdefault(task, set()).update(ids)
            eval_logger.info(f"Loaded Tier 2 IDs from {path}: "
                           f"{sum(len(v) for v in result.values())} samples across {len(result)} tasks")
            return result
    eval_logger.warning(f"Could not find sensitive_samples.json at any of: {paths_to_try}")
    return {}


def filter_tier2(dataset, task_name):
    """Filter dataset to only Tier 2 samples for this task."""
    tier2_ids = _load_tier2_ids()
    ids = tier2_ids.get(task_name, set())
    if not ids:
        eval_logger.warning(f"No Tier 2 IDs for task {task_name}, returning full dataset")
        return dataset
    # Filter by doc index position in the dataset
    filtered = dataset.select([i for i in range(len(dataset)) if i in ids])
    eval_logger.info(f"Tier 2 filter {task_name}: {len(dataset)} → {len(filtered)} samples")
    return filtered


# Generate per-task filter functions
# Each YAML references: process_docs: !function utils.filter_<task_suffix>
def _make_filter(task_name):
    def _filter(dataset):
        return filter_tier2(dataset, task_name)
    _filter.__name__ = f"filter_{task_name}"
    return _filter


# Tasks requiring chaining: base process_docs must run before tier2 filtering.
# When tier2 YAMLs override process_docs, they REPLACE (not chain) the base.
# Any task whose base process_docs (a) creates new fields for doc_to_text, or
# (b) filters/reindexes the dataset (so IDs are relative to filtered dataset)
# must be handled here explicitly.

# GPQA: base process_docs renames raw columns ("Incorrect Answer 1" etc.) into
# choice1-4 fields used by the Jinja doc_to_text template.
def filter_gpqa_diamond_aligned_1x(dataset):
    from lm_eval.tasks.gpqa.aligned import utils as _gpqa_utils
    dataset = _gpqa_utils.process_docs(dataset)
    return filter_tier2(dataset, "gpqa_diamond_aligned_1x")


# LiveCodeBench: base process_docs creates format_prompt field used by doc_to_text.
def filter_livecodebench_v6only_aligned(dataset):
    from lm_eval.tasks.livecodebench.aligned import utils as _lcb_utils
    dataset = _lcb_utils.process_docs(dataset)
    return filter_tier2(dataset, "livecodebench_v6only_aligned")


# MMLU-Pro: base process_docs filters dataset to a single subject category.
# Tier2 IDs are indices into the subject-filtered dataset, so subject filtering
# must run before the tier2 index selection.
def filter_mmlu_pro_math_aligned(dataset):
    from lm_eval.tasks.mmlu_pro.aligned.utils import process_math
    dataset = process_math(dataset)
    return filter_tier2(dataset, "mmlu_pro_math_aligned")


def filter_mmlu_pro_physics_aligned(dataset):
    from lm_eval.tasks.mmlu_pro.aligned.utils import process_physics
    dataset = process_physics(dataset)
    return filter_tier2(dataset, "mmlu_pro_physics_aligned")


def filter_mmlu_pro_computer_science_aligned(dataset):
    from lm_eval.tasks.mmlu_pro.aligned.utils import process_computer_science
    dataset = process_computer_science(dataset)
    return filter_tier2(dataset, "mmlu_pro_computer_science_aligned")


# Pre-generate filter functions for tasks that don't need base process_docs chaining
_TASK_NAMES = [
    "aime24_1x",
    "ifeval_aligned",
    "humaneval_plus_aligned_1x",
    # gpqa_diamond_aligned_1x handled separately above (needs base process_docs chaining)
    # mmlu_pro_* handled separately above (needs subject filter chaining)
    # livecodebench_v6only_aligned handled separately above (needs format_prompt creation)
    "mmlu_redux_formal_logic_aligned",
    "mmlu_redux_econometrics_aligned",
    "mmlu_redux_college_mathematics_aligned",
    "ceval_aligned_advanced_mathematics",
    "ceval_aligned_chinese_language_and_literature",
    "ceval_aligned_logic",
    "bbh_aligned_boolean_expressions",
    "bbh_aligned_causal_judgement",
    "bbh_aligned_date_understanding",
    "bbh_aligned_formal_fallacies",
    "bbh_aligned_logical_deduction_seven_objects",
    "bbh_aligned_multistep_arithmetic_two",
    "bbh_aligned_navigate",
    "bbh_aligned_object_counting",
    "bbh_aligned_tracking_shuffled_objects_seven_objects",
    "bbh_aligned_web_of_lies",
]

for _tn in _TASK_NAMES:
    globals()[f"filter_{_tn}"] = _make_filter(_tn)
