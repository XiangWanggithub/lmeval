"""Aligned IFEval utilities — channel-aware response extraction.

GPT-OSS outputs multi-channel responses:
  <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...

When served via vLLM's /v1/chat/completions endpoint, the harmony parser
separates analysis (→ reasoning_content) from the final response (→ content).
Evalscope uses this API, so it only scores the final channel.

In lm-eval offline mode, the raw output includes all channel tags. This filter
extracts only the "final" channel content to match evalscope's behavior.
"""

import re
from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval import instructions_registry


# ── Channel extraction ──────────────────────────────────────────────

# Pattern: <|end|><|start|>assistant<|channel|>final<|message|>CONTENT
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.DOTALL,
)

# Cleanup: strip any trailing special tokens from the extracted content
_TRAILING_SPECIAL_RE = re.compile(
    r"<\|(end|start|channel|message|return|im_end|endoftext|eot_id)\|>.*$",
    re.DOTALL,
)


def extract_final_channel(response: str) -> str:
    """Extract the 'final' channel content from a GPT-OSS multi-channel response.

    If no channel tags are found, returns the original response unchanged
    (non-GPT-OSS models or already-clean responses).
    """
    m = _FINAL_CHANNEL_RE.search(response)
    if m:
        content = m.group(1)
        # Strip trailing special tokens
        content = _TRAILING_SPECIAL_RE.sub("", content)
        return content.strip()

    # No channel tags — check if the response starts with analysis channel
    # but never reached the final channel (truncated during analysis)
    if "<|channel|>analysis" in response:
        # Model exhausted token budget during analysis — no actual response
        return ""

    # No channel tags at all — return as-is (non-GPT-OSS model)
    return response


# ── Filter for lm-eval filter_list ──────────────────────────────────

def strip_channel_tags(resps, docs):
    """lm-eval filter function: extract final channel from each response."""
    filtered = []
    for resp_group in resps:
        filtered.append([extract_final_channel(r) for r in resp_group])
    return filtered


# ── IFEval scoring (same as upstream, operating on filtered response) ──

import dataclasses


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(inp, response):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp, response):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    # Strip GPT-OSS analysis channel before scoring
    response = extract_final_channel(response)

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
