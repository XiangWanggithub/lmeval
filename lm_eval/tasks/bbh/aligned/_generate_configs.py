"""Generate per-subtask YAML configs for BBH aligned evaluation.

Embeds evalscope's curated 3-shot CoT prompts as the description field,
so they are prepended to every test question.
"""

import os
import yaml

TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# Curated CoT prompts (copied from evalscope)
COT_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cot_prompts")

PROMPT = (
    "Q: {{input}}\n"
    "A: Let's think step by step. "
    'Put your final answer in the format of "So the answer is [ANSWER]" '
    "(without quotes) where [ANSWER] is your answer."
)


def load_cot_prompt(task_name: str) -> str:
    """Load CoT prompt from evalscope's cot_prompts directory."""
    cot_path = os.path.join(COT_PROMPTS_DIR, f"{task_name}.txt")
    if os.path.exists(cot_path):
        with open(cot_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def main():
    group_tasks = []

    for task_name in TASKS:
        yaml_task_name = f"bbh_aligned_{task_name}"
        group_tasks.append(yaml_task_name)

        # Load curated 3-shot CoT prompt from evalscope
        cot_prompt = load_cot_prompt(task_name)

        if cot_prompt:
            # Append the answer format instruction to each CoT example's A: line
            # The CoT prompt already ends with "So the answer is X." for each example
            # We add two newlines to separate from the current question
            description = cot_prompt + "\n\n"
        else:
            description = ""

        config = {
            "dataset_name": task_name,
            "description": description,
            "doc_to_text": PROMPT,
            "include": "_bbh_aligned_template_yaml",
            "task": yaml_task_name,
        }

        with open(f"{task_name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                      width=10000)

    # Write group file
    group_config = {
        "group": "bbh_aligned",
        "task": group_tasks,
        "aggregate_metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "weight_by_size": True,
            }
        ],
        "metadata": {"version": 2.0},
    }

    with open("_bbh_aligned.yaml", "w") as f:
        yaml.dump(group_config, f, default_flow_style=False, allow_unicode=True)

    print(f"Generated {len(TASKS)} subtask configs + 1 group config")


if __name__ == "__main__":
    main()
