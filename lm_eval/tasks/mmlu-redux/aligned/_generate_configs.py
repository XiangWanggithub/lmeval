"""Generate per-subject YAML configs for MMLU-Redux aligned evaluation."""

import yaml

SUBJECT_GROUPS = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "social_sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology", "human_sexuality",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}


def main():
    all_group_tasks = {}

    for group_name, subjects in SUBJECT_GROUPS.items():
        group_task_names = []
        tag = f"mmlu_redux_{group_name}_aligned"

        for subject in subjects:
            task_name = f"mmlu_redux_{subject}_aligned"
            group_task_names.append(task_name)

            desc_subject = subject.replace("_", " ")
            config = {
                "dataset_name": subject,
                "description": f"The following are multiple choice questions (with answers) about {desc_subject}.\n\n",
                "tag": tag,
                "include": "_default_template_yaml",
                "task": task_name,
                "task_alias": subject,
            }

            with open(f"mmlu_{subject}.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        all_group_tasks[group_name] = group_task_names

    # Write top-level group file
    group_config = {
        "group": "mmlu_redux_aligned",
        "group_alias": "mmlu_redux (aligned)",
        "task": [],
        "aggregate_metric_list": [
            {
                "aggregation": "mean",
                "metric": "exact_match",
                "weight_by_size": True,
            }
        ],
        "metadata": {"version": 1},
    }

    for group_name, task_names in all_group_tasks.items():
        group_config["task"].append({
            "group": group_name,
            "task": [f"mmlu_redux_{group_name}_aligned"],
            "aggregate_metric_list": [
                {
                    "metric": "exact_match",
                    "weight_by_size": True,
                }
            ],
        })

    with open("_mmlu.yaml", "w") as f:
        yaml.dump(group_config, f, default_flow_style=False, allow_unicode=True)

    total = sum(len(v) for v in SUBJECT_GROUPS.values())
    print(f"Generated {total} subject configs + 1 group config")


if __name__ == "__main__":
    main()
