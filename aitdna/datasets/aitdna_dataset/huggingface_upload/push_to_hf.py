import os
import json
import datasets
from datasets import Dataset

AITDNA_ROOT = "AITDNA_new"
NOTIONS = ["original", "sentence", "document", "boundary", "intent", "content", "span", "membership"] 

def get_path(task_path, notion):
    match notion:
        case "sentence":
            return os.path.join(task_path, "notions", "final_text_by_user_sentence_level.json")
        case "document":
            return os.path.join(task_path, "notions", "final_text_by_user_document_level.json")
        case "boundary":
            return os.path.join(task_path, "notions", "boundary_level",
                                "final_text_by_user_boundary_level_ilp_5seg_1lp_1ip.json")
        case "intent":
            return os.path.join(task_path, "notions", "final_text_by_user_intent_based.json")
        case "content":
            return os.path.join(task_path, "notions", "final_text_by_user_content_based.json")
        case "membership":
            return os.path.join(task_path, "notions", "final_text_by_user_membership_based.json")
        case "span":
            return os.path.join(task_path, "notions", "final_text_by_user_span_level.json")
        case "original":
            return os.path.join(task_path, "edits.json")

def collect_examples(notion):
    all_data = []
    for study in sorted(os.listdir(AITDNA_ROOT)):
        for user in sorted(os.listdir(os.path.join(AITDNA_ROOT, study))):
            for task in sorted(os.listdir(os.path.join(AITDNA_ROOT, study, user))):
                task_path = os.path.join(AITDNA_ROOT, study, user, task)
                path = get_path(task_path, notion)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metadata_path = os.path.join(task_path, "statistics", "metadata.json")
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                all_data.append({"data": data, "metadata": meta})
    return all_data

metadata = datasets.Features({
        "author": datasets.Value("string"),
        "human_only": datasets.Value("bool"),
        "model": datasets.Value("string"),
        "temperature": datasets.Value("float32"),
        "setting": datasets.Value("string"),
        "task": datasets.Value("string")
    })

features = datasets.Features(
    {"data": [
        {
            "text": datasets.Value("string"),
            "author": datasets.Value("string"),
            "queries": [datasets.Value("string")]
        }
    ],
    "metadata": metadata
    }
)
original_features = datasets.Features({
    "data": [datasets.Json()],
    "metadata": metadata
})

for notion in NOTIONS:
    print(f"Processing {notion}...")
    examples = collect_examples(notion)
    if notion == "original":
        ds = Dataset.from_list(examples, features=original_features)
    else:
        ds = Dataset.from_list(examples, features=features)
    ds.push_to_hub("marinajim/AITDNA", config_name=notion, split="test")
    print(f"  pushed {len(ds)} examples")