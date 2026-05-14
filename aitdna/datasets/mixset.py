import random
import os
import json
import argparse

from .aitdna_dataset.processing.format_data import get_and_save_notions,\
    get_and_save_final_text, create_folders_for_analysis
from datasets import load_dataset
from fast_diff_match_patch import diff


def recreate_edits_fast_diff(original: str, revised: str):
    """
    Takes the difference between original and revised version, and returns the edits.
    
    :param diff: quill diff delta
    :type diff: list[dict]
    """
    delta = diff(original, revised)
    current_index_original = 0
    current_index_revised = 0
    
    final_entries = [{
        "user": "User",
        "operationType": "insert",
        "text": original,
        "offset": 0,
        "span": len(original)
    }]

    for (op, length) in delta:
        final_entry = {}
        if op == "=":
            current_index_original += length
            current_index_revised += length
        if op == "+":
            final_entry["user"] = "Bot"
            final_entry["operationType"] = "insert"
            final_entry["text"] = revised[current_index_revised:current_index_revised+length]
            final_entry["offset"] = current_index_original
            final_entry["span"] = length
            current_index_revised += length
            current_index_original += length
        if op == "-":
            final_entry["user"] = "Bot"
            final_entry["operationType"] = "delete"
            final_entry["text"] = None
            final_entry["offset"] = current_index_original
            final_entry["span"] = length
        if final_entry:
            final_entries.append(final_entry)
    return final_entries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dst_root', type=str, default="data/other_datasets/processed/mixset")
    args = parser.parse_args(argv)

    DST_ROOT = args.dst_root
    random.seed(444)
    mixset = load_dataset("ONE-Lab/MixSet")

    for split in ("train", "test"):
        data = mixset[split]
        for i, data_point in enumerate(data):
            folder_name = split + "_" + str(i)
            dst_folder = os.path.join(DST_ROOT, folder_name)

            if not os.path.exists(DST_ROOT):
                os.mkdir(DST_ROOT)
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)

            edits = recreate_edits_fast_diff(data_point["original"], data_point["revised"])
            stats_path, notions_path, boundary_path = create_folders_for_analysis(dst_folder)

            dst_file_path = os.path.join(dst_folder, "edits.json")
            with open(dst_file_path, "w", encoding="utf-8") as f:
                json.dump(edits, f)

            _, _, _, text_by_user, _ = get_and_save_notions(edits,
                                                                    boundary_path,
                                                                    notions_path)
        
            get_and_save_final_text(text_by_user, dst_folder)