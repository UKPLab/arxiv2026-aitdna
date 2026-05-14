import os
import json
import argparse
import pandas as pd

from .aitdna_dataset.processing.format_data import get_and_save_notions,\
    get_and_save_final_text, create_folders_for_analysis


def get_final_data(src_path):
    df = pd.read_excel(src_path)
    final_data = []
    for essay, essay_id, author, split in zip(df["sent_and_label"], df["essay_id"], df["author_seq"], df["train_ix"]):
        essay_data = []
        essay_length = 0
        evaluated_essay = eval(essay)
        for j, (sent, label) in enumerate(evaluated_essay):
            text = sent
            if j != len(evaluated_essay) - 1:
                text += " "
            user = "User" if label == "human" else "Bot"
            offset = essay_length
            span = len(text)
            edit = {
                "user": user,
                "operationType": "insert",
                "offset": offset,
                "text": text,
                "span": span
            }
            essay_length += len(text)
            essay_data.append(edit)
        final_data.append({
                "essay_id": essay_id,
                "authorSeq": author,
                "split": split,
                "edits": essay_data
            })
    return final_data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--src_path', type=str,
                        default="data/other_datasets/original/boundary_detection/data.xlsx")
    parser.add_argument("-d", '--dst_root', type=str, default="data/other_datasets/processed/boundary_detection")
    args = parser.parse_args(argv)

    SRC_PATH = args.src_path
    DST_ROOT = args.dst_root
    if not os.path.exists(DST_ROOT):
        os.mkdir(DST_ROOT)

    essays = get_final_data(src_path=SRC_PATH)

    for i, essay in enumerate(essays):

        dst_folder = os.path.join(DST_ROOT, str(i))

        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        stats_path, notions_path, boundary_path = create_folders_for_analysis(dst_folder)
        dataset_related_stats = {
            "essay_id": essay["essay_id"],
            "authorSeq": essay["authorSeq"],
            "split": essay["split"]
        }
        with open(os.path.join(stats_path, "dataset_related_stats.json"), "w") as f:
            json.dump(dataset_related_stats, f)
        edits = essay["edits"]
        dst_file_path = os.path.join(dst_folder, "edits.json")
        with open(dst_file_path, "w", encoding="utf-8") as f:
            json.dump(edits, f)

        _, _, _, text_by_user, _ = get_and_save_notions(edits,
                                                                boundary_path,
                                                                notions_path)

        get_and_save_final_text(text_by_user, dst_folder)