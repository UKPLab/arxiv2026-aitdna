import random
import os
import json
import argparse

from .aitdna.processing.format_data import get_and_save_notions,\
    get_and_save_final_text, create_folders_for_analysis
from .mixset import recreate_edits_fast_diff


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--src_root', type=str, default="data/other_datasets/original/detectRL")
    parser.add_argument("-d", '--dst_root', type=str, default="data/other_datasets/processed/detectRL")
    args = parser.parse_args(argv)

    random.seed(444)

    SRC_ROOT = args.src_root
    DST_ROOT = args.dst_root

    for dataset in os.listdir(SRC_ROOT):
        dataset_name = dataset.replace("_2800.json", "")
        human_key = ""
        if dataset_name == "arxiv":
            human_key = "abstract"
        elif dataset_name == "xsum":
            human_key = "document"
        elif dataset_name == "writing_prompt":
            human_key = "story"
        elif dataset_name == "yelp_review":
            human_key = "content"
        with open(os.path.join(SRC_ROOT, dataset), "r") as f:
            data = json.load(f)
        for i, data_point in enumerate(data):
            edits = recreate_edits_fast_diff(data_point[human_key], data_point["paraphrase_polish_human"])
            dst_folder = os.path.join(DST_ROOT, f"{dataset_name}_{str(i)}")
            if not os.path.exists(DST_ROOT):
                os.mkdir(DST_ROOT)
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)
            stats_path, notions_path, boundary_path = create_folders_for_analysis(dst_folder)


            dst_file_path = os.path.join(dst_folder, "edits.json")
            with open(dst_file_path, "w", encoding="utf-8") as f:
                json.dump(edits, f)

            _, _, _, text_by_user, _ = get_and_save_notions(edits,
                                                                    boundary_path,
                                                                    notions_path)
        
            get_and_save_final_text(text_by_user, dst_folder)