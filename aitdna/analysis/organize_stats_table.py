import json
import os
import pandas as pd

rename_map = {
    "avg_readability_scores": "avg_read_ease",
    "flesch_reading_ease": "FRE",
    "flesch_kincaid_grade_level": "FKGL",
    "gunning_fog": "GF",
    "dalle_chall": "DC"
}


def rename_key(k):
    parts = k.split(".")
    if parts[0] in rename_map:
        parts[0] = rename_map[parts[0]]
    if len(parts) > 1 and parts[1] in rename_map:
        parts[1] = rename_map[parts[1]]
    return ".".join(parts)


def process_and_save_stats(stats_dir: str):
    stats = []
    for stat_file in os.listdir(stats_dir):
        with open(os.path.join(stats_dir, stat_file), "r", encoding="utf-8") as f:
            dataset = json.load(f)
    
        dataset["stats_per_user"].pop("pos")
        df = pd.json_normalize(dataset, sep=".")
        d_flat = df.to_dict(orient='records')[0]
        d_flat = {
            ".".join(k.split(".")[1:]) if "." in k else k: v
            for k, v in d_flat.items()
        }
        d_flat = {rename_key(k): v for k, v in d_flat.items()}

        for stat, value in d_flat.items():
            d_flat[stat] = round(value, 3)
        d_flat["dataset"] = stat_file.replace(".json", "")
        stats.append(d_flat)

    df = pd.DataFrame(stats)
    df = df.set_index("dataset")
    df = df.transpose()
    df.index.names = ['metric']
    df = df.round(3)
    df.to_csv("overall_stats.csv")
    df.to_markdown("overall_stats.txt", colalign=("left", "left", "left", "left", "left", "left", "left"))


def process_and_save_base_stats(stats_dir: str, dst_file_path: str):
    stats = []
    for stat_file in os.listdir(stats_dir):
        with open(os.path.join(stats_dir, stat_file), "r", encoding="utf-8") as f:
            dataset = json.load(f)
            dataset["dataset"] = stat_file.replace(".json", "")
            stats.append(dataset)

    df = pd.DataFrame(stats)
    df = df.set_index("dataset")
    df = df.transpose()
    df = df.round(3)
    df.index.names = ['metric']
    df.to_csv(dst_file_path)


def main():
    # TODO add cli command
    stats_dir = "data/stats_and_plots/base_stats"
    dst_file_path = "data/stats_and_plots/tables/base_stats.csv"
    process_and_save_base_stats(stats_dir, dst_file_path)


if __name__ == "__main__":
    main()