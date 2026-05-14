import os
from txaitd.analysis.StatsComputer import StatsComputer
from txaitd.notions.data_loading import DatasetName
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

def test_avg_n_tokens():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    tokens = stats.get_avg_n_tokens_per_span()
    print(tokens)

def test_avg_n_chars():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    sents = stats.get_avg_n_sentences_per_span()
    print(sents)

def test_avg_n_boud_charwise():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    n = stats.get_avg_n_boundaries_span_level()
    print(n)

def test_avg_n_boud_sentwise():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    n = stats.get_avg_n_boundaries_sentence_level()
    print(n)

def test_ling_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    ling_stats = stats.get_linguistic_stats_all()
    print(ling_stats)

def test_ling_trees():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    trees = stats.get_linguistic_trees_all()
    print(trees)

def test_sentence_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    sent = stats.get_avg_stats_per_sentence()
    print(sent)

def test_user_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    sent = stats.get_avg_stats_per_user()
    print(sent)

def test_span_stats():
    stats = StatsComputer(dataset_name=DatasetName.AITDNA, dataset_root="data/aitdna_anonymized/formatted")
    sent = stats.get_avg_stats_per_span()
    print(sent)

def test_plot_ai_percentage():
    dataset_and_root = [(DatasetName.AITDNA, "data/aitdna/formatted"),
                        (DatasetName.AITDNA_SYNTHETIC, "data/llm_dataset"),
                        (DatasetName.BOUNDARY_DETECTION , "data/other_datasets/processed/boundary_detection"),
                        (DatasetName.COAUTHOR , "data/other_datasets/processed/coauthor-v1.0"),
                        (DatasetName.DETECTRL , "data/other_datasets/processed/detectRL"),
                        (DatasetName.MIXSET , "data/other_datasets/processed/mixset"),
                        ]
    name_map = {
        "AITDNA": "AITDNA",
        "MIXSET": "Mixset",
        "COAUTHOR": "CoAuthor",
        "BOUNDARY_DETECTION": "BD",
        "DETECTRL": "DetectRL",
        "AITDNA_SYNTHETIC": "AITDNA-S"
    }
    style_map = {
        "AITDNA": "-",
        "MIXSET": ":",
        "COAUTHOR": "-",
        "BOUNDARY_DETECTION": "-.",
        "DETECTRL": "--",
        "AITDNA_SYNTHETIC": "-"
    }
    sns.set_theme(style="darkgrid") 
    plt.title("KDE of AI Token Percentage per Dataset")
    for i, (name, path) in enumerate(dataset_and_root):
        stats = StatsComputer(dataset_name=name, dataset_root=path)
        perc = stats.get_perc_ai_tokens()["Bot"]
        ds_name = str(name).split(".")[-1]
        short_ds_name = name_map[ds_name]
        sns.kdeplot(perc, label=short_ds_name, linestyle=style_map[ds_name])
    plt.legend()
    plt.xlabel("Smoothed AI Token Percentage")
    plt.savefig("data/stats_and_plots/plots/ai_perc_overalls.png")


def test_ai_percentage_stats():
    dataset_and_root = [(DatasetName.AITDNA, "data/aitdna_anonymized/formatted"),
                        (DatasetName.AITDNA_SYNTHETIC, "data/llm_dataset"),
                        (DatasetName.BOUNDARY_DETECTION , "data/other_datasets/processed/boundary_detection"),
                        (DatasetName.COAUTHOR , "data/other_datasets/processed/coauthor-v1.0"),
                        (DatasetName.DETECTRL , "data/other_datasets/processed/detectRL"),
                        (DatasetName.MIXSET , "data/other_datasets/processed/mixset"),
                        ]

    fig, ax = plt.subplots(3, 2, figsize=(20, 15))
    ax = ax.flatten()
    fig.suptitle("Percentage of AI tokens per text")
    for i, (name, path) in enumerate(dataset_and_root):
        stats = StatsComputer(dataset_name=name, dataset_root=path)
        perc = stats.get_perc_ai_tokens()["Bot"]
        print("Dataset: ", name)
        print("Mean: ", np.mean(perc))
        print("Std: ", np.std(perc))
        print("Skew: ", scipy.stats.skew(perc))
        print()


def rename_files():
    renaming = {
        "final_text_by_user_authorshipwise.json": "final_text_by_user_span_level.json",
        "final_text_by_user_sentencewise.json": "final_text_by_user_sentence_level.json",
        "final_text_by_user_textwise.json": "final_text_by_user_document_level.json",
        "final_text_by_user_tokenwise.json": "final_text_by_user_token_level.json",
        "final_text_by_user_boundarywise_ilp_2seg_1lp_1ip.json": "final_text_by_user_boundary_level_ilp_2seg_1lp_1ip.json",
        "final_text_by_user_boundarywise_ilp_5seg_1lp_1ip.json": "final_text_by_user_boundary_level_ilp_5eg_1lp_1ip.json",
        "final_text_by_user_boundarywise_ilp_10seg_1lp_1ip.json": "final_text_by_user_boundary_level_ilp_10seg_1lp_1ip.json",

    }
    root = "data/llm_dataset"
    for dataset in os.listdir(root):
        for data_point in os.listdir(os.path.join(root, dataset)):
            notions = os.path.join(root, dataset, data_point, "notions")
            for file in os.listdir(notions):
                if os.path.isdir(os.path.join(notions, file)):
                    for boundary in os.listdir(os.path.join(notions, file)):
                        try:
                            os.rename(os.path.join(notions, file, boundary), os.path.join(notions, file, renaming[boundary]))
                        except KeyError:
                            continue
                else:
                        try:
                            os.rename(os.path.join(notions, file), os.path.join(notions, renaming[file]))
                        except KeyError:
                            continue

test_plot_ai_percentage()