import os
import json
import logging
import math
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import numpy as np
import pandas as pd
import nltk
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

import seaborn as sns
from datasets import Dataset, concatenate_datasets

from aitdna.analysis.StatsComputer import StatsComputer
from aitdna.experiments.mgtd.mgtd_datasets.DetectionDataset import DetectionDataset
from aitdna.experiments.mgtd.methods.generation import MGTDMethod
from aitdna.notions.data_loading import  DatasetName


def check_time_per_user(root: str) -> dict[str, dict[str, float]]:
    """
    Compute how much time each user spent in total.
    """
    stats = {}
    for study in os.listdir(root):
        stats[study] = {}
        for user in os.listdir(os.path.join(root, study)):
            total_time = 0
            user_root = os.path.join(root, study, user)
            for task in os.listdir(user_root):
                with open(os.path.join(user_root, task, "edits.json"), "r", encoding="utf-8") as f:
                    edits = json.load(f)
                edits = [edit for edit in edits if not ("operationType" in edit and edit["operationType"] == 0 and edit["text"] is None)]
                latest_edit = datetime.fromisoformat(edits[-1]["createdAt"])
                earliest_edit = datetime.fromisoformat(edits[0]["createdAt"])
                time_spent = (latest_edit - earliest_edit).total_seconds()
                total_time += time_spent
            stats[study][user] = f"{math.ceil(total_time / 60)}:{int(total_time) % 60}"
    return stats

def get_ai_perception_vs_real(root) -> list[float]:
    """Return a list of actual AI percentage - perceived AI percentage

    Args:
        root (_type_): root

    Returns:
        _type_: _description_
    """
    diff = []
    for study in os.listdir(root):
        study_path = os.path.join(root, study)
        for user in os.listdir(study_path):
            for task in os.listdir(os.path.join(study_path, user)):
                perception_path = os.path.join(study_path, user, task, "statistics", "ai_perception.json")
                if not os.path.exists(perception_path):
                    continue
                with open(perception_path, "r", encoding="utf-8") as f:
                    perceptions = json.load(f)
                perception = 100 - perceptions["ai_perception"][0]["perceivedHumanPercentage"]

                stats_path = os.path.join(study_path, user, task, "notions", "final_text_by_user_token_level.json")
                with open(stats_path, "r", encoding="utf-8") as f:
                    tokens = json.load(f)
                actual = len([token for token in tokens if token["author"] == "Bot"]) / len(tokens) * 100
                diff.append(actual - perception)
    return diff

def plot_ai_perception_vs_real(diff, figpath):
    plt.hist(diff, bins=20, edgecolor="black", color="#4BB05C")
    plt.title("Difference between Actual and Perceived AI Percentage")
    plt.ylabel("# Occurences")
    plt.xlabel("Actual - Perceived AI Percentage")
    plt.savefig(figpath)

def get_violations(root):
    n = 0
    total = 0
    study_path = root
    for user in os.listdir(study_path):
        for task in os.listdir(os.path.join(study_path, user)):
            total += 1
            stats_path = os.path.join(study_path, user, task, "statistics", "violations.txt")
            if os.path.exists(stats_path):
                #print(study)
                print(user)
                print(task)
                print()
                n += 1
    print(n)
    print(total)

def get_prolific_no_violations(study_path):
    for user in os.listdir(study_path):
        good = True
        tasks = os.listdir(os.path.join(study_path, user))
        if len(tasks) != 4:
            continue
        for task in tasks:
            stats_path = os.path.join(study_path, user, task, "statistics")
            if os.path.exists(os.path.join(stats_path, "violations.txt")):
               good = False
            if not os.path.exists(os.path.join(stats_path, "background.json")):
                good = False
            if not os.path.exists(os.path.join(stats_path, "ux_survey.json")):
                good = False
        if good:
            print(user)


def survey_missing(study_root):
    ux_missing = set()
    bg_missing = set()
    for user in os.listdir(study_root):
        for task in os.listdir(os.path.join(study_root, user)):
            stats = os.listdir(os.path.join(study_root, user, task, "statistics"))
            if "ux_survey.json" not in stats:
                ux_missing.add(user)
            if "background.json" not in stats:
                bg_missing.add(user)
    print("Background survey missing for: ", bg_missing)
    print("UX survey missing for: ", ux_missing)

def get_ux_survey_results(root):
    questions = {"Did the usage of LLMs help you improve text quality?": [],
                 "Were you satisfied with the resulting text?": [],
                 "Did the LLM provide meaningful and relevant\u00a0answers?": [],
                "Did the LLMs help you come up with new ideas?": [],
                "Did the usage of LLMs speed up your writing?": []
                }
    for study in os.listdir(root):
        study_root = os.path.join(root, study)
        for user in os.listdir(study_root):
            for task in os.listdir(os.path.join(study_root, user)):
                stats = os.listdir(os.path.join(study_root, user, task, "statistics"))
                if "ux_survey.json" not in stats:
                    continue
                with open(os.path.join(study_root, user, task, "statistics", "ux_survey.json"), "r") as f:
                    ux_survey = json.load(f)
                for key, value in ux_survey.items():
                    if key in questions:
                        questions[key].append(value)
    return questions

def save_ux_survey_results(results: dict):
    """Plots the results of the survey

    Args:
        results (dict): results in form: {"Question": [5-likert-options], ..}
    """
    labels = ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"]
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0, 1, len(labels)))
    plt.figure(constrained_layout=True)

    for i, (question, values) in enumerate(results.items()):
        counts = []
        for answer in labels:
            counts.append(len([v for v in values if v == answer]) / len(values))
        plt.bar(labels, counts, color=colors)
        plt.title(question)
        plt.xticks(labels, rotation=25, ha="right", rotation_mode="anchor")
        plt.savefig(f"data/stats_and_plots/plots/survey_results/{str(i)}.png")
        plt.clf()

def get_avg_n_chars(root):
    n_chars = 0
    n_texts = 0
    for study in os.listdir(root):
        study_path = os.path.join(root, study)
        for user in os.listdir(study_path):
            for task in os.listdir(os.path.join(study_path, user)):
                with open(os.path.join(study_path, user, task, "final_text.txt"),
                "r",
                encoding="utf-8") as f:
                    text = f.read()
                n_texts += 1
                n_chars += len(text)
    return n_chars / n_texts

def get_n_queries_by_task(root):
    avg_n_queries = {
        "Argumentative": {
            "text_revision": 0,
            "text_continuation": 0,
        },
        "Creative": {
            "text_revision": 0,
            "text_continuation": 0,
        },
        "Explanatory": {
            "text_revision": 0,
            "text_continuation": 0,
        }
    }

    revision_times = {
        "Argumentative": {
            "first_revision_time": [],
            "first_continuation_time": []
        },
        "Creative": {
            "first_revision_time": [],
            "first_continuation_time": []
        },
        "Explanatory": {
            "first_revision_time": [],
            "first_continuation_time": []
        }
    }
    total_n_texts = {
        "Argumentative": 0,
        "Creative": 0,
        "Explanatory": 0
    }
    for study in os.listdir(root):
        study_path = os.path.join(root, study)
        for user in os.listdir(study_path):
            for task in os.listdir(os.path.join(study_path, user)):
                if "Human" in task or "Peer" in task:
                    continue
                with open(os.path.join(study_path, user, task, "edits.json"),
                "r",
                encoding="utf-8") as f:
                    edits = json.load(f)
                if "Argumentative" in task:
                    key = "Argumentative"
                elif "Creative" in  task:
                    key = "Creative"
                elif "Explanatory" in task:
                    key = "Explanatory"
                else:
                    continue
                total_n_texts[key] += 1
                first_revision_found = False
                first_continuation_found = False
                for edit in edits:
                    if "nlpService" in edit:
                        avg_n_queries[key][edit["nlpService"]] += 1

                        if edit["nlpService"] == "text_revision" and not first_revision_found:
                            first_revision_found = True
                            revision_times[key]["first_revision_time"].append(edit["createdAt"])
                        if edit["nlpService"] == "text_continuation" and not first_continuation_found:
                            first_continuation_found = True
                            revision_times[key]["first_continuation_time"].append(edit["createdAt"])
    
    for setup, info in avg_n_queries.items():
        for skill, n_requests in info.items():
            avg_n_queries[setup][skill] = round(n_requests / total_n_texts[setup], 2)
    
    for setup, info in revision_times.items():
        for skill, times in info.items():
            revision_times[setup][skill] = round(float(np.median(times)), 0)
    return avg_n_queries, revision_times


def compute_overall_paper_stats(src_root, formatted_root, ignore_src=False):
    if not ignore_src:
        n_src = 0
        for study in os.listdir(src_root):
            study_path = os.path.join(src_root, study)
            for user in os.listdir(study_path):
                for task in os.listdir(os.path.join(study_path, user)):
                    n_src += 1
    else:
        n_src = -1
    
    n_dst = 0
    n_human_only = 0
    n_human_only_actual = 0

    n_texts = {
        "Standard": {
            "Argumentative":   {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            },
            "Creative": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            },
            "Explanatory": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            }
        },
        "Scientific": {
            "Argumentative": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            },
            "Creative": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            },
            "Explanatory": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            },
            "Peer Review": {
                "Total": 0,
                "Collaborative": 0,
                "Human-Only": 0
            }
        }
    }

    n_models = defaultdict(lambda: defaultdict(int))
    for study in os.listdir(formatted_root):
        study_path = os.path.join(formatted_root, study)
        for user in os.listdir(study_path):
            for task in os.listdir(os.path.join(study_path, user)):
                if "Human" in task:
                    n_human_only += 1
                n_dst += 1
                with open(
                    os.path.join(study_path, user, task, "notions", "final_text_by_user_token_level.json"),
                    "r",
                    encoding="utf-8") as f:
                    tokens = json.load(f)
                is_human_only = all(token["author"] == "User" for token in tokens)
                n_human_only_actual += 1 if is_human_only else 0

                if study in ["session_2", "session_4"]:
                    setup = "Scientific"
                else:
                    setup = "Standard"

                if "Argumentative" in task:
                    task_type = "Argumentative"
                elif "Creative" in task:
                    task_type = "Creative"
                elif "Explanatory" in task:
                    task_type = "Explanatory"
                elif "Peer" in task:
                    task_type = "Peer Review"
                else:
                    raise ValueError("Unknown task type")

                n_texts[setup][task_type]["Total"] += 1
                if is_human_only:
                    n_texts[setup][task_type]["Human-Only"] += 1
                else:
                    n_texts[setup][task_type]["Collaborative"] += 1

                assignment_path = os.path.join(study_path, user, task, "statistics", "user_task_assignment.json")
                if not os.path.exists(assignment_path):
                    continue
                with open(assignment_path, "r", encoding="utf-8") as f:
                    assignment_info = json.load(f)
                    if "model" in assignment_info and assignment_info["model"]:
                        n_models[str(assignment_info["temperature"])][assignment_info["model"]] += 1


    stats = {
        "n_all_data_points": n_src,
        "n_data_points_after_filtering": n_dst,
        "n_human_only_tasks": n_human_only,
        "n_actual_human_only": n_human_only_actual
    }
    return stats, n_texts, n_models

    
def n_texts_to_csv(data, csv_path):
    rows = []
    for category, subcats in data.items():
        for subcat, values in subcats.items():
            rows.append((subcat, values['Total'], values['Collaborative'], values['Human-Only']))

    df = pd.DataFrame(rows, columns=['Task', 'Total', 'Collaborative', 'Human-Only'])
    df = df.set_index(['Task'])
    df.to_markdown(csv_path)
    

def n_models_to_csv(data, csv_path):
    rows = []
    cols = list(set(k for settings in data.values() for k in settings.keys()))
    for temperature, settings in data.items():
        row = {"temperature": temperature}
        for col in cols:
            row[col] = settings.get(col)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def get_total_n_words(root_paths):
    total_tokens = 0
    for root in root_paths:
        if "aitdna" in root:
            for study in os.listdir(root):
                study_root = os.path.join(root, study)
                for user in os.listdir(study_root):
                    for task in os.listdir(os.path.join(study_root, user)):
                        with open(os.path.join(study_root, user, task, "final_text.txt"), "r") as f:
                            text = f.read()
                            tokens = nltk.word_tokenize(text)
                            total_tokens += len(tokens)
        else:
            for model in os.listdir(root):
                for task in os.listdir(os.path.join(root, model)):
                    with open(os.path.join(root, model, task, "final_text.txt"), "r") as f:
                        text = f.read()
                        tokens = nltk.word_tokenize(text)
                        total_tokens += len(tokens)
    return total_tokens

def get_stats(root):
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for file in os.listdir(root):
        if "metrics" not in file:
            continue
        with open(os.path.join(root, file), "r") as f:
            metrics = json.load(f)
        detector_name = file.replace("metrics_", "").replace(".json", "").replace("AITDNA_SYNTHETIC", "AITDNA-S")
        dataset_name = detector_name.split("_")[-1]
        detector_name = detector_name.replace(f"_{dataset_name}", "")
        for metric, value in metrics.items():
            stats[metric][detector_name][dataset_name] = value
    return stats

def stats_to_csv(data, csv_root):
    for metric, metric_data in data.items():
        df = pd.DataFrame.from_dict(metric_data, orient="index")
        df = df.round(3)
        df.to_csv(os.path.join(csv_root, f"{metric}.csv"))


def plot_ai_percentage():
    dataset_and_root = [(DatasetName.AITDNA, "data/aitdna/formatted"),
                        (DatasetName.COAUTHOR , "data/other_datasets/processed/coauthor/coauthor-v1.0"),
                        (DatasetName.SENDETEX, "data/other_datasets/processed/senDetEx_processed/senDetEx"),
                        (DatasetName.BOUNDARY_DETECTION , "data/other_datasets/processed/bd/boundary_detection"),
                        (DatasetName.DETECTRL , "data/other_datasets/processed/detectRL"),
                        (DatasetName.MIXSET , "data/other_datasets/processed/mixset"),
                        ]
    name_map = {
        "AITDNA": "AITDNA",
        "COAUTHOR": "CoAuthor",
        "MIXSET": "Mixset",
        "BOUNDARY_DETECTION": "BD",
        "DETECTRL": "DetectRL",
        "SENDETEX": "SenDetEx"
    }
    color_map = {
        "AITDNA": "#1f77b4",
        "COAUTHOR": "#ff7f0e",
        "MIXSET": "#2ca02c",
        "BOUNDARY_DETECTION": "#d62728",
        "DETECTRL": "#9467bd",
        "SENDETEX": "#8c564b"
    }
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    #plt.title("AI Token Percentage per Dataset")
    for i, (name, path) in enumerate(dataset_and_root):
        stats = StatsComputer(dataset_name=name, dataset_root=path)
        perc = stats.get_perc_ai_tokens()["Bot"]
        ds_name = str(name).split(".")[-1]
        short_ds_name = name_map[ds_name]
        sns.histplot(perc, label=short_ds_name, stat='density', element='step', fill=False, linewidth=1, color=color_map[ds_name])

    plt.legend(fontsize=12)
    plt.xlabel("% AI tokens", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.savefig("data/stats_and_plots/aipercent_histogram.png", dpi=300, bbox_inches='tight')


def plot_ai_percentage_cumulative():
    dataset_and_root = [(DatasetName.AITDNA, "data/aitdna/formatted"),
                        (DatasetName.COAUTHOR , "data/other_datasets/processed/coauthor/coauthor-v1.0"),
                        (DatasetName.SENDETEX, "data/other_datasets/processed/senDetEx_processed/senDetEx"),
                        (DatasetName.BOUNDARY_DETECTION , "data/other_datasets/processed/bd/boundary_detection"),
                        (DatasetName.DETECTRL , "data/other_datasets/processed/detectRL"),
                        (DatasetName.MIXSET , "data/other_datasets/processed/mixset"),
                        ]
    name_map = {
        "AITDNA": "AITDNA",
        "COAUTHOR": "CoAuthor",
        "MIXSET": "Mixset",
        "BOUNDARY_DETECTION": "BD",
        "DETECTRL": "DetectRL",
        "SENDETEX": "SenDetEx"
    }
    color_map = {
        "AITDNA": "#1f77b4",
        "COAUTHOR": "#ff7f0e",
        "MIXSET": "#2ca02c",
        "BOUNDARY_DETECTION": "#d62728",
        "DETECTRL": "#9467bd",
        "SENDETEX": "#8c564b"
    }
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    #plt.title("AI Token Percentage per Dataset")
    for i, (name, path) in enumerate(dataset_and_root):
        stats = StatsComputer(dataset_name=name, dataset_root=path)
        perc = stats.get_perc_ai_tokens()["Bot"]

        print(name)
        print("min", min(perc), "max", max(perc), "mean", sum(perc) / len(perc))

        ds_name = str(name).split(".")[-1]
        short_ds_name = name_map[ds_name]

        if max(perc) < 100:
            perc.append(100)

        sns.ecdfplot(data=perc, label=short_ds_name, stat='proportion', complementary=False, linewidth=1, color=color_map[ds_name])

    ax.set_xlim(left=0, right=1)

    plt.legend(fontsize=12)
    plt.xlabel("% AI tokens", fontsize=14)
    plt.ylabel("Cumulative proportion", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 1.05)
    plt.savefig("data/stats_and_plots/aipercent_cumulative.png", dpi=300, bbox_inches='tight')

def plot_ai_percentage_sentencewise():
    dataset_and_root = [(DatasetName.AITDNA, "data/aitdna/formatted"),
                        (DatasetName.COAUTHOR , "data/other_datasets/processed/coauthor/coauthor-v1.0"),
                        (DatasetName.SENDETEX, "data/other_datasets/processed/senDetEx_processed/senDetEx"),
                        (DatasetName.BOUNDARY_DETECTION , "data/other_datasets/processed/bd/boundary_detection"),
                        (DatasetName.DETECTRL , "data/other_datasets/processed/detectRL"),
                        (DatasetName.MIXSET , "data/other_datasets/processed/mixset"),
                        ]
    name_map = {
        "AITDNA": "AITDNA",
        "COAUTHOR": "CoAuthor",
        "MIXSET": "Mixset",
        "BOUNDARY_DETECTION": "BD",
        "DETECTRL": "DetectRL",
        "SENDETEX": "SenDetEx"
    }
    color_map = {
        "AITDNA": "#1f77b4",
        "COAUTHOR": "#ff7f0e",
        "MIXSET": "#2ca02c",
        "BOUNDARY_DETECTION": "#d62728",
        "DETECTRL": "#9467bd",
        "SENDETEX": "#8c564b"
    }
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    #plt.title("AI Token Percentage per Dataset")
    for i, (name, path) in enumerate(dataset_and_root):
        stats = StatsComputer(dataset_name=name, dataset_root=path)
        perc = stats.get_perc_sent_ai_tokens()["Bot"]
        ds_name = str(name).split(".")[-1]
        short_ds_name = name_map[ds_name]
        sns.ecdfplot(data=perc, label=short_ds_name, stat='proportion', complementary=False, linewidth=1, color=color_map[ds_name])

    plt.legend(fontsize=12)
    plt.xlabel("% AI tokens", fontsize=14)
    plt.ylabel("Cumulative proportion", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 100)
    plt.savefig("data/stats_and_plots/aipercent_cumulative_sentencewise.png", dpi=300, bbox_inches='tight')

def organize_results_table():
    root = "data/changed_threshold_prediction_output"
    data = defaultdict(dict)
    datasets = ["AITDNA", "DETECTRL", "SENDETEX", "MIXSET", "COAUTHOR", "DETECTION"]
    methods = [ "log_rank", "likelihood", "min_k", "binoculars", "fastdetectgpt", "modernBERT"]

    for file in os.listdir(root):
        dataset = file.split("_")[-2]
        detector = ""
        for method in methods:
            if method in file:
                detector = method
                break
        if detector == "":
            print("Detector not found")

        with open(os.path.join(root, file), "r") as f:
            metrics = json.load(f)
        auroc = round(metrics["AUROC"], 3)
        f1 = round(metrics["F-1"], 3)
        data[dataset][detector] = {
            "auroc": auroc,
            "f1": f1
        }
    rows = []
    for method in methods:
        row = []
        row.append(method)
        for dataset in datasets:
            row.append(data[dataset][method]["auroc"])
            row.append(data[dataset][method]["f1"])
        rows.append(row)

    with open("AITDNA.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def organize_results_table_detector():
    root = "data/predictor_test/local"
    detector = "fastdetectgpt"
    rows = []
    for file in os.listdir(root):
        if detector not in file:
            continue
        dataset = file.replace("metrics_fastdetectgpt_", "").replace(".json", "")
        with open(os.path.join(root, file), "r") as f:
            data = json.load(f)
            auroc = round(data["AUROC"], 3)
            f1 = round(data["F-1"], 3)
            rows.append([dataset, auroc, f1])
    
    with open("fastdetectgpt.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Detector", "AUROC", "F-1"])
        writer.writerows(rows)


def plot_heatmap_notion_performance_stylish(metric="F-1"):
    root = "data/heatmap_data"
    # options metric: Accuracy, AUROC, F-1, Precision, Recall

    notions = ["document", "boundary", "sentence", "intent", "content", "membership"]
    methods = ["Log Rank", "Likelihood", "Min K", "Binoculars", "fdGPT", "BERT", "GPTZero", "Pangram"]
    if metric == "AUROC":
        methods = ["Log Rank", "Likelihood", "Min K", "Binoculars", "fdGPT", "BERT"]


    plot_path = "aitd-notions-heatmap.png"

    rows = []
    for file in os.listdir(root):
        method_name = ""
        if "fastdetect" in file:
            method_name = "fdGPT"
        else:
            for method in methods:
                if method.lower() in file.lower().replace("_", " "):
                    method_name = method
                    break

        notion = file.split("_")[-1].replace(".json", "")
        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        if metric in data:
            rows.append({
                "Notion": notion,
                "Method": method_name,
                "metric": data[metric]
            })


    df = pd.DataFrame(rows)

    # ── Build pivot + medians ──────────────────────────────────────────────────────
    heatmap_data = df.pivot(index="Notion", columns="Method", values="metric")
    heatmap_data.columns.name = None
    heatmap_data.index.name = None

    # Median column (per notion row, across methods)
    heatmap_data["median"] = heatmap_data.median(axis=1)

    # Median row (per method col, across notions) — computed before adding gap rows
    median_row = heatmap_data.median(axis=0)
    median_row.name = "median"
    heatmap_data = pd.concat([heatmap_data, median_row.to_frame().T])

    # ── Separate main block from median row/col ────────────────────────────────────
    # Main block: all notions x all methods (no median row/col)
    main_notions = notions  # original notion rows
    main_methods = methods  # original method columns

    main_block = heatmap_data.loc[main_notions, main_methods].astype(float)
    med_col_vals = heatmap_data.loc[main_notions, "median"].astype(float)  # median column, notion rows
    med_row_vals = heatmap_data.loc["median", main_methods].astype(float)  # median row, method cols
    med_corner = float(heatmap_data.loc["median", "median"])

    # ── Color maps ────────────────────────────────────────────────────────────────
    # Light grey → light green (gentle, pastel)
    main_cmap = LinearSegmentedColormap.from_list("gg", ["#e0e0e0", "#a8d5a2"])
    med_cmap1 = LinearSegmentedColormap.from_list("gg2", ["#FADFBE", "#FAB664"])
    med_cmap2 = LinearSegmentedColormap.from_list("gg3", ["#C7E5FC", "#65B8F7"])

    def norm_row(row_vals):
        """Return normalised values in [0,1] per row, ignoring NaNs."""
        if np.all(np.isnan(row_vals)):
            return row_vals # Return as is (all NaNs)
        
        lo = np.nanmin(row_vals)
        hi = np.nanmax(row_vals)
        
        if hi == lo:
            return np.where(np.isnan(row_vals), np.nan, 0.5)
        
        # Calculate normalization only for non-NaN indices
        return (row_vals - lo) / (hi - lo)

    def norm_global(vals):
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return np.full_like(vals, 0.5, dtype=float)
        return (vals - lo) / (hi - lo)

    # ── Compute colour matrices ───────────────────────────────────────────────────
    n_notions = len(main_notions)
    n_methods = len(main_methods)

    # Main block: row-normalised
    main_colors = np.zeros((n_notions, n_methods, 4))
    for i in range(n_notions):
        row_norm = norm_row(main_block.iloc[i].values)
        for j in range(n_methods):
            main_colors[i, j] = main_cmap(row_norm[j])

    # Median column: global normalisation across notion rows
    med_col_norm = norm_global(med_col_vals.values)
    med_col_colors = np.array([med_cmap1(v) for v in med_col_norm]).reshape(n_notions, 1, 4)

    # Median row: global normalisation across method cols
    med_row_norm = norm_global(med_row_vals.values)
    med_row_colors = np.array([med_cmap2(v) for v in med_row_norm]).reshape(1, n_methods, 4)

    # Corner cell: mid-point of med_cmap
    corner_color = (1.0, 1.0, 1.0, 1.0)

    # ── Layout geometry ───────────────────────────────────────────────────────────
    GAP = 0.35  # width of white gap column / row in data-units
    CELL_W = 1.0
    CELL = 1.0  # each cell is 1 unit

    # x positions (left edge of each column)
    x_main = np.arange(n_methods) * CELL_W
    x_gap_c = n_methods * CELL_W
    x_med_c = x_gap_c + GAP  # median column

    # y positions (bottom edge, rows drawn top-to-bottom so we invert)
    y_main = np.arange(n_notions)  # 0 … n_notions-1
    y_gap_r = n_notions  # gap row
    y_med_r = n_notions/2 + GAP  # median row

    total_w = x_med_c + CELL
    total_h = y_med_r + CELL/2

    fig_w = max(10, total_w * 1.2)
    fig_h = max(4, total_h * 0.85)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Draw helper ───────────────────────────────────────────────────────────────
    FONT_MAIN = 12
    FONT_MED = 12

    def draw_cell(ax, x, y, color, value, fontsize=FONT_MAIN, bold=False):
        rect = plt.Rectangle((x, y), CELL, 0.5, facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "#333333" if lum > 0.45 else "#111111"
        ax.text(x + 0.5, y + 0.25, f"{value:.3f}",
                ha="center", va="center", fontsize=fontsize,
                color=txt_color,
                fontweight="bold" if bold else "normal",
                fontfamily="monospace")

    # ── Main block ────────────────────────────────────────────────────────────────
    for i in range(n_notions):
        for j in range(n_methods):
            draw_cell(ax, x_main[j], y_main[i]/2.0, main_colors[i, j], main_block.iloc[i, j])

    # ── Median column (notion rows) ───────────────────────────────────────────────
    for i in range(n_notions):
        draw_cell(ax, x_med_c, y_main[i]/2.0, med_col_colors[i, 0], med_col_vals.iloc[i],
                  fontsize=FONT_MED, bold=True)

    # ── Median row (method cols) ──────────────────────────────────────────────────
    for j in range(n_methods):
        draw_cell(ax, x_main[j], y_med_r, med_row_colors[0, j], med_row_vals.iloc[j],
                  fontsize=FONT_MED, bold=True)

    # ── Corner cell ───────────────────────────────────────────────────────────────
    draw_cell(ax, x_med_c, y_med_r, corner_color, med_corner, fontsize=0, bold=True)

    # ── Column headers (method names) ────────────────────────────────────────────
    for j, m in enumerate(main_methods):
        ax.text(x_main[j] + 0.5, -0.25, m,
                ha="center", va="bottom", fontsize=11, color="#444444")

    ax.text(x_med_c + 0.5, -0.25, "median",
            ha="center", va="bottom", fontsize=11, color="#444444")

    # ── Row headers (notion names) ────────────────────────────────────────────────
    for i, n in enumerate(main_notions):
        ax.text(-0.15, y_main[i]/2.0 + 0.25, n,
                ha="right", va="center", fontsize=11, color="#444444")

    ax.text(-0.15, y_med_r + 0.25, "median",
            ha="right", va="center", fontsize=11, color="#444444")

    plt.tight_layout(pad=0.3)
    plot_path = f"data/stats_and_plots/aitd-notions-heatmap-{metric}.pdf"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {plot_path}")

def add_membership_heatmap_data():
    predictions_root = "data/all_predictions"
    metrics_root = "data/heatmap_data"

    ds = get_dataset(dataset_name=DatasetName.AITDNA,
                dataset_path="data/aitdna/formatted",
                threshold=0.5, detection_level="membership")

    method = MGTDMethod()
    for file in os.listdir(predictions_root):
        if "AITDNA" not in file or "predictions" not in file:
            continue
        with open(os.path.join(predictions_root, file), "r") as f:
            predictions = json.load(f)
        result = method.evaluate(predictions, ds)
        metrics_file_name = file.replace("predictions_", "metrics_").replace("_document.json", "_membership.json")
        with open(os.path.join(metrics_root, metrics_file_name), "w") as f:
            json.dump(result, f)


def get_dataset(dataset_name, dataset_path, threshold, detection_level: str = "document"):
      all_datasets = []
      dataset = DetectionDataset(data_path=dataset_path,
            dataset_name=dataset_name,
            detection_level=detection_level,
            threshold=threshold)
      dataset = Dataset.from_list(dataset)
      all_datasets.append(dataset)

      return concatenate_datasets(all_datasets)


def evaluate_thresholds(predictions, dataset_name):
    results = {}
    path_mapping = {
        "AITDNA": "data/aitdna/formatted",
        "COAUTHOR": "data/other_datasets/processed/coauthor-v1.0",
        "MIXSET": "data/other_datasets/processed/mixset",
        "DETECTRL": "data/other_datasets/processed/detectRL",
        "SENDETEX": "data/other_datasets/processed/senDetEx",
        "BOUNDARY_DETECTION": "data/other_datasets/processed/boundary_detection",
    }

    for i in range(1, 10):
        threshold = i / 10
        ds = get_dataset(DatasetName[dataset_name],
                            path_mapping[dataset_name],
                            threshold)
        method = MGTDMethod()
        result = method.evaluate(predictions, ds)
        results[threshold] = result
    return results

def test_detection_ds():
    root ="data/all_predictions"
    data = defaultdict(dict)
    metrics = defaultdict(dict)
    methods = [ "log_rank", "likelihood", "min_k", "binoculars", "fastdetectgpt", "modernBERT"]

    for file in os.listdir(root):
        if "metrics" in file:
            continue
        dataset = file.split("_")[-2]
        if dataset == "DETECTION":
            dataset = "BOUNDARY_DETECTION"
        detector = ""
        for method in methods:
            if method in file:
                detector = method
                break
        if detector == "":
            print("Detector not found")

        with open(os.path.join(root, file), "r") as f:
            predictions = json.load(f)
        data[dataset][detector] = predictions
    
    for ds, methods in data.items():
        for method, preds in methods.items():
            if os.path.exists(f"data/threshold_varying/{method}_{ds}.json"):
                continue
            if ds == "SENDETEX" and method == "fastdetectgpt":
                continue
            results = evaluate_thresholds(preds, ds)
            metrics[dataset][detector] = results
            with open(f"data/threshold_varying/{method}_{ds}.json", "w") as f:
                json.dump(results, f)
    
    return metrics

def create_thresholds_table(method, root):
    datasets = ["AITDNA", "DETECTRL", "SENDETEX", "MIXSET", "COAUTHOR", "DETECTION"]
    results = {}
    for file in os.listdir(root):
        if method not in file:
            continue
        if "_new.json" not in file:
            continue
        dataset = file.split("_")[-2]
        with open(os.path.join(root, file), "r") as f:
            data = json.load(f)
        results[dataset] = data
    
    final_data = []
    for i in range(1, 10):
        threshold_data = []
        threshold = i / 10
        threshold_data.append(threshold)
        for dataset in datasets:
            threshold_data.append(results[dataset][str(threshold)][metric])
        final_data.append(threshold_data)
    
    cols = ["Threshold"] + datasets
    df = pd.DataFrame(columns=cols, data=final_data)
    df = df.round(3)
    df.to_csv(f"thresholds_{method}_{metric}.csv", index=False)


def evaluate_for_heatmap():
    dataset_root = "data/aitdna/formatted"
    result_root = "data/new_heatmap"
    method = MGTDMethod()

    membership_ds = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                     data_path=dataset_root,
                                     detection_level="membership")

    for approach in ["min_k", "binoculars", "fastdetectgpt", "log_rank", "modernBERT", "likelihood"]:
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_document.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, membership_ds)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_membership.json"), "w") as f:
            json.dump(results, f)

        if os.path.exists(os.path.join(result_root, f"metrics_{approach}_AITDNA_boundary.json")):
            continue

        dataset = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                   data_path=dataset_root,
                                   detection_level="document")
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_document.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, dataset)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_document.json"), "w") as f:
            json.dump(results, f)

        dataset = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                   data_path=dataset_root,
                                   detection_level="intent")

        dataset = list(dataset)
        dataset_list = []
        for sample in dataset:
            for local_sample in sample:
                dataset_list.append(local_sample)
        dataset = dataset_list
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_document.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, dataset)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_intent.json"), "w") as f:
            json.dump(results, f)

        dataset = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                   data_path=dataset_root,
                                   detection_level="content")
        dataset = list(dataset)
        dataset_list = []
        for sample in dataset:
            for local_sample in sample:
                dataset_list.append(local_sample)
        dataset = dataset_list
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_document.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, dataset)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_content.json"), "w") as f:
            json.dump(results, f)

        dataset = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                   data_path=dataset_root,
                                   detection_level="sentence")
        dataset = list(dataset)
        dataset_list = []
        for sample in dataset:
            for local_sample in sample:
                dataset_list.append(local_sample)
        dataset = dataset_list
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_sentence.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, dataset)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_sentence.json"), "w") as f:
            json.dump(results, f)

        dataset = DetectionDataset(dataset_name=DatasetName.AITDNA,
                                   data_path=dataset_root,
                                   detection_level="boundary")
        dataset = list(dataset)
        dataset_list = []
        for sample in dataset:
            for local_sample in sample:
                dataset_list.append(local_sample)
        dataset = dataset_list
        with open(f"data/all_predictions/predictions_{approach}_AITDNA_boundary.json", "r") as f:
            data = json.load(f)
        results = method.evaluate(data, dataset)
        with open(os.path.join(result_root, f"metrics_{approach}_AITDNA_boundary.json"), "w") as f:
            json.dump(results, f)


def plot_auroc_thresholds(metric: str = "AUROC", no_legend=False):
    """Plot metric values vs thresholds as a seaborn line graph with markers.

    Args:
        metric: One of "AUROC", "F-1", or "FPR".
    """

    metric_config = {
        "AUROC": {
            "file": "data/stats_and_plots/thresholds_likelihood_AUROC.csv",
            "ylabel": "AUROC",
            "ylim": (0.0, 1.05),
        },
        "F-1": {
            "file": "data/stats_and_plots/thresholds_likelihood_F-1.csv",
            "ylabel": "F1-Score",
            "ylim": (0.0, 1.05),
        },
        "FPR": {
            "file": "data/stats_and_plots/thresholds_likelihood_FPR.csv",
            "ylabel": "FPR",
            "ylim": (0.0, 1.05),
        },
    }

    if metric not in metric_config:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(metric_config.keys())}")

    config = metric_config[metric]

    df = pd.read_csv(config["file"])

    # Rename columns
    df = df.rename(columns={
        "COAUTHOR": "CoAuthor",
        "SENDETEX": "SenDetEx",
        "DETECTRL": "DetectRL",
        "MIXSET": "Mixset",
        "DETECTION": "BD",
    })

    df_long = df.melt(id_vars="Threshold", var_name="Method", value_name=metric)

    # Extend CoAuthor: duplicate its 0.8 value at 0.9
    coauthor_08 = df_long[(df_long["Method"] == "CoAuthor") & (df_long["Threshold"] == 0.8)]
    coauthor_09 = coauthor_08.copy()
    coauthor_09["Threshold"] = 0.9
    df_long = pd.concat([df_long, coauthor_09], ignore_index=True)

    # Fix method order
    method_order = ["AITDNA", "CoAuthor", "SenDetEx", "BD", "DetectRL", "Mixset"]
    df_long["Method"] = pd.Categorical(df_long["Method"], categories=method_order, ordered=True)
    df_long = df_long.sort_values(["Method", "Threshold"])

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    if not no_legend:
        plt.subplots_adjust(top=0.85)

    # Explicitly map each method to its tab10 color by position
    palette = {
        "AITDNA": "#1f77b4",
        "CoAuthor": "#ff7f0e",
        "Mixset": "#2ca02c",
        "BD": "#d62728",
        "DetectRL": "#9467bd",
        "SenDetEx": "#8c564b"
    }

    sns.lineplot(
        data=df_long,
        x="Threshold",
        y=metric,
        hue="Method",
        hue_order=method_order,
        marker="o",
        markersize=7,
        linewidth=2,
        palette=palette,
        ax=ax,
    )

    ax.set_xlabel("threshold", fontsize=14)
    ax.set_ylabel(config["ylabel"], fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_xticks([round(x * 0.1, 1) for x in range(1, 10)])
    ax.set_xlim(0.09, 0.91)
    ax.set_ylim(*config["ylim"])
    ax.tick_params(axis="both", labelsize=14)
    ax.get_legend().remove()
    if not no_legend:
        fig.legend(
            title=None,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(method_order) //2,
            fontsize=12,
            framealpha=0.4,
        )
    plt.tight_layout()

    output_path = f"data/stats_and_plots/thresholds_{metric}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def plot_dual_metric_thresholds(metric_left: str = "AUROC", metric_right: str = "FPR"):
    """Plot two metrics vs thresholds with dual y-axes.

    Left axis: solid lines with circle markers.
    Right axis: dashed lines with square markers.
    Legend is placed outside the plot to the right, showing methods by color only.

    Args:
        metric_left: Metric for the left y-axis. One of "AUROC", "F-1", "FPR".
        metric_right: Metric for the right y-axis. One of "AUROC", "F-1", "FPR".
    """

    metric_config = {
        "AUROC": {
            "file": "data/stats_and_plots/thresholds_likelihood_AUROC.csv",
            "ylabel": "AUROC",
            "ylim": (0.0, 1.05),
        },
        "F-1": {
            "file": "data/stats_and_plots/thresholds_likelihood_F-1.csv",
            "ylabel": "F1-Score",
            "ylim": (0.0, 1.05),
        },
        "FPR": {
            "file": "data/stats_and_plots/thresholds_likelihood_FPR.csv",
            "ylabel": "FPR",
            "ylim": (0.0, 1.05),
        },
    }

    if metric_left not in metric_config:
        raise ValueError(f"Unknown metric '{metric_left}'. Choose from: {list(metric_config.keys())}")
    if metric_right not in metric_config:
        raise ValueError(f"Unknown metric '{metric_right}'. Choose from: {list(metric_config.keys())}")

    method_order = ["AITDNA", "CoAuthor", "SenDetEx", "BD", "DetectRL", "Mixset"]
    tab10 = sns.color_palette("tab10")
    palette = {method: tab10[i] for i, method in enumerate(method_order)}

    rename_map = {
        "COAUTHOR": "CoAuthor",
        "SENDETEX": "SenDetEx",
        "DETECTRL": "DetectRL",
        "MIXSET": "Mixset",
        "DETECTION": "BD",
    }

    def load_metric(metric):
        df = pd.read_csv(metric_config[metric]["file"])
        df = df.rename(columns=rename_map)
        df_long = df.melt(id_vars="Threshold", var_name="Method", value_name=metric)
        # Extend CoAuthor: duplicate its 0.8 value at 0.9
        coauthor_08 = df_long[(df_long["Method"] == "CoAuthor") & (df_long["Threshold"] == 0.8)]
        coauthor_09 = coauthor_08.copy()
        coauthor_09["Threshold"] = 0.9
        df_long = pd.concat([df_long, coauthor_09], ignore_index=True)
        df_long["Method"] = pd.Categorical(df_long["Method"], categories=method_order, ordered=True)
        return df_long.sort_values(["Method", "Threshold"])

    df_left = load_metric(metric_left)
    df_right = load_metric(metric_right)

    sns.set_theme(style="darkgrid")

    fig, ax_left = plt.subplots(figsize=(9, 6))
    ax_right = ax_left.twinx()

    # Plot left axis: solid lines, circle markers
    for method in method_order:
        data = df_left[df_left["Method"] == method]
        ax_left.plot(
            data["Threshold"],
            data[metric_left],
            color=palette[method],
            linestyle="-",
            marker="x",
            markersize=4,
            linewidth=1,
        )

    # Plot right axis: dashed lines, square markers
    for method in method_order:
        data = df_right[df_right["Method"] == method]
        ax_right.plot(
            data["Threshold"],
            data[metric_right],
            color=palette[method],
            linestyle="--",
            marker="^",
            markersize=4,
            linewidth=1,
        )

    # Axis formatting
    for ax, metric, side in [(ax_left, metric_left, "left"), (ax_right, metric_right, "right")]:
        ax.set_ylim(*metric_config[metric]["ylim"])
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel(metric_config[metric]["ylabel"], fontsize=14)

    ax_left.set_xlabel("Threshold", fontsize=12)
    ax_left.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_left.set_xticks([round(x * 0.1, 1) for x in range(1, 10)])
    ax_left.set_xlim(0.09, 0.91)
    ax_left.tick_params(axis="x", labelsize=14)

    # Legend: one entry per method (color only), placed outside to the right
    legend_handles = [
        mlines.Line2D([], [], color=palette[method], linewidth=2, label=method)
        for method in method_order
    ]
    # Add a style legend for the two axes
    #style_handles = [
    #    mlines.Line2D([], [], color="gray", linewidth=2, linestyle="-",  marker="o", markersize=7, label=metric_left),
    #    mlines.Line2D([], [], color="gray", linewidth=2, linestyle="--", marker="s", markersize=7, label=metric_right),
    #]

    ax_left.legend(
        handles=legend_handles, #+ style_handles,
        title=None,
        loc="upper left",
        bbox_to_anchor=(1.08, 1),
        fontsize=12,
        framealpha=0.7,
        borderaxespad=0,
    )

    plt.tight_layout()

    output_path = f"data/stats_and_plots/thresholds_{metric_left}_{metric_right}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

#plot_dual_metric_thresholds("F-1", "FPR")
#plot_auroc_thresholds("F-1")
#plot_auroc_thresholds("FPR", no_legend=True)

plot_heatmap_notion_performance_stylish()
#plot_heatmap_notion_performance_stylish("AUROC")
#plot_heatmap_notion_performance_stylish("Precision")
#plot_heatmap_notion_performance_stylish("Recall")
plot_heatmap_notion_performance_stylish("FPR", invert_colorscheme=True)
