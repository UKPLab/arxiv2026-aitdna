import os
import json
import argparse
from collections import defaultdict
from aitdna.notions.data_loading import DatasetName, AitdDataset, Notion, Population
from aitdna.experiments.mgtd.mgtd_datasets.DetectionDataset import DetectionDataset

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, auc,  roc_curve, confusion_matrix, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, auc,  roc_curve, confusion_matrix, precision_recall_curve

def evaluate_document_level_pangram(gt_token: list[dict], gt_document: list[dict], prediction: dict):
    res = {}
    gt = {}
    authors = list(set(token["author"] for token in gt_token))
    gt["document_HAM"] = "Mixed" if len(authors) > 1 else authors[0]

    res["document_HAM"] = "Mixed" if prediction["prediction_short"] in ["Mixed", "AI-Assisted"] else "Bot" if prediction["prediction_short"] == "AI" else "User"
    
    gt_ai_percentage = len([token for token in gt_token if token["author"] != "User"]) / len(gt_token)
    predicted_ai_percentage = prediction["fraction_ai"] + prediction["fraction_ai_assisted"]
    res["ai_percentage"] = predicted_ai_percentage
    gt["ai_percentage"] = gt_ai_percentage

    if "AI-generated" in gt_document:
        gt["document_HA"] = "Bot" if gt_document["AI-generated"] else "User"
    else:
        gt["document_HA"] = gt_document[0]["author"]
    res["document_HA"] = "User" if prediction["prediction_short"] == "Human" else "Bot"
    return res, gt


def evaluate_document_level_gptzero(gt_token: list[dict], gt_document: list[dict], prediction: dict):
    prediction = prediction["documents"][0]
    res = {}
    gt = {}
    authors = set(token["author"] for token in gt_token)
    authors = list(set(token["author"] for token in gt_token))
    gt["document_HAM"] = "Mixed" if len(authors) > 1 else authors[0]
    res["document_HAM"] = "Mixed" if prediction["document_classification"] == "MIXED" else "User" if prediction["document_classification"] == "HUMAN_ONLY" else "Bot"

    if "AI-generated" in gt_document:
        gt["document_HA"] = "Bot" if gt_document["AI-generated"] else "User"
    else:
        gt["document_HA"] = gt_document[0]["author"]
    res["document_HA"] = "User" if prediction["document_classification"] == "HUMAN_ONLY" else "Bot"
    return res, gt


def evaluate_sentence_level_pangram(gt_sentence: list[dict], prediction: dict, index):
    results = defaultdict(list)
    gt = defaultdict(list)
    for sent in gt_sentence:
        sent_text = sent["text"].replace('"', '').replace("''", "").replace("*", "")
        found = False
        for window in prediction["windows"]:
            window_text = window["text"].replace('"', '').replace("''", "").replace("*", "")
            if sent_text in window_text:
                found = True
                res = "User" if window["label"] == "Human Written" else "Bot"
                true_label = "User" if sent["author"] == "User" else "Bot"
                results["sentence_HA"].append(res)
                gt["sentence_HA"].append(true_label)

                res = "User" if window["label"] == "Human Written" else "Bot" if window["label"] == "AI-Generated" else "Mixed"
                results["sentence_HAM"].append(res)
                gt["sentence_HAM"].append(sent["author"])
                break
        if not found:
            print(index)
    return results, gt


def evaluate_sentence_level_gptzero(gt_sentence: list[dict], prediction: dict, sentence_threshold: float, index):
    prediction = prediction["documents"][0]
    results = defaultdict(list)
    gt = defaultdict(list)
    for sent in gt_sentence:
        gt_sent_text = sent["text"].replace('"', '').replace("''", "").replace("*", "")
        found = False
        for sentence in prediction["sentences"]:
            pred_sent_text = sentence["sentence"].replace('"', '').replace("''", "").replace("*", "")
            if gt_sent_text in pred_sent_text or pred_sent_text in gt_sent_text:
                found = True
                results["sentence_HA"].append("User" if sentence["generated_prob"] < sentence_threshold else "Bot")
                gt["sentence_HA"].append("User" if sent["author"] == "User" else "Bot")
                
                author = max(sentence["class_probabilities"], key=sentence["class_probabilities"].get)
                results["sentence_HAM"].append("User" if author == "human" else "Bot" if author == "ai" else "Mixed")
                gt["sentence_HAM"].append(sent["author"])
                break
        if not found:
            gt["sentence_HA"].append(sent["author"])
            results["sentence_HA"].append("Bot" if sent["author"] == "User" else "User")

            gt["sentence_HAM"].append(sent["author"])
            results["sentence_HAM"].append("Bot" if sent["author"] == "User" else "User")
    return results, gt


def multi_class_metric_to_dict(metric: callable, gt: list, pred: list):
    scores = metric(gt, pred, labels=["Bot", "User", "Mixed"], average=None, zero_division=1)
    return {
        "Bot": round(float(scores[0]), 3),
        "User": round(float(scores[1]), 3),
        "Mixed": round(float(scores[2]), 3)
    }

def evaluate_document_level(eval_root, token_level_dataset, document_level_dataset, framework_type):
    results = defaultdict(list)
    gt = defaultdict(list)

    for file in os.listdir(eval_root):
        with open(os.path.join(eval_root, file), "r") as f:
            prediction = json.load(f)

        index = int(file.replace(".json", ""))
        gt_token = token_level_dataset[index]
        gt_document = document_level_dataset[index]
        if framework_type == "pangram":
            document_level_result, document_level_gt = evaluate_document_level_pangram(gt_token, gt_document, prediction)
        else:
            document_level_result, document_level_gt = evaluate_document_level_gptzero(gt_token, gt_document, prediction)
        
        for k, v in document_level_result.items():
            results[k].append(v)
        for k, v in document_level_gt.items():
            gt[k].append(v)

    for k, v in results.items():
        if k == "document_HA":
            conf_matrix = confusion_matrix(gt[k], v, labels=["User", "Bot"])
            tn, fp, fn, tp = conf_matrix.ravel()
            fpr = fp/(fp+tn)

            return {
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                "Precision": round(precision_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "Recall": round(recall_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "F-1": round(f1_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "FPR": round(fpr, 3)
                "F-1": round(f1_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "FPR": round(fpr, 3)
            }

def evaluate_sentence_level(eval_root, sentence_level_dataset, framework_type, sentence_threshold=None):
    results = defaultdict(list)
    gt = defaultdict(list)

    for file in os.listdir(eval_root):
        with open(os.path.join(eval_root, file), "r") as f:
            prediction = json.load(f)

        index = int(file.replace(".json", ""))
        gt_sentence = sentence_level_dataset[index]
        if framework_type == "pangram":
            sentence_level_result, sentence_level_gt = evaluate_sentence_level_pangram(gt_sentence, prediction, index)
        else:
            sentence_level_result, sentence_level_gt = evaluate_sentence_level_gptzero(gt_sentence, prediction, sentence_threshold, index)
        
        for k, v in sentence_level_result.items():
            results[k].extend(v)
        for k, v in sentence_level_gt.items():
            gt[k].extend(v)

    for k, v in results.items():
        if k == "sentence_HA":
            conf_matrix = confusion_matrix(gt[k], v, labels=["User", "Bot"])
            tn, fp, fn, tp = conf_matrix.ravel()
            fpr = fp/(fp+tn)
            return {
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                "Precision": round(precision_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "Recall": round(recall_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "F-1": round(f1_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "FPR": round(fpr, 3)
            }


def evaluate(eval_root, dataset_root, framework_type, sentence_threshold=None):
    results = defaultdict(list)
    gt = defaultdict(list)

    token_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.TOKEN_LEVEL)
    document_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.DOCUMENT_LEVEL)
    sentence_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.SENTENCE_LEVEL)

    for file in os.listdir(eval_root):
        with open(os.path.join(eval_root, file), "r") as f:
            prediction = json.load(f)

        index = int(file.replace(".json", ""))
        gt_token = token_level_dataset[index]
        gt_document = document_level_dataset[index]
        gt_sentence = sentence_level_dataset[index]
        if framework_type == "pangram":
            document_level_result, document_level_gt = evaluate_document_level_pangram(gt_token, gt_document, prediction)
            sentence_level_result, sentence_level_gt = evaluate_sentence_level_pangram(gt_sentence, prediction, index)
        else:
            document_level_result, document_level_gt = evaluate_document_level_gptzero(gt_token, gt_document, prediction)
            sentence_level_result, sentence_level_gt = evaluate_sentence_level_gptzero(gt_sentence, prediction, sentence_threshold, index)
        
        for k, v in document_level_result.items():
            results[k].append(v)
        for k, v in document_level_gt.items():
            gt[k].append(v)
        for k, v in sentence_level_result.items():
            results[k].extend(v)
        for k, v in sentence_level_gt.items():
            gt[k].extend(v)


    final_results = []
    for k, v in results.items():
        if k== "document_HAM":
            final_results.append({
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                "Precision": multi_class_metric_to_dict(precision_score, gt[k], v),
                "Recall": multi_class_metric_to_dict(recall_score, gt[k], v),
                "F-1": multi_class_metric_to_dict(f1_score, gt[k], v)
            })
        elif k == "document_HA":
            gt_binary = [1 if g == "Bot" else 0 for g in gt[k]]
            pred_binary = [1 if g == "Bot" else 0 for g in v]
            fpr, tpr, _ = roc_curve(gt_binary, pred_binary, pos_label="Bot")
            final_results.append({
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                # "auc": round(auc(fpr, tpr), 3),
                "Precision": round(precision_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "Recall": round(recall_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "F-1": round(f1_score(gt[k], v, pos_label="Bot", zero_division=1), 3)
            })
        elif k == "ai_percentage":
            final_results.append({
                "Model": framework_type,
                "Evaluation": k,
                "MAE": mean_absolute_error(gt[k], v)
            })
        elif k == "sentence_HA":
            final_results.append({
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                "Precision": round(precision_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "Recall": round(recall_score(gt[k], v, pos_label="Bot", zero_division=1), 3),
                "F-1": round(f1_score(gt[k], v, pos_label="Bot", zero_division=1), 3)
            })
        elif k == "sentence_HAM":
            final_results.append({
                "Model": framework_type,
                "Evaluation": k,
                "Accuracy": round(accuracy_score(gt[k], v), 3),
                "Precision": multi_class_metric_to_dict(precision_score, gt[k], v),
                "Recall": multi_class_metric_to_dict(recall_score, gt[k], v),
                "F-1": multi_class_metric_to_dict(f1_score, gt[k], v)
            })
    return final_results


def evaluate_for_heatmap(eval_root, dataset_root, framework_type, sentence_threshold=None):
    root = "data/new_heatmap_commercials"
    token_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.TOKEN_LEVEL)
    document_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.DOCUMENT_LEVEL)

    document_level_results = evaluate_document_level(eval_root=eval_root,
                                      token_level_dataset=token_level_dataset,
                                      document_level_dataset=document_level_dataset,
                                      framework_type=framework_type)

    with open(os.path.join(root, f"metrics_{framework_type}_AITDNA_document.json"), "w") as f:
        json.dump(document_level_results, f)

    intent_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.INTENT_BASED)
    intent_level_results = evaluate_document_level(eval_root=eval_root,
                                      token_level_dataset=token_level_dataset,
                                      document_level_dataset=intent_level_dataset,
                                      framework_type=framework_type)
    with open(os.path.join(root, f"metrics_{framework_type}_AITDNA_intent.json"), "w") as f:
        json.dump(intent_level_results, f)
    
    content_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.CONTENT_BASED)
    content_level_results = evaluate_document_level(eval_root=eval_root,
                                      token_level_dataset=token_level_dataset,
                                      document_level_dataset=content_level_dataset,
                                      framework_type=framework_type)
    with open(os.path.join(root, f"metrics_{framework_type}_AITDNA_content.json"), "w") as f:
        json.dump(content_level_results, f)

    sentence_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                root_dir=dataset_root,
                notion=Notion.SENTENCE_LEVEL)
    sentence_level_results = evaluate_sentence_level(eval_root=eval_root,
                                      sentence_level_dataset=sentence_level_dataset,
                                      sentence_threshold=sentence_threshold,
                                      framework_type=framework_type)
    with open(os.path.join(root, f"metrics_{framework_type}_AITDNA_sentence.json"), "w") as f:
        json.dump(sentence_level_results, f)

    membership_dataset = DetectionDataset(data_path=dataset_root,
            dataset_name=DatasetName.AITDNA,
            detection_level="membership",
            threshold=0.5)

    membership_level_results = evaluate_document_level(eval_root=eval_root,
                                      token_level_dataset=token_level_dataset,
                                      document_level_dataset=membership_dataset,
                                      framework_type=framework_type)
    with open(os.path.join(root, f"metrics_{framework_type}_AITDNA_membership.json"), "w") as f:
        json.dump(membership_level_results, f)

    
def main(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--path_to_eval_root", type=str,
                           help="Path to the root of prediction files",
                           default="data/evaluation_results/GPTZero")
    argparser.add_argument("-d", "--path_to_dataset_root", type=str,
                           help="Path to the dataset root",
                           default="data/aitdna/formatted")
    argparser.add_argument("-r", "--path_to_results", type=str,
                           help="Path to the json where the metrics will be saved",
                           default="data/predictor_test/gptzero.json")
    argparser.add_argument("-f", "--framework_type", type=str,
                           help="pangram or gptzero",
                           default="gptzero")
    argparser.add_argument("-t", "--sentence_threshold", type=float,
                           help="Prediction threshold",
                           default=0.5)
    argparser.add_argument("-m", "--results_for_heatmap", action="store_true")

    argv = argparser.parse_args(argv)
    
    path_to_eval_root = argv.path_to_eval_root
    path_to_dataset_root = argv.path_to_dataset_root
    framework_type = argv.framework_type
    sentence_threshold = float(argv.sentence_threshold)
    path_to_results = argv.path_to_results
    results_for_heatmap = argv.results_for_heatmap

    if not results_for_heatmap:
        results = evaluate(eval_root=path_to_eval_root, dataset_root=path_to_dataset_root, framework_type=framework_type, sentence_threshold=sentence_threshold)
        with open(path_to_results, "w", encoding="utf-8") as f:
            json.dump(results, f)
    else:
        evaluate_for_heatmap(eval_root=path_to_eval_root,
                             dataset_root=path_to_dataset_root,
                             framework_type=framework_type,
                             sentence_threshold=sentence_threshold)

if __name__ =="__main__":
    main()