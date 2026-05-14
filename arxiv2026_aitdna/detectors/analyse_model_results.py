import os
import random
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from .ZeroGPT import evaluate_zero_gpt_tokenwise, get_labels_sentencewise
from ..dataset.notions.AITDNotions import AITDNotions

ROOT = "datasets/formatted/"
study = "txaitd_data_2026_01_22"
THRESHOLD = 0.5


def build_results_df_per_model(root: str, threshold: float):
    """
    Returns result df containing model prediction and ground truth for all model-user-task combinations.
    :returns pd.DataFrame
    """
    records = {}
    for study in os.listdir(root):
        study_path = os.path.join(root, study)
        for user in os.listdir(study_path):
            user_path = os.path.join(study_path, user)
            for task in os.listdir(user_path):
                task_path = os.path.join(user_path, task)
                model_info_path = os.path.join(task_path, "statistics", "user_task_assignment.json")
                if not os.path.exists(model_info_path):
                    continue
                with open(model_info_path, "r") as f:
                    model = json.load(f)
                    if "model" not in model:
                        continue
                    model = model["model"]
                if model not in records:
                    records[model] = []
                task_df = pd.read_csv(os.path.join(task_path, "model_results", "model_results.csv"))
                with open(os.path.join(task_path, "statistics", "stats.json"), "r") as f:
                    true_results = json.load(f)
                ai_generated = 1 if true_results["Bot"]["percentage_authorship"] > threshold else 0
                for _, row in task_df.iterrows():
                    records[model].append({
                        "Model": row["Model"],
                        "User": user,
                        "Task": task,
                        "Prediction": row["Class"],
                        "Ground Truth": ai_generated
                    })
                records[model].append({
                    "Model": "Random",
                    "User": user,
                    "Task": task,
                    "Prediction": random.randint(0, 1),
                    "Ground Truth": ai_generated
                })
        
    return {model: pd.DataFrame(data) for model, data in records.items()}

def build_results_df(root: str, threshold: float):
    """
    Returns result df containing model prediction and ground truth for all model-user-task combinations.
    :returns pd.DataFrame
    """
    records = []
    for study in os.listdir(root):
        study_path = os.path.join(root, study)
        for user in os.listdir(study_path):
            user_path = os.path.join(study_path, user)
            for task in os.listdir(user_path):
                task_path = os.path.join(user_path, task)
                task_df = pd.read_csv(os.path.join(task_path, "model_results", "model_results.csv"))
                with open(os.path.join(task_path, "statistics", "stats.json"), "r") as f:
                    true_results = json.load(f)
                ai_generated = 1 if true_results["Bot"]["percentage_authorship"] > threshold else 0
                for _, row in task_df.iterrows():
                    records.append({
                        "Model": row["Model"],
                        "User": user,
                        "Task": task,
                        "Prediction": row["Class"],
                        "Ground Truth": ai_generated
                    })
                records.append({
                    "Model": "Random",
                    "User": user,
                    "Task": task,
                    "Prediction": random.randint(0, 1),
                    "Ground Truth": ai_generated
                })
        
    return pd.DataFrame(records)

def compute_metrics(df):
    metrics = []
    for model, g in df.groupby("Model"):
        y_true = g["Ground Truth"]
        y_pred = g["Prediction"]

        metrics.append({
            "Model": model,
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=1), 3),
            "Recall": round(recall_score(y_true, y_pred, zero_division=1), 3),
            "F1": round(f1_score(y_true, y_pred, zero_division=1), 3)
        })
    return pd.DataFrame(metrics)


def compute_metrics_given_labels(gt: list[str], pred: list[str]) -> tuple[float, float, float, float, float]:
    """
    Computes accuracy, precision, and recall for gt-prediction pairs
    
    :param gt: list with ground truth labels
    :type gt: list[str]
    :param pred: list with predicted labels
    :type pred: list[str]
    :returns tuple of metrics: accuracy, precision for bot, recall for bot, precision for user, recall for user
    """
    accuracy = round(accuracy_score(gt, pred), 3)
    precision_bot = round(precision_score(gt, pred, zero_division=1, pos_label="Bot"), 3)
    recall_bot = round(recall_score(gt, pred, zero_division=1, pos_label="Bot"), 3)

    precision_user = round(precision_score(gt, pred, zero_division=1, pos_label="User"), 3)
    recall_user = round(recall_score(gt, pred, zero_division=1, pos_label="User"), 3)
    return  (accuracy, precision_bot, recall_bot, precision_user, recall_user)

def compute_macro_metrics(results: dict[str, list[str]]) -> tuple[float, float, float, float, float]:
    """
    Compute macro accuracy, precision, recall
    
    :param results: {pred: list of prediction labels, gt: list of GT labels}
    :returns tuple of metrics: accuracy, precision for bot, recall for bot, precision for user, recall for user
    """
    metrics = []
    for res in results:
        gt = res["gt"]
        pred = res["pred"]
        metrics.append(compute_metrics_given_labels(gt, pred))
    macro_accuracy = round(sum(result[0] for result in metrics) / len(metrics), 3)
    macro_precision_bot = round(sum(result[3] for result in metrics) / len(metrics), 3)
    macro_recall_bot = round(sum(result[4] for result in metrics) / len(metrics), 3)
    macro_precision_user = round(sum(result[1] for result in metrics) / len(metrics), 3)
    macro_recall_user = round(sum(result[2] for result in metrics) / len(metrics), 3)
    return  {"MaAccuracy": [macro_accuracy], "MaP User": [macro_precision_user],
                    "MaP Bot": [macro_precision_bot], "MaR User": [macro_recall_user], "MaR Bot": [macro_recall_bot]}

def compute_micro_metrics(results):
    """
    Compute micro accuracy, precision, recall
    
    :param results: {pred: list of prediction labels, gt: list of GT labels}
    :returns tuple of metrics: accuracy, precision for bot, recall for bot, precision for user, recall for user
    """
    gt = [g for result in results for g in result["gt"]]
    pred = [p for result in results for p in result["pred"]]
    (micro_accuracy, micro_precision_bot, micro_recall_bot, micro_precision_user, micro_recall_user) = compute_metrics_given_labels(gt, pred)
    return  {"MiAccuracy": [micro_accuracy], "MiP User": [micro_precision_user],
                    "MiP Bot": [micro_precision_bot], "MiR User": [micro_recall_user], "MiR Bot": [micro_recall_bot]}

def compute_tokenwise_results(root):
    """
    Compute and save results of the tokenwise evaluation, macro and micro
    
    :param root: path to root folder
    :returns DataFrame with computed metrics
    """
    aitd = AITDNotions()
    
    results = []
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "final_text.txt"), "r", encoding="utf-8") as f:
                text = f.read()
            with open(os.path.join(task_path, "edits.json"), "r", encoding="utf-8") as f:
                edits = json.load(f)
            with open(os.path.join(task_path, "model_results", "fine-grained_results.json", encoding="utf-8"), "r") as f:
                model_results = json.load(f)
            ai_generated = model_results["ai_generated"]
            gt_tokenwise = aitd.get_final_text_by_user_tokenwise(edits)
            result_tokenwise = evaluate_zero_gpt_tokenwise(text, gt_tokenwise, ai_generated)
            results.append(result_tokenwise)

    macro = compute_macro_metrics(results)
    micro = compute_micro_metrics(results)
    return pd.DataFrame({"Model": ["ZeroGPT"], **macro, **micro})

def compute_sentencewise_results(root: str) -> pd.DataFrame:
    """
    Compute and save results of the sentencewise evaluation, macro and micro
    
    :param root: path to root folder
    :returns DataFrame with computed metrics
    """
    results = []
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "fine-grained_results.json"), "r", encoding="utf-8") as f:
                model_results = json.load(f)
            ai_generated = model_results["ai_generated"]

            with open(os.path.join(task_path, "final_text_by_user_sentencewise.json"), "r", encoding="utf-8") as f:
                gt_sentencewise = json.load(f)
            result_sentencewise = get_labels_sentencewise(gt_sentencewise, ai_generated, count_mixed_as_bot=True)
            results.append(result_sentencewise)
    macro = compute_macro_metrics(results)
    micro = compute_micro_metrics(results)
    return pd.DataFrame({"Model": ["ZeroGPT"], **macro, **micro})


def compute_acceptance_stats(root: str) -> tuple[int, int]:
    """
    Compute # acceptance of LLM suggestions and total answers
    
    :param root: path to root folder 
    :type root: str
    :returns tuple: number of accepted answers, total number of answers
    """
    total_answers = 0
    accepted = 0
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "edits.json"), "r") as f:
                data = json.load(f)
            for entry in data:
                if "requestId" in entry and entry["accepted"] is not None:
                    total_answers += 1
                    accepted += 1 if entry["accepted"] else 0
    return (accepted, total_answers)


def main():
    
    # df = build_results_df(ROOT, THRESHOLD)
    # df.to_csv(os.path.join("model_results", "overall", "result_table.csv"))
    # metrics_df = compute_metrics(df)
    # metrics_df.to_csv(os.path.join("model_results", "overall", "metrics_table.csv"))

    df_per_model = build_results_df_per_model(ROOT, THRESHOLD)
    for model, data in df_per_model.items():
        data.to_csv(os.path.join("model_results", "overall", "metrics_per_model", f"results_{model}.csv"))
        metrics_df = compute_metrics(data)
        metrics_df.to_csv(os.path.join("model_results", "overall", f"metrics_{model}.csv"))
    # result = compute_tokenwise_results(root)
    # result.to_csv("model_results/finegrained_result_table.csv")

    # result = compute_sentencewise_results(root)
    # result.to_csv("model_results/finegrained_result_sentencewise_table.csv")

    # (accepted, total_answers) = compute_acceptance_stats(ROOT)
    # rate = accepted / total_answers
    # with open(os.path.join("model_results", study, "acceptance.json"), "w") as f:
    #     json.dump({"total_answers": total_answers, "accepted": accepted, "acceptance_rate": rate}, f)

if __name__ == "__main__":
    main()