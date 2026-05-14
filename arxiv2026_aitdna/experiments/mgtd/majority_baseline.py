import os
import json
from txaitd.notions.data_loading import DatasetName, AitdDataset, Notion, Population
from txaitd.experiments.mgtd.methods.generation import MGTDMethod
from txaitd.experiments.mgtd.mgtd_datasets.DetectionDataset import DetectionDataset
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, \
                                    accuracy_score, f1_score
datasets = {
    "AITDNA": "data/aitdna/formatted",
    "COAUTHOR": "data/other_datasets/processed/coauthor-v1.0",
    "MIXSET": "data/other_datasets/processed/mixset",
    "DETECTRL": "data/other_datasets/processed/detectRL",
    "BOUNDARY_DETECTION": "data/other_datasets/processed/boundary_detection",
    "SENDETEX": "data/other_datasets/processed/senDetEx"
}

def majority_baseline():
    mgtd = MGTDMethod()
    root = "data/all_predictions"
    for ds, path in datasets.items():
        dataset = DetectionDataset(data_path=path, dataset_name=DatasetName[ds],
                                   detection_level="document", threshold=0.5)
        ai_texts = [text for text in dataset if text["author"] == "Bot"]
        majority_label = "Bot" if len(ai_texts) / len(dataset) > 0.5 else "User"
        predictions = [1 if majority_label == "Bot" else 0 for _ in range(len(dataset))]
        results = mgtd.evaluate(predictions, dataset)

        with open(os.path.join(root, f"metrics_majBase_{ds}_document.json"), "w") as f:
            json.dump(results, f)



def majority_baseline_implemented():
    root = "data/all_predictions"
    for ds, path in datasets.items():
        dataset = DetectionDataset(data_path=path, dataset_name=DatasetName[ds],
                                   detection_level="document", threshold=0.5)
        real_labels = [1  if text["AI-generated"] else 0 for text in dataset]
        majority_label = "Bot" if sum(real_labels) / len(dataset) > 0.5 else "User"

        predictions = [1 if majority_label == "Bot" else 0 for _ in range(len(dataset))]

        conf_matrix = confusion_matrix(real_labels, predictions, labels=[0, 1])
        precision = precision_score(real_labels, predictions)
        recall = recall_score(real_labels, predictions)
        f1 = f1_score(real_labels, predictions)
        accuracy = accuracy_score(real_labels, predictions)

        results = {
            "f-1": f1,
            "recall": recall
        }

        with open(os.path.join(root, f"metrics_majBase_{ds}_document.json"), "w") as f:
            json.dump(results, f)

majority_baseline()