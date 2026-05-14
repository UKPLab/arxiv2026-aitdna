import json
import os
import nltk
import requests
from dotenv import load_dotenv
from typing import List
import re

from sklearn.metrics import precision_score, recall_score, accuracy_score

load_dotenv()
API_KEY = os.getenv("ZEROGPT_API_KEY")

def predict_zero_gpt(text):
    url = "https://api.zerogpt.com/api/detect/detectText"

    payload = json.dumps({
    "input_text": text})

    headers = {
    'ApiKey': API_KEY,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=120)
    answer = response.json()["data"]
    ai_percentage = answer["fakePercentage"]
    ai_generated = answer["h"]
    return ("ZeroGPT", ai_generated, ai_percentage)

def tokenize_with_spans(text: str):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    return [{'token': text[s:e], 'start': s, 'end': e} for s,e in spans]

def label_tokens_from_pred_substrings(text: str, prediction: List[str]):
    text = re.sub(' +', ' ', text)
    prediction = [re.sub(' +', ' ', pred) for pred in prediction]
    tokens = tokenize_with_spans(text)
    pred_labels = ["User"] * len(tokens)
    pos = 0
    for sub in prediction:
        idx = text.find(sub, pos)
        end = idx + len(sub)
        for i, t in enumerate(tokens):
            if t['end'] > idx and t['start'] < end:
                pred_labels[i] = "Bot"
        pos = end
    for i, t in enumerate(tokens):
        t['pred_label'] = pred_labels[i]
    return tokens

def compute_metrics(gt, pred):
    accuracy = round(accuracy_score(gt, pred), 3)
    precision_bot = round(precision_score(gt, pred, zero_division=1, pos_label="Bot"), 3)
    recall_bot = round(recall_score(gt, pred, zero_division=1, pos_label="Bot"), 3)

    precision_user = round(precision_score(gt, pred, zero_division=1, pos_label="User"), 3)
    recall_user = round(recall_score(gt, pred, zero_division=1, pos_label="User"), 3)
    return {"Accuracy": accuracy,"Bot": {"Precision": precision_bot, "Recall": recall_bot},
            "User": {"Precision": precision_user, "Recall": recall_user} }

def evaluate_zero_gpt_tokenwise(text: str, gt_labels, prediction: list[str]):
    """
    Evaluate ZeroGPT tokenwise
    
    :param text: Description
    :param gt_labels: Description
    :param prediction: list
    """

    tokens = label_tokens_from_pred_substrings(text, prediction)
    gt = []
    for i, t in enumerate(tokens):
        label = gt_labels[i][1] if gt_labels[i][1] != "Mixed" else "Bot"
        gt.append(label)
    pred = [t["pred_label"] for t in tokens]
    return {"gt": gt, "pred": pred}

def get_labels_sentencewise(gt: list[dict[str, object]], prediction: list[str], count_mixed_as_bot: True) -> dict[str, list[str]]:
    """
    Returns a dict with gt and prediction labels.

    :param gt: ground truth labels for each sentence
    :param prediction: Bot prediction. Strings in this list are predicted as Bot
    :param count_mixed_as_bot: whether mixed sentences count as llm
    """
    # fix double space problem (in gt, some sentences have double spaces between words; prediction doesn't)
    prediction = [" ".join(pred.split()) for pred in prediction]
    for pred in prediction:
        pred_in_txt = False
        for text in gt:
            txt = " ".join(text["text"].split())
            if pred in txt or txt in pred:
                pred_in_txt = True
        if not pred_in_txt:
            raise Exception("Pred sentence not found in gt!")

    gt_labels = []
    pred_labels = []
    for sentence in gt:
        author = sentence["author"]
        gt_text = " ".join(sentence["text"].split())
        if  author == "Mixed":
            author = "Bot" if count_mixed_as_bot else "User"
        gt_labels.append(author)

        if gt_text.strip() in prediction:
            pred_labels.append("Bot")
        else:
            if any(gt_text in pred or pred in gt_text for pred in prediction):
                pred_labels.append("Bot")
            else:
                pred_labels.append("User")
    return {"gt": gt_labels, "pred": pred_labels}
    
