import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score


def test_content_based():
    root_path = "."
    with open(f"{root_path}/rsc/eval/content-labels.json", "r") as file:
        gold = json.load(file)

    AITDNA_DATA_DIR = Path(root_path) / "data" / "aitdna" / "formatted"

    all_pred, all_gold = [],[]
    for gid in gold:
        pred_path = AITDNA_DATA_DIR / gid / "notions" / "content_based_labels_gpt-5.4-nano.json"
        with open(pred_path, "r") as file:
            pred = json.load(file)

        sentence_wise = AITDNA_DATA_DIR / gid / "notions" / "final_text_by_user_sentence_level.json"
        with open(sentence_wise, "r") as file:
            sentences = json.load(file)

        sentences = [s for s in sentences if s["author"] != "User"]

        pred_filtered = []
        for sent, p in zip(sentences, pred):
            if sent["author"] == "Bot":
                pred_filtered.append(p)

        pred = pred_filtered

        assert len(pred) == len(gold[gid]), print(f"predictions and gold dont have the same target sentences. {len(pred)} vs gold {len(gold[gid])}")

        all_pred += pred
        all_gold += gold[gid]

    # evaluate rule identification performance (f1-macro on multilabel)
    ALL_LABELS = ["C1", "C2", "C3", "C4"]

    def to_indicator(label_list: list[str]) -> list[int]:
        """Convert a list of label strings to a fixed-order binary indicator vector."""
        return [1 if l in label_list else 0 for l in ALL_LABELS]

    def macro_f1_multilabel(example, predicted, trace=None) -> float:
        y_true = [to_indicator(l) for l in example]  # e.g. list of [1, 0, 1, 1, 0]
        y_pred = [to_indicator(p) for p in predicted]  # e.g.  list of [0, 0, 1, 1, 1]

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return float(f1)

    ma_f1_rules = macro_f1_multilabel(all_gold, all_pred)

    # evaluate classification performance
    strictness_level = 3

    def matches_slevel(x: str):
        if type(x) is not str or len(x) < 2 or not x.startswith("C"):
            return True
        return int(x[1:]) > strictness_level

    def to_binary(multilabels):
        r = []
        for m in multilabels:
            no_violation = all(map(matches_slevel, m))
            r += [0 if no_violation else 1]

        return r

    all_pred_binary = to_binary(all_pred)
    all_gold_binary = to_binary(all_gold)

    print("pred", Counter(all_pred_binary), "gold", Counter(all_gold_binary))

    f1_classification = f1_score(all_gold_binary, all_pred_binary, zero_division=0)
    acc_classification = accuracy_score(all_gold_binary, all_pred_binary)

    return {
        "macro_f1_multilabel_rules": ma_f1_rules,
        "f1_binary_classification": f1_classification,
        "acc_binary_classification": acc_classification
    }

print("CONTENT")
r = test_content_based()
print(r)


def test_intent_based():
    root_path = "."
    with open(f"{root_path}/rsc/eval/intent-labels.json", "r") as file:
        gold = json.load(file)

    AITDNA_DATA_DIR = Path(root_path) / "data" / "aitdna" / "formatted"

    all_pred, all_gold = [],[]
    for gid in gold:
        pred_path = AITDNA_DATA_DIR / gid / "notions" / "intent_based_labels_gpt-5.4-nano.json"
        with open(pred_path, "r") as file:
            pred = json.load(file)

        assert len(pred) == len(gold[gid]), print(f"predictions and gold dont have the same target sentences. {len(pred)} vs gold {len(gold[gid])}")

        all_pred += pred
        all_gold += gold[gid]

    # evaluate rule identification performance (f1-macro on multilabel)
    ALL_LABELS = ["P1", "P2", "P3", "P4"]

    def to_indicator(label_list: list[str]) -> list[int]:
        """Convert a list of label strings to a fixed-order binary indicator vector."""
        return [1 if l in label_list else 0 for l in ALL_LABELS]

    def macro_f1_multilabel(example, predicted, trace=None) -> float:
        y_true = [to_indicator(l) for l in example]  # e.g. list of [1, 0, 1, 1, 0]
        y_pred = [to_indicator(p) for p in predicted]  # e.g.  list of [0, 0, 1, 1, 1]

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return float(f1)

    ma_f1_rules = macro_f1_multilabel(all_gold, all_pred)

    # evaluate classification performance
    looseness_level = 1

    def matches_llevel(x: str):
        if type(x) is not str or len(x) < 2:
            return True

        return int(x[1:]) <= looseness_level

    def to_binary(multilabels):
        r = []
        for m in multilabels:
            permission = any(map(matches_llevel, m))
            r += [0 if permission else 1]

        return r

    all_pred_binary = to_binary(all_pred)
    all_gold_binary = to_binary(all_gold)

    print("pred", Counter(all_pred_binary), "gold", Counter(all_gold_binary))

    f1_classification = f1_score(all_gold_binary, all_pred_binary, zero_division=0)
    acc_classification = accuracy_score(all_gold_binary, all_pred_binary)

    return {
        "macro_f1_multilabel_rules": ma_f1_rules,
        "f1_binary_classification": f1_classification,
        "acc_binary_classification": acc_classification,
    }

print("INTENT")
r = test_intent_based()
print(r)