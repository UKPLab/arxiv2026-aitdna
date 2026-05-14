import os
from collections import Counter
import pandas as pd
import json

ROOT = "datasets/formatted/txaitd_data_2025_11_26"
MODELS = ["TrustSafeAI/RADAR-Vicuna-7B", "desklib/ai-text-detector-v1.01", "ShantanuT01/BERT-tiny-RAID"]
THRESHOLD = 0.5


def compute_pos(root: str):
    """
    Compute POS stats across all user and bot parts of the final text
    :param root(str): root folder with the dataset
    :returns: pd.DataFrame, containing POS and their percentage in user's and bot's texts
    """
    counter_user = Counter()
    counter_bot = Counter()

    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "morph_analysis.json"), "r") as f:
                data = json.load(f)
            counter_user += Counter(data["User"]["pos"])
            counter_bot += Counter(data["Bot"]["pos"])
    df_user = pd.DataFrame(counter_user.items(), columns=["POS", "User Count"])
    df_user["User Count"] = (df_user["User Count"] / df_user["User Count"].sum()).round(3)

    df_bot = pd.DataFrame(counter_bot.items(), columns=["POS", "Bot Count"])
    df_bot["Bot Count"] = (df_bot["Bot Count"] / df_bot["Bot Count"].sum()).round(3)
    return df_user.merge(df_bot, on="POS", how="outer")


def compare_lemma_occurence(root):
    user_morph = 0
    bot_morph = 0
    total_docs_user = 0
    total_docs_bot = 0
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "morph_analysis.json"), "r") as f:
                data = json.load(f)
                user_morph += data["User"]["perc_dist_lemmas"]
                total_docs_user += 1
                if data["Bot"]["perc_dist_lemmas"] != 0:
                    bot_morph += data["Bot"]["perc_dist_lemmas"]
                    total_docs_bot += 1
    return (user_morph / total_docs_user, bot_morph / total_docs_bot)

def compute_averages(author: str, keys: list[str], all_sentences: list[dict[str, object]]):
    """
    Computes averages for each of keys for user
    
    :param author: user to compute average for
    :param keys: for which keys to compute stats
    :param all_sentences: sentences to extract stats from
    """

    sentences = [sentence for sentence in all_sentences if sentence["author"] == author]
    averages = {}
    averages["author"] = author

    if len(sentences) == 0:
        for key in keys:
            averages[key] = 0
        return averages

    for key in keys:
        averages[key] = sum(sentence[key] for sentence in sentences) / len(sentences)
    return averages
    

def compute_tree_stats_per_user(root):
    all_sentences = []
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        for task in os.listdir(user_path):
            task_path = os.path.join(user_path, task)
            with open(os.path.join(task_path, "final_text_by_user_sentencewise.json"), "r") as f:
                data = json.load(f)
                all_sentences.extend(data)

    keys = ["depth", "width", "leaves"]
    user = compute_averages("User", keys, all_sentences)
    bot = compute_averages("Bot", keys, all_sentences)
    mixed = compute_averages("Mixed", keys, all_sentences)

    return pd.DataFrame([user, bot, mixed])

def main():
    
    res = compute_tree_stats_per_user(ROOT)
    res.to_csv("model_results/tree_stats.csv")

    pos = compute_pos(ROOT)
    pos.to_csv("model_results/pos_analysis.csv")

    (user_morph, bot_morph) = compare_lemma_occurence(ROOT)
    with open("model_results/lemma_occurence.json", "w") as f:
        json.dump({"user_morphs": user_morph, "bot_morphs": bot_morph}, f)

if __name__ == "__main__":
    main()