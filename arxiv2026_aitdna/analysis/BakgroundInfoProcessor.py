import os
import json
import textwrap
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def map_proficiency(key):
    prof_map = {
        "Absolute Beginner - Have (almost) never written texts with scientific content, very low confidence": "Absolute Beginner",
        "Beginner: Have written scientific texts a few times, low confidence": "Beginner",
        "Intermediate: Have some experience writing scientific texts; moderate confidence": "Intermediate",
        "Advanced: Have written multiple high-quality scientific texts; high confidence": "Advanced",
        "Expert: Have written and advised on multiple high-quality scientific texts; very high confidence": "Expert",
        "Expert: Have written and advised on multiple high-quality scietific texts; very high confidence": "Expert",
    }
    return prof_map[key]

def map_field_of_study(key):
    field_map = {
        "Formal science (math, computer science)": "Formal science",
        "Natural science (biology, chemistry, physics)": "Natural science",
        "Humanities and social sciences (psychology, linguistics, law, history)": "Humanities and social sciences",
        "Applied science (electrical/mechanical engineering, medicine)": "Applied science",
        "Other": "Other"
    }
    return field_map[key]

def get_info_from_surveys(surveys: list[dict], survey_type: str):
    answers = defaultdict(list)
    usernames = set()
    for survey in surveys:
        if survey_type == "bg":
            for key in ["What is your field of study/work? [Other]", "What is your highest level of education? [Other]"]:
                survey.pop(key)
        else:
            time_data = [k for k in survey.keys() if "Question time" in k or "Group time" in k or "Total time" in k]
            for key in time_data:
                survey.pop(key)


        for k, v in survey.items():
            if k == "What is your CARE username?":
                if v not in usernames:
                    usernames.add(v)
                    continue
                break
            if k in ["What problems and technical issues did you encounter during the interaction?",
                     "What limitations did you encounter during the interaction?",
                     "What improvements would you suggest?",
                     "Do you have any other feedback?"]:
                continue
            if k == "How would you rate your proficiency in scientific writing?":
                answers[k].append(map_proficiency(v))
            if k == "What is your field of study/work?":
                answers[k].append(map_field_of_study(v))
            elif v and v != "Other":
                answers[k].append(v)
    return answers

def get_labels_from_values(values):
        labels = list(set(values))
        if "Somewhat Agree" in labels:
            return ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"]
        if "Disagree" in labels:
            return ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        if "B. Sc." in labels:
            return ["High school", "B. Sc.", "M. Sc.", "PhD"]
        if "Native" in labels:
            return ["B1-B2", "C1-C2", "Native"]
        if "1-2" in labels:
            return ["1-2", "3-5", "5-10", "10+"]
        if "True (absolutely certain)" in labels:
            return ["True (absolutely certain)", "True (rather certain)", "I don't know", "False (rather certain)", "False (absolutely certain)"]
        if "Rarely (a few times a year)" in labels:
            return ["Never", "Rarely (a few times a year)", "Occasionally (a few times a month)", "Frequently (several times a week)", "Very Frequently (almost daily or daily)"]
        if "25-30" in labels:
            return ["18-24", "25-30", "31-40", "41-50", "51+"]
        if "Beginner" in labels:
            return ["Absolute Beginner", "Beginner", "Intermediate","Advanced", "Expert"]
        return labels


def plot_results(results: dict[str, list[str]], rows: int, cols: int, path_to_file: str):
    fig, ax = plt.subplots(rows, cols, figsize=(20, rows*5), constrained_layout=True)
    ax = ax.flatten()
    plt.subplots_adjust(hspace=0.5)

    cmap = plt.get_cmap("Spectral")

    for i, (question, values) in enumerate(results.items()):
        labels = get_labels_from_values(values)
        colors = cmap(np.linspace(0, 1, len(labels)))
        counts = []
        for answer in labels:
            counts.append(len([v for v in values if v == answer]) / len(values))
        ax[i].bar(labels, counts, color=colors)
        wrapped_title = "\n".join(textwrap.wrap(question, width=90))
        ax[i].set_title(wrapped_title)
        ax[i].set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")

    plt.savefig(path_to_file)


def get_surveys(root: str):
    bg_surveys = []
    ux_surveys = []
    for study in os.listdir(root):
        for user in os.listdir(os.path.join(root, study)):
            for task in os.listdir(os.path.join(root, study, user)):
                bg_path = os.path.join(root, study, user, task, "statistics", "background_survey.json")
                ux_path = os.path.join(root, study, user, task, "statistics", "ux_survey.json")
                for survey_path in [bg_path, ux_path]:
                    if not os.path.exists(survey_path):
                        continue
                    with open(survey_path, "r", encoding="utf-8") as f:
                        survey_data = json.load(f)
                        if "ux_survey" in survey_path:
                            ux_surveys.append(survey_data)
                        else:
                            bg_surveys.append(survey_data)
    bg_data = get_info_from_surveys(bg_surveys, "bg")
    ux_data = get_info_from_surveys(ux_surveys, "ux")
    plot_results(ux_data, int(len(ux_data) / 2), 2, "ux.png")
    plot_results(bg_data, int(len(bg_data) / 2), 2, "bg.png")


def main():
    # TODO add cli command
    get_surveys("data/aitdna_anonymized/formatted")
