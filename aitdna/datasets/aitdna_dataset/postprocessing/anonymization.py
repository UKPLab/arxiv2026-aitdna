import os
import json
import random
import argparse
import nltk

# requires nltk corpus words!
from nltk.corpus import words

random.seed(5330192)

def anonymize(root, same_person_file_path):

    if same_person_file_path == "":
        same_person = {}
    else:
        if not os.path.exists(same_person_file_path):
            raise ValueError("File with same person mapping not found!")
        with open(same_person_file_path, "r") as f:
            same_person = json.load(f)

    mapping = {}
    for value in same_person.values():
        mapping[value] = random.choice(words.words()).lower()

    for i, study in enumerate(sorted(os.listdir(root))):
        os.rename(src=os.path.join(root, study), dst=os.path.join(root, f"session_{str(i)}"))

    for study in sorted(os.listdir(root)):
        for user in os.listdir(os.path.join(root, study)):
            if user in mapping:
                new_name = mapping[user]
            elif user in same_person:
                new_name = mapping[same_person[user]]
            else:
                new_name = random.choice(words.words()).lower()
            
            for task in os.listdir(os.path.join(root, study, user)):
                bg_path = os.path.join(root, study, user, task, "statistics", "background.json")
                if os.path.exists(bg_path):
                    with open(bg_path, "r", encoding="utf-8") as f:
                        bg = json.load(f)
                    bg["What is your CARE username?"] = new_name
                    with open(bg_path, "w", encoding="utf-8") as f:
                        json.dump(bg, f)
                
                ux_survey_path = os.path.join(root, study, user, task, "statistics", "ux_survey.json")
                if os.path.exists(ux_survey_path):
                    with open(ux_survey_path, "r", encoding="utf-8") as f:
                        ux = json.load(f)
                    ux["What is your CARE username?"] = new_name
                    with open(ux_survey_path, "w", encoding="utf-8") as f:
                        json.dump(ux, f)

            n_repetitions = 1
            try:
                os.rename(src=os.path.join(root, study, user),
                        dst=os.path.join(root, study, new_name))
            except OSError:
                while os.path.exists(os.path.join(root, study, new_name + "_" + str(n_repetitions))):
                    n_repetitions += 1
                new_name = new_name + "_" + str(n_repetitions)
                os.rename(src=os.path.join(root, study, user),
                        dst=os.path.join(root, study, new_name))

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CLI for anonymizing the AITDNA dataset"
    )
    parser.add_argument("-r", "--root", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("-s", "--same_person_file_path",
                        type=str, default="",
                        help="Path to the file specifying user name mappings if the same person participated >1 times. If not passed, assume all users are distinct.")
    argv = parser.parse_args(argv)
    root = argv.root
    same_person_file_path = argv.same_person_file_path
    anonymize(root=root, same_person_file_path=same_person_file_path)
