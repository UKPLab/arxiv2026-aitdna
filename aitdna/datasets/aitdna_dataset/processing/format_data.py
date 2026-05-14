import argparse
import os
import json
import shutil
import pandas as pd
import numpy as np
import imgkit
from aitdna.notions import AITDNotions
from .DataFormatter import DataFormatter

UNRELEVANT_KEYS = ("studySessionId", "studyStepId", "draft",
                   "order", "requestType", "deletedAt", "deleted")
NO_STATS = []

def create_folders_for_analysis(folder_path: str):
    """
    Creates folders for analysis:
    - folder_path/statistics for linguistic analysis
    - folder_path/notions for text in different notions
    - folder_path/model_results for results of different models
    """
    stats_path = os.path.join(folder_path, "statistics")
    if not os.path.exists(stats_path):
        os.mkdir(stats_path)

    notions_path = os.path.join(folder_path, "notions")
    if not os.path.exists(notions_path):
        os.mkdir(notions_path)

    boundary_path = os.path.join(notions_path, "boundary_level")
    if not os.path.exists(boundary_path):
        os.mkdir(boundary_path)

    model_result_path = os.path.join(folder_path, "model_results")
    if not os.path.exists(model_result_path):
        os.mkdir(model_result_path)

    return stats_path, notions_path, boundary_path


def create_folders_for_user(user: str, task: str, root: str):
    """
    Creates folders for the user:
    - user folder as root for user
    - user/task as root for task
    - user/task/statistics for linguistic analysis
    - user/task/notions for text in different notions
    - user/task/model_results for results of different models
    """
    if user not in os.listdir(root):
        os.mkdir(os.path.join(root, user))
    folder_path = os.path.join(root, user, task)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    stats_path, notions_path, boundary_path = create_folders_for_analysis(folder_path)
    return folder_path, stats_path, notions_path, boundary_path


def get_user_task_assignment(user: str,
                             task: str, user_task_assignment_path: str):
    """
    Gets the user-task-llm mapping for the given user and task.
    
    :param text_analyser: text analyser instance
    :param user: user from the mapping
    :type user: str
    :param task: task from the mapping
    :type task: str
    :param original_folder: original folder path (where the mapping is stored)
    :type original_folder: str
    """
    with open(user_task_assignment_path, "r", encoding="utf-8") as f:
        assignment = json.load(f)
    for user_info in assignment:
        if user_info["name"] == user:
            task_stats = None
            if "2.1" in task:
                task_stats = user_info["modelTaskMapping"]["LLM"]["1"]
            elif "2.2" in task:
                task_stats = user_info["modelTaskMapping"]["LLM"]["2"]
            elif "2.3" in task:
                task_stats = user_info["modelTaskMapping"]["LLM"]["3"]
            elif "Peer" in task:
                task_stats = {"model": "gpt-5.2",
                              "temperature": 1,
                              "task": "Peer Review"}
            else:
                task_stats = {"task": user_info["modelTaskMapping"]["noLLM"]}

            if "Human" in task and ("Argumentative" in task or "Creative" in task) and "human" in user_info:
                task_stats["questions"] = user_info["human"]
            elif "Argumentative" in task and "argumentative" in user_info:
                task_stats["questions"] = user_info["argumentative"]
            elif "Creative" in task and "creative" in user_info:
                task_stats["questions"] = user_info["creative"]
            
            task_stats["setting"] = user_info["setting"]

            return task_stats
    return None


def get_and_save_notions(edits: list[dict[str|int]], boundary_path: str,
                         notions_path: str, save_png: bool = False,
                         n_segments: list[int] = [2, 5, 10], length_penalty: float = 1,
                         impurity_penalty: float = 1, document_level_threshold: float = 0.5,
                         sentence_level_threshold: float = 0.5):
    """
    Computes and saves text in different notions.
    Available notions:
    - SPAN_LEVEL
    - TOKEN_LEVEL
    - DOCUMENT_LEVEL
    - SENTENCE_LEVEL
    - BOUNDARY_LEVEL
    
    :param edits: text by user in format [["Text snippet", author, query], ..]
    :type text_by_user: list[dict[str|int]]
    :param boundary_path: folder to save the boundary-level to
    :type boundary_path: str
    :param notions_path: folder to save the notions (except boundary-level) to
    :type notions_path: str
    """
    notions = AITDNotions()
    for n in n_segments:
        segments = notions.get_final_text_by_user_boundary_level(edits, n_seg=n,
                                                                length_penalty=length_penalty,
                                                                impurity_penalty=impurity_penalty)
        with open(
            os.path.join(
                boundary_path,
                f"final_text_by_user_boundary_level_ilp_{n}seg_{length_penalty}lp_{impurity_penalty}ip.json"), "w", encoding="utf-8") as f:
                json.dump(segments, f)
        if save_png:
            notions.evaluate_segments(segments,
                    os.path.join(
                        boundary_path,
                        f"final_text_by_user_boundary_level_ilp_{n}seg_{length_penalty}lp_{impurity_penalty}ip.png"
                        ))

    document_level = notions.get_final_text_by_user_document_level(edits, document_level_threshold)
    with open(os.path.join(notions_path, "final_text_by_user_document_level.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(document_level, f)

    token_level = notions.get_final_text_by_user_token_level(edits)
    with open(os.path.join(notions_path, "final_text_by_user_token_level.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(token_level, f)

    text_by_user = notions.get_final_text_by_user_span_level(edits)
    # save text data by user
    with open(os.path.join(notions_path, "final_text_by_user_span_level.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(text_by_user, f)

    sentence_level = notions.get_final_text_by_user_sentence_level(edits, sentence_level_threshold)
    with open(os.path.join(notions_path, "final_text_by_user_sentence_level.json"),
              "w",
    encoding="utf-8") as f:
        json.dump(sentence_level, f)

    return segments, document_level, token_level, text_by_user, sentence_level


def get_and_save_final_text(text_by_user: list[tuple[str, str, list[str]]],
                            target_folder: str,
                            save_png: bool = False):
    """
    Generates and saves final text version.
    
    :param text_analyser: text analyser instance
    :param text_by_user: text by user in format [["Text snippet", author, query], ..]
    :type text_by_user: list[tuple[str, str, list[str]]]
    :param target_folder: folder to save to
    :type target_folder: str
    """
    # save color-coded version
    if save_png:
        generate_and_save_final_text_png(text_by_user,
                target_folder)

    # save final text version
    final_text = DataFormatter.get_final_text(text_by_user)
    with open(os.path.join(target_folder, "final_text.txt"), "w", encoding="utf-8") as f:
        f.write(final_text)

def generate_and_save_final_text_png(text_by_user: list[tuple[str, str, list[str]]],
                                     target_folder: str) -> None:
    """
    Generate and save a color-coded representation of text
    :param target_folder(str): where to create color_coded.png
    """
    color_map = {
        "Bot": "#d9ca6c",
        "User": "#6cd97b"
    }
    html = """<html>
        <head>
            <style>
                body { font-size: 16px; line-height: 1.6; }
                span { padding: 1px 2px; border-radius: 3px; }
                .legend-color-box { width: 10px; height: 10px; }
            </style>
        </head>
        <body><div>"""

    html += '''<div class="container">
                <div class="d-inline-block">'''

    authors = {data_point["author"] for data_point in text_by_user}

    for author in authors:
        color = color_map[author]
        html += f'''<div
                        class="d-inline-block align-items-center">
                    <span
                        class="legend-color-box d-inline-block"
                        style="background: {color}"
                    > </span>
                    <span class="legend-label">{author}</span>
                    </div>'''
    html += '</div><br>'

    for data_point in text_by_user:
        text, author, _ = data_point["text"], data_point["author"], data_point["queries"]
        text = text.replace("\n", "<br>")
        color = color_map[author]
        html += f'<span style="background:{color}">{text}</span>'
    html += "</div></body></html>"
    html_path = os.path.join(target_folder, "color_coded.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    imgkit.from_file(html_path,
                    os.path.join(target_folder, "color_coded.png"))
    os.remove(html_path)

def get_survey_info(user: str, df_path: pd.DataFrame) -> dict[str, str]:
    """Get informations from the surveys

    Args:
        user (str): user to find info for
        df_path (pd.DataFrame): path to data frame

    Returns:
        dict[str, str]: survey info for user
    """
    df = pd.read_csv(df_path, sep=";")
    df = df.replace({np.nan:None})
    df_stats = df.loc[(df['What is your CARE username?'] == user) & df['Date submitted'].notnull()]
    stats = df_stats.to_dict(orient="records")
    if len(stats) == 0:
        return None
    if len(stats) > 1:
        all_stats = []
        for stat in all_stats:
            for key in ["Response ID", "Date submitted", "Last page", "Start language", "Seed", "Date started", "Date last action"]:
                stat.pop(key)
        return all_stats

    stats = stats[0]
    for key in ["Response ID", "Date submitted", "Last page", "Start language", "Seed", "Date started", "Date last action"]:
        stats.pop(key)
    return stats


def generate_all_information(user: str, task: str, target_root: str, original_folder: str,
                            data_formatter: DataFormatter,
                            n_segments: list[int] = [2, 5, 10], length_penalty: float = 1,
                            impurity_penalty: float = 1, document_level_threshold: float = 0.5,
                            sentence_level_threshold: float = 0.5,
                            survey_paths: list[str] = ["data/surveys_data/processed/background.csv",
                                                       "data/surveys_data/processed/ux_survey.csv"],
                            user_task_assignment_path: str = "data/raw/user_task_assignment"):
    """
    Processes the given text and generates all information.
    
    :param user: user name
    :type user: str
    :param task: task name
    :type task: str
    :param target_root: target root
    :type target_root: str
    :param original_folder: original folder
    :type original_folder: str
    """
    target_folder, stats_path, notions_path, boundary_path = create_folders_for_user(user, task, target_root)
    # process the data and save the new version
    target_file_path = os.path.join(target_folder, "edits.json")
    original_file_path = os.path.join(original_folder, user, task, "edits.json")


    violations = data_formatter.format_and_save(original_file_path, target_file_path)
    if violations != "":
        shutil.rmtree(target_folder)
        return

    for survey_path in survey_paths:
        user_info = get_survey_info(user, survey_path)
        if user_info:
            file_name = survey_path.split("/")[-1].replace(".csv", ".json")
            with open(os.path.join(stats_path, file_name), "w", encoding="utf-8") as f:
                json.dump(user_info, f)


    user_task_assignment = get_user_task_assignment(user, task,
                                                    user_task_assignment_path)
    if user_task_assignment:
        with open(os.path.join(stats_path, "user_task_assignment.json"),
                  "w",
        encoding="utf-8") as f:
            json.dump(user_task_assignment, f)

    # get text in format [{"user": "Once upon"}, {"bot": " a time"},..]
    with open(target_file_path, "r", encoding="utf-8") as f:
        edits = json.load(f)

    _, _, _, text_by_user, _ = get_and_save_notions(edits=edits,
                                                    boundary_path=boundary_path,
                                                    notions_path=notions_path,
                                                    n_segments=n_segments,
                                                    length_penalty=length_penalty,
                                                    impurity_penalty=impurity_penalty,
                                                    document_level_threshold=document_level_threshold,
                                                    sentence_level_threshold=sentence_level_threshold,
                                                    save_png=False)

    get_and_save_final_text(text_by_user, target_folder)
    
    ai_perception_file_path = os.path.join(original_folder, user, task, "ai_perception.json")
    if os.path.exists(ai_perception_file_path):
        shutil.copyfile(ai_perception_file_path, os.path.join(stats_path, "ai_perception.json"))


def format_data(original_folder: str, target_root: str,
                n_segments: list[int] = [2, 5, 10], length_penalty: float = 1,
                impurity_penalty: float = 1, document_level_threshold: float = 0.5,
                sentence_level_threshold: float = 0.5,
                survey_paths: str = "data/surveys_data/background_survey.csv",
                user_task_assignment_path: str ="data/raw/user_task_assignment",
                guidelines_path: str = "rsc/guidelines.txt"):
    """
    Formats all original data from a given study.
    
    :param original_folder: original study folder
    :type original_folder: str
    :param target_root: target root
    :type target_root: str
    """
    data_formatter = DataFormatter(25, 10000, UNRELEVANT_KEYS, guidelines_path=guidelines_path)
    for user in os.listdir(original_folder):
        for task in os.listdir(os.path.join(original_folder, user)):
           generate_all_information(user=user, task=task, target_root=target_root,
                                    original_folder=original_folder,
                                    data_formatter=data_formatter,
                                    n_segments=n_segments,
                                    length_penalty=length_penalty,
                                    impurity_penalty=impurity_penalty,
                                    document_level_threshold=document_level_threshold,
                                    sentence_level_threshold=sentence_level_threshold,
                                    survey_paths=survey_paths,
                                    user_task_assignment_path=user_task_assignment_path)
        # if the user cheated in all tasks, delete their folder
        if not os.listdir(os.path.join(target_root, user)):
            shutil.rmtree(os.path.join(target_root, user))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--src_root', type=str, required=True)
    parser.add_argument("-d", '--dst_root', type=str, required=True)
    parser.add_argument('-n', '--ns_segments', nargs='+',
                        help='Each value corresponds to a number of segments to optimize boundary notion for.', required=True)
    parser.add_argument("-l", "--length_penalty", type=float, default=1)
    parser.add_argument("-i", "--impurity_penalty", type=float, default=1)
    parser.add_argument("-t", "--document_level_threshold", type=float, default=0.5)
    parser.add_argument("-c", "--sentence_level_threshold", type=float, default=0.5)
    parser.add_argument("-a", "--process_all", action="store_true")
    parser.add_argument("-u", "--user_task_assignment", type=str,
                        default="data/raw/user_task_assignment/new_user_task_assignment.json")
    parser.add_argument("-v", "--survey_paths", nargs='+', help="Each value corresponds to a survey path")
    parser.add_argument("-g", "--guidelines_path", type=str, default="rsc/guidelines.txt", help="Path to guidelines text.")
    args = parser.parse_args(argv)

    src_root = args.src_root
    dst_root = args.dst_root
    n_segments = [int(n) for n in args.ns_segments]
    length_penalty = args.length_penalty
    impurity_penalty = args.impurity_penalty
    document_level_threshold = args.document_level_threshold
    sentence_level_threshold = args.sentence_level_threshold
    process_all = args.process_all
    user_task_assignment_path = args.user_task_assignment
    survey_paths = [path for path in args.survey_paths]
    guidelines_path = args.guidelines_path
    if process_all:
        for original_folder in os.listdir(src_root):
            original_path = os.path.join(src_root, original_folder)
            target_root = os.path.join(dst_root, original_folder)

            if not os.path.exists(target_root):
                os.mkdir(target_root)
            format_data(original_path, target_root,
                        n_segments=n_segments, length_penalty=length_penalty,
                        impurity_penalty=impurity_penalty, document_level_threshold=document_level_threshold,
                        sentence_level_threshold=sentence_level_threshold,
                        survey_paths=survey_paths,
                        user_task_assignment_path=user_task_assignment_path,
                        guidelines_path=guidelines_path)
    else:
        if not os.path.exists(dst_root):
            os.mkdir(dst_root)
        format_data(src_root, dst_root,
                    n_segments=n_segments, length_penalty=length_penalty,
                    impurity_penalty=impurity_penalty, document_level_threshold=document_level_threshold,
                    sentence_level_threshold=sentence_level_threshold,
                    survey_paths=survey_paths,
                    user_task_assignment_path=user_task_assignment_path,
                    guidelines_path=guidelines_path)