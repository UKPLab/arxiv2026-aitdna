import json
import os
import requests
from nltk.metrics import edit_distance

def use_ollama_client(url: str, model: str, prompt: str) -> str:
    """
    Send a query using OpenAI client
    
    :param self: 
    :param model: model name
    :type model: str
    :param url: server to connect to
    :type url: str
    :param prompt: user prompt
    :type prompt: str
    :return: model answer
    :rtype: str
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    result = requests.post(url, data=json.dumps(payload), timeout=30)

    return result.json()["response"]

def generate_answer(text, model):
    prompt = """You are a tool that extracts pro- and contra-arguments from the provided input text.

Rules (follow exactly):
1. Extract only arguments that appear VERBATIM in the input. Copy exact characters (case, punctuation, spaces) — do not paraphrase or normalize.
2. Do NOT add or remove characters. Each extracted item must be an exact substring of the input.
3. Output ONLY valid JSON: an array of strings. Each string must be the exact argument substring.
4. No other output or commentary.

Input:
Online learning offers several advantages. One pro is flexibility, as students can study anytime and from anywhere. This flexibility makes it easier for people to balance education with work or family responsibilities. Another pro is access to many resources, such as videos and digital materials. On the other hand, students may feel less motivated without direct interaction with teachers and classmates.

Output:
[
    "One pro is flexibility, as students can study anytime and from anywhere. This flexibility makes it easier for people to balance education with work or family responsibilities.",
    "Another pro is access to many resources, such as videos and digital materials.",
    "On the other hand, students may feel less motivated without direct interaction with teachers and classmates."
]

Your turn:
Input:
""" + text + "\nOutput:\n"

    url = "http://10.167.31.203:11434/api/generate"
    result = use_ollama_client(url=url, model=model, prompt=prompt)
    return result

def get_all_arguments(root):
    model = "llama4:scout"
    for study in os.listdir(root):
        for user in os.listdir(os.path.join(root, study)):
            for task in os.listdir(os.path.join(root, study, user)):
                if "Argumentative" not in task:
                    continue
                final_text_path = os.path.join(root, study, user, task, "final_text.txt")
                with open(final_text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                arguments = generate_answer(text, model)
                with open(os.path.join(root, study, user, task, "arguments.txt"), "w", encoding="utf-8") as f:
                    f.write(arguments)

def evaluate_all_arguments(root):
    all_authors = {}
    for study in os.listdir(root)[:1]:
        for user in os.listdir(os.path.join(root, study))[:10]:
            for task in os.listdir(os.path.join(root, study, user)):
                task_authors = {
                    "User": [],
                    "Bot": [],
                    "Mixed": []
                }
                if "Argumentative" not in task:
                    continue
                arguments_path = os.path.join(root, study, user, task, "arguments.txt")
                with open(arguments_path, "r", encoding="utf-8") as f:
                    arguments_txt = f.read()
                arguments = eval(arguments_txt)
                with open(os.path.join(root, study, user, task, "notions", "final_text_by_user_span_level.json"),
                          "r", encoding="utf-8") as f:
                    text_by_user = json.load(f)
                
                author_map = []
                full_text = ""
                for seg in text_by_user:
                    text = seg["text"]
                    author = seg["author"]
                    full_text += text
                    author_map.extend([author]*len(text))
                
                for argument in arguments:
                    start = 0
                    idx = full_text.find(argument, start)
                    if idx == -1:
                        min_levenshtein = 100000
                        min_levelstein_index = -1
                        i = 0
                        while i < len(full_text) - len(argument):
                            snippet = full_text[i:i+len(argument)]
                            distance = edit_distance(snippet, argument)
                            if distance > 40:
                                i += 20
                            else:
                                i += 1
                            if distance < min_levenshtein:
                                min_levenshtein = distance
                                min_levelstein_index = i
                                if min_levenshtein == 1:
                                    break
                        span_authors = list(set(author_map[min_levelstein_index:min_levelstein_index+len(argument)]))
                        if len(span_authors) > 1:
                            task_authors["Mixed"].append(argument)
                        else:
                            task_authors[span_authors[0]].append(argument)
                    else:
                        span_authors = list(set(author_map[idx:idx+len(argument)]))
                        if len(span_authors) > 1:
                            task_authors["Mixed"].append(argument)
                        else:
                            task_authors[span_authors[0]].append(argument)
                    start = idx + 1
                all_authors[f"{study}_{user}_{task}"] = task_authors

    with open("authors.json", "w") as f:
        json.dump(all_authors, f)
    

def evaluate_all_arguments_by_edit(root):
    for study in os.listdir(root):
        for user in os.listdir(os.path.join(root, study)):
            for task in os.listdir(os.path.join(root, study, user)):
                task_authors = {
                    "User": [],
                    "Bot": [],
                }
                if "Argumentative" not in task:
                    continue

                arguments_path = os.path.join(root, study, user, task, "arguments.txt")
                with open(arguments_path, "r", encoding="utf-8") as f:
                    arguments_txt = f.read()
                arguments = eval(arguments_txt)
                with open(os.path.join(root, study, user, task, "edits.json"),
                          "r", encoding="utf-8") as f:
                    edits = json.load(f)
                response_texts = []
                for edit in edits:
                    if "requestId" in edit:
                        model_requests = [req for req in edits if "documentEditId" in req and req["id"] == edit["requestId"]]
                        if len(model_requests) != 1:
                            raise ValueError("expected to find 1 request")
                        request = model_requests[0]
                        if request["nlpService"] == "text_continuation":
                            response_texts.append(edit["response"])
                full_model_text = "".join(response_texts)
                
                for argument in arguments:
                    llm_produced = False
                    start = 0
                    idx = full_model_text.find(argument, start)
                    if idx == -1:
                        min_levenshtein = 100000
                        i = 0
                        while i < len(full_model_text) - len(argument):
                            snippet = full_model_text[i:i+len(argument)]
                            distance = edit_distance(snippet, argument)
                            if distance > 40:
                                i += 20
                            if distance < min_levenshtein:
                                min_levenshtein = distance
                                if min_levenshtein < 10:
                                    llm_produced = True
                            i += 1
                    else:
                        llm_produced = True
                    if llm_produced:
                        task_authors["Bot"].append(argument)
                    else:
                        task_authors["User"].append(argument)
                with open(os.path.join(root, study, user, task, "arguments_per_user.json"), "w") as f:
                    json.dump(task_authors, f)



def analyse_argument_number(root):
    avg_scores = {
        "Human-only": 0,
        "Human in LLM Setup": 0,
        "LLM in LLM Setup": 0
    }

    total_texts_human_only = 0
    total_texts_llm_setup = 0
    for study in os.listdir(root):
        for user in os.listdir(os.path.join(root, study)):
            for task in os.listdir(os.path.join(root, study, user)):
                if "Argumentative" not in task:
                    continue
                if not os.path.exists(os.path.join(root, study, user, task, "arguments_per_user.json")):
                   continue
                with open(os.path.join(root, study, user, task, "arguments_per_user.json"), "r") as f:
                    authors = json.load(f)
                if "Human-only" in task:
                    total_texts_human_only += 1
                    avg_scores["Human-only"] += len(authors["User"])
                else:
                    total_texts_llm_setup += 1
                    avg_scores["Human in LLM Setup"] += len(authors["User"])
                    avg_scores["LLM in LLM Setup"] += len(authors["Bot"])
    avg_scores["Human-only"] = round(avg_scores["Human-only"] / total_texts_human_only, 2)
    avg_scores["Human in LLM Setup"] = round(avg_scores["Human in LLM Setup"] / total_texts_llm_setup, 2)
    avg_scores["LLM in LLM Setup"] = round(avg_scores["LLM in LLM Setup"] / total_texts_llm_setup, 2)
    return avg_scores


def main():
    scores = analyse_argument_number("data/arguments/formatted")
    print(scores)

if __name__ == "__main__":
    main()