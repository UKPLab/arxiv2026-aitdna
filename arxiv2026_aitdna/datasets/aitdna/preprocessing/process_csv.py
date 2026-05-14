import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from txaitd.notions.AITDNotions import AITDNotions

MAX_LEN = 3000
USERNAME_COL_NAME = "I have read and understood the privacy statement and agree to participate in this study. [CARE Username]"

def compute_text(edits, max_len, selection_index, selection_length):
    notions = AITDNotions()
    text_by_user = notions.get_final_text_by_user_span_level(edits=edits)
    final_text = "".join([data_point["text"] for data_point in text_by_user])
    if selection_length > 0:
        final_text = final_text[selection_index:selection_index + selection_length]
    else:
        final_text = final_text[max(0, selection_index - max_len):selection_index]
    return final_text

def convert_time(time):
    return datetime.fromisoformat(time)

def modify_full_query(time_sorted):
    for i, edit in enumerate(time_sorted):
        if "nlpService" in edit:
            user_prompt = edit["query"]
            text = compute_text(time_sorted[:i], MAX_LEN, edit["selectionIndex"], edit["selectionLength"])
            if edit["nlpService"] == "text_continuation":
                if not user_prompt:
                    user_prompt = "Continue this text:\n"

                time_sorted[i]["fullQuery"] = f"""You are a writing assistant. Your role is to continue user text according to their queries.
You MUST NOT add greetings, explanations, confirmations, or follow-up offers.
DO NOT repeat user text, just continue it.
ALWAYS answer in English!

Example
---------

User prompt:
Write an argument to support my claim.

Text:
In my opinion, the transformers architecture was revolutionary for natural language processing. The reason for that is

Continuation:
the fundamental change of how models understand and generate language. Unlike previous architectures such as RNNs or LSTMs, transformers rely entirely on self-attention mechanisms, allowing them to capture long-range dependencies and contextual relationships within text more effectively.

Your turn
----------

User prompt:
{user_prompt}
Text:
{text}
Continuation:

"""
            elif edit["nlpService"] == "text_revision":
                if not user_prompt:
                    user_prompt = "Fix spelling errors in this text\n"
                time_sorted[i]["fullQuery"] = f"""You are a writing assistant. Your role is to modify user text according to their queries.
You MUST NOT add greetings, explanations, confirmations, or follow-up offers. 
When responding, start DIRECTLY with the transformed text. Do not include labels, headings, or introductions.
ALWAYS answer in English!

Example
---------

User prompt:
Make this text sound better and fix my grammar

Text:
Main advantage of the transformer architecture lays in the ability to handle long-range dependences between different parts of text.

Revision:
The main advantage of the transformer architecture lies in its ability to handle long-range dependencies between different parts of a text.

Your turn
---------
User prompt:
{user_prompt}
Text:
{text}
Revision:
"""
    return time_sorted

def filter_entries(edit):
    if "operationType" in edit and edit["operationType"] == 0 and edit["text"] is None:
        return False
    return True

def process_edits(src_root: str, dst_root: str,
                  earliest_cutoff_date: str, latest_cutoff_date: str,
                  should_modify_full_query: bool =False, consent_form_path: str = ""):
    documents = pd.read_csv(f"{src_root}/document.csv", index_col="id")
    edits = pd.read_csv(f"{src_root}/document_edit.csv").replace([np.nan], [None], regex=False).to_dict(orient="records")
    requests = pd.read_csv(f"{src_root}/nlp_editor_request.csv").replace([np.nan], [None], regex=False).to_dict(orient="records")
    responses = pd.read_csv(f"{src_root}/nlp_editor_response.csv").replace([np.nan], [None], regex=False).to_dict(orient="records")
    users = pd.read_csv(f"{src_root}/user.csv").replace([np.nan], [None], regex=False).to_dict(orient="records")
    ai_perceptions = pd.read_csv(f"{src_root}/human_ai_perception.csv").replace([np.nan], [None], regex=False).to_dict(orient="records")

    if os.path.exists(consent_form_path):
        consent_form = pd.read_csv(consent_form_path, sep=";")
        consented_users = consent_form[USERNAME_COL_NAME]
        consented_users = [user.lower() for user in consented_users if not pd.isnull(user)]
    else:
        consented_users = [user["userName"] for user in users]

    response_by_edit = defaultdict(list)
    for rsp in responses:
        response_by_edit[rsp["requestId"]].append(rsp)

    request_by_edit = defaultdict(list)
    for request in requests:
        request_by_edit[request["documentEditId"]].append(request)
        if request["id"] in response_by_edit:
            request_by_edit[request["documentEditId"]].extend(response_by_edit[request["id"]])

    data_by_document = defaultdict(list)
    for edit in edits:
        data_by_document[edit["documentId"]].append(edit)
        if edit["id"] in request_by_edit:
            data_by_document[edit["documentId"]].extend(request_by_edit[edit["id"]])

    ai_perceptions_by_document = defaultdict(list)
    for perception in ai_perceptions:
        ai_perceptions_by_document[perception["documentId"]].append(perception)

    no_consent = set()
    for key, value in data_by_document.items():
        if convert_time(value[0]["createdAt"]) < convert_time(earliest_cutoff_date) or \
            convert_time(value[0]["createdAt"]) > convert_time(latest_cutoff_date):
            continue
        if len(value) == 1:
            continue
        userid = documents.at[key, "userId"]
        doc_name = documents.at[key, "name"]
        if "Warm-up" in doc_name:
            continue
        user = [usr for usr in users if usr["id"] == userid][0]["userName"]
        if user == "admin":
            continue
        if consented_users and user not in consented_users:
            no_consent.add(user)
            continue
        value = filter(filter_entries, value)
        time_sorted = sorted(value, key=lambda v: convert_time(v["createdAt"]))
        if should_modify_full_query:
            time_sorted = modify_full_query(time_sorted)
        user_path = os.path.join(dst_root, user)
        if not os.path.exists(user_path):
            os.mkdir(user_path)
        doc_path = os.path.join(user_path, doc_name)
        if not os.path.exists(doc_path):
            os.mkdir(doc_path)
        with open(os.path.join(doc_path, "edits.json"), "w") as f:
            json.dump(time_sorted, f)
        
        ai_perception = ai_perceptions_by_document.get(key)
        if not ai_perception:
            continue
        with open(os.path.join(doc_path, "ai_perception.json"), "w") as f:
            json.dump({"ai_perception": ai_perception}, f)
    print("No consent for: ", no_consent)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--src_root', type=str)
    parser.add_argument("-d", '--dst_root', type=str)
    parser.add_argument("-e", '--earliest_cutoff_date', type=str, default="2020-01-01 13:00:00.00+00")
    parser.add_argument("-l", '--latest_cutoff_date', type=str, default="2030-01-01 13:00:00.00+00")
    parser.add_argument("-c", "--consent_form_path", type=str, help="path to the consent form file", default="")

    args = parser.parse_args(argv)
    EARLIEST_CUTOFF_DATE = args.earliest_cutoff_date
    LATEST_CUTOFF_DATE = args.latest_cutoff_date
    SRC_ROOT = args.src_root
    DST_ROOT = args.dst_root
    CONSENT_FORM_PATH = args.consent_form_path
    if not os.path.exists(DST_ROOT):
        os.mkdir(DST_ROOT)
    process_edits(src_root=SRC_ROOT, dst_root=DST_ROOT, earliest_cutoff_date=EARLIEST_CUTOFF_DATE,
                  latest_cutoff_date=LATEST_CUTOFF_DATE, should_modify_full_query=True,
                  consent_form_path=CONSENT_FORM_PATH)
