import os
import json
from pathlib import Path
from collections import defaultdict

from aitdna.notions.AITDNotions import AITDNotions
import inspect
from aitdna.notions.data_loading import Population, AitdDataset, DatasetName, Notion
from torch.utils.data import DataLoader
print(inspect.getfile(AitdDataset))

def test_population_based():
    notions = AITDNotions()
    pop = Population(cache_dir="data/cache")
    root = "data/aitdna_anonymized/formatted"
    for study in os.listdir(root):
        for user in os.listdir(os.path.join(root, study)):
            for task in os.listdir(os.path.join(root, study, user)):
                with open(os.path.join(root, study, user, task, "edits.json"), "r") as f:
                    edits = json.load(f) 
                value = notions.get_final_text_by_user_population_based(edits, pop, 3)
                print(value)

def filter_by_username(meta: dict[str, str], user: str):
    return meta["author"] == user

def test_membership_based():
    PROJECT_ROOT = "/home/dycke/Projects/TXAITD/aitdna-paper1"

    popdata = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=PROJECT_ROOT + "/data/aitdna/formatted",
                          notion=Notion.DOCUMENT_LEVEL)

    pop = Population(dataset=popdata,cache_dir=None)#PROJECT_ROOT + "/data/cache")

    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=PROJECT_ROOT + "/data/aitdna/formatted",
                          notion=Notion.MEMBERSHIP_BASED,
                          population=pop)

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for snippet in text:
                print(snippet)

def test_different_ns_membership_based():
    
    threshold = 0.5
    token_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                           root_dir="data/aitdna/formatted",
                           notion=Notion.TOKEN_LEVEL,
                           document_level_threshold=0,
                           with_meta=True)
    pop = Population(dataset=token_level_dataset)
    ai_percentages = {}
    for n in range(4, 8):
        dataset = AitdDataset(dataset=DatasetName.AITDNA,
                            root_dir="data/aitdna/formatted",
                            notion=Notion.MEMBERSHIP_BASED,
                            n_gram_len=n,
                            population=pop,
                            with_meta=True)
        ai_texts = 0
        n_total_texts = 0
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for (text, meta) in batch:
                if "model" not in meta:
                    continue
                n_total_texts += 1
                ai_tokens = 0
                for token in text:
                    if token["author"] == "Bot":
                        ai_tokens += 1
                if ai_tokens / len(text) >= threshold:
                    ai_texts += 1
                else:
                    k = 0
        ai_percentages[n] = ai_texts / n_total_texts
    print(ai_percentages)
                    
def test_different_ns_authorship_based():
    
    threshold = 0.5
    token_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                           root_dir="data/aitdna/formatted",
                           notion=Notion.TOKEN_LEVEL,
                           document_level_threshold=0,
                           with_meta=True)
    loader = DataLoader(token_level_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    users = defaultdict(int)
    for batch in loader:
        for (text, meta) in batch:
            users[meta["author"]] += 1
    user = max(users, key=users.get)
    print("User: ", user)

    pop = Population(dataset=token_level_dataset, filter_fun=filter, user=user)
    ai_percentages = {}
    for n in range(1, 5):
        dataset = AitdDataset(dataset=DatasetName.AITDNA,
                            root_dir="data/aitdna/formatted",
                            notion=Notion.AUTHORSHIP_BASED,
                            n_gram_len=n,
                            population=pop,
                            with_meta=True)
        ai_texts = 0
        n_total_texts = 0
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for (text, meta) in batch:
                if meta["author"] == user:
                    continue
                n_total_texts += 1
                ai_tokens = 0
                for token in text:
                    if token["author"] == "Bot":
                        ai_tokens += 1
                if ai_tokens / len(text) >= threshold:
                    ai_texts += 1
                else:
                    k = 0
        ai_percentages[n] = ai_texts / n_total_texts
    print(ai_percentages)
                    

def test_authorship_based():
    document_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                           root_dir="data/aitdna/formatted",
                           notion=Notion.DOCUMENT_LEVEL,
                           document_level_threshold=0,
                           with_meta=True)
    pop = Population(dataset=document_level_dataset, filter_fun=filter_by_username, user="zira")

    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir="data/aitdna/formatted",
                          notion=Notion.AUTHORSHIP_BASED,
                          population=pop)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for snippet in text:
                print(snippet)


def test_contentbased():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # this runs classification and stores it on disk
    content_based_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                                         root_dir=f"{PROJECT_ROOT}/data/aitdna/formatted",
                                         notion=Notion.CONTENT_BASED,
                                         with_meta=True,
                                         llm_type="gpt-5.4-nano",
                                         strictness_level=3)

    loader = DataLoader(content_based_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for sentence in text:
                print(sentence)
            break


def test_intentbased():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # this runs classification and stores it on disk
    intent_based_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                                         root_dir=f"{PROJECT_ROOT}/data/aitdna/formatted",
                                         notion=Notion.INTENT_BASED,
                                         with_meta=True,
                                         llm_type="gpt-5.4-nano",
                                         strictness_level=3)

    loader = DataLoader(intent_based_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for sentence in text:
                print(sentence)
            break

def test_document_level():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                                         notion=Notion.DOCUMENT_LEVEL,
                                         document_level_threshold=0.1,
                                         with_meta=False)
    ai_texts = 0
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for snippet in text:
                if snippet["author"] == "Bot":
                    ai_texts += 1
    print(ai_texts / len(dataset))
        

def test_sentence_level():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                                         notion=Notion.SENTENCE_LEVEL,
                                         sentence_level_threshold=0.6,
                                         with_meta=False)

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for sentence in text:
                print(sentence)
            exit(0)

def test_boundary_level():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                                         notion=Notion.BOUNDARY_LEVEL,
                                         with_meta=False)

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for text in batch:
            for sentence in text:
                print(sentence)
            exit(0)

