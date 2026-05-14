import os
import json
from torch.utils.data import Dataset
from ..AITDNotions import AITDNotions
from .DatasetName import DatasetName
from .Notion import Notion


class AitdDataset(Dataset):
    def __init__(self, dataset: DatasetName, root_dir: str, notion: Notion, with_meta: bool = False, **kwargs):
        """Creates a dataset for the specified notion.
        Uses data from all user studies inside root_dir.

        Args:
            dataset (DatasetName): dataset to load.
            root_dir (str): root directory. Structure inside it: root/study/user/task
            notion (Notion): notion to use
            **kwargs: Depends on the notion. Available combinations:
            Notion.BOUNDARY_LEVEL: 
                n_segments (int): number of boundaries. Default: 5
                length_penalty (float): segment length penalty. Default: 1
                impurity_penalty (float): segment impurity penalty. Default: 1
            Notion.DOCUMENT_LEVEL:
                document_level_threshold (float): threshold for AI-generated label. Default: 0.5
            Notion.SENTENCE_LEVEL:
                sentence_level_threshold (float): threshold for AI-generated label. Default: 0.5
            Notion.MEMBERSHIP_BASED:
                n_gram_len (int): n for n-gram to search in corpus for. Default: 3
                population (Population): the corpus to search for n-grams in.
        """
        data = []
        meta = []
        if dataset == DatasetName.AITDNA:
            for study in os.listdir(root_dir):
                for user in os.listdir(os.path.join(root_dir, study)):
                    for task in os.listdir(os.path.join(root_dir, study, user)):
                        task_path = os.path.join(root_dir, study, user, task)
                        data.append(self.extract_data_for_text(task_path, notion, user, task=task, **kwargs))
                        if with_meta:
                            user_task_assignment_path = os.path.join(task_path, "statistics", "user_task_assignment.json")
                            with open(user_task_assignment_path, "r") as f:
                                task_stats = json.load(f)
                            # fix for users that participated twice in the same US
                            if "_" in user:
                                user_name = user.split("_")[0]
                            else:
                                user_name = user
                            metadata = {
                                "author": user_name
                            }
                            if "model" in task_stats:
                                metadata["model"] = task_stats["model"]
                                metadata["temperature"] = task_stats["temperature"]
                                metadata["human_only"] = False
                            else:
                                metadata["human_only"] = True
                            metadata["setting"] = task_stats["setting"] if "setting" in task_stats else None
                            metadata["task"] = task_stats["task"] if "task" in task_stats else None
                            meta.append(metadata)
        elif dataset == DatasetName.AITDNA_SYNTHETIC:
            for model in os.listdir(root_dir):
                for task in os.listdir(os.path.join(root_dir, model)):
                    task_path = os.path.join(root_dir, model, task)
                    data.append(self.extract_data_for_text(task_path, notion, **kwargs))
                    if with_meta:
                        model_name, temperature = model.split("_")
                        temperature = float(temperature.replace("t", ""))
                        meta.append({
                            "author": model_name,
                            "model": model_name,
                            "temperature": temperature,
                            "task": task
                        })
        else:
            for data_point in os.listdir(root_dir):
                task_path = os.path.join(root_dir, data_point)
                data.append(self.extract_data_for_text(task_path, notion, **kwargs))
                if with_meta:
                    # todo could add metadata to other datasets too
                    meta.append({})

        self.data = data
        self.with_meta = with_meta
        if self.with_meta:
            self.meta = meta
        self.notion = notion
    

    def extract_data_for_text(self, task_path: str, notion: Notion, user: str = None, **kwargs):
        match notion:
            case Notion.SPAN_LEVEL:
                with open(os.path.join(task_path, "notions", "final_text_by_user_span_level.json"),
                          "r", encoding="utf-8") as f:
                    text = json.load(f)
                    return text

            case Notion.TOKEN_LEVEL:
                with open(os.path.join(task_path, "notions", "final_text_by_user_token_level.json"),
                          "r", encoding="utf-8") as f:
                    text = json.load(f)
                    return text

            case Notion.SENTENCE_LEVEL:
                threshold = kwargs.get("sentence_level_threshold")
                if not threshold or threshold == 0.5:
                    with open(os.path.join(task_path, "notions", "final_text_by_user_sentence_level.json"),
                            "r", encoding="utf-8") as f:
                        text = json.load(f)
                        return text
                if threshold < 0 or threshold > 1:
                    raise ValueError("sentence_level_threshold has to be between 0 and 1!")
                with open(os.path.join(task_path, "edits.json"),
                            "r", encoding="utf-8") as f:
                    text = json.load(f)
                notions = AITDNotions()
                text = notions.get_final_text_by_user_sentence_level(edits=text, threshold=threshold)
                return text

            case Notion.DOCUMENT_LEVEL:
                threshold = kwargs.get("document_level_threshold")
                if not threshold or threshold == 0.5:
                    with open(os.path.join(task_path, "notions", "final_text_by_user_document_level.json"),
                              "r", encoding="utf-8") as f:
                        text = json.load(f)
                    return text

                if threshold < 0 or threshold > 1:
                    raise ValueError("document_level_threshold has to be between 0 and 1!")
                with open(os.path.join(task_path, "edits.json"),
                            "r", encoding="utf-8") as f:
                    text = json.load(f)
                notions = AITDNotions()
                text = notions.get_final_text_by_user_document_level(edits=text, threshold=threshold)
                return text

            case Notion.BOUNDARY_LEVEL:
                n_segments = kwargs.get("n_segments", 5)
                length_penalty = kwargs.get("length_penalty", 1)
                impurity_penalty = kwargs.get("impurity_penalty", 1)
                if n_segments in [2, 5, 10] and length_penalty == impurity_penalty == 1:
                    with open(os.path.join(task_path, "notions",
                                           "boundary_level",
                                           f"final_text_by_user_boundary_level_ilp_{n_segments}seg_1lp_1ip.json"),
                                "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    with open(os.path.join(task_path, "edits.json"),
                              "r", encoding="utf-8") as f:
                        text = json.load(f)
                    notions = AITDNotions()
                    text = notions.get_final_text_by_user_boundary_level(text, n_seg=n_segments,
                                                                        length_penalty=length_penalty,
                                                                        impurity_penalty=impurity_penalty)
                    return text

            case Notion.CONTENT_BASED:
                llm_type = kwargs.get("llm_type", "gpt-5.4-nano")
                strictness_level = kwargs.get("strictness_level", 3)
                task = kwargs.get("task")[len("Task X.Y: "):]

                notions = AITDNotions()
                content_label_path = os.path.join(task_path, "notions", f"content_based_labels_{llm_type}.json")

                # load sentence-level
                with open(os.path.join(task_path, "notions", "final_text_by_user_sentence_level.json"),
                          "r", encoding="utf-8") as f:
                    sentences = json.load(f)

                if not os.path.exists(content_label_path):
                    labels = notions.get_content_based_labels(sentences, llm_type, task)
                    with open(content_label_path, "w", encoding="utf-8") as f:
                        json.dump(labels, f)
                else:
                    with open(content_label_path, "r", encoding="utf-8") as f:
                        labels = json.load(f)

                def matches_slevel(x:str):
                    if type(x) is not str or len(x) < 2 or not x.startswith("C"):
                        return True
                    return int(x[1:]) > strictness_level

                i = 0
                for s in sentences:
                    if s["author"] in ["Bot", "Mixed"]:
                        multi_label = labels[i]
                        i += 1

                        no_violation = all(map(matches_slevel, multi_label))
                        if no_violation:
                            s["author"] = "User"

                return sentences
            case Notion.INTENT_BASED:
                llm_type = kwargs.get("llm_type", "gpt-5.4-nano")
                looseness_level = kwargs.get("looseness_level", 1)
                task = kwargs.get("task")[len("Task X.Y: "):]

                notions = AITDNotions()
                intent_label_path = os.path.join(task_path, "notions", f"intent_based_labels_{llm_type}.json")

                # load sentence-level
                with open(os.path.join(task_path, "notions", "final_text_by_user_sentence_level.json"),
                          "r", encoding="utf-8") as f:
                    sentences = json.load(f)

                if not os.path.exists(intent_label_path):
                    labels = notions.get_intent_based_labels(sentences, llm_type, task)
                    with open(intent_label_path, "w", encoding="utf-8") as f:
                        json.dump(labels, f)
                else:
                    with open(intent_label_path, "r", encoding="utf-8") as f:
                        labels = json.load(f)

                def matches_llevel(x:str):
                    if type(x) is not str or len(x) < 2:
                        return True

                    return int(x[1:]) <= looseness_level

                i = 0
                for s in sentences:
                    if s["author"] in ["Bot", "Mixed"]:
                        multi_label = labels[i]
                        i += 1

                        permission = any(map(matches_llevel, multi_label))
                        if permission:
                            s["author"] = "User"

                return sentences
            case Notion.MEMBERSHIP_BASED:
                n_gram_len = kwargs.get("n_gram_len", 3)
                notions = AITDNotions()
                population = kwargs.get("population", None)
                if not population:
                    raise ValueError("Population not found!")
                with open(os.path.join(task_path, "edits.json"),
                              "r", encoding="utf-8") as f:
                        edits = json.load(f)
                text = notions.get_final_text_by_user_population_based(edits, population, n_gram_len)
                return text
            case Notion.AUTHORSHIP_BASED:
                n_gram_len = kwargs.get("n_gram_len", 3)
                notions = AITDNotions()
                population = kwargs.get("population", None)
                if not population:
                    raise ValueError("Population not found!")
                with open(os.path.join(task_path, "edits.json"),
                              "r", encoding="utf-8") as f:
                        edits = json.load(f)
                text = notions.get_final_text_by_user_population_based(edits, population, n_gram_len)
                return text

            case _:
                raise ValueError(f"Unknown notion type {notion}. Cannot create dataset")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        d = self.data[index]

        if self.with_meta:
            return d, self.get_meta(index)

        return d

    def get_meta(self, index):
        return self.meta[index]