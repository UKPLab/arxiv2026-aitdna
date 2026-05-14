import sys
from aitdna.notions.data_loading.AitdDataset import AitdDataset
from aitdna.notions.data_loading.DatasetName import DatasetName
from aitdna.notions.data_loading.Notion import Notion
from aitdna.notions.data_loading.Population import Population
import nltk

import logging
logger = logging.getLogger(__name__)

class DetectionDataset(object):

    def __init__(self, data_path: str,
                 dataset_name: DatasetName,
                 detection_level: str = "document",
                 threshold: float | None = None):

        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = 0.5
        self.doc_level_dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.DOCUMENT_LEVEL,
                                             document_level_threshold=self.threshold, with_meta=True)
        if detection_level == "document":
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.DOCUMENT_LEVEL,
                                             document_level_threshold=self.threshold, with_meta=True)

        elif detection_level == "sentence":
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.SENTENCE_LEVEL, with_meta=True)
        elif detection_level == "boundary":
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.BOUNDARY_LEVEL, n_segments=5, with_meta=True)
        elif detection_level == "content":
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.CONTENT_BASED, with_meta=True)
        elif detection_level == "intent":
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.INTENT_BASED, with_meta=True)
        elif detection_level == "membership":
            token_level_dataset = AitdDataset(dataset=DatasetName.AITDNA,
                           root_dir="data/aitdna/formatted",
                           notion=Notion.TOKEN_LEVEL,
                           document_level_threshold=0,
                           with_meta=True)
            population = Population(dataset=token_level_dataset)
            self.dataset = AitdDataset(dataset=dataset_name, root_dir=data_path,
                                             notion=Notion.MEMBERSHIP_BASED,
                                             n_gram_len=2,
                                             population=population,
                                             with_meta=True)
            self.detokenizer = nltk.tokenize.TreebankWordDetokenizer()
            self.threshold = threshold if threshold else 0.5
        
        self.detection_level = detection_level

    def __getitem__(self, idx):
        if self.detection_level == "document":
            sample = {
                "input": self.dataset[idx][0][0]["text"]
            }
            for k, v in self.dataset[idx][0][0].items():
                sample[k] = v
            sample["label"] = str(sample["author"] == "Bot")
            sample["AI-generated"] = sample["author"] == "Bot"
            meta = self.doc_level_dataset[idx][1]
            if "model" in meta:
                sample["model"] = "human"
                sample["temperature"] = 0
        elif self.detection_level == "membership":
            text = self.detokenizer.detokenize([tok["text"] for tok in self.dataset[idx][0]])
            bot_tokens = [tok for tok in self.dataset[idx][0] if tok["author"] == "Bot"]
            ai_generated = len(bot_tokens) / len(self.dataset[idx][0]) >= self.threshold
            sample = {
                "input": text,
                "label": str(ai_generated),
                "AI-generated": ai_generated
            }
            meta = self.doc_level_dataset[idx][1]
            if "model" in meta:
                sample["model"] = "human"
                sample["temperature"] = 0
        else:
            sample = []
            for local_sample in self.dataset[idx][0]:
                data = {
                    "input": local_sample["text"],
                    "label": str(local_sample["author"] != "User"),
                    "AI-generated": local_sample["author"] != "User",
                }
                meta = self.doc_level_dataset[idx][1]
                if "model" in meta:
                    data["model"] = "human"
                    data["temperature"] = 0
                sample.append(data)
        return sample


    def __len__(self):
        return len(self.dataset)