import os
from pathlib import Path
import json
import dotenv
from txaitd.notions.NotionConverter import Converter
from txaitd.notions.data_loading import DatasetName, AitdDataset, Notion, Population
from torch.utils.data import DataLoader

# todo directories
def test_splitting():
    AITDNA_DATA_DIR = "data/aitdna_anonymized/new_formatted"
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=AITDNA_DATA_DIR,
                          notion=Notion.SPAN_LEVEL)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for data_point in batch:
            with open("span_level.json", "w") as f:
                json.dump(data_point, f)
            break
        break
    converter = Converter()
    new_dataset = converter.convert(dataset, notion = Notion.SENTENCE_LEVEL)
    new_loader = DataLoader(new_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in new_loader:
        for data_point in batch:
            with open("sentence_level.json", "w") as f:
                json.dump(data_point, f)
        break


def test_aggregating():
    AITDNA_DATA_DIR = "data/aitdna_anonymized/new_formatted"
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=AITDNA_DATA_DIR,
                          notion=Notion.TOKEN_LEVEL)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for data_point in batch:
            print("Here")
    converter = Converter()
    new_dataset = converter.convert(dataset, notion = Notion.SENTENCE_LEVEL)
    new_loader = DataLoader(new_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in new_loader:
        for data_point in batch:
            print("Here")