import os
from pathlib import Path
import json
import dotenv
from aitdna.notions.NotionConverter import Converter
from aitdna.notions.data_loading import DatasetName, AitdDataset, Notion, Population
from torch.utils.data import DataLoader

# todo directories
def test_splitting():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          notion=Notion.SPAN_LEVEL)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for data_point in batch:
            break
        break
    converter = Converter()
    new_dataset = converter.convert(dataset, notion = Notion.SENTENCE_LEVEL)
    new_loader = DataLoader(new_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in new_loader:
        for data_point in batch:
            break
        break


def test_aggregating():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          notion=Notion.TOKEN_LEVEL)
    converter = Converter()
    new_dataset = converter.convert(dataset, notion = Notion.SENTENCE_LEVEL)
    new_loader = DataLoader(new_dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in new_loader:
        for data_point in batch:
            break
