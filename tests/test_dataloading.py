import os
from pathlib import Path

import dotenv
from torch.utils.data import DataLoader
from aitdna.notions.data_loading import DatasetName, AitdDataset, Notion, Population

# PARAMS
dotenv.load_dotenv(".env")
AITDNA_DATA_DIR = Path(os.environ.get("DATA_ROOT_PATH", "./data")) / "aitdna/formatted"


def IGNORE_test_membership_loading():
    print("data", AITDNA_DATA_DIR)

    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=AITDNA_DATA_DIR,
                          notion=Notion.MEMBERSHIP_BASED)
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda data_point: data_point)
    cnt = 0
    for batch in loader:
        for data_point in batch:
            for snippet in data_point:
                cnt += len(snippet)

    assert cnt > 0, "failed to load aitdna data points for membership notion"


def test_population():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=AITDNA_DATA_DIR,
                          notion=Notion.DOCUMENT_LEVEL)

    p = Population(dataset=dataset)

    assert "mechanistic interpretability" in p
    assert "mechanistic interpretability leads" not in p


def test_with_meta():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          root_dir=str(AITDNA_DATA_DIR),
                          root_dir=AITDNA_DATA_DIR,
                          with_meta=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for (data_point, metadata) in batch:
            print(data_point)
            print(metadata)
            break


def test_content_based_loading():
