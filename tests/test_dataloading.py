from torch.utils.data import DataLoader
from aitdna.notions.data_loading import DatasetName, AitdDataset, Notion, Population

def test_population():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          notion=Notion.DOCUMENT_LEVEL)

    p = Population(dataset=dataset)

    assert ("mechanistic interpretability", 2) in p
    assert ("mechanistic interpretability leads", 3) not in p


def test_with_meta():
    dataset = AitdDataset(dataset=DatasetName.AITDNA,
                          notion=Notion.DOCUMENT_LEVEL,
                          with_meta=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    for batch in loader:
        for (data_point, metadata) in batch:
            print(data_point)
            print(metadata)
            break
