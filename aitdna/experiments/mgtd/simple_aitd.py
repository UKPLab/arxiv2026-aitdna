import contextlib
import io
import logging
import sys

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class SimpleAITD(object):

    def __init__(
        self,
        model=None,
        data_collator=None,
        dataset=None,
        method=None,
        data_args=None,
    ):
        self.data_collator = data_collator
        self.dataset = dataset
        self.method = method

        self.data_args = data_args
        if model:
            self.model = model.bfloat16().to("cuda:0").eval()

    def _get_dataloader(self):

        dataloader = DataLoader(
            self.dataset, 
            shuffle=False, 
            batch_size=self.data_args.eval_batch_size,
            collate_fn=self.data_collator
        )
        return dataloader


    @torch.no_grad()
    def predict(self, test_dataset_raw):
        all_outputs = []
        test_dataloader = self._get_dataloader()
        for idx, batch in tqdm(enumerate(test_dataloader)):
            outputs = self.method.predict(batch, self.model, test_dataset_raw[idx*self.data_args.eval_batch_size: idx*self.data_args.eval_batch_size + self.data_args.eval_batch_size])
            all_outputs.extend(outputs)
        return all_outputs
