import contextlib
import copy
import io
import logging
import os
import json
import sys
from contextlib import nullcontext

import numpy as np
import torch
from huggingface_hub import login
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from vllm.inputs.data import TokensPrompt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

class SimpleTrainer(object):

    def __init__(
        self,
        model=None,
        args=None,
        tokenizer=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        method=None,
        data_args=None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.method = method

        self.data_args = data_args
        if model:
            self.model = model.bfloat16().to("cuda:0").eval()

    def _get_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator
        )

        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False, 
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator
        )
        return train_dataloader, eval_dataloader

    def _get_test_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False, 
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator
        )
        return eval_dataloader


    @torch.no_grad()
    def predict(self, test_dataset, test_dataset_raw):
        final_outputs = []
        all_outputs = []
        test_dataloader = self._get_test_dataloader()
        for idx, batch in tqdm(enumerate(test_dataloader)):
            outputs = self.method.predict(batch, self.model, test_dataset_raw[idx*self.args.eval_batch_size: idx*self.args.eval_batch_size + self.args.eval_batch_size])
            all_outputs.extend(outputs)
        return all_outputs
