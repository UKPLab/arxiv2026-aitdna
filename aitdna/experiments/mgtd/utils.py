import json
import logging
import sys

import transformers
import numpy as np
import torch
from transformers.hf_argparser import DataClass
from transformers.trainer_utils import is_main_process

from transformers import PreTrainedModel
from .arguments import ModelArguments, DataPredictionArguments


logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default_dict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = default_dict2dict(v)
    return dict(d)


def randrange_excluding(a, b, excluded):
    """
    Samples a random number x, with a <= x < b with x != excluded and a <= excluded < b
    """
    assert a < b
    assert a <= excluded < b
    random_int = torch.randint(a, b - 1, ()).item()
    if random_int >= excluded:
        random_int += 1
    return random_int


def iterate_values_in_nested_dict(nested_dict):
    for value in nested_dict.values():
        if isinstance(value, dict):
            yield from iterate_values_in_nested_dict(value)
        else:
            yield value
