import copy
import dataclasses
import argparse
import itertools
import json
import logging
import os
import subprocess
import sys


import time
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

import torch
import transformers
from sacrebleu.metrics import BLEU
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser
)
from transformers.trainer_utils import is_main_process, PredictionOutput

from aitdna.experiments.mgtd.arguments import *
from aitdna.experiments.mgtd.methods.base import Method
from aitdna.experiments.mgtd.methods.generation import CausalSeq2SeqMethod, LikelihoodMethod, \
LogRankMethod, BinocularsMethod, \
 MinKMethod, FastDetectGPTMethod, \
 PangramPredictor, GPTZeroPredictor, ModernBERTPredictor
from aitdna.experiments.mgtd.utils import NumpyEncoder
from aitdna.notions.data_loading import DatasetName, AitdDataset, Notion, Population

logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)

method_classes = [
    LikelihoodMethod,
    LogRankMethod,
    BinocularsMethod,
    MinKMethod,
    ModernBERTPredictor,
    FastDetectGPTMethod,
    # PangramPredictor,
    # GPTZeroPredictor,
]


def get_config_class(model_args):
    return AutoConfig

def get_tokenizer_class(config, model_args):
    return AutoTokenizer

def get_tokenizer_name(config, model_args):
    if model_args.tokenizer_name:
        return model_args.tokenizer_name
    else:
        return model_args.model_name_or_path

def run_and_evaluate_local(method_definition, trainer, test_dataset, test_dataset_raw, data_args, model_args):
    model = model_args.model_name_or_path[0]
    file_name = f"predictions_{model_args.method}_{data_args.dataset_name}_{data_args.detection_level}.json"
    file_path = os.path.join(data_args.metric_output_dir, file_name)
    if os.path.exists(file_path):
        return
    pred_type = f"{model_args.method}_{data_args.dataset_name}"
    logger.info(f"Predicting for {pred_type}")
    results = trainer.predict(test_dataset, test_dataset_raw)

    with open(file_path, 'w') as f:
        json.dump(
            dataclasses.asdict(results) if type(
                results) == PredictionOutput else results,
            f,
            cls=NumpyEncoder
        )
    results = method_definition.evaluate(
        results,
        test_dataset_raw
    )

    if data_args.metric_output_dir is not None:
        file_name = f"metrics_{model_args.method}_{data_args.dataset_name}_{data_args.detection_level}.json"
        with open(os.path.join(data_args.metric_output_dir, file_name), 'wt') as f:
            try:
                json.dump(
                    dataclasses.asdict(results) if type(
                        results) == PredictionOutput else results,
                    f,
                    cls=NumpyEncoder
                )
            except:
                json.dump({}, f)


def run_prediction(method_definition, trainer, data_args, model_args):
    test_dataset = method_definition.get_test_dataset()

    test_dataset_raw = method_definition.get_test_dataset(process=False)
    run_and_evaluate_local(method_definition=method_definition, trainer=trainer, test_dataset=test_dataset,
            test_dataset_raw=test_dataset_raw, data_args=data_args, model_args=model_args)


def run_predict_api(model_args, data_args):
    evaluation_folder = data_args.evaluation_folder
    dataset = AitdDataset(dataset=DatasetName(data_args.dataset_name),
                          root_dir=data_args.dataset_path,
                          notion=Notion.DOCUMENT_LEVEL)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
    method_class = next(
                (m for m in method_classes if m.name == model_args.method), None)
    method_definition: Method = method_class(
            model_args, data_args)

    for idx, batch in enumerate(loader):
            preprocessed_batch = method_definition.preprocess_features(batch)
            for text in preprocessed_batch:
                if os.path.exists(os.path.join(evaluation_folder, f"{str(idx)}.json")):
                    continue
                try:
                    outputs = method_definition.predict(text)
                    with open(os.path.join(evaluation_folder, f"{str(idx)}.json"), "w") as f:
                        json.dump(outputs, f)
                except ValueError as e:
                    logger.warning(f"{str(idx)}: {str(e)}")
                


def run_predict(model_args, data_args):
    
    logging.info(data_args)

    method_class = next(
        (m for m in method_classes if m.name == model_args.method), None)
    
    if method_class and method_class.predictor_type == "api":
        run_predict_api(model_args, data_args)
    else:
        config_class = get_config_class(model_args)

        config = config_class.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
        if model_args.num_labels is not None:
            config.num_labels = model_args.num_labels


        tokenizer_class = get_tokenizer_class(config, model_args)
        tokenizer = tokenizer_class.from_pretrained(
            get_tokenizer_name(config, model_args),
            padding_side="left",
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if method_class is None:
            raise Exception(f"No method class for name {model_args.method}.")
        method_definition: Method = method_class(
            model_args, data_args, config, tokenizer)
        # Set seed before initializing model.

        model = None
        if method_definition.predictor_type != "api":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = method_definition.get_model(config).to(device)

        if model is not None:
            model.config.keys_to_ignore_at_inference = [
                "decoder_attentions"
            ]

        eval_dataset = method_definition.get_test_dataset()

        data_collator = method_definition.get_data_collator()
        trainer_class = method_definition.get_trainer_class()

        trainer = trainer_class(
            model=model,
            tokenizer=method_definition.tokenizer,
            method=method_definition,
            data_collator=data_collator,
            data_args=data_args,
            dataset=eval_dataset
        )

        run_prediction(method_definition, trainer, data_args, model_args)


def set_envs(cache_dir):
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ["VLLM_CACHE_ROOT"] = cache_dir
    os.environ["VLLM_CONFIG_ROOT"] = cache_dir
    os.environ["FLASHINFER_WORKSPACE_BASE"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir

    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    # os.environ["VLLM_USE_V1"] = "0"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_TOKEN"] = "TODO"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(argv=None):
    parser_arguments = (ModelArguments, DataPredictionArguments)
    parser = HfArgumentParser(parser_arguments)

    argparser = argparse.ArgumentParser(
        description="CLI for generating datasets of the benchmark. Exhausts all model-temperature combinations for dataset creation."
    )
    argparser.add_argument("-p", "--path_to_config_json", type=str, required=True, help="Path to the json file with evaluation settings")
    argparser.add_argument("-m", "--evaluate_all_methods", action="store_true", help="Evaluate all detectors")
    argparser.add_argument("-a", "--evaluate_all_datasets", action="store_true", help="Evaluate all datasets")
    argparser.add_argument("-d", "--path_to_datasets_json", type=str, help="Only relevant if evaluate_all == True. Path to the json that contains all dataset names to be evaluated and the corresponding root directory.")
    argparser.add_argument("-c", "--cache_dir", type=str, help="Cache directory for triton etc")

    argv = argparser.parse_args(argv)
    
    path_to_json = argv.path_to_config_json
    evaluate_all_methods = argv.evaluate_all_methods
    evaluate_all_datasets = argv.evaluate_all_datasets
    cache_dir = argv.cache_dir
    set_envs(cache_dir)
    
    if path_to_json.endswith(".json"):
        with open(path_to_json) as fp:
            json_args_dict = json.load(fp)

        model_args, data_args = parser.parse_dict(
            json_args_dict, allow_extra_keys=True)
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    if evaluate_all_methods:
        methods = [method.name for method in method_classes]
    else:
        methods = [model_args.method]
    
    if evaluate_all_datasets:
        path_to_datasets_json = argv.path_to_datasets_json
        with open(path_to_datasets_json, "r") as f:
            datasets_paths = json.load(f)
        datasets = [(dataset, path) for dataset, path in datasets_paths.items()]
    else:
        datasets = [(data_args.dataset_name, data_args.dataset_path)]
    
    for (dataset, path) in datasets:
        for method in methods:
            data_args.dataset_name = dataset
            data_args.dataset_path = path
            model_args.method = method
            run_predict(model_args, data_args)
