import abc

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoConfig, DataCollatorWithPadding, EvalPrediction

from aitdna.experiments.mgtd.mgtd_datasets.DetectionDataset import DetectionDataset
from aitdna.notions.data_loading.DatasetName import DatasetName

class Method(abc.ABC):
    
    def __init__(self, model_args, data_args, config = None, tokenizer = None):
        self.model_args = model_args
        self.data_args = data_args
        self.config = config
        self.tokenizer = tokenizer
        self.metrics = []

    @abc.abstractmethod
    def get_model_class(self, config):
        raise NotImplementedError()

    def get_model(self, config):
        model_class = self.get_model_class(config)
        if self.model_args.model_name_or_path is not None:
            model = model_class.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                torch_dtype=torch.bfloat16 if not "Scout" in self.model_args.model_name_or_path else torch.int4
            )
            model.config.attention_probs_dropout_prob = 0.0
            model.config.hidden_dropout_prob = 0.0
        else:
            print("Initializing model from scratch")
            model_config = AutoConfig.from_pretrained(self.model_args.config_name)
            model = model_class.from_config(model_config)
        # model.resize_token_embeddings(len(self.tokenizer))
        print(f"# Parameters: {model.num_parameters()}")
        return model

    @abc.abstractmethod
    def preprocess_features(self, features):
        raise NotImplementedError()

    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)

    @abc.abstractmethod
    def get_predictor_class(self):
        raise NotImplementedError()

    def postprocess_predictions(self, p, dataset):
        return p

    @abc.abstractmethod
    def compute_metrics(self, p: EvalPrediction):
        raise NotImplementedError()

    def _get_dataset(self, config_name=None, train=False, process=True):
        all_datasets = []
        if config_name is None:
            for dataset_name in self.data_args.dataset_name.split(";"):
                dataset_name = DatasetName[self.data_args.dataset_name]
                dataset = DetectionDataset(self.data_args.dataset_path,
                                            dataset_name,
                                            self.data_args.detection_level,
                                            self.data_args.dataset_threshold)
                if self.data_args.detection_level in ["span", "sentence", "boundary", "content", "intent"]:
                    # then each item is a list of spans
                    dataset = list(dataset)
                    dataset_list = []
                    for sample in dataset:
                        for local_sample in sample:
                            dataset_list.append(local_sample)
                    dataset = dataset_list
                dataset = Dataset.from_list(dataset)
                if self.data_args.dataset_restrict_to is not None:
                    allowed_authors = self.data_args.dataset_restrict_to.split(";")
                    dataset = dataset.filter(lambda sample: sample["model"] in allowed_authors)
                all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)

        old_eval_column_names = dataset.column_names

        if process:
            dataset = dataset.map(
                self.preprocess_features,
                batched=True,
                batch_size=1000,
                writer_batch_size=1000,
                load_from_cache_file=False,
                remove_columns=old_eval_column_names,
                fn_kwargs={"train": train}
            )
        return dataset

    def get_test_dataset(self, process=True):
        return self._get_dataset(train=False, process=process)
