import copy
import itertools

from transformers import GemmaTokenizerFast

from aitdna.experiments.mgtd.methods.preprocessing.base import Preprocessor

class Seq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences = self.tokenizer(
            [self.model_args.prompt_prefix + source for source in features["input"]], 
            max_length=self.model_args.max_input_length,
            truncation=True
        )["input_ids"]
        labels = self.tokenizer(
            features["output"], 
            max_length=self.model_args.max_input_length,
            truncation=True
        )["input_ids"]

        return sequences, labels


class CausalSeq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels, sequences_no_labels = [], [], []
        output_key = "output" if "output" in features else "label"
        for source, target in zip(features["input"], features[output_key]):
            if "qwen" in self.model_args.model_name_or_path.lower():
                sys_id = "assistant"
            elif "llama" in self.model_args.model_name_or_path.lower():
                sys_id = "model"
            else:
                sys_id = "system"
            if source is not None and target is not None:
                # hack to truncate to max_input_length..
                source = self.tokenizer.batch_decode(
                    [self.tokenizer(
                        source,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.model_args.max_input_length
                    )["input_ids"]],
                    skip_special_tokens=True
                )[0]
                try:
                    messages = []
                    if self.model_args.system_prompt != "":
                        messages.append({
                            "role": "system",
                            "content": self.model_args.system_prompt,
                        })
                    messages.append({
                        "role": "user",
                        "content": self.model_args.prompt_prefix + " " + source
                    })
                    input_no_label = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=not "llama" in self.model_args.model_name_or_path, 
                        date_string = "25 Dec 2025", # some models have a date string and if you finetune with one date only and then use a different one it throws them off...
                        enable_thinking=False)
                    messages.append({
                        "role": sys_id,
                        "content": target
                    })
                    input_label = self.tokenizer.apply_chat_template(
                        messages,
                        date_string = "25 Dec 2025", 
                        enable_thinking=False
                    )
                    tokenized_target = input_label
                except ValueError:
                    # GPT2 trained from scratch
                    input_no_label = self.tokenizer(source, add_special_tokens=False)["input_ids"] 
                    input_label = self.tokenizer(source + target, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

            sequences.append(input_label)
            sequences_no_labels.append(input_no_label)
            labels.append(self.tokenizer(target, add_special_tokens=False)["input_ids"])

        return sequences_no_labels, sequences if train else labels

class LanguageModelingPreprocessor(Preprocessor):

    def _group_texts(self, examples, max_input_length):
        block_size = max_input_length
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result

    def preprocess(self, features, train=True, max_input_length=128):
        sequences, labels = [], []
        all_samples = {"input_ids": []}
        for sample in features["text"]:
            sample = self.tokenizer([sample], add_special_tokens=False)
            all_samples["input_ids"].extend(self._group_texts(sample, max_input_length)["input_ids"])

        return all_samples["input_ids"], None #all_samples["labels"]


class LanguageModelingPreprocessorWithEOSToken(Preprocessor):

    def _group_texts(self, examples, max_input_length):
        block_size = max_input_length
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result

    def preprocess(self, features, train=True, max_input_length=1e10):
        sequences, labels = [], []
        all_samples = {"input_ids": []}
        for sample in features["text"]:
            sample = self.tokenizer([sample], add_special_tokens=False)
            all_samples["input_ids"].extend(self._group_texts(sample, max_input_length)["input_ids"])

        for idx, sample in enumerate(all_samples["input_ids"]):
            all_samples["input_ids"][idx] = sample + [self.tokenizer.eos_token_id]

        return all_samples["input_ids"], None #all_samples["labels"]

class MGTDPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        samples = []
        for sample in features["input"]:
            samples.append(self.tokenizer(sample, truncation=True, max_length=self.model_args.max_input_length)["input_ids"])
        return samples, None
