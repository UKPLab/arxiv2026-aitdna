import dataclasses
import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, \
                                    accuracy_score, f1_score
from scipy.stats import norm
from pangram import Pangram
from gptzero import GPTZeroAPI

from aitdna.experiments.mgtd.methods.base import Method
from aitdna.experiments.mgtd.methods.preprocessing.preprocessing import MGTDPreprocessor
from aitdna.experiments.mgtd.simple_aitd import SimpleAITD

os.environ["TORCH_COMPILE_DISABLE"] = "1"

class MGTDMethod(object):

    predictor_type = "zero-shot"

    def _get_roc_metrics(self, real_preds, sample_preds):
        real_labels = [0] * len(real_preds) + [1] * len(sample_preds)
        predicted_probs = real_preds + sample_preds

        fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)

        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(real_labels, predicted_probs)
        auprc = auc(recall_vals, precision_vals)

        # Calculate F1 for every possible threshold
        # Note: precision and recall have one more element than thresholds
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx]

        # # Youden's J statistic
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        # handle single-label case
        if len(set(real_labels)) == 1:
            predictions = [real_labels[0]] * len(predicted_probs)
        else:
            predictions = [int(prob >= optimal_threshold) for prob in predicted_probs]
        conf_matrix = confusion_matrix(real_labels, predictions, labels=[0, 1])
        precision = precision_score(real_labels, predictions)
        recall = recall_score(real_labels, predictions)
        f1 = f1_score(real_labels, predictions)
        accuracy = accuracy_score(real_labels, predictions)
        tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)

        tn, fp, fn, tp = conf_matrix.ravel()
        fpr = fp/(fp+tn)

        return float(roc_auc), float(optimal_threshold), conf_matrix.tolist(), float(
            precision), float(recall), float(f1), float(accuracy), float(tpr_at_fpr_0_01), float(fpr), float(auprc)

    def evaluate(self, predictions, dataset):
        human_predictions, machine_predictions = [], []
        for prediction, sample in zip(predictions, dataset):
            if sample["AI-generated"]:
                machine_predictions.append(prediction)
            else:
                human_predictions.append(prediction)

        auroc, _, _, precision, recall, f1, accuracy, _, fpr , auprc = self._get_roc_metrics(human_predictions, machine_predictions)

        metrics = {
            "Accuracy": accuracy,
            "AUROC": auroc,
            "F-1": f1,
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
            "AUPRC": auprc
        }

        print(metrics)
        return metrics

class CausalSeq2SeqMethod(Method):
    name="causal_seq2seq"
    predictor_type = "zero-shot"
    peft_task_type = "CAUSAL_LM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            # load_metric(metric, cache_dir=self.model_args.cache_dir) for metric in ["sacrebleu"]
        ]

        self.tokenizer.padding_side = "left"
    
    def preprocess_features(self, features):
        processor = MGTDPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, _ = processor.preprocess(features)
        return_dict = {
            "input_ids": input_ids,
            # "labels": labels
        }

        return return_dict

    def compute_metrics(self, predictions):
        return predictions
        
    def get_data_collator(self):
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def get_model_class(self):
        return AutoModelForCausalLM

    def get_predictor_class(self):
        return SimpleAITD

    def postprocess_predictions(self, predictions, dataset, input_ids=None, num_beams=1):
        out = []
        # None for RL-training but for eval it's the eval dataset
        if dataset is not None:
            reference_strings = self.tokenizer.batch_decode([sample["labels"] for sample in dataset], skip_special_tokens=True)
        model_class = self.get_model_class()

        if isinstance(predictions[0][0], str):
            decoded_predictions = []
            for batch in predictions:
                for prediction in batch:
                    decoded_predictions.append(prediction)
        else:
            decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        if dataset is not None:
            decoded_inputs = self.tokenizer.batch_decode(
                [sample["input_ids"] for sample in dataset], skip_special_tokens=True
            )
        else:
            assert input_ids is not None
            decoded_inputs = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

        for idx, prediction in enumerate(decoded_predictions):
            input_idx = math.floor(idx / self.model_args.num_return_sequences)
            if dataset is not None:
                out.append({
                    "sequence": prediction[len(decoded_inputs[input_idx]):].strip(),
                    "reference": reference_strings[input_idx],
                    "source": decoded_inputs[input_idx]
                })
            else:
                input_idx = math.floor(idx / num_beams)
                out.append({
                    "sequence": prediction[len(decoded_inputs[input_idx]):].strip(),
                    "source": decoded_inputs[input_idx].replace("user\n", "").replace("\nmodel\n", "").replace("\nsystem\n", "").replace("You are a helpful assistant. ", "") 
                })
        return out

class LikelihoodMethod(CausalSeq2SeqMethod, MGTDMethod):
    name = "likelihood"
    predictor_type = "zero-shot"

    @torch.no_grad
    def predict(self, batch, model, data_examples):
        predictions = []
        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)
        logits = model(**batch)["logits"]
        for sample in range(logits.shape[0]):
            # make sure pad tokens don't contribute to loss
            batch["input_ids"][sample][batch["attention_mask"][sample] == 0] = -100
            # shifting is already done inside the loss function so no need to do it here!
            loss = (-model.loss_function(logits[sample], batch["input_ids"][sample], model.config.vocab_size)).exp().item()
            predictions.append(loss)
        return predictions

class LogRankMethod(CausalSeq2SeqMethod, MGTDMethod):
    name = "log_rank"
    predictor_type = "zero-shot"

    @torch.no_grad
    def predict(self, batch, model, data_examples):
        predictions = []
        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)

        logits = model(**batch)["logits"]
        for sample in range(logits.shape[0]):
            num_pad_tokens = batch["attention_mask"][sample].shape[0] - batch["attention_mask"][sample].sum()
            # here we need to shift manually
            local_logits = logits[sample][num_pad_tokens:,:][:-1,:]
            local_input_ids = batch["input_ids"][sample][num_pad_tokens:][1:]

            sorted_logits = local_logits.argsort(-1, descending=True)
            matches = (sorted_logits == local_input_ids.unsqueeze(-1)).nonzero()

            ranks = matches[:,-1]
            ranks = ranks.float() + 1
            ranks = torch.log(ranks).float().mean().item()
            
            predictions.append(-ranks)

        return predictions


class BinocularsMethod(CausalSeq2SeqMethod, MGTDMethod):
    name = "binoculars"
    predictor_type = "zero-shot"

    @torch.no_grad()
    def predict(self, batch, model, data_examples):
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        predictions = []
        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)

        logits = model(**batch)["logits"]
        logits_ref = self.reference_model(**batch)["logits"]
        for sample in range(logits.shape[0]):
            log_ppl = model.loss_function(logits[sample], batch["input_ids"][sample], model.config.vocab_size).item()
            softmax = nn.Softmax(dim=1)
            # log_probs = softmax(logits[sample])
            log_probs = softmax(logits_ref[sample])
            # log_cross_ppl = loss(logits_ref[sample][:,:-1], log_probs[:,:-1]) 
            log_cross_ppl = loss(logits[sample][:,:-1], log_probs[:,:-1]) 
            log_cross_ppl = log_cross_ppl.mean().float().item()
            binoculars_score = log_ppl / log_cross_ppl   
            predictions.append(-binoculars_score)

        return predictions

    def get_model(self, config):
        model = super().get_model(config)
        model_class = self.get_model_class()
        self.reference_model = model_class.from_pretrained(self.model_args.reference_model_name_or_path).to("cuda:0").eval()
        print(f"# Parameters: {model.num_parameters()}")
        return model


class MinKMethod(CausalSeq2SeqMethod, MGTDMethod):
    name = "min_k"
    predictor_type = "zero-shot"

    @torch.no_grad
    def predict(self, batch, model, data_examples):
        loss_function = nn.CrossEntropyLoss(reduction="none")
        predictions = []
        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)

        logits = model(**batch)["logits"]
        for sample in range(logits.shape[0]):
            shift_logits = logits[sample][:-1,:]
            batch["input_ids"][sample][batch["attention_mask"][sample] == 0] = -100
            shift_input_ids = batch["input_ids"][sample][1:]
            seq_len = (shift_input_ids != -100).sum()
            loss = loss_function(shift_logits, shift_input_ids)
            k = int(max(1, 0.2*seq_len))
            loss = -(loss.topk(k = k)[0]).mean().item()
            predictions.append(loss)
        return predictions


class FastDetectGPTMethod(CausalSeq2SeqMethod, MGTDMethod):
    name = "fastdetectgpt"
    predictor_type = "zero-shot"
    
    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def compute_crit(self, tokens, model):
        labels = tokens["input_ids"][:, 1:]
        with torch.no_grad():
            logits_score = model(**tokens).logits[:, :-1]
            if self.model_args.sampling_model_name == self.model_args.model_name_or_path:
                logits_ref = logits_score
            else:
                assert torch.all(tokens["input_ids"][:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = model(**tokens).logits[:, :-1]
            crit = self.get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        return crit, labels.size(1)
    
    def compute_prob_norm(self, x, mu0, sigma0, mu1, sigma1):
        pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
        pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
        prob = pdf_value1 / (pdf_value0 + pdf_value1)
        return prob

    @torch.no_grad()
    def predict(self, batch, model, data_examples):
        classifier = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
            'llama3-8b_llama3-8b-instruct': {'mu0': 0.1603, 'sigma0': 1.0791, 'mu1': 2.4686, 'sigma1': 2.1582},
        }
        key = 'gpt-neo-2.7B_gpt-neo-2.7B'
        mu0 = classifier[key]['mu0']
        sigma0 = classifier[key]['sigma0']
        mu1 = classifier[key]['mu1']
        sigma1 = classifier[key]['sigma1']

        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)

        predictions = []
        bs = batch["input_ids"].size(0)
        for i in range(bs):
            single = {k: v[i:i+1] for k, v in batch.items()}
            crit, ntoken = self.compute_crit(single, model)
            prob = self.compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
            predictions.append(prob)

        return predictions


class APIMethod(Method):
    predictor_type = "api"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_features(self, features):
        return [feature[0]["text"] for feature in features]

    def compute_metrics(self, predictions):
        return predictions
        
    def get_model_class(self):
        return AutoModelForCausalLM
    
    def predict(self, text):
        raise NotImplementedError()


class PangramPredictor(APIMethod):
    name = "pangram"
    predictor_type = "api"

    def predict(self, text):
        pangram_client = Pangram()
        return pangram_client.predict(text)


class GPTZeroPredictor(APIMethod):
    name = "gpt_zero"
    predictor_type = "api"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def predict(self, text):
        gpt_zero_api_key = os.getenv("GPT_ZERO_API_KEY")
        gpt_zero = GPTZeroAPI(gpt_zero_api_key)
        return gpt_zero.text_predict(text)


class ModernBERTPredictor(CausalSeq2SeqMethod, MGTDMethod):
    name = "modernBERT"
    predictor_type = "fine-tuned"

    def preprocess_features(self, features):
        input_ids = []
        attention_mask = []

        tokenizer_class = self.get_tokenizer_class()
        tokenizer = tokenizer_class.from_pretrained(self.model_args.tokenizer_name)
        for text in features["input"]:
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids.append(encoded["input_ids"].squeeze(0))
            attention_mask.append(encoded["attention_mask"].squeeze(0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @torch.no_grad
    def predict(self, batch, model, data_examples):
        import torch._inductor.config as config
        if "labels" in batch:
            del batch["labels"]
        for k, v in batch.items():
            if k != "labels":
                batch[k] = v.to(model.device)
        outputs = model(**batch)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[:, 1].tolist()

    def get_tokenizer_class(self):
        return AutoTokenizer
    
    def get_model_class(self):
        return AutoModelForSequenceClassification

    def get_model(self, config):
        print("Config: ", config)
        model = super().get_model(config)
        print(f"# Parameters of BERT: {model.num_parameters()}")
        return model
