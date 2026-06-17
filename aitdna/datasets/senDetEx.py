import random
import os
import json
import re
from nltk.metrics import edit_distance
import argparse
import math
import itertools
import openai
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from .aitdna_dataset.processing.format_data import get_and_save_notions,\
    get_and_save_final_text, create_folders_for_analysis

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
HF = os.getenv("HF")

def compute_perplexity(text, perplexity_model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(perplexity_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

    
def call_generator(text, model_name, temperature):
    prompt = f"Fill in each [MASK] in the following document with a single sentence to ensure overall fluency, coherence, and logic. Original document: {text}. New completed document:"
    if "gpt" in model_name:
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif "deepseek" in model_name:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError("Unsupported model!")
    response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature
        )
    return response.choices[0].message.content


def random_mask_sentences(document, gamma=0.35):
    sentences = document.replace('.\n', '. ').split('. ')
    num_to_mask = max(1, int(len(sentences) * gamma))
    mask_indices = random.sample(range(len(sentences)), num_to_mask)
    for i in mask_indices:
        sentences[i] = "[MASK]"
    return '. '.join(sentences)


def create_dataset(dataset_name, generator_name, temperature, max_samples,
                    perplexity_model, tokenizer, processed_dataset_path, gamma, seed):
    dataset = load_dataset(dataset_name, split='train')
    selected_data = dataset.shuffle(seed=seed).select(range(max_samples))
    dataset_short_name = dataset_name.split("/")[-1]

    all_dp = os.listdir(processed_dataset_path)
    relevant_dp = [dp for dp in all_dp if generator_name in dp and dataset_short_name in dp]
    computed_dp = [int(dp.split("_")[-1].replace(".json", "")) for dp in relevant_dp]
    last_dp = max(computed_dp, default=0)
    for i, example in enumerate(selected_data):
        if i <= last_dp:
            continue
        for j in range(10):
            doc = example['document'] if 'xsum' in dataset_name else example['prompt'] + ' ' + example['story']
            masked_doc = random_mask_sentences(doc, gamma=gamma)
            try:
                generated_doc = call_generator(masked_doc, generator_name, temperature)
            except openai.BadRequestError as e:
                print(f"{i}: {e}\n")
                continue
            ppl_original = compute_perplexity(doc, perplexity_model, tokenizer)
            ppl_generated = compute_perplexity(generated_doc, perplexity_model, tokenizer)
            if ppl_generated < ppl_original:
                with open(os.path.join(processed_dataset_path, f"{dataset_short_name}_{generator_name}_{str(i)}.json"), "w") as f:
                    data = {
                        "original": doc,
                        "masked": masked_doc,
                        "generated": generated_doc
                    }
                    json.dump(data, f)
                break

def create_sendetex(dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    seed = 42
    random.seed(seed)

    PERPLEXITY_MODEL = 'meta-llama/Meta-Llama-3-8B'
    DATASETS = ["euclaise/writingprompts", "EdinburghNLP/xsum"]
    MODELS = ["gpt-4o", "deepseek-chat"]
    gamma = 0.35
    m = {"EdinburghNLP/xsum": 5000, "euclaise/writingprompts": 3500}
    temperature = 0.7

    perplexity_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL, token=HF, device_map="cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL, token=HF)

    for (dataset, model) in itertools.product(DATASETS, MODELS):
        create_dataset(dataset, model, temperature, m[dataset], perplexity_model, tokenizer, dataset_path, gamma, seed)

def normalize(text):
    text = re.sub(r'\s+([,?.!])', r'\1', text)
    text = text.replace("`` ", '"').replace("``", '"').replace("''", '"').replace("[ WP ]", "[WP]")
    text = text.replace("“", '"'). replace("”", '"').replace("( ", "(").replace(" )", ")")
    text = text.replace(".** **", ". ").replace("** **", ". ").replace("**", ". ").replace(".\n", ". ")
    text = text.replace(" n't", "n't")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_min_edit_distance(masked, snippet):
    if len(snippet) < 15:
        return False
    min_levenshtein = 100000
    i = 0
    while i < len(masked) - len(snippet):
        substring = masked[i:i+len(snippet)]
        distance = edit_distance(substring, snippet)
        if distance < min_levenshtein:
            min_levenshtein = distance
        if min_levenshtein <= 3:
            return True
        # speed up
        if distance > 20:
            i += 10
        else:
            i += 1

    if min_levenshtein > 5:
        return False
    return True


def process_text(data):
    edits = []
    masked = normalize(data["masked"])
    generated_sentences = [sent for sent in normalize(data["generated"]).split(". ") if sent]
    len_text = 0
    for i, sent in enumerate(generated_sentences):
        if sent.strip() in masked:
            author = "User"
        else:
            found = find_min_edit_distance(masked, sent)
            author = "User" if found else "Bot"
        if i != len(generated_sentences) - 1:
            sent = sent + ". "
        edits.append({
            "user": author,
            "operationType": "insert",
            "text": sent,
            "offset": len_text,
            "span": len(sent)
        })
        len_text += len(sent)
    return edits


def process_dataset(orig_ds_root, dst_root):
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)
    files = sorted(os.listdir(orig_ds_root))
    for file in files:
        folder_name = file.replace(".json", "")
        dst_folder = os.path.join(dst_root, folder_name)
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)  
        stats_path, notions_path, boundary_path = create_folders_for_analysis(dst_folder)

        with open(os.path.join(orig_ds_root, file), "r") as f:
            data = json.load(f)
        edits = process_text(data)
        with open(os.path.join(dst_folder, "edits.json"), "w") as f:
            json.dump(edits, f)
        
        _, _, _, text_by_user, _ = get_and_save_notions(edits,
                                    boundary_path,
                                    notions_path)

        get_and_save_final_text(text_by_user, dst_folder)

        dataset, model, n = folder_name.split("_")
        stats = {
            "model": model,
            "dataset": dataset,
            "n": n
        }
        with open(os.path.join(stats_path, "dataset_related_stats.json"), "w") as f:
            json.dump(stats, f)
    
    
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--orig_ds_root', type=str, default="data/other_datasets/original/sendetex")
    parser.add_argument("-d", '--dst_root', type=str, default="data/other_datasets/processed/sendetex")
    args = parser.parse_args(argv)

    ORIG_DS_ROOT = args.orig_ds_root
    DST_DS_ROOT = args.dst_root
    create_sendetex(ORIG_DS_ROOT)
    process_dataset(ORIG_DS_ROOT, DST_DS_ROOT)
