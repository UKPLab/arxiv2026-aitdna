import json
import os
import itertools
import argparse

from ....utils import send_request

def generate_texts(prompt_file_path: str, models: list[str], temperatures: list[float], dst_root: str):
    with open(prompt_file_path, "r") as f:
        all_prompts = json.load(f)
    for model, temperature in itertools.product(models, temperatures):
        model_folder = f"{model}_t{temperature}"
        if not os.path.exists(os.path.join(dst_root, model_folder)):
            os.mkdir(os.path.join(dst_root, model_folder))
        for i, (prompt_type, prompts) in enumerate(all_prompts.items()):
            for prompt in prompts:
                text = send_request(model=model, prompt=prompt, temperature=temperature)
                task_name = f"{prompt_type}_{str(i)}"
                if not os.path.exists(os.path.join(dst_root, model_folder, task_name)):
                    os.mkdir(os.path.join(dst_root, model_folder, task_name))
                with open(os.path.join(dst_root, model_folder, task_name, "final_text.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(text)
                with open(os.path.join(dst_root, model_folder, task_name, "prompt.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(prompt)

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CLI for generating datasets of the benchmark. Exhausts all model-temperature combinations for dataset creation."
    )
    parser.add_argument("-p", "--prompt_file_path", type=str, required=True, help="Path to the json file with prompts")
    parser.add_argument("-d", "--dst_root", type=str, required=True, help="Path to the destination root folder")
    parser.add_argument("-m", "--models", nargs="+", required=True, help="List of models to use for the generation")
    parser.add_argument("-t", "--temperatures", nargs="+", required=True, help="List of temperatures to use for the generation")
    argv = parser.parse_args(argv)
    
    prompt_file_path = argv.prompt_file_path
    dst_root = argv.dst_root
    models = argv.models
    temperatures = [float(t) for t in argv.temperatures]
    generate_texts(prompt_file_path=prompt_file_path,
                   models=models, temperatures=temperatures,
                   dst_root=dst_root)
