from .DesklibAiDetectionModel import load_and_predict_desklib
from .RADAR import load_and_predict_radar
from .bertraid import load_and_predict_bertraid
from .ZeroGPT import predict_zero_gpt

import pandas as pd
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DATASET_PATH = "datasets/formatted/txaitd_data_2026_01_22"
DESKLIB_THRESHOLD = 0.5
    

def use_yes_no_models(text):
    try:
        radar = load_and_predict_radar(text)
        bert = load_and_predict_bertraid(text)
        desklib = load_and_predict_desklib(text, DESKLIB_THRESHOLD)
        return pd.DataFrame([radar, desklib, bert], columns=["Model", "Class", "Probability"])
    except Exception as e:
        print(e)
        desklib = load_and_predict_desklib(text, DESKLIB_THRESHOLD)
        return pd.DataFrame([desklib], columns=["Model", "Class", "Probability"])

def use_finegrained_models(text):
    (model_name, ai_generated, ai_percentage) = predict_zero_gpt(text)
    return {"model_name": model_name, "ai_generated": ai_generated, "ai_percentage": ai_percentage}


for user in os.listdir(DATASET_PATH):
    for task in os.listdir(os.path.join(DATASET_PATH, user)):
        task_path = os.path.join(DATASET_PATH, user, task)
        if os.path.exists(os.path.join(task_path, "model_results", "model_results.csv")):
            continue
        text_path = os.path.join(task_path, "final_text.txt")
        with open(text_path, "r") as f:
            text = f.read()
        model_path = os.path.join(task_path, "model_results")
        output_path = os.path.join(model_path, "model_results.csv")
        if os.path.exists(output_path):
            continue
        yes_no_models = use_yes_no_models(text)
        yes_no_models.to_csv(output_path, index=False)

        # output_path = os.path.join(model_path, "fine_grained_results.json")
        # fine_grained_models = use_finegrained_models(text)
        # with open(output_path, "w") as f:
        #     json.dump(fine_grained_models, f)
