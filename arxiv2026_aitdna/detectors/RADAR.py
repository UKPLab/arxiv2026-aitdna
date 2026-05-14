from transformers import pipeline

def load_and_predict_radar(text):
    """
    Main prediction function.
    @param text text to predict authorship for
    @returns (predicted_label, probability) - label (1 if AI-gen, 0 if not) and probability of AI-gen
    """
    pipe = pipeline("text-classification", model="TrustSafeAI/RADAR-Vicuna-7B")
    ans = pipe(text, truncation=True, max_length=512)
    label = 1 if ans[0]['label'] == 'LABEL_1' else 0
    score = ans[0]['score']
    return ("TrustSafeAI/RADAR-Vicuna-7B", label, score)