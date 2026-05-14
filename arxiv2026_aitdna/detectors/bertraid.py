from transformers import pipeline

def load_and_predict_bertraid(text):
    """
    Main prediction function.
    @param text text to predict authorship for
    @returns (predicted_label, probability) - label (1 if AI-gen, 0 if not) and probability of AI-gen
    """
    pipe = pipeline("text-classification", model="ShantanuT01/BERT-tiny-RAID")
    ans = pipe(text)
    label = 1 if ans[0]['label'] == 'LABEL_1' else 0
    score = ans[0]['score']
    return ("ShantanuT01/BERT-tiny-RAID", label, score)