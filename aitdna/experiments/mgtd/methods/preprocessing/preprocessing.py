class MGTDPreprocessor:

    def __init__(self, config, data_args, model_args, tokenizer):
        self.config = config
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

    def preprocess(self, features):
        samples = []
        for sample in features["input"]:
            samples.append(self.tokenizer(sample, truncation=True, max_length=self.model_args.max_input_length)["input_ids"])
        return samples, None
