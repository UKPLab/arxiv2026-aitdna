# create base classifier program
import os

import dspy


class WritingTaskRelevance(dspy.Signature):
    """
    __Role__
    You are an expert evaluator analyzing student essays. Your task is to identify the most relevant sentences for evaluation to a
    given writing task by applying zero, one or more classification labels to a given sentence.

    __Rules__
    - A sentence can receive **multiple labels** if more than one applies.
    - A sentence can receive **no labels** if none apply.
    - Apply labels strictly based on the definitions below; do not infer intent.
    - C4 applies even if the claim is likely true, as long as no evidence is provided; except if the claim is personal or general knowledge or justified within the sentence.
    """
    sentence: str = dspy.InputField(desc="A sentence extracted from the response to the writing task.")
    task: str = dspy.InputField(desc="The task/topic of the writing assignment")

    relevance_labels: list[str] = dspy.OutputField(
        prefix="""
        __Labels__
        Apply all labels that apply to the sentence:
        """,
        desc="""
        | Label | Apply when the sentence... |
        |-------|---------------------------|
        | C1 | Directly answers the writing task |
        | C2 | Contains an idea (a creative thought, concept, perspective) that is directly relevant to the writing task |
        | C3 | Contains an argument (reasoning from premises to some conclusion) that is directly relevant to the writing task  |
        | C4 | Contains a factual claim presented without supporting evidence |
        """)


class ExtractWritingTopic(dspy.Signature):
    """
    Given the following text written in response to a writing task, extract the topic/assignment
    the text is tackling. Carefully check the beginning of the text where the authors should
    restate the topic they are tackling. If not, infer it from the text itself. Keep the topic
    very general. Match the topic with the general task/genre to summarize the writing assignment.
    """
    text: str = dspy.InputField(desc="the input text")
    task: str = dspy.InputField(desc="the genre/task of the text")

    topic: str = dspy.OutputField(desc="a one sentence summary on a high-level and in simple language of the writing assignment underlying the text.")


class RelevanceProgram(dspy.Module):
    def __init__(self):
        super().__init__()

        self.extract_topic = dspy.Predict(ExtractWritingTopic)
        self.relevance_judgement = dspy.Predict(WritingTaskRelevance)

    def forward(self, document: str, ai_sentences: list[str], task: str) -> list[str]:
        topic = self.extract_topic(text=document, task=task).topic
        labels_per_sentence = []
        for s in ai_sentences:
            labels_per_sentence += [{"labels": self.relevance_judgement(sentence=s, task=topic).relevance_labels, "topic": topic}]

        return labels_per_sentence


class ContentPolicy:
    prog: RelevanceProgram = None

    def __init__(self, llm:str):
        super().__init__()

        if "deepseek" in llm:
            apikey = os.environ.get("DEEPSEEK_API_KEY", None)
            assert apikey is not None

            lm = dspy.LM(model=f"deepseek/{llm}",
                         api_key=apikey)
        elif "gpt" in llm:
            apikey = os.environ.get("OPENAI_API_KEY", None)
            assert apikey is not None

            lm = dspy.LM(model=f"openai/{llm}",
                         api_key=apikey)
        else:
            raise ValueError("Provided llm type not supported")

        dspy.configure(lm=lm)

        self.prog = RelevanceProgram()

    def __call__(self, document, ai_sentences, task):
        return self.prog(document=document, ai_sentences=ai_sentences, task=task)
