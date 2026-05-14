# create base classifier program
import os

import dspy

from aitdna.notions.ContentPolicy import ExtractWritingTopic


class WritingTaskIntentRelevance(dspy.Signature):
    """
    __Role__

    You are an evaluator assessing student LLM use in a writing assignment. Classify the student's prompt(s) using the labels below.
    Apply **all that fit**; if multiple prompts exist, labels that match any of them should be included.

    __Rules__
    - The prompts can receive **multiple labels** if more than one applies.
    - The prompts can receive **no labels** if none apply.
    - Apply labels strictly based on the definitions below
    - **Special case**: if the prompt is empty or asks for continuation explicitly, always assign NO label
     """
    prompts: str = dspy.InputField(desc="One or multiple prompts separated by ; as posed by the students")
    task: str = dspy.InputField(desc="The task/topic of the writing assignment")

    intent_labels: list[str] = dspy.OutputField(
        prefix="""
        __Labels__
        """,
        desc="""
        | Label | Apply when one of the prompts… |
        |-------|------------------------|
        | P1 | Requests language polish (grammar, style, clarity) |
        | P2 | Asks to add a minor fact not directly relevant to the assignment |
        | P3 | Asks to revise an existing argument or idea providing substantial information on how to do so |
        | P4 | Asks to introduce a new argument or idea providing substantial information on how to do so |
        """)


class RelevanceIntentProgram(dspy.Module):
    def __init__(self):
        super().__init__()

        self.extract_topic = dspy.Predict(ExtractWritingTopic)
        self.relevance_judgement = dspy.Predict(WritingTaskIntentRelevance)

    def forward(self, document: str, prompts: list[str], task: str) -> list[str]:
        topic = self.extract_topic(text=document, task=task).topic
        labels_per_sentence = []
        for s in prompts:
            labels_per_sentence += [{"labels": self.relevance_judgement(prompts=s, task=topic).intent_labels, "topic": topic}]

        return labels_per_sentence


class IntentPolicy:
    prog: RelevanceIntentProgram = None

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

        self.prog = RelevanceIntentProgram()

    def __call__(self, document, prompts, task):
        """
        Classify student prompts by intent relevance.

        If prompts are empty, automatically assign P4 (write from scratch).
        Otherwise, process all prompts through the relevance intent program.
        """
        # Handle empty prompts case (prompts is an empty string)
        if prompts == "":
            return [{"labels": [], "topic": None}]

        # Call the program with prompts and return results
        return self.prog(document=document, prompts=prompts, task=task)
