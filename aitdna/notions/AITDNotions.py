from collections import Counter
import itertools
from itertools import groupby
import nltk
import ruptures as rpt
import imgkit
import numpy as np
import random

from .IntentPolicy import IntentPolicy
from .CostFunction import CostFunction
from .ContentPolicy import ContentPolicy

class AITDNotions:
    last_spans: list[dict] = []

    def __init__(self):
        random.seed(595100)

    def get_final_text_by_user_document_level(self, edits, threshold: float):
        """
        Returns a binary label of whether the text belongs to the AI or to the human.
        Threshold determines the minimum percentage of AI tokens to be classified as AI-generated
        """
        tokens = self.get_final_text_by_user_token_level(edits)
        n_user = len([token for token in tokens if token["author"] == "User"])

        if 1 - n_user / len(tokens) >= threshold:
            author = "Bot"
        else:
            author = "User"

        text = self._get_text_query_author(edits)
        final_text = "".join(letter for letter, _, _ in text)

        return [{"text": final_text, "author": author, "queries": list(set([query for t in tokens for query in t["queries"]]))}]


    def get_final_text_by_user_token_level(self, edits):
        """
        Returns the resulting text (end state of editor)
            in form: [(token, author), (token, author), ...]
        """
        text = self._get_text_query_author(edits)
        results = []
        final_text = "".join(letter for letter, _, _ in text)
        tokens = self.tokenize_with_spans(final_text)
        self.last_spans = tokens
        for token in list(tokens):
            _, authors, queries = zip(*text[token["start"]:token["end"]])
            author_count = Counter(authors)
            if len(author_count) > 1:
                final_author = "Mixed"
            else:
                final_author = authors[0]
            results.append({"text": token["token"], "author": final_author, "queries": list(set(query for query in queries if query))})
        return results

    def get_final_text_by_user_sentence_level(self, edits, threshold: float = 0.5):
        """
        Returns the resulting text (end state of editor)
            in form: [(sentence, author), (sentence, author), ...]
        """
        sentences_by_author = []
        text_by_user_span_level = self.get_final_text_by_user_span_level(edits)
        parts = []
        authors_by_char = []
        queries_by_char = []
        for data_point in text_by_user_span_level:
            text, author, query = data_point["text"], data_point["author"], data_point["queries"]
            parts.append(text)
            authors_by_char.extend([author] * len(text))
            queries_by_char.extend([query] * len(text))
        final_text = "".join([snippet["text"] for snippet in text_by_user_span_level])
        sent_index = 0
        for sentence in nltk.sent_tokenize(final_text):
            sent_index = sent_index + final_text[sent_index:].find(sentence)
            sent_tokens = self.tokenize_with_spans(sentence)
            sent_authors = []
            sent_queries = []
            for sent_token in sent_tokens:
                start = sent_token["start"]
                end = sent_token["end"]
                token_authors = set(authors_by_char[sent_index + start: sent_index + end])
                sent_authors.append("User" if token_authors == {"User"} else "Bot")
                sent_queries.extend(queries_by_char[sent_index + start: sent_index + end])
    
            author = "Bot" if len([token for token in sent_authors if token == "Bot"]) / len(sent_authors) >= threshold else "User"
            query = list(set(query for arr in sent_queries for query in arr if query))
            sentences_by_author.append({"text": sentence, "author": author, "queries": query})
        return sentences_by_author


    def get_final_text_by_user_span_level(self, edits):
        """
        Returns the resulting text (end state of editor)
            in form: [(span1, author), (span2, author), ...]
        """
        text = self._get_text_query_author(edits)
        groups_by_user = groupby(text, lambda x: x[1])
        merged_text = []
        for author, group in groups_by_user:
            i1, i2 = itertools.tee(group)
            snippet = "".join(letter for letter, _, _ in i1)
            queries = list(set(query for _, _, query in i2 if query))
            merged_text.append({"text": snippet, "author": author, "queries": queries})
        return merged_text

    def _get_text_query_author(self, edits):
        text = []
        latest_request = None
        for edit in edits:
            if "documentEditId" in edit:
                latest_request = edit
            if "operationType" not in edit or edit["operationType"] == "retain":
                continue
            snippet = edit["text"]
            offset = edit["offset"]
            span = edit["span"]
            if "user" in edit:
                author = edit["user"]
            else:
                author = "Bot" if edit["userId"] == 2 else "User"
            if author != "Bot":
                query = None
            else:
                if latest_request:
                    query = latest_request["query"]
                else:
                    query = ""
            if edit["operationType"] == "insert" and snippet is not None:
                text[offset:offset] = [(letter, author, query) for letter in snippet]
            elif edit["operationType"] == "delete":
                text = text[:offset] + text[offset + span:]
        return text
    
    def _get_text_by_snippet(self, text: str, tokens: list[dict], segments: list[int], ai_threshold: float = 0.5):
        spans = []
        segment_start = 0
        segment_end = 0
        segment_authors = []
        segment_queries = []
        label = 1
        for i, (token, segment) in enumerate(zip(tokens, segments)):
            if segment == label:
                segment_end = text.index(token["text"], segment_end) + len(token["text"])
                segment_authors.append(token["author"])
                segment_queries.extend(token["queries"])
                if i == len(tokens) - 1:
                    author = "Bot" if len([author for author in segment_authors if author != "User"]) / len(segment_authors) >= ai_threshold else "User"
                    spans.append({"text": text[segment_start:segment_end], "author": author, "queries": list(set(segment_queries))})
            else:
                author = "Bot" if len([author for author in segment_authors if author != "User"]) / len(segment_authors) >= ai_threshold else "User"
                spans.append({"text": text[segment_start:segment_end], "author": author, "queries": list(set(segment_queries))})
                segment_start = segment_end
                label += 1
                segment_authors = []
                segment_queries = []
        return spans

    def get_final_text_by_user_boundary_level(self, edits, n_seg: int,
                                            length_penalty: float = 0.1,
                                            impurity_penalty: float=0.1):
        """
        Divides the text into n segments by authorship.
        
        :param self: Description
        :param edits: Description
        :param n_seg: Description
        :type n_seg: int
        :param algorithm: algorithm name. Currently supported: ILP and DP
        :type algorithm: str
        """
        tokens = self.get_final_text_by_user_token_level(edits=edits)
        labels = []
        for token in tokens:
            if token["author"] == "User":
                labels.append(0)
            else:
                labels.append(1)
        segments = self.ruptures_partition(labels=labels, n_seg=n_seg, length_penalty=length_penalty, impurity_penalty=impurity_penalty)
        
        final_text = "".join([text["text"] for text in self.get_final_text_by_user_span_level(edits)])
        return self._get_text_by_snippet(final_text, tokens, segments)

    
    def evaluate_segments(self, segments: list[dict], file_name: str):
        """
        Displays the predicted segments
        
        :param self
        :param edits: The list of edits
        :param segments: Predicted segments in form [1, 1, 1, 2, 2, ..]
        """
        html = self.get_html_color_codes(segments)
        html_path = file_name.replace(".png", ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        imgkit.from_file(html_path,
                        file_name)
        # os.remove(html_path)

    
    def get_html_color_codes(self, segments: list[str]) -> str:
        """
        Get HTML with color-coded text
        """

        html = """<html>
            <head>
                <style>
                    body { font-size: 16px; line-height: 1.6; }
                    span { padding: 1px 2px; border-radius: 3px; }
                    .legend-color-box { width: 10px; height: 10px; }
                </style>
            </head>
            <body><div>"""

        html += '''<div class="container">
                    <div class="d-inline-block">'''
        
        color_map = {}
        for i, segment in enumerate(segments):
            text = segment["text"]
            hue = (i * 137) % 360
            color_map[i] = f"hsl({hue}, 80%, 80%)"

            html += f'''<div
                            class="d-inline-block align-items-center">
                        <span
                            class="legend-color-box d-inline-block"
                            style="background: {color_map[i]}"
                        > </span>
                        <span class="legend-label">{str(i)}</span>
                        </div>'''
        html += '</div><br>'

        for i, segment in enumerate(segments):
            text = segment["text"]
            color = color_map[i]
            html += f'<span style="background:{color}">{text}</span>'
        html += "</div></body></html>"
        return html
    
    def tokenize_with_spans(self, text: str):
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(text))
        return [{'token': text[s:e], 'start': s, 'end': e} for s,e in spans]

    def ruptures_partition(self, labels: list[int], n_seg: int, length_penalty: float = 0.1, impurity_penalty: float=0.1):
        """
        Partitioning with dynamic programming
        
        :param self: Description
        :param labels: labels per token in form [0, 1, 0,..]
        :type labels: list[int]
        :param n_seg: number of segments
        :type n_seg: int
        """
        n_bkps = n_seg - 1
        algo = rpt.Dynp(custom_cost=CostFunction(n_segments=n_seg, length_penalty=length_penalty, impurity_penalty=impurity_penalty), jump=1).fit(np.array(labels))
        prediction = algo.predict(n_bkps=n_bkps)
        # prediction is in form: [last index of 1st segment, of 2nd, ..], 1st index is 1
        segments = []
        current_partition = 1
        for i in range(len(labels)):
            if i in prediction:
                current_partition += 1
            segments.append(current_partition)
        return segments

    def get_content_based_labels(self, sentence_level_text: list, llm_type:str, task:str):
        document = " ".join(s["text"] for s in sentence_level_text)
        ai_sentences = [s["text"] for s in sentence_level_text if s["author"] in ["Bot", "Mixed"]]

        if len(ai_sentences) > 0:
            policy = ContentPolicy(llm_type)
            labels_per_sentence = policy(document=document, ai_sentences=ai_sentences, task=task)

            return [d["labels"] for d in labels_per_sentence]
        else:
            return []

    def get_intent_based_labels(self, sentence_level_text:list, llm_type:str, task:str):
        document = " ".join(s["text"] for s in sentence_level_text)
        # Extract all prompts/queries from the document
        queries = ["; ".join(s["queries"]) for s in sentence_level_text if s["author"] in ["Bot", "Mixed"]]

        if len(queries) > 0:
            policy = IntentPolicy(llm_type)
            labels_per_prompt = policy(document=document, prompts=queries, task=task)

            return [d["labels"] for d in labels_per_prompt]
        else:
            return []

    def get_final_text_by_user_population_based(self, edits: list[dict],
                                                reference_corpus: 'Population', n_gram_len: int) -> list[dict]:
        """Returns a list with len() == n tokens in text. For each token, author is User if an n-gram containing this token
        is present in reference corpus.

        Can be used to generate both Membership-based and Authorship-based notions.
        For membership-based, pass the human-only corpus of all users.
        For authorship-based, pass the human-only corpus of the target user.

        Args:
            edits (list[dict]): document edits
            reference_corpus (Population): reference corpus to search n-grams in
            n (int): n for n-gram search

        Returns:
            list[dict]: list of tokens in form {"text": token, "author": "User", "queries": [..]}
        """

        text, authors, queries = zip(*self._get_text_query_author(edits))
        final_text = "".join(text)
        text_by_user = []

        tokens = self.tokenize_with_spans(final_text)
        user_token_indexes = []
        for i in range(len(tokens) - n_gram_len + 1):
            n_gram_start = tokens[i]["start"]
            n_gram_end = tokens[i+n_gram_len-1]["end"]
            n_gram = final_text[n_gram_start:n_gram_end]
            contains = (n_gram, n_gram_len) in reference_corpus
            if contains:
                for j in range(i, i+n_gram_len):
                    if j not in user_token_indexes:
                        text_by_user.append({
                            "text": tokens[j]["token"],
                            "author": "User",
                            "queries": []
                        })
                        user_token_indexes.append(j)
            else:
                if i not in user_token_indexes:
                    text_by_user.append({
                        "text": tokens[i]["token"],
                        "author": "Bot",
                        "queries": list(set([query for query in queries[n_gram_start:n_gram_end] if query]))
                    })

        return text_by_user
