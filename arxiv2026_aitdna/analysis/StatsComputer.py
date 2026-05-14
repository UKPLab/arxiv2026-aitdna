import argparse
import json
from collections import Counter

import numpy as np
import spacy
import nltk
import readability
from nltk import Tree
from spacy.tokens.token import Token
from torch._C._jit_tree_views import Return
from tqdm import tqdm

from .DependencyTree import DependencyTree
from ..notions.data_loading import AitdDataset, Notion, DatasetName
from torch.utils.data import DataLoader

nltk.data.path.append("/storage/ukp/work/sakharova/nltk/models")
NLP = spacy.load("en_core_web_lg")

class StatsComputer():
    def __init__(self, dataset_name: DatasetName, dataset_root: str):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root


    def get_all_stats(self):
        return {
            "stats_per_sentence": self.get_avg_stats_per_sentence(),
            "stats_per_span": self.get_avg_stats_per_span(),
            "stats_per_user": self.get_avg_stats_per_user()
        }


    def get_avg_stats_per_sentence(self) -> dict[str, dict[str, float]]:
        """Computes average statistics per sentence. Includes:
        - Avg syntax tree depth, width, and number of leaves
        - Avg number of tokens and chars per sentence
        - Total number of sentences
        All stats are computed per author (User, Bot, Mixed)

        Returns:
            dict[str, dict[str, float]]: Computed stats
        """

        stats_per_sentence = {
            "avg_syntax_tree_depth": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            },
            "avg_syntax_tree_width": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            },
            "avg_syntax_tree_n_leaves": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            },
            "avg_n_tokens": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            },
            "avg_n_chars": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            },
            "total_sentences": {
                "User": 0,
                "Bot": 0,
                "Mixed": 0
            }
        }
        trees = self.get_linguistic_trees_all()
        for tree in trees:
            author = tree["author"]
            stats_per_sentence["avg_syntax_tree_depth"][author] += tree["depth"]
            stats_per_sentence["avg_syntax_tree_width"][author] += tree["width"]
            stats_per_sentence["avg_syntax_tree_n_leaves"][author] += tree["leaves"]
            stats_per_sentence["avg_n_tokens"][author] += len(tree["words"])
            stats_per_sentence["avg_n_chars"][author] += len(tree["text"])
            stats_per_sentence["total_sentences"][author] += 1

        for stat, users in stats_per_sentence.items():
            if stat == "total_sentences":
                continue
            for user, n in users.items():
                if stats_per_sentence["total_sentences"][user] != 0:
                    users[user] = round(n/stats_per_sentence["total_sentences"][user], 3)

        return stats_per_sentence

    def get_avg_stats_per_user(self) -> dict[str, dict[str, int]]:
        """Returns average statistics per user for the dataset. Includes:
        - Avg percentage of distinct lemmas
        - Total vocabulary size of a user
        - Average n distinct lemmas
        - POS and their count
        - Total data points per user

        Returns:
            dict[str, dict[str, int]]: _description_
        """
        stats_per_user = {
            "total_n_tokens": {
                "User": 0,
                "Bot": 0
            },
            "avg_n_tokens": {
                "User": 0,
                "Bot": 0
            },
            "total_vocab_size": {
                "User": 0,
                "Bot": 0
            },
            "avg_n_distinct_lemmas": {
                "User": 0,
                "Bot": 0
            },
            "pos": {
                "User": {},
                "Bot": {},
            },
            "avg_readability_scores": {
                    "flesch_reading_ease": {
                    "User": [],
                    "Bot": []
                },
                    "flesch_kincaid_grade_level": {
                    "User": [],
                    "Bot": []
                },
                    "gunning_fog": {
                    "User": [],
                    "Bot": []
                },
                    "dalle_chall": {
                    "User": [],
                    "Bot": []
                }
            },
            "total_n_stats": {
                "User": 0,
                "Bot": 0
            }
        }
        total_vocabs = {
            "User": set(),
            "Bot": set(),
        }
        n_lemmas = {
                "User": 0,
                "Bot": 0
        }

        all_stats = self.get_linguistic_stats_all()
        for text in all_stats:
            for author, stat in text.items():
                if stat["percentage_authorship"] == 0:
                    continue
                stats_per_user["total_n_stats"][author] += 1
                total_vocabs[author].update(list(set(stat["lemmas"])))

                n_lemmas[author] += len(stat["lemmas"])
                stats_per_user["total_n_tokens"][author] += stat["n_tokens"]

                for pos, value in stat["pos"].items():
                    if pos in stats_per_user["pos"][author]:
                        stats_per_user["pos"][author][pos] += value
                    else:
                        stats_per_user["pos"][author][pos] = value
                for metric, values in stat["readability_scores"].items():
                    stats_per_user["avg_readability_scores"][metric][author].extend(values)

        for author in ["User", "Bot"]:
            stats_per_user["total_vocab_size"][author] = len(total_vocabs[author])
            if stats_per_user["total_vocab_size"][author] == 0:
                stats_per_user["avg_readability_scores"][metric][author] = 0
                continue
            stats_per_user["avg_n_distinct_lemmas"][author] = stats_per_user["total_vocab_size"][author] / n_lemmas[author]
            stats_per_user["avg_n_tokens"][author] = stats_per_user["total_n_tokens"][author] / stats_per_user["total_n_stats"][author]
            for metric in stats_per_user["avg_readability_scores"]:
                stats_per_user["avg_readability_scores"][metric][author] = sum(stats_per_user["avg_readability_scores"][metric][author]) / len(stats_per_user["avg_readability_scores"][metric][author])
            for pos, value in stats_per_user["pos"][author].items():
                stats_per_user["pos"][author][pos] = value / stats_per_user["total_n_tokens"][author]
        return stats_per_user


    def get_avg_stats_per_span(self):
        """
        Returns average stats that are computed across spans.
        """
        return {
        # in one span, how many sentences/tokens?
        "avg_n_sentences_per_span": self.get_avg_n_sentences_per_span(),
        "avg_n_tokens_per_span": self.get_avg_n_tokens_per_span(),
        # for sentence-level notion, how often does the authorship change?
        "avg_n_boundaries_sentence_level": self.get_avg_n_boundaries_sentence_level(),
        # how many boundaries in span-level?
        "avg_n_boundaries_span_level": self.get_avg_n_boundaries_span_level(),
        }


    def get_n_texts(self) -> int:
        """Get total number of texts in the dataset

        Returns:
            int: # data points in dataset
        """
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.DOCUMENT_LEVEL)
        return len(dataset)


    def get_base_stats(self):
        texts = {
            "n_texts_total": 0,
            "n_texts_User": 0,
            "n_texts_Bot": 0,
            "n_texts_Mixed": 0,
            "avg_n_tokens": 0
        }
        total_n_tokens = 0
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.TOKEN_LEVEL)   
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        
        for batch in loader:
            for text in batch:
                texts["n_texts_total"] += 1
                authors = []
                for token in text:
                    if token["text"].strip() == "":
                        continue
                    total_n_tokens += 1
                    authors.append(token["author"])
                if len(set(authors)) > 1:
                    texts["n_texts_Mixed"] += 1
                else:
                    texts["n_texts_" + authors[0]] += 1
        texts["avg_n_tokens"] = total_n_tokens / texts["n_texts_total"]
        return texts

    def get_percentage_human_texts(self) -> float:
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SPAN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)

        n_total = len(dataset)
        n_human = 0
        for batch in loader:
            for text in batch:
                human = True
                for snippet in text:
                    if snippet["author"] == "Bot":
                        human = False
                        break
                if human:
                    n_human += 1
        return n_human / n_total

    def get_avg_ai_token_ratio_per_text(self) -> tuple[float,float]:
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.TOKEN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)

        ai_ratios = [
            sum(1 for token in text if token["author"] in ["Bot", "Mixed"]) / len(text) if len(text) > 0 else 0.0
            for batch in loader
            for text in batch
        ]

        return float(np.mean(ai_ratios)), float(np.std(ai_ratios))

    def get_linguistic_trees_all(self):
        """
        Generates trees for each sentence in text
        """
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SENTENCE_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)

        return self.get_linguistic_trees(loader)


    def get_linguistic_stats_all(self):
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SPAN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        all_stats = []
        for batch in loader:
            for text in batch:
                stats = self.get_linguistic_stats(text)
                all_stats.append(stats)
        return all_stats


    def get_avg_n_boundaries_sentence_level(self):
        n_boundaries = 0
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SENTENCE_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                n_boundaries += StatsComputer.get_n_boundaries_sentence_level(text)
        return n_boundaries / len(dataset)

    
    def get_avg_n_tokens_per_span(self):
        tokens_per_author = {
            "User": 0,
            "Bot": 0
        }
        total_spans = {
            "User": 0,
            "Bot": 0
        }
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SPAN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                tokens_per_span = StatsComputer.get_n_tokens_for_span(text)
                for author in ["User", "Bot"]:
                    tokens_per_author[author] += tokens_per_span[author]
                    total_spans[author] += len([span for span in text if span["author"] == author])

        for author in ["User", "Bot"]:
            if total_spans[author] == 0:
                continue
            tokens_per_author[author] = tokens_per_author[author] / total_spans[author]

        return tokens_per_author, (tokens_per_author["User"] + tokens_per_author["Bot"]) / (total_spans['User'] + total_spans['Bot'])


    def get_avg_n_sentences_per_span(self):
        avg_n_sents_per_span = {
            "User": 0,
            "Bot": 0
        }
        total_spans = {
            "User": 0,
            "Bot": 0
        }
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SPAN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                sents_per_span = StatsComputer.get_n_sents_for_span(text)
                for author in ["User", "Bot"]:
                    avg_n_sents_per_span[author] += sents_per_span[author]
                    total_spans[author] += len([span for span in text if span["author"] == author])
        
        for author in ["User", "Bot"]:
            if total_spans[author] == 0:
                continue
            avg_n_sents_per_span[author] = avg_n_sents_per_span[author] / total_spans[author]
        return avg_n_sents_per_span


    def get_avg_n_boundaries_span_level(self):
        n_boundaries = 0
        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.SPAN_LEVEL)
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                n_boundaries += StatsComputer.get_n_boundaries_span_level(text)
        return n_boundaries / len(dataset)


    def get_perc_ai_sent_tokens(self, no_human_only=False):
        ai_percentages = []
        user_percentages = []

        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.TOKEN_LEVEL, with_meta=self.dataset_name == DatasetName.AITDNA and no_human_only)
        #todo fix this for plot

        # discard human-only condition for analysis
        if self.dataset_name == DatasetName.AITDNA and no_human_only:
            filtered_data = [data for i, data in enumerate(dataset.data) if not dataset.meta[i].get("human_only", False)]
        else:
            filtered_data = dataset.data

        loader = DataLoader(filtered_data, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                ai_tokens = 0
                user_tokens = 0
                total_tokens = 0
                for token in text:
                    total_tokens += 1
                    if token["author"] == "Bot" or token["author"] == "Mixed":
                        ai_tokens += 1
                    else:
                        user_tokens += 1

                ai_percentages.append(ai_tokens / total_tokens * 100)
                user_percentages.append(user_tokens / total_tokens * 100)
        return {
            "Bot": ai_percentages,
            "User": user_percentages
        }

    def get_perc_ai_tokens(self, no_human_only=False):
        ai_percentages = []
        user_percentages = []

        dataset = AitdDataset(self.dataset_name, self.dataset_root,
                              Notion.TOKEN_LEVEL, with_meta=self.dataset_name == DatasetName.AITDNA and no_human_only)

        # discard human-only condition for analysis
        if self.dataset_name == DatasetName.AITDNA and no_human_only:
            filtered_data = [data for i, data in enumerate(dataset.data) if not dataset.meta[i].get("human_only", False)]
        else:
            filtered_data = dataset.data

        loader = DataLoader(filtered_data, batch_size=1, collate_fn=lambda data_point: data_point)
        for batch in loader:
            for text in batch:
                ai_tokens = 0
                user_tokens = 0
                total_tokens = 0
                for token in text:
                    total_tokens += 1
                    if token["author"] == "Bot" or token["author"] == "Mixed":
                        ai_tokens += 1
                    else:
                        user_tokens += 1

                ai_percentages.append(ai_tokens / total_tokens * 100)
                user_percentages.append(user_tokens / total_tokens * 100)
        return {
            "Bot": ai_percentages,
            "User": user_percentages
        }

    @staticmethod
    def get_linguistic_trees(batches: list[list[dict]]) -> list[dict]:
        """Generates linguistic trees for sentence-level text representation.

        Args:
            text (list[dict]): text in form: [{"text": "sentence 1", "author": "", "queries": ["", ..]}]

        Returns:
            _type_: list of trees
        """
        nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "attribute_ruler"])

        def to_nltk_tree(node: Token, words: dict):
            words[node.text] = {"POS": node.pos_, "DEP": node.dep_}
            children = list(node.children)
            if children:
                return Tree(node.orth_, [to_nltk_tree(child, words) for child in children])
            else:
                return node.orth_

        sentences = []
        authors = []
        for batch in batches:
            for text in batch:
                for sentence in text:
                    sentences.append(sentence["text"])
                    authors.append(sentence["author"])

        res = []
        for doc, author in tqdm(zip(nlp.pipe(sentences, batch_size=8), authors), desc="extracting syntax of sentences"):
            for sent in doc.sents:
                if not sent:
                    continue

                words = {}
                try:
                    tree = to_nltk_tree(sent.root, words)
                except RecursionError:
                    print("Error on parsing sentence as syntax tree", sent.text)
                    continue

                if isinstance(tree, str) or len(tree) == 0:
                    continue

                res.append({
                    "text": sent.text[:],
                    "depth": tree.height(),
                    "width": len(tree),
                    "leaves": len(tree.leaves()),
                    "words": words,
                    "author": author,
                })

        return res

    @staticmethod
    def get_pos_count_by_user(text: list[tuple[str, str, list[str]]]) -> dict[str, int]:
        """
        Get count of all POS in the text in format
        
        :param text: Text to identify POS in
        :returns dict[str, int]: dictionary of POS as key and #occurences as value
        """
        counters = {
            "User": Counter(),
            "Bot": Counter()
        }
        for data_point in text:
            text, author, _ = data_point["text"], data_point["author"], data_point["queries"]
            doc = NLP(text)
            counter = Counter([token.pos_ for token in doc])
            counters[author] += counter
        return {
            "User": dict(counters["User"].most_common()),
            "Bot": dict(counters["Bot"].most_common())
        }

    @staticmethod
    def get_lemma_counter_by_user(text: list[tuple[str, str, list[str]]]) -> dict[str, int]:
        """
        Get lemmas counter for each user
        
        :param text: Text to identify POS in
        :returns dict[str, int]: dictionary of POS as key and #occurences as value
        """
        lemmas = {
            "User": [],
            "Bot": []
        }
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for data_point in text:
            text, author, _ = data_point["text"], data_point["author"], data_point["queries"]
            tokens = nltk.word_tokenize(text)
            lemmas[author].extend(lemmatizer.lemmatize(t) for t in tokens)
        return lemmas

    @staticmethod
    def compute_text_percentage_by_user(text: list[tuple[str, str, list[str]]]):
        """
        Compute percentages of users usage
        """
        result = {"Bot": 0, "User": 0}
        total_len = 0
        for data_point in text:
            text, author, _ = data_point["text"], data_point["author"], data_point["queries"]
            result[author] += len(text)
            total_len += len(text)
        if total_len == 0:
            return {"Bot": 0, "User": 0}
        return {author: round(length / total_len, 4) for author, length in result.items()}

    @staticmethod
    def get_readability_scores(text):
        reading_ease = {
            "User": {
                "flesch_reading_ease": [],
                "flesch_kincaid_grade_level": [],
                "gunning_fog": [],
                "dalle_chall": []
            },
            "Bot": {
                "flesch_reading_ease": [],
                "flesch_kincaid_grade_level": [],
                "gunning_fog": [],
                "dalle_chall": []
            }
        }
        for snippet in text:
            text = snippet["text"]
            r = readability.Readability(text)
            try:
                reading_ease[snippet["author"]]["flesch_reading_ease"].append(r.flesch().score)
                reading_ease[snippet["author"]]["flesch_kincaid_grade_level"].append(r.flesch_kincaid().score)
                reading_ease[snippet["author"]]["gunning_fog"].append(r.gunning_fog().score)
                reading_ease[snippet["author"]]["dalle_chall"].append(r.dale_chall().score)
            except readability.exceptions.ReadabilityException:
                # occurs if # words in snippet < 100
                continue
        return reading_ease

    @staticmethod
    def compute_n_tokens(text):
        tokens = {
            "User": 0,
            "Bot": 0
        }
        for snippet in text:
            text = snippet["text"]
            tok = nltk.tokenize.word_tokenize(text)
            tokens[snippet["author"]] += len(tok)
        return tokens

    @staticmethod
    def get_linguistic_stats(text: list[dict]) -> dict[str, dict[dict, list]]:
        pos = StatsComputer.get_pos_count_by_user(text)
        lemmas = StatsComputer.get_lemma_counter_by_user(text)
        percentage_by_authorship = StatsComputer.compute_text_percentage_by_user(text)
        readability_scores = StatsComputer.get_readability_scores(text)
        tokens_by_user = StatsComputer.compute_n_tokens(text)

        return {
                "User": {
                    "pos": pos["User"],
                    "lemmas": lemmas["User"],
                    "percentage_authorship": percentage_by_authorship["User"],
                    "readability_scores": readability_scores["User"],
                    "n_tokens": tokens_by_user["User"]
                },
                "Bot": {
                    "pos": pos["Bot"],
                    "lemmas": lemmas["Bot"],
                    "percentage_authorship": percentage_by_authorship["Bot"],
                    "readability_scores": readability_scores["Bot"],
                    "n_tokens": tokens_by_user["Bot"]
                }
            }

    @staticmethod
    def get_n_boundaries_sentence_level(text):
        n_boundaries = 0
        current_author = None
        for sent in text:
            if not current_author:
                current_author = sent["author"]
            if current_author != sent["author"]:
                n_boundaries += 1
                current_author = sent["author"]
        return n_boundaries

    @staticmethod
    def get_n_boundaries_span_level(text):
        return len(text)

    @staticmethod
    def get_n_tokens_for_span(text):
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokens_by_author = {
            "User": 0,
            "Bot": 0
        }
        for snippet in text:
            tokens_by_author[snippet["author"]] += len(tokenizer.tokenize(snippet["text"]))
        return tokens_by_author

    @staticmethod
    def get_n_sents_for_span(text):
        sents_by_author = {
            "User": 0,
            "Bot": 0
        }
        for snippet in text:
            sents_by_author[snippet["author"]] += len(nltk.sent_tokenize(snippet["text"]))
        return sents_by_author

def main(argv):
    parser = argparse.ArgumentParser(description="Computes various statistics for a dataset.")
    parser.add_argument("-r", "--dataset_root", type=str, help="Path to dataset root")
    parser.add_argument("-n", "--dataset_name", type=str, help="Name of dataset to load", choices=set([d.value for d in DatasetName]))
    parser.add_argument("-d", "--dst_file_path", type=str, help="File to save the stats in")
    parser.add_argument("-b", "--base_stats", action="store_true", help="If specified, base stats are computed. Otherwise, detailed stats per user/sentence/span")
    argv = parser.parse_args(argv)
    dataset_root = argv.dataset_root
    dataset_name = argv.dataset_name
    base_stats = argv.base_stats
    if dataset_name not in DatasetName:
        raise ValueError(f"Dataset {dataset_name} not found!")
    dataset = DatasetName(dataset_name)

    dst_file_path = argv.dst_file_path
    stats_computer = StatsComputer(dataset_root=dataset_root, dataset_name=dataset)
    if base_stats:
        stats = stats_computer.get_base_stats()
    else:
        stats = stats_computer.get_all_stats()
    with open(dst_file_path, "w", encoding="utf-8") as f:
        json.dump(stats, f) 