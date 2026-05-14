import spacy
from spacy.tokens.token import Token
from nltk import Tree


class DependencyTree():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "attribute_ruler"])

    def to_nltk_tree(self, node: Token, words: dict):
        words[node.text] = {"POS": node.pos_, "DEP": node.dep_}
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [self.to_nltk_tree(child, words) for child in node.children])
        else:
            return node.orth_

    def get_tree_info(self, sent):
        words = {}
        tree = self.to_nltk_tree(sent.root, words)
        if isinstance(tree, str):
            return None
        return {
            "text": sent.text[:],
            "depth": tree.height(),
            "width": len(tree),
            "leaves": len(tree.leaves()),
            "words": words
        }

    def get_trees(self, text: str) -> list[dict[str, object]]:
        """
        For each sentence in text, get its dependency tree information:
        1. Max depth
        2. Width (# children of root)
        3. # Leaves (final width of the tree)
        4. Each word tagged with POS (ADV, NOUN) and dependency (ROOT, nsubj) tag

        Output format:
        [{
            "text": sent1,
            "depth": depth,
            "width": width,
            "leaves": #leaves,
            "words": {
                "is": {
                    "POS": "verb",
                    "DEP": "ROOT"
                }
            }
        }, ...]
        """
        doc = self.nlp(text)
        result = []
        for sent in doc.sents:
            tree_info =self.get_tree_info(sent)
            result.append(tree_info)
        del doc

        return result
    

